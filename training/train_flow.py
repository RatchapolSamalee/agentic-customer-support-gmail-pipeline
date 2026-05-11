import json
import logging
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlflow
import numpy as np
import torch
from datasets import Dataset
from prefect import flow, task
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.config import (
    BERT_MODEL_PATH,
    BERT_MODEL_NAME,
    MLFLOW_TRACKING_URI,
)

logger = logging.getLogger(__name__)

MODEL_NAME = "airesearch/wangchanberta-base-att-spm-uncased"
LABEL_COLS = ["is_policy", "is_product", "is_store_info"]
MAX_LENGTH = 256
DATA_PATH = "training/data/train_set_v1.json"
OUTPUT_DIR = BERT_MODEL_PATH


def _latest_data_path() -> str:
    import glob as _glob
    files = sorted(_glob.glob("training/data/train_set_v*.json"))
    return files[-1] if files else DATA_PATH


def _compute_metrics(eval_pred, threshold: float = 0.5) -> dict:
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)

    tp = (preds * labels).sum(axis=0)
    fp = (preds * (1 - labels)).sum(axis=0)
    fn = ((1 - preds) * labels).sum(axis=0)

    precision = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
    recall = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
    f1_per_label = np.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        0.0,
    )
    return {
        "macro_f1": float(f1_per_label.mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
    }


@task(name="load_dataset")
def load_dataset(data_path: str) -> list[dict]:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Loaded %d samples from %s", len(data), data_path)
    return data


@task(name="prepare_splits")
def prepare_splits(data: list[dict]) -> tuple[Dataset, Dataset, Dataset]:
    texts = [d["text"] for d in data]
    labels = [
        [float(d.get("is_policy", 0)), float(d.get("is_product", 0)), float(d.get("is_store_info", 0))]
        for d in data
    ]

    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42
    )

    def _make_ds(t, l):
        return Dataset.from_dict({"text": t, "labels": l})

    return _make_ds(train_texts, train_labels), _make_ds(val_texts, val_labels), _make_ds(test_texts, test_labels)


@task(name="tokenize")
def tokenize(train_ds: Dataset, val_ds: Dataset, test_ds: Dataset) -> tuple[Dataset, Dataset, Dataset]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def _tokenize(batch):
        enc = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)
        enc["labels"] = [list(map(float, l)) for l in batch["labels"]]
        return enc

    cols_to_remove = ["text"]
    train_ds = train_ds.map(_tokenize, batched=True, remove_columns=cols_to_remove)
    val_ds = val_ds.map(_tokenize, batched=True, remove_columns=cols_to_remove)
    test_ds = test_ds.map(_tokenize, batched=True, remove_columns=cols_to_remove)

    for ds in (train_ds, val_ds, test_ds):
        ds.set_format("torch")

    return train_ds, val_ds, test_ds


@task(name="train_model")
def train_model(train_ds: Dataset, val_ds: Dataset) -> str:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_COLS),
        problem_type="multi_label_classification",
        id2label={i: l for i, l in enumerate(LABEL_COLS)},
        label2id={l: i for i, l in enumerate(LABEL_COLS)},
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        warmup_ratio=0.1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
        num_train_epochs=5,
        weight_decay=0.01,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=_compute_metrics,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    for ckpt in [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]:
        shutil.rmtree(os.path.join(OUTPUT_DIR, ckpt), ignore_errors=True)

    return OUTPUT_DIR


@task(name="evaluate_model")
def evaluate_model(test_ds: Dataset, model_dir: str) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        compute_metrics=_compute_metrics,
    )
    metrics = trainer.evaluate(test_ds)
    logger.info("Test metrics: %s", metrics)
    return metrics


def _compute_label_distribution(data_path: str) -> dict:
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        total = len(data)
        if total == 0:
            return {}
        dist = {}
        for col in LABEL_COLS:
            count = sum(1 for row in data if row.get(col, 0) == 1)
            dist[f"dist_{col}"] = count
            dist[f"dist_{col}_pct"] = round(count / total * 100, 1)
        dist["dist_none"] = sum(1 for row in data if all(row.get(c, 0) == 0 for c in LABEL_COLS))
        return dist
    except Exception:
        return {}


@task(name="log_to_mlflow")
def log_to_mlflow(metrics: dict, model_dir: str, data_path: str, labeled_count: int = 0) -> dict:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(BERT_MODEL_NAME)

    label_dist = _compute_label_distribution(data_path)

    import torch as _torch
    import transformers as _transformers
    import sklearn as _sklearn

    with mlflow.start_run() as run:
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("max_length", MAX_LENGTH)
        mlflow.log_param("labeled_count", labeled_count)
        mlflow.log_param("python_version", sys.version.split()[0])
        mlflow.log_param("torch_version", _torch.__version__)
        mlflow.log_param("transformers_version", _transformers.__version__)
        mlflow.log_param("sklearn_version", _sklearn.__version__)
        for k, v in label_dist.items():
            mlflow.log_param(k, v)
        mlflow.log_metrics({k.replace("eval_", ""): v for k, v in metrics.items() if isinstance(v, float)})
        mlflow.log_artifacts(model_dir, artifact_path="model")
        run_id = run.info.run_id
        logger.info("MLflow run logged: %s", run_id)

    client = mlflow.tracking.MlflowClient()
    artifact_uri = client.get_run(run_id).info.artifact_uri
    try:
        client.create_registered_model(BERT_MODEL_NAME)
    except Exception:
        pass
    mv = client.create_model_version(
        name=BERT_MODEL_NAME,
        source=f"{artifact_uri}/model",
        run_id=run_id,
    )
    logger.info("Registered %s version %s", BERT_MODEL_NAME, mv.version)

    meta = {
        "mlflow_run_id": run_id,
        "mlflow_model_version": mv.version,
        "metrics": metrics,
        "data_path": data_path,
    }
    with open(os.path.join(model_dir, "train_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return {"run_id": run_id, "model_version": mv.version}


@flow(name="bert-email-tagger-training")
def training_flow(data_path: str = "", labeled_count: int = 0) -> dict:
    if not data_path:
        data_path = _latest_data_path()
    data = load_dataset(data_path)
    train_ds, val_ds, test_ds = prepare_splits(data)
    train_ds, val_ds, test_ds = tokenize(train_ds, val_ds, test_ds)
    model_dir = train_model(train_ds, val_ds)
    metrics = evaluate_model(test_ds, model_dir)
    result = log_to_mlflow(metrics, model_dir, data_path, labeled_count)
    logger.info("Training flow complete. run=%s version=%s", result["run_id"], result["model_version"])
    return result


if __name__ == "__main__":
    training_flow()
