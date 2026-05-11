import json
import logging
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from prefect import flow, task
from sklearn.metrics import classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from mlops.promotion import should_promote
from src.config import BERT_MODEL_NAME, BERT_THRESHOLD, MLFLOW_TRACKING_URI

logger = logging.getLogger(__name__)

LABELS = ["is_policy", "is_product", "is_store_info"]
BASE_MODEL = "airesearch/wangchanberta-base-att-spm-uncased"
DATASET_PATH = "training/data/train_set.json"


class EmailDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


@task
def verify_data(dataset_path: str) -> dict:
    data = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    total = len(data)
    stats = {"total": total}
    for label in LABELS:
        pos = sum(1 for d in data if d.get(label) == 1)
        stats[f"pos_{label}"] = pos
    logger.info("Dataset verified: %s", stats)
    return stats


@task
def train_model(dataset_path: str, params: dict) -> tuple[object, object, dict]:
    data = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    np.random.shuffle(data)
    n = len(data)
    train_data = data[:int(n * 0.8)]
    val_data = data[int(n * 0.8):int(n * 0.9)]
    test_data = data[int(n * 0.9):]

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def encode(items):
        return tokenizer(
            [d["text"] for d in items],
            truncation=True, padding="max_length",
            max_length=params.get("max_length", 256),
        )

    def get_labels(items):
        return [[d.get(l, 0) for l in LABELS] for d in items]

    train_ds = EmailDataset(encode(train_data), get_labels(train_data))
    val_ds = EmailDataset(encode(val_data), get_labels(val_data))
    test_ds = EmailDataset(encode(test_data), get_labels(test_data))

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
    )

    training_args = TrainingArguments(
        output_dir="./mlops/tmp_bert",
        num_train_epochs=params.get("epochs", 5),
        per_device_train_batch_size=params.get("batch_size", 16),
        per_device_eval_batch_size=params.get("batch_size", 16),
        learning_rate=params.get("lr", 3e-5),
        warmup_ratio=0.1,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds)
    trainer.train()

    preds = torch.sigmoid(torch.tensor(trainer.predict(test_ds).predictions))
    pred_labels = (preds > BERT_THRESHOLD).int().numpy()
    true_labels = np.array(get_labels(test_data))

    metrics = {}
    for i, label in enumerate(LABELS):
        f1 = f1_score(true_labels[:, i], pred_labels[:, i], zero_division=0)
        metrics[f"f1_{label}"] = round(f1, 4)
    metrics["macro_f1"] = round(float(np.mean(list(metrics.values()))), 4)

    return model, tokenizer, metrics


@task
def log_to_mlflow(model, tokenizer, params: dict, metrics: dict, stats: dict) -> str:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("bert-email-tagger")

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_metrics({f"dataset_{k}": v for k, v in stats.items()})
        mlflow.pytorch.log_model(model, "model")
        tokenizer.save_pretrained("./mlops/tmp_tokenizer")
        mlflow.log_artifacts("./mlops/tmp_tokenizer", artifact_path="tokenizer")
        return run.info.run_id


@task
def register_and_promote(run_id: str, metrics: dict) -> Optional[str]:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri, BERT_MODEL_NAME)
    version = mv.version

    try:
        prod_versions = client.get_latest_versions(BERT_MODEL_NAME, stages=["Production"])
        prod_metrics = None
        if prod_versions:
            prod_run = client.get_run(prod_versions[0].run_id)
            prod_metrics = prod_run.data.metrics
    except Exception:
        prod_metrics = None

    if should_promote(metrics, prod_metrics):
        client.transition_model_version_stage(BERT_MODEL_NAME, version, "Production", archive_existing_versions=True)
        logger.info("Model version %s promoted to Production", version)
        return version
    return None


@task
def notify_bert_api(reload_url: str) -> None:
    import httpx
    try:
        resp = httpx.post(reload_url, timeout=10)
        logger.info("BERT API reload: %s", resp.status_code)
    except Exception as e:
        logger.warning("BERT API reload failed: %s", e)


@flow(name="train-bert")
def train_bert_flow(
    dataset_path: str = DATASET_PATH,
    bert_api_reload_url: str = "http://localhost:8002/reload",
):
    params = {"lr": 3e-5, "epochs": 5, "batch_size": 16, "max_length": 256, "threshold": BERT_THRESHOLD}

    stats = verify_data(dataset_path)
    model, tokenizer, metrics = train_model(dataset_path, params)
    run_id = log_to_mlflow(model, tokenizer, params, metrics, stats)
    version = register_and_promote(run_id, metrics)

    if version:
        notify_bert_api(bert_api_reload_url)

    logger.info("train_bert_flow done. run_id=%s promoted_version=%s metrics=%s", run_id, version, metrics)


if __name__ == "__main__":
    train_bert_flow()
