import logging
import os
from typing import Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

LABEL_COLS = ["is_policy", "is_product", "is_store_info"]
_instance: Optional["BERTTagger"] = None


def get_tagger() -> "BERTTagger":
    global _instance
    if _instance is None:
        from src.config import BERT_MODEL_PATH
        tagger = BERTTagger()
        local_config = os.path.join(BERT_MODEL_PATH, "config.json")
        if os.path.exists(local_config):
            try:
                tagger.load_from_path(BERT_MODEL_PATH)
            except Exception:
                logger.warning("Local model load failed — API running in no-model mode")
        else:
            try:
                tagger.load_from_mlflow()
            except Exception:
                logger.warning("No BERT model available — API running in no-model mode")
        _instance = tagger
    return _instance


def reload_tagger() -> "BERTTagger":
    global _instance
    _instance = None
    logger.info("BERTTagger singleton reset — reloading from disk")
    return get_tagger()


class BERTTagger:
    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.tokenizer = None
        self.model = None
        self.model_version: Optional[str] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_from_path(self, local_dir: str = "models/bert-email-tagger") -> "BERTTagger":
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(local_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(local_dir)
            self.model.to(self.device)
            self.model.eval()
            self.model_version = "local"
            logger.info("BERTTagger loaded from %s on %s", local_dir, self.device)
        except Exception:
            logger.exception("Failed to load BERTTagger from %s", local_dir)
            raise
        return self

    def load_from_mlflow(self, model_name: str = "bert-email-tagger", stage: str = "Production") -> "BERTTagger":
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
            from src.config import MLFLOW_REGISTRY_URI, MLFLOW_TRACKING_URI
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_registry_uri(MLFLOW_REGISTRY_URI)
            client = MlflowClient()
            versions = client.get_latest_versions(model_name, stages=[stage])
            if not versions:
                raise ValueError(f"No model version found for {model_name} stage={stage}")
            run_id = versions[0].run_id
            local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model")
            tokenizer_path = os.path.join(local_path, "components", "tokenizer")
            model_path = os.path.join(local_path, "model")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            self.model_version = f"{model_name}/v{versions[0].version}"
            logger.info("BERTTagger loaded from MLflow %s v%s on %s", model_name, versions[0].version, self.device)
        except Exception:
            logger.exception("MLflow load failed, falling back to local path")
            self.load_from_path()
        return self

    def predict(self, text: str) -> dict:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded - call load_from_path() or load_from_mlflow() first")

        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=256,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits.cpu().numpy()[0]

            probs = 1 / (1 + np.exp(-logits))
            tags = [label for label, prob in zip(LABEL_COLS, probs) if prob >= self.threshold]
            scores = {label: float(prob) for label, prob in zip(LABEL_COLS, probs)}

            return {
                "tags": tags,
                "scores": scores,
                "model_version": self.model_version,
            }
        except Exception:
            logger.exception("BERTTagger.predict failed for text: %.80s", text)
            raise
