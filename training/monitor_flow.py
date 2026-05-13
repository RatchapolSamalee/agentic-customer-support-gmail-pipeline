import logging
import os
import sys

import httpx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlflow
from mlflow.tracking import MlflowClient
from prefect import flow, task

from src.config import BERT_MODEL_NAME, MLFLOW_TRACKING_URI
from src.db.base import get_connection
from training.export_task import export_training_data
from training.train_flow import training_flow

logger = logging.getLogger(__name__)

TRAIN_THRESHOLD = int(os.getenv("TRAIN_THRESHOLD", "50"))


def _get_last_trained_count() -> int:
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{BERT_MODEL_NAME}'")
        if not versions:
            return 0
        latest = max(versions, key=lambda v: int(v.version))
        run = client.get_run(latest.run_id)
        val = run.data.params.get("labeled_count", "0")
        return int(val)
    except Exception:
        return 0


@task(name="count_labeled")
def count_labeled() -> int:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM feedback_logs WHERE include_in_training = TRUE"
            )
            count = cur.fetchone()[0]
        logger.info("Labeled count: %d / %d threshold", count, TRAIN_THRESHOLD)
        return count
    finally:
        conn.close()


@task(name="promote_to_production")
def promote_to_production(model_version: str) -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    client.transition_model_version_stage(
        name=BERT_MODEL_NAME,
        version=str(model_version),
        stage="Production",
        archive_existing_versions=True,
    )
    logger.info("Model %s v%s promoted to Production", BERT_MODEL_NAME, model_version)


@task(name="reload_model")
def reload_model() -> None:
    bert_api_url = os.getenv("BERT_API_URL", "http://localhost:8002")
    try:
        resp = httpx.post(f"{bert_api_url}/reload", timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
        logger.info("BERT API reloaded: %s", data)
    except Exception as e:
        logger.warning("BERT API /reload failed (%s) -- skipping hot-reload", e)


@flow(name="monitor-and-train")
def monitor_flow() -> None:
    count = count_labeled()
    last_trained_count = _get_last_trained_count()

    if count < TRAIN_THRESHOLD:
        logger.info("Threshold not reached (%d/%d) -- skipping", count, TRAIN_THRESHOLD)
        return

    if count < last_trained_count + TRAIN_THRESHOLD:
        logger.info("Not enough new data (%d/%d needed) -- skipping", count, last_trained_count + TRAIN_THRESHOLD)
        return

    logger.info("New labeled data sufficient (%d >= %d) -- export + train", count, last_trained_count + TRAIN_THRESHOLD)

    data_path = export_training_data()
    result = training_flow(data_path=data_path, labeled_count=count)
    promote_to_production(model_version=result["model_version"])
    reload_model()

    logger.info("monitor_flow complete -- v%s now in Production", result["model_version"])


if __name__ == "__main__":
    import time
    INTERVAL = int(os.getenv("MONITOR_INTERVAL_SEC", "3600"))
    while True:
        try:
            monitor_flow()
        except Exception as e:
            logger.error("monitor_flow error: %s", e)
        time.sleep(INTERVAL)
