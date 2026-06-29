"""
Model Rollback Utility
======================
ใช้งาน:
  python training/rollback.py --list
  python training/rollback.py --version 2
"""
import argparse
import logging
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlflow
from mlflow.tracking import MlflowClient

from src.config import BERT_MODEL_NAME, BERT_MODEL_PATH, MLFLOW_TRACKING_URI

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _client() -> MlflowClient:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return MlflowClient()


def list_versions() -> None:
    client = _client()
    versions = client.search_model_versions(f"name='{BERT_MODEL_NAME}'")
    if not versions:
        print(f"No registered versions for '{BERT_MODEL_NAME}'")
        return
    print(f"\n{'Ver':<6} {'Stage':<14} {'Run ID':<36} {'Data path'}")
    print("-" * 80)
    for v in sorted(versions, key=lambda x: int(x.version)):
        run = client.get_run(v.run_id)
        data_path = run.data.params.get("data_path", "-")
        f1 = run.data.metrics.get("macro_f1", None)
        f1_str = f"  F1={f1:.4f}" if f1 else ""
        print(f"v{v.version:<5} {v.current_stage:<14} {v.run_id}  {data_path}{f1_str}")
    print()


def rollback(version: int) -> None:
    client = _client()

    mv = client.get_model_version(BERT_MODEL_NAME, str(version))
    run_id = mv.run_id
    logger.info("Downloading model v%d from run %s", version, run_id)

    tmp_dir = f"models/_rollback_v{version}"
    local_path = mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{run_id}/model",
        dst_path=tmp_dir,
    )

    # Replace current serving model
    if os.path.exists(BERT_MODEL_PATH):
        shutil.rmtree(BERT_MODEL_PATH)
    shutil.copytree(local_path, BERT_MODEL_PATH)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    logger.info("Replaced %s with v%d artifacts", BERT_MODEL_PATH, version)

    # Promote in Registry
    client.transition_model_version_stage(
        name=BERT_MODEL_NAME,
        version=str(version),
        stage="Production",
        archive_existing_versions=True,
    )
    logger.info("v%d promoted to Production in MLflow Registry", version)

    # Reload singleton
    from src.classifier import reload_tagger
    reload_tagger()
    logger.info("Pipeline now uses model v%d", version)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BERT model rollback utility")
    parser.add_argument("--list", action="store_true", help="List all registered versions")
    parser.add_argument("--version", type=int, help="Rollback to specific version number")
    args = parser.parse_args()

    if args.list:
        list_versions()
    elif args.version:
        rollback(args.version)
    else:
        parser.print_help()
