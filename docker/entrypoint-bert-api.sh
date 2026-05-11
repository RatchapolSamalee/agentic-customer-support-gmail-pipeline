#!/bin/bash
set -e

echo "[BERT API] Waiting for PostgreSQL..."
until python3 -c "from src.db.base import get_connection; get_connection()" 2>/dev/null; do
  sleep 2
done

echo "[BERT API] Creating tables..."
python3 -c "from src.db.base import create_tables; create_tables()"

echo "[BERT API] Waiting for MLflow..."
until curl -sf http://mlflow:5000/ > /dev/null 2>&1; do
  sleep 3
done

echo "[BERT API] Checking model..."
python3 -c "
import mlflow
from mlflow.tracking import MlflowClient
import os, sys

mlflow.set_tracking_uri('http://mlflow:5000')
client = MlflowClient()

try:
    versions = client.get_latest_versions('bert-email-tagger', stages=['Production'])
    if versions:
        print('[BERT API] Model found in MLflow Production stage.')
        sys.exit(0)
except Exception:
    pass

if os.path.exists('models/bert-email-tagger/config.json'):
    print('[BERT API] Local model found.')
    sys.exit(0)

print('[BERT API] No model found — starting initial training...')
import subprocess
result = subprocess.run(['python3', 'training/train_flow.py'], capture_output=False)
if result.returncode != 0:
    print('[BERT API] Training failed — API will start without model.')
else:
    print('[BERT API] Training complete.')
"

echo "[BERT API] Starting BERT API..."
exec python3 -m uvicorn src.api:app --host 0.0.0.0 --port 8002
