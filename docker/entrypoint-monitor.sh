#!/bin/bash
set -e

echo "[Monitor] Waiting for PostgreSQL..."
until python3 -c "from src.db.base import get_connection; get_connection()" 2>/dev/null; do
  sleep 2
done

echo "[Monitor] Waiting for MLflow..."
until curl -sf http://mlflow:5000/ > /dev/null 2>&1; do
  sleep 3
done

echo "[Monitor] Starting monitor flow..."
exec python3 training/monitor_flow.py
