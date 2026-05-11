#!/bin/bash
set -e

echo "[Pipeline] Preflight checks..."
for f in token.json credentials.json .env; do
  if [ ! -e "$f" ]; then
    echo "[Pipeline] ERROR: '$f' is missing. Please copy it to the project root on the host."
    exit 1
  fi
  if [ -d "$f" ]; then
    echo "[Pipeline] ERROR: '$f' is a directory (Docker created it). Run on host:"
    echo "  docker compose down"
    echo "  rm -rf $f"
    echo "  # then re-upload the real $f file and start again"
    exit 1
  fi
done
echo "[Pipeline] Preflight OK."

echo "[Pipeline] Waiting for PostgreSQL..."
until python3 -c "from src.db.base import get_connection; get_connection()" 2>/dev/null; do
  sleep 2
done
echo "[Pipeline] PostgreSQL ready."

echo "[Pipeline] Creating tables..."
python3 -c "from src.db.base import create_tables; create_tables()"

echo "[Pipeline] Checking knowledge base..."
python3 -c "
import chromadb, os
client = chromadb.PersistentClient(path=os.getenv('CHROMA_DB_PATH', './chroma_db'))
cols = client.list_collections()
count = cols[0].count() if cols else 0
print(f'[Pipeline] Chroma docs: {count}')
if count == 0:
    print('[Pipeline] Ingesting knowledge base...')
    import subprocess
    subprocess.run(['python3', 'scripts/ingest_knowledge_base.py'], check=True)
"

echo "[Pipeline] Starting pipeline..."
exec python3 scripts/run_pipeline.py --mode watch
