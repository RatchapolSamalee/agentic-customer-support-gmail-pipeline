import glob
import json
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prefect import task

from src.db.base import get_connection

logger = logging.getLogger(__name__)

DATA_DIR = "training/data"
BASE_TAGS = ["policy", "product", "store_info"]


def _latest_version() -> int:
    files = glob.glob(os.path.join(DATA_DIR, "train_set_v*.json"))
    if not files:
        return 1
    nums = []
    for f in files:
        base = os.path.basename(f)
        try:
            nums.append(int(base.replace("train_set_v", "").replace(".json", "")))
        except ValueError:
            pass
    return max(nums) if nums else 1


def _load_existing(version: int) -> list[dict]:
    path = os.path.join(DATA_DIR, f"train_set_v{version}.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fetch_labeled_from_db() -> list[dict]:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT e.email_id, e.body, f.corrected_tags
                FROM feedback_logs f
                JOIN emails e ON f.email_id = e.email_id
                WHERE f.include_in_training = TRUE
                  AND f.corrected_tags IS NOT NULL
                ORDER BY e.email_id ASC
            """)
            rows = cur.fetchall()
    finally:
        conn.close()

    records = []
    for email_id, body, tags_json in rows:
        if not body:
            continue
        tags = tags_json if isinstance(tags_json, list) else (
            json.loads(tags_json) if tags_json else []
        )
        norm_tags = [t[3:] if t.startswith("is_") else t for t in tags]
        records.append({
            "_email_id": email_id,
            "text": body.strip(),
            "is_policy":     int("policy" in norm_tags),
            "is_product":    int("product" in norm_tags),
            "is_store_info": int("store_info" in norm_tags),
        })
    return records


@task(name="export_training_data")
def export_training_data() -> str:
    current_version = _latest_version()
    existing = _load_existing(current_version)
    existing_texts = {r["text"] for r in existing}
    max_id = max((r["id"] for r in existing if "id" in r), default=0)

    new_records = _fetch_labeled_from_db()
    added = 0
    for rec in new_records:
        if rec["text"] in existing_texts:
            continue
        max_id += 1
        existing.append({
            "id": max_id,
            "text": rec["text"],
            "is_policy":     rec["is_policy"],
            "is_product":    rec["is_product"],
            "is_store_info": rec["is_store_info"],
        })
        existing_texts.add(rec["text"])
        added += 1

    if added == 0:
        logger.info("No new labeled records to add — skipping export")
        return os.path.join(DATA_DIR, f"train_set_v{current_version}.json")

    new_version = current_version + 1
    out_path = os.path.join(DATA_DIR, f"train_set_v{new_version}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    logger.info(
        "Exported train_set_v%d.json — total %d records (+%d new)",
        new_version, len(existing), added,
    )
    return out_path
