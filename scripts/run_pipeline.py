import argparse
import logging
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.gmail_client import get_gmail_service, fetch_all_emails
from src.pipeline import build_pipeline
from src.db.emails import fetch_pending_emails_for_retry, update_email_status, get_processed_message_ids
from src.models import EmailData
from src.config import LOG_LEVEL

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

WATCH_INTERVAL_SECONDS = 30


def run_batch(service, pipeline) -> None:
    logger.info("Batch mode: fetching all emails from inbox")
    emails = fetch_all_emails(service)
    logger.info("Found %d emails", len(emails))

    results = {"replied": 0, "pending": 0, "error": 0}

    for email in emails:
        logger.info("Processing: [%s] %s", email.gmail_message_id, email.subject[:50])
        try:
            initial_state = {
                "email": email,
                "service": service,
                "db_email_id": None,
                "full_context": "",
                "tags": [],
                "bert_scores": {},
                "product_query": "",
                "rag_query": "",
                "rag_context": "",
                "draft_subject": "",
                "draft_body": "",
                "review_approved": False,
                "review_feedback": None,
                "needs_human_review": False,
                "retry_count": 0,
                "sent": False,
                "status": "processing",
                "error": None,
            }
            final_state = pipeline.invoke(initial_state)
            status = final_state.get("status", "unknown")
            results[status] = results.get(status, 0) + 1
            logger.info("Done: %s -> %s", email.gmail_message_id, status)

        except Exception as e:
            logger.error("Pipeline failed for %s: %s", email.gmail_message_id, e)
            results["error"] += 1

    logger.info("Batch complete. Results: %s", results)


def _make_state(email: EmailData, service, db_email_id=None, full_context="") -> dict:
    return {
        "email": email,
        "service": service,
        "db_email_id": db_email_id,
        "full_context": full_context,
        "tags": [],
        "bert_scores": {},
        "product_query": "",
        "rag_query": "",
        "rag_context": "",
        "rag_refined_query": None,
        "draft_subject": "",
        "draft_body": "",
        "review_action": "",
        "review_reason": "",
        "needs_human_review": False,
        "retry_count": 0,
        "sent": False,
        "status": "processing",
        "error": None,
    }


def _retry_pending(service, pipeline) -> None:
    pending = fetch_pending_emails_for_retry()
    if not pending:
        return
    logger.info("Retrying %d pending emails", len(pending))
    for row in pending:
        try:
            email = EmailData(
                gmail_message_id=row["gmail_message_id"],
                thread_id=row["gmail_message_id"],
                sender=row["sender"],
                subject=row["subject"],
                body=row["body"],
                has_attachments=row["has_attachments"],
            )
            update_email_status(row["email_id"], "processing")
            state = _make_state(email, service, db_email_id=row["email_id"], full_context=row["full_context"] or row["body"])
            pipeline.invoke(state)
            logger.info("Pending retry done: email_id=%d", row["email_id"])
        except Exception as e:
            logger.error("Pending retry failed email_id=%d: %s", row["email_id"], e)
            update_email_status(row["email_id"], "pending")


def run_watch(service, pipeline) -> None:
    logger.info("Watch mode: polling every %ds for new emails", WATCH_INTERVAL_SECONDS)

    while True:
        try:
            _retry_pending(service, pipeline)

            processed_ids = get_processed_message_ids()
            new_emails = fetch_all_emails(service, processed_ids=processed_ids)

            if new_emails:
                new_emails.sort(key=lambda e: e.received_at or datetime(2000, 1, 1))
                logger.info("Found %d unprocessed emails", len(new_emails))
                for i, email in enumerate(new_emails, 1):
                    logger.info("Starting email %d/%d: %s", i, len(new_emails), email.subject[:50])
                    pipeline.invoke(_make_state(email, service))
                    logger.info("Finished email %d/%d", i, len(new_emails))
            else:
                logger.debug("No new emails")

        except Exception as e:
            logger.error("Watch loop error: %s", e)

        time.sleep(WATCH_INTERVAL_SECONDS)


def main():
    parser = argparse.ArgumentParser(description="Run AutoGmaiVibe pipeline")
    parser.add_argument(
        "--mode",
        choices=["batch", "watch"],
        default="batch",
        help="batch: process all inbox emails | watch: poll for new emails",
    )
    args = parser.parse_args()

    service = get_gmail_service()
    pipeline = build_pipeline()

    if args.mode == "batch":
        run_batch(service, pipeline)
    else:
        run_watch(service, pipeline)


if __name__ == "__main__":
    main()
