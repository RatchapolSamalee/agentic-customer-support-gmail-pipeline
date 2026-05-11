import json
import logging
from typing import Optional

from src.db.base import get_connection

logger = logging.getLogger(__name__)


def insert_screening(
    email_id: int,
    is_related: bool,
    reason: str,
    method: str = "llm",
    version: Optional[str] = None,
) -> int:
    sql = """
        INSERT INTO screening_results (email_id, is_related, reason, method, version)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING screening_id
    """
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, (email_id, is_related, reason, method, version))
                return cur.fetchone()[0]
    finally:
        conn.close()


def insert_prediction(
    email_id: int,
    predicted_tags: list,
    policy_score: float = 0.0,
    product_score: float = 0.0,
    store_info_score: float = 0.0,
    model_version: Optional[str] = None,
) -> int:
    sql = """
        INSERT INTO predictions
            (email_id, policy_score, product_score, store_info_score, predicted_tags, model_version)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING prediction_id
    """
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, (email_id, policy_score, product_score, store_info_score,
                                  json.dumps(predicted_tags), model_version))
                return cur.fetchone()[0]
    finally:
        conn.close()


def insert_generated_reply(
    email_id: int,
    draft_text: str,
    review_approved: bool,
    review_feedback: Optional[str],
    needs_human_review: bool = False,
    review_reason: Optional[str] = None,
    retry_count: int = 0,
    final_text: str = "",
    llm_model: Optional[str] = None,
    prompt_version: Optional[str] = None,
    retrieved_doc_count: int = 0,
) -> int:
    sql = """
        INSERT INTO generated_replies
            (email_id, draft_text, review_approved, review_feedback,
             needs_human_review, review_reason, retry_count,
             final_text, llm_model, prompt_version, retrieved_doc_count)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING reply_id
    """
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, (email_id, draft_text, review_approved, review_feedback,
                                  needs_human_review, review_reason, retry_count,
                                  final_text, llm_model, prompt_version, retrieved_doc_count))
                return cur.fetchone()[0]
    finally:
        conn.close()


def update_prediction_llm_tags(email_id: int, llm_tags: list) -> None:
    sql = "UPDATE predictions SET llm_refined_tags = %s WHERE email_id = %s"
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, (json.dumps(llm_tags), email_id))
    finally:
        conn.close()


def insert_sent_log(
    email_id: int,
    recipient: str,
    subject: str,
    sent_status: str,
    error_message: Optional[str] = None,
) -> int:
    sql = """
        INSERT INTO sent_email_logs (email_id, recipient, subject, sent_status, error_message)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING sent_id
    """
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, (email_id, recipient, subject, sent_status, error_message))
                return cur.fetchone()[0]
    finally:
        conn.close()
