import logging
import psycopg2.extras
from typing import Optional

from src.db.base import get_connection

logger = logging.getLogger(__name__)


def insert_email(
    gmail_message_id: str,
    gmail_thread_id: str,
    sender: str,
    subject: str,
    body: str,
    has_attachments: bool = False,
    received_at=None,
) -> int:
    sql = """
        INSERT INTO emails (gmail_message_id, gmail_thread_id, sender, subject, body, has_attachments, received_at)
        VALUES (%s, %s, %s, %s, %s, %s, COALESCE(%s, NOW() AT TIME ZONE 'Asia/Bangkok'))
        ON CONFLICT (gmail_message_id) DO UPDATE SET gmail_message_id = EXCLUDED.gmail_message_id
        RETURNING email_id
    """
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, (gmail_message_id, gmail_thread_id, sender, subject, body, has_attachments, received_at))
                return cur.fetchone()[0]
    finally:
        conn.close()


def get_email(email_id: int) -> Optional[dict]:
    sql = "SELECT * FROM emails WHERE email_id = %s"
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (email_id,))
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()


def fetch_pending_emails_for_retry() -> list[dict]:
    sql = """
        SELECT email_id, gmail_message_id, sender, subject, body, full_context, has_attachments
        FROM emails
        WHERE status = 'pending'
        ORDER BY email_id ASC
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
            cols = ["email_id", "gmail_message_id", "sender", "subject", "body", "full_context", "has_attachments"]
            return [dict(zip(cols, row)) for row in rows]
    finally:
        conn.close()


def get_processed_message_ids(limit: int = 500) -> set[str]:
    # fetch only the most recent N records to keep the set small
    sql = "SELECT gmail_message_id FROM emails ORDER BY email_id DESC LIMIT %s"
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (limit,))
            return {row[0] for row in cur.fetchall()}
    finally:
        conn.close()


def update_email_status(email_id: int, status: str) -> None:
    sql = "UPDATE emails SET status = %s WHERE email_id = %s"
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, (status, email_id))
    finally:
        conn.close()


def update_email_context(email_id: int, ocr_text: Optional[str], full_context: str) -> None:
    sql = "UPDATE emails SET ocr_text = %s, full_context = %s WHERE email_id = %s"
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, (ocr_text, full_context, email_id))
    finally:
        conn.close()


def insert_attachment_meta(
    email_id: int,
    file_index: int,
    original_filename: str,
    stored_path: str,
    mime_type: str,
    file_size_bytes: Optional[int] = None,
) -> int:
    sql = """
        INSERT INTO email_attachments
            (email_id, file_index, original_filename, stored_path, mime_type, file_size_bytes)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING attachment_id
    """
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, (email_id, file_index, original_filename, stored_path, mime_type, file_size_bytes))
                return cur.fetchone()[0]
    finally:
        conn.close()


def update_attachment_ocr(attachment_id: int, ocr_text: str) -> None:
    sql = "UPDATE email_attachments SET ocr_extracted = TRUE, ocr_text = %s WHERE attachment_id = %s"
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, (ocr_text, attachment_id))
    finally:
        conn.close()
