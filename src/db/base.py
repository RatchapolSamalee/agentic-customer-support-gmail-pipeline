import logging
import psycopg2
from psycopg2.extensions import connection as PgConnection

from src.config import DATABASE_URL

logger = logging.getLogger(__name__)


def get_connection() -> PgConnection:
    return psycopg2.connect(DATABASE_URL)


def create_tables() -> None:
    sql_statements = [
        """
        CREATE TABLE IF NOT EXISTS emails (
            email_id         SERIAL PRIMARY KEY,
            gmail_message_id VARCHAR(255) UNIQUE NOT NULL,
            gmail_thread_id  VARCHAR(255),
            sender           VARCHAR(255),
            subject          TEXT,
            body             TEXT,
            has_attachments  BOOLEAN DEFAULT FALSE,
            ocr_text         TEXT,
            full_context     TEXT,
            received_at      TIMESTAMP DEFAULT NOW(),
            status           VARCHAR(50) DEFAULT 'received'
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS email_attachments (
            attachment_id     SERIAL PRIMARY KEY,
            email_id          INTEGER REFERENCES emails(email_id),
            file_index        INTEGER,
            original_filename VARCHAR(255),
            stored_path       TEXT,
            mime_type         VARCHAR(100),
            file_size_bytes   INTEGER,
            ocr_extracted     BOOLEAN DEFAULT FALSE,
            ocr_text          TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id    SERIAL PRIMARY KEY,
            email_id         INTEGER REFERENCES emails(email_id),
            policy_score     FLOAT,
            product_score    FLOAT,
            store_info_score FLOAT,
            predicted_tags   JSONB,
            model_version    VARCHAR(50)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS retrieved_documents (
            retrieval_id     SERIAL PRIMARY KEY,
            email_id         INTEGER REFERENCES emails(email_id),
            document_path    TEXT,
            retrieval_method VARCHAR(50),
            similarity_score FLOAT,
            selected_by_tag  VARCHAR(50)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS generated_replies (
            reply_id             SERIAL PRIMARY KEY,
            email_id             INTEGER REFERENCES emails(email_id),
            draft_text           TEXT,
            review_approved      BOOLEAN,
            review_feedback      TEXT,
            needs_human_review   BOOLEAN DEFAULT FALSE,
            review_reason        TEXT,
            retry_count          INTEGER DEFAULT 0,
            final_text           TEXT,
            llm_model            VARCHAR(100),
            prompt_version       VARCHAR(50),
            retrieved_doc_count  INTEGER DEFAULT 0
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS sent_email_logs (
            sent_id       SERIAL PRIMARY KEY,
            email_id      INTEGER REFERENCES emails(email_id),
            recipient     VARCHAR(255),
            subject       TEXT,
            sent_status   VARCHAR(20),
            error_message TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS feedback_logs (
            feedback_id    SERIAL PRIMARY KEY,
            email_id       INTEGER REFERENCES emails(email_id),
            prediction_id  INTEGER,
            predicted_tags JSONB,
            corrected_tags JSONB,
            action         VARCHAR(20),
            is_correct     BOOLEAN,
            reviewer       VARCHAR(100),
            created_at     TIMESTAMP DEFAULT NOW()
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS benchmark_results (
            result_id        SERIAL PRIMARY KEY,
            question_id      INTEGER,
            predicted_answer INTEGER,
            model_version    VARCHAR(50),
            is_correct       BOOLEAN,
            run_date         TIMESTAMP DEFAULT NOW()
        )
        """,
    ]

    migrations = [
        "ALTER TABLE feedback_logs ADD COLUMN IF NOT EXISTS include_in_training BOOLEAN DEFAULT FALSE",
        "ALTER TABLE emails ADD COLUMN IF NOT EXISTS manual_replied BOOLEAN",
        "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS llm_refined_tags JSONB",
    ]

    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                for sql in sql_statements:
                    cur.execute(sql)
                for sql in migrations:
                    cur.execute(sql)
        logger.info("All tables created successfully")
    except Exception as e:
        logger.error("Failed to create tables: %s", e)
        raise
    finally:
        conn.close()
