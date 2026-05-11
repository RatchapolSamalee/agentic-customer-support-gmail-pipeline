import os
import re
import base64
import logging
from email.mime.text import MIMEText
from datetime import datetime
from typing import Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from src.config import (
    GMAIL_CREDENTIALS_FILE,
    GMAIL_TOKEN_FILE,
    GMAIL_SCOPES,
    GMAIL_FETCH_MAX_RESULTS,
    ATTACHMENTS_DIR,
)
from src.models import AttachmentMeta, EmailData

logger = logging.getLogger(__name__)


def get_gmail_service():
    creds = None

    if os.path.exists(GMAIL_TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(GMAIL_TOKEN_FILE, GMAIL_SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(GMAIL_CREDENTIALS_FILE, GMAIL_SCOPES)
            creds = flow.run_local_server(port=0)

        with open(GMAIL_TOKEN_FILE, "w") as token_file:
            token_file.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def _extract_body(payload: dict) -> str:
    # Try plain text part first, then fall back to full body data
    parts = payload.get("parts", [])
    for part in parts:
        if part.get("mimeType") == "text/plain":
            data = part.get("body", {}).get("data", "")
            if data:
                return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")

    body_data = payload.get("body", {}).get("data", "")
    if body_data:
        return base64.urlsafe_b64decode(body_data).decode("utf-8", errors="replace")

    return ""


def _extract_attachment_meta(parts: list) -> list[dict]:
    # Collect attachment metadata from all parts recursively
    attachments = []
    for part in parts:
        if part.get("filename") and part["body"].get("attachmentId"):
            attachments.append({
                "filename": part["filename"],
                "mime_type": part.get("mimeType", "application/octet-stream"),
                "attachment_id": part["body"]["attachmentId"],
                "size": part["body"].get("size", 0),
            })
        if part.get("parts"):
            attachments.extend(_extract_attachment_meta(part["parts"]))
    return attachments


def _parse_message(msg: dict) -> Optional[EmailData]:
    payload = msg.get("payload", {})
    headers = payload.get("headers", [])

    header_map = {h["name"].lower(): h["value"] for h in headers}
    subject = header_map.get("subject", "")
    sender = header_map.get("from", "")

    body = _extract_body(payload)

    all_parts = payload.get("parts", [])
    raw_attachments = _extract_attachment_meta(all_parts)

    attachment_meta_list = [
        AttachmentMeta(
            file_index=i + 1,
            original_filename=att["filename"],
            stored_path="",
            mime_type=att["mime_type"],
            file_size_bytes=att["size"],
        )
        for i, att in enumerate(raw_attachments)
    ]

    internal_date_ms = msg.get("internalDate")
    received_at = None
    if internal_date_ms:
        from datetime import datetime, timezone, timedelta
        utc_dt = datetime.fromtimestamp(int(internal_date_ms) / 1000, tz=timezone.utc)
        received_at = utc_dt.astimezone(timezone(timedelta(hours=7))).replace(tzinfo=None)

    return EmailData(
        gmail_message_id=msg["id"],
        sender=sender,
        subject=subject,
        body=body,
        thread_id=msg.get("threadId", ""),
        has_attachments=len(attachment_meta_list) > 0,
        attachments=attachment_meta_list,
        received_at=received_at,
    )


def fetch_all_emails(
    service,
    max_results: int = GMAIL_FETCH_MAX_RESULTS,
    processed_ids: set | None = None,
) -> list[EmailData]:
    emails = []
    next_page_token = None
    processed_ids = processed_ids or set()

    try:
        while True:
            response = service.users().messages().list(
                userId="me",
                labelIds=["INBOX"],
                maxResults=min(max_results - len(emails), 100),
                pageToken=next_page_token,
            ).execute()

            messages = response.get("messages", [])
            if not messages:
                break

            for msg_ref in messages:
                if msg_ref["id"] in processed_ids:
                    continue

                full_msg = service.users().messages().get(
                    userId="me", id=msg_ref["id"], format="full"
                ).execute()

                email_data = _parse_message(full_msg)
                if email_data:
                    emails.append(email_data)

                if len(emails) >= max_results:
                    break

            next_page_token = response.get("nextPageToken")
            if not next_page_token or len(emails) >= max_results:
                break

    except HttpError as e:
        logger.error("Failed to fetch emails: %s", e)

    logger.info("Fetched %d new emails from inbox", len(emails))
    return emails


def fetch_new_emails(service, after_timestamp: Optional[int] = None) -> list[EmailData]:
    # Trigger mode: fetch only emails newer than after_timestamp (Unix seconds)
    query = "label:INBOX"
    if after_timestamp:
        query += f" after:{after_timestamp}"

    emails = []
    try:
        response = service.users().messages().list(
            userId="me", q=query, maxResults=50
        ).execute()

        for msg_ref in response.get("messages", []):
            full_msg = service.users().messages().get(
                userId="me", id=msg_ref["id"], format="full"
            ).execute()
            email_data = _parse_message(full_msg)
            if email_data:
                emails.append(email_data)

    except HttpError as e:
        logger.error("Failed to fetch new emails: %s", e)

    return emails


def download_attachments(service, message_id: str, email_id: int) -> list[AttachmentMeta]:
    if not os.path.exists(ATTACHMENTS_DIR):
        os.makedirs(ATTACHMENTS_DIR)

    full_msg = service.users().messages().get(
        userId="me", id=message_id, format="full"
    ).execute()

    all_parts = full_msg.get("payload", {}).get("parts", [])
    raw_attachments = _extract_attachment_meta(all_parts)

    saved = []
    for i, att in enumerate(raw_attachments):
        ext = os.path.splitext(att["filename"])[1] or ""
        stored_filename = f"email_{email_id}_file_{i + 1}{ext}"
        stored_path = os.path.join(ATTACHMENTS_DIR, stored_filename)

        try:
            attachment = service.users().messages().attachments().get(
                userId="me", messageId=message_id, id=att["attachment_id"]
            ).execute()

            file_data = base64.urlsafe_b64decode(attachment["data"])
            with open(stored_path, "wb") as f:
                f.write(file_data)

            saved.append(AttachmentMeta(
                file_index=i + 1,
                original_filename=att["filename"],
                stored_path=stored_path,
                mime_type=att["mime_type"],
                file_size_bytes=len(file_data),
            ))
            logger.info("Downloaded attachment: %s -> %s", att["filename"], stored_path)

        except HttpError as e:
            logger.error("Failed to download attachment %s: %s", att["filename"], e)

    return saved


def send_reply(
    service,
    thread_id: str,
    to: str,
    subject: str,
    body: str,
    in_reply_to: Optional[str] = None,
) -> bool:
    try:
        _match = re.search(r'<([^>]+)>', to)
        to_addr = _match.group(1) if _match else to.strip()
        mime_msg = MIMEText(body, "plain", "utf-8")
        mime_msg["To"] = to_addr
        mime_msg["Subject"] = subject
        if in_reply_to:
            mime_msg["In-Reply-To"] = in_reply_to
            mime_msg["References"] = in_reply_to

        raw = base64.urlsafe_b64encode(mime_msg.as_bytes()).decode()
        message = {"raw": raw, "threadId": thread_id}

        service.users().messages().send(userId="me", body=message).execute()
        logger.info("Reply sent to %s in thread %s", to, thread_id)
        return True

    except HttpError as e:
        logger.error("Failed to send reply: %s", e)
        return False


def mark_as_read(service, message_id: str) -> None:
    try:
        service.users().messages().modify(
            userId="me",
            id=message_id,
            body={"removeLabelIds": ["UNREAD"]},
        ).execute()
        logger.info("Marked message %s as read", message_id)
    except HttpError as e:
        logger.error("Failed to mark message as read: %s", e)
