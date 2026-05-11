from datetime import datetime
from pydantic import BaseModel, Field
from typing import Literal, Optional


class ScreeningResult(BaseModel):
    is_related: bool
    reason: str = Field(min_length=1)


class TaggingResult(BaseModel):
    tags: list[str] = Field(default_factory=list)
    product_query: str
    confidence: Literal["low", "medium", "high"]


class GeneratedReply(BaseModel):
    subject: str
    body: str


class ReviewResult(BaseModel):
    action: str
    reason: str
    refined_query: Optional[str] = None


class AttachmentMeta(BaseModel):
    file_index: int
    original_filename: str
    stored_path: str
    mime_type: str
    file_size_bytes: Optional[int] = None
    ocr_extracted: bool = False
    ocr_text: Optional[str] = None


class OCRResult(BaseModel):
    file_path: str
    extracted_text: str
    success: bool


class EmailData(BaseModel):
    email_id: Optional[str] = None
    gmail_message_id: str
    sender: str
    subject: str
    body: str
    thread_id: str
    has_attachments: bool = False
    attachments: list[AttachmentMeta] = Field(default_factory=list)
    ocr_text: Optional[str] = None
    full_context: Optional[str] = None
    received_at: Optional[datetime] = None
