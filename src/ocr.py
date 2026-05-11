import base64
import logging
from pathlib import Path

from src.config import (
    GEMINI_API_KEY, GEMINI_VISION_MODEL,
)
from src.models import AttachmentMeta, OCRResult

logger = logging.getLogger(__name__)

_SUPPORTED_OCR_MIMES = {"image/jpeg", "image/png", "image/webp", "image/gif", "application/pdf"}
_PLAIN_TEXT_MIMES = {"text/plain", "text/csv", "application/json", "text/html"}
_OCR_PROMPT = "Extract all text from this document. Return only the extracted text, no commentary."


def should_ocr(filename: str, mime_type: str) -> bool:
    mime = mime_type.lower()
    if mime in _PLAIN_TEXT_MIMES:
        return False
    if mime in _SUPPORTED_OCR_MIMES:
        return True
    ext = Path(filename).suffix.lower()
    return ext in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".pdf"}


def _ocr_gemini(file_path: str, mime: str) -> str:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=GEMINI_API_KEY)
    with open(file_path, "rb") as f:
        data = f.read()
    response = client.models.generate_content(
        model=GEMINI_VISION_MODEL,
        contents=[
            types.Part.from_bytes(data=data, mime_type=mime),
            _OCR_PROMPT,
        ],
    )
    return response.text.strip() if response.text else ""


def _run_ocr(file_path: str, mime: str) -> str:
    return _ocr_gemini(file_path, mime)


def extract_text_from_image(image_path: str) -> OCRResult:
    ext = Path(image_path).suffix.lower().lstrip(".")
    mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp", "gif": "image/gif"}
    mime = mime_map.get(ext, "image/png")
    try:
        text = _run_ocr(image_path, mime)
        return OCRResult(file_path=image_path, extracted_text=text, success=bool(text))
    except Exception as e:
        logger.warning("OCR failed for %s: %s", image_path, e)
        return OCRResult(file_path=image_path, extracted_text="", success=False)


def extract_text_from_pdf(pdf_path: str) -> OCRResult:
    try:
        text = _run_ocr(pdf_path, "application/pdf")
        return OCRResult(file_path=pdf_path, extracted_text=text, success=bool(text))
    except Exception as e:
        logger.error("OCR failed for %s: %s", pdf_path, e)
        return OCRResult(file_path=pdf_path, extracted_text="", success=False)


def process_attachments(attachments: list[AttachmentMeta]) -> str:
    ocr_texts = []

    for att in attachments:
        if not should_ocr(att.original_filename, att.mime_type):
            logger.info("Skipping OCR for %s (%s)", att.original_filename, att.mime_type)
            continue

        mime = att.mime_type.lower()
        if mime == "application/pdf":
            result = extract_text_from_pdf(att.stored_path)
        else:
            result = extract_text_from_image(att.stored_path)

        if result.success and result.extracted_text:
            ocr_texts.append(f"[{att.original_filename}]\n{result.extracted_text}")

    return "\n\n".join(ocr_texts)
