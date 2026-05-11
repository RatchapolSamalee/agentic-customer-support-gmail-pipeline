import base64
import json
import logging
import threading
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel

from src.classifier import get_tagger, reload_tagger
from src.rag import retrieve_documents, format_context

logger = logging.getLogger(__name__)

app = FastAPI(title="Email Tagger API", version="2.0")


class TagRequest(BaseModel):
    text: str


class TagResponse(BaseModel):
    tags: list[str]
    scores: dict[str, float]
    model_version: Optional[str]


class RetrieveRequest(BaseModel):
    query: str
    tags: list[str]


class RetrieveResponse(BaseModel):
    context: str
    chunk_count: int


@app.get("/health")
def health() -> dict:
    tagger = get_tagger()
    return {"status": "ok", "model_version": tagger.model_version}


@app.post("/reload")
def reload_model() -> dict:
    try:
        reload_tagger()
        tagger = get_tagger()
        return {"status": "reloaded", "model_version": tagger.model_version}
    except Exception as e:
        logger.error("reload failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tag", response_model=TagResponse)
def tag_email(req: TagRequest) -> TagResponse:
    try:
        tagger = get_tagger()
        result = tagger.predict(req.text)
        return TagResponse(**result)
    except Exception as e:
        logger.error("Tag endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    try:
        docs = retrieve_documents(query=req.query, tags=req.tags)
        context = format_context(docs)
        return RetrieveResponse(context=context, chunk_count=len(docs))
    except Exception as e:
        logger.error("Retrieve endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


def _run_pipeline_for_message(history_id: str) -> None:
    try:
        from src.gmail_client import get_gmail_service, fetch_emails
        from src.pipeline import build_pipeline
        from src.models import EmailData

        logger.info("Pub/Sub trigger: historyId=%s", history_id)
        service = get_gmail_service()
        emails = fetch_emails(service, max_results=5)
        if not emails:
            logger.info("No unread emails found for historyId=%s", history_id)
            return

        pipeline = build_pipeline()
        for email in emails:
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
                "status": "pending",
                "error": None,
            }
            result = pipeline.invoke(initial_state)
            logger.info("Pub/Sub pipeline done: %s -> %s", email.gmail_message_id, result.get("status"))
    except Exception as e:
        logger.error("_run_pipeline_for_message failed: %s", e)


@app.post("/webhook/gmail")
async def gmail_webhook(request: Request, background_tasks: BackgroundTasks):
    """Gmail Pub/Sub push notification endpoint."""
    try:
        body = await request.json()
        message = body.get("message", {})
        data_b64 = message.get("data", "")
        if data_b64:
            data = json.loads(base64.b64decode(data_b64).decode("utf-8"))
            history_id = str(data.get("historyId", "unknown"))
        else:
            history_id = "unknown"

        logger.info("Gmail Pub/Sub received: historyId=%s", history_id)
        background_tasks.add_task(_run_pipeline_for_message, history_id)
        return {"status": "accepted"}
    except Exception as e:
        logger.error("gmail_webhook failed: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
