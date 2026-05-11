import logging
import re
from typing import Optional, TypedDict

from langgraph.graph import StateGraph, END

from src.chains import generation_chain, review_chain, rag_eval_chain, llm_tagging_chain
from src.classifier_client import predict_tags
from src.gmail_client import download_attachments, send_reply, mark_as_read
from src.ocr import process_attachments
from src.rag import retrieve_documents, format_context
from src.db import (
    insert_email,
    update_email_status,
    update_email_context,
    insert_attachment_meta,
    insert_prediction,
    insert_generated_reply,
    insert_sent_log,
    update_prediction_llm_tags,
)
from src.config import MAX_REVIEW_RETRIES
from src.models import EmailData

logger = logging.getLogger(__name__)


def _strip_html(text: str) -> str:
    if not text or "<" not in text:
        return text
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&[a-z]+;", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class PipelineState(TypedDict):
    email: EmailData
    service: object
    db_email_id: Optional[int]
    full_context: str
    tags: list[str]
    bert_scores: dict
    product_query: str
    rag_query: str
    rag_context: str
    draft_subject: str
    draft_body: str
    review_action: str
    review_reason: str
    rag_refined_query: Optional[str]
    retry_count: int
    sent: bool
    status: str
    error: Optional[str]


def inspect_node(state: PipelineState) -> PipelineState:
    email = state["email"]
    service = state["service"]

    try:
        db_email_id = insert_email(
            gmail_message_id=email.gmail_message_id,
            gmail_thread_id=email.thread_id,
            sender=email.sender,
            subject=email.subject,
            body=email.body,
            has_attachments=email.has_attachments,
            received_at=email.received_at,
        )
        ocr_text = ""
        if email.has_attachments:
            attachments = download_attachments(service, email.gmail_message_id, db_email_id)
            for att in attachments:
                insert_attachment_meta(
                    email_id=db_email_id,
                    file_index=att.file_index,
                    original_filename=att.original_filename,
                    stored_path=att.stored_path,
                    mime_type=att.mime_type,
                    file_size_bytes=att.file_size_bytes,
                )
            ocr_text = process_attachments(attachments)

        clean_body = _strip_html(email.body)
        full_context = clean_body
        if ocr_text:
            full_context = f"{clean_body}\n\n[Attachments]\n{ocr_text}"

        update_email_context(db_email_id, ocr_text or None, full_context)

        return {**state, "db_email_id": db_email_id, "full_context": full_context}

    except Exception as e:
        logger.error("inspect_node failed: %s", e)
        return {**state, "status": "error", "error": str(e)}


def tag_node(state: PipelineState) -> PipelineState:
    try:
        result = predict_tags(state["full_context"])
        tags = result["tags"]
        scores = result["scores"]

        insert_prediction(
            email_id=state["db_email_id"],
            predicted_tags=tags,
            policy_score=scores.get("is_policy", 0.0),
            product_score=scores.get("is_product", 0.0),
            store_info_score=scores.get("is_store_info", 0.0),
            model_version=result.get("model_version"),
        )

        if not tags:
            logger.info("BERT returned no tags — trying LLM fallback tagging")
            llm_result = llm_tagging_chain(state["full_context"])
            llm_tags = llm_result.get("tags", [])
            if llm_tags:
                logger.info("LLM fallback tags: %s", llm_tags)
                update_prediction_llm_tags(state["db_email_id"], llm_tags)
                norm_llm_tags = ["is_" + t if not t.startswith("is_") else t for t in llm_tags]
                return {**state, "tags": norm_llm_tags, "product_query": llm_result.get("product_query", ""), "bert_scores": scores}
            update_email_status(state["db_email_id"], "pending")
            return {**state, "tags": [], "product_query": "", "bert_scores": scores, "status": "pending"}

        return {**state, "tags": tags, "product_query": result.get("product_query", ""), "bert_scores": scores}

    except Exception as e:
        logger.error("tag_node failed: %s", e)
        return {**state, "status": "error", "error": str(e)}


def _norm_tag(tag: str) -> str:
    return tag[3:] if tag.startswith("is_") else tag


BERT_NEAR_MISS_LOW = 0.3
BERT_NEAR_MISS_HIGH = 0.5


def _has_near_miss(bert_scores: dict) -> bool:
    return any(BERT_NEAR_MISS_LOW <= v < BERT_NEAR_MISS_HIGH for v in bert_scores.values())


def retrieve_node(state: PipelineState) -> PipelineState:
    try:
        norm_tags = [_norm_tag(t) for t in state["tags"]]
        product_query = state.get("product_query", "")
        query = state.get("rag_refined_query") or product_query or state["full_context"][:200]
        docs = retrieve_documents(query, norm_tags, product_query=product_query)
        rag_context = format_context(docs)

        if not state.get("rag_refined_query"):
            eval_result = rag_eval_chain(state["full_context"], query, rag_context)
            bert_scores = state.get("bert_scores", {})
            should_llm_retag = not eval_result["sufficient"] or _has_near_miss(bert_scores)

            if should_llm_retag:
                llm_result = llm_tagging_chain(state["full_context"])
                llm_tags = llm_result.get("tags", [])
                llm_query = llm_result.get("product_query", "") or eval_result.get("refined_query", "") or query
                llm_norm_tags = [_norm_tag(t) for t in llm_tags] if llm_tags else norm_tags
                reason = "RAG insufficient" if not eval_result["sufficient"] else "BERT near-miss"
                logger.info("LLM re-tag triggered (%s) — tags: %s, query: %s", reason, llm_tags, llm_query)
                docs = retrieve_documents(llm_query, llm_norm_tags, product_query=llm_result.get("product_query", ""))
                rag_context = format_context(docs)
                query = llm_query
                if state.get("db_email_id") and llm_tags:
                    update_prediction_llm_tags(state["db_email_id"], llm_tags)

        return {**state, "rag_context": rag_context, "rag_query": query, "rag_refined_query": None}

    except Exception as e:
        logger.error("retrieve_node failed: %s", e)
        return {**state, "status": "error", "error": str(e)}


def generate_node(state: PipelineState) -> PipelineState:
    try:
        reply = generation_chain(state["full_context"], state["rag_context"], state["tags"])
        return {
            **state,
            "draft_subject": reply.subject,
            "draft_body": reply.body,
            "retry_count": state.get("retry_count", 0) + 1,
        }
    except Exception as e:
        logger.error("generate_node failed: %s", e)
        return {**state, "status": "error", "error": str(e)}


def review_node(state: PipelineState) -> PipelineState:
    try:
        result = review_chain(state["full_context"], state["draft_body"], state["rag_context"])
        logger.info("Agent decision: action=%s reason=%s", result.action, result.reason)
        return {
            **state,
            "review_action": result.action,
            "review_reason": result.reason,
            "rag_refined_query": result.refined_query,
        }
    except Exception as e:
        logger.error("review_node failed: %s", e)
        return {**state, "status": "error", "error": str(e)}


def send_node(state: PipelineState) -> PipelineState:
    email = state["email"]
    service = state["service"]

    try:
        ok = send_reply(
            service,
            thread_id=email.thread_id,
            to=email.sender,
            subject=state["draft_subject"],
            body=state["draft_body"],
        )
        mark_as_read(service, email.gmail_message_id)

        insert_generated_reply(
            email_id=state["db_email_id"],
            draft_text=state["draft_body"],
            review_approved=state.get("review_action") == "send",
            review_feedback=state.get("review_reason"),
            needs_human_review=state.get("review_action") == "human_review",
            retry_count=state["retry_count"],
            final_text=state["draft_body"],
        )
        insert_sent_log(
            email_id=state["db_email_id"],
            recipient=email.sender,
            subject=state["draft_subject"],
            sent_status="sent" if ok else "failed",
        )
        update_email_status(state["db_email_id"], "replied" if ok else "send_failed")

        return {**state, "sent": ok, "status": "replied" if ok else "send_failed"}

    except Exception as e:
        logger.error("send_node failed: %s", e)
        return {**state, "status": "error", "error": str(e)}


def human_review_node(state: PipelineState) -> PipelineState:
    db_email_id = state.get("db_email_id")
    try:
        if db_email_id:
            update_email_status(db_email_id, "awaiting_review")
        logger.info(
            "Email %s flagged for human review. Reason: %s",
            db_email_id,
            state.get("review_reason"),
        )
        return {**state, "status": "awaiting_review"}
    except Exception as e:
        logger.error("human_review_node failed: %s", e)
        return {**state, "status": "error", "error": str(e)}


def log_node(state: PipelineState) -> PipelineState:
    db_email_id = state.get("db_email_id")
    status = state.get("status", "unknown")

    if db_email_id:
        try:
            update_email_status(db_email_id, status)
        except Exception as e:
            logger.error("log_node failed to update status: %s", e)

    logger.info("Pipeline finished for email %s with status: %s", db_email_id, status)
    return state


def route_after_tag(state: PipelineState) -> str:
    if state.get("status") in ("error", "pending"):
        return "log"
    return "retrieve"


def route_after_review(state: PipelineState) -> str:
    if state.get("status") == "error":
        return "log"
    action = state.get("review_action", "human_review")
    if action == "send":
        return "send"
    if action == "retry_generate" and state.get("retry_count", 0) < MAX_REVIEW_RETRIES:
        return "generate"
    if action == "retry_rag" and state.get("retry_count", 0) < MAX_REVIEW_RETRIES:
        return "retrieve"
    return "human_review"


def build_pipeline() -> StateGraph:
    graph = StateGraph(PipelineState)

    graph.add_node("inspect", inspect_node)
    graph.add_node("tag", tag_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("review", review_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("send", send_node)
    graph.add_node("log", log_node)

    graph.set_entry_point("inspect")
    graph.add_edge("inspect", "tag")
    graph.add_conditional_edges("tag", route_after_tag, {"log": "log", "retrieve": "retrieve"})
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "review")
    graph.add_conditional_edges(
        "review",
        route_after_review,
        {"send": "send", "generate": "generate", "retrieve": "retrieve", "log": "log", "human_review": "human_review"},
    )
    graph.add_edge("human_review", "log")
    graph.add_edge("send", "log")
    graph.add_edge("log", END)

    return graph.compile()
