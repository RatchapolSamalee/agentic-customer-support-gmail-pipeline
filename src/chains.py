import logging
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from src.config import (
    GEMINI_API_KEY, GEMINI_MODEL,
    SCREENING_LLM_PROVIDER, SCREENING_MODEL,
    RAG_EVAL_LLM_PROVIDER, RAG_EVAL_MODEL,
    GENERATION_LLM_PROVIDER, GENERATION_MODEL,
    REVIEW_LLM_PROVIDER, REVIEW_MODEL,
)
from src.models import ScreeningResult, TaggingResult, GeneratedReply, ReviewResult

logger = logging.getLogger(__name__)

SCREENING_PROMPT = """คุณเป็น AI คัดกรองอีเมลของร้านฟ้าใหม่ (FahMai) ร้านขายอิเล็กทรอนิกส์ออนไลน์ของไทย

ตอบคำถามเดียว: อีเมลนี้เกี่ยวข้องกับร้านฟ้าใหม่ไหม?
ตอบเป็น JSON เท่านั้น ห้ามมีข้อความอื่น:
{{
  "is_related": true/false,
  "reason": "เหตุผลสั้นๆ"
}}

กฎ:
- is_related: true ถ้าอีเมลเกี่ยวข้องกับสินค้า/นโยบาย/การบริการ/ร้านฟ้าใหม่ ไม่ว่าจะเป็นคำถาม ร้องเรียน หรือขอความช่วยเหลือ
- is_related: false ถ้าไม่เกี่ยวกับฟ้าใหม่เลย เช่น spam โฆษณา สมัครงาน หรือคำถามทั่วไปที่ไม่เกี่ยวกับร้าน
- reason ต้องไม่ว่าง

อีเมล:
{email_content}"""

TAGGING_PROMPT = """จัดหมวดหมู่อีเมลลูกค้าร้านฟ้าใหม่ ตอบเป็น JSON เท่านั้น ห้ามมีข้อความอื่น:
{{
  "tags": ["policy", "product", "store_info"],
  "product_query": "คำค้นสั้นๆ สำหรับค้นหาสินค้า (ถ้าไม่มี tag product ให้ใส่ string ว่าง)",
  "confidence": "low/medium/high"
}}

tag ที่เลือกได้ (เลือกได้มากกว่า 1):
- policy: ถามเรื่องนโยบาย คืนสินค้า ประกัน Care+ จัดส่ง ยกเลิก สมาชิก Points
- product: ถามเรื่องสินค้า สเปค ราคา เปรียบเทียบ ความเข้ากันได้ แนะนำสินค้า
- store_info: ถามเรื่องร้าน สาขา เวลาเปิด ติดต่อ วิธีสั่งซื้อ ช่องทางชำระเงิน

อีเมล:
{email_content}"""


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class _ThinkingStripper:
    def invoke(self, message):
        content = message.content if hasattr(message, "content") else str(message)
        message.content = _strip_thinking(content)
        return message

    def __or__(self, other):
        from langchain_core.runnables import RunnableLambda
        return RunnableLambda(lambda x: self.invoke(x)) | other


def _json_parser():
    from langchain_core.runnables import RunnableLambda
    stripper = RunnableLambda(lambda msg: (
        setattr(msg, "content", _strip_thinking(msg.content)) or msg
        if hasattr(msg, "content") else msg
    ))
    return stripper | JsonOutputParser()


_CHAIN_CONFIG = {
    "screening": (SCREENING_LLM_PROVIDER, SCREENING_MODEL),
    "rag_eval": (RAG_EVAL_LLM_PROVIDER, RAG_EVAL_MODEL),
    "generation": (GENERATION_LLM_PROVIDER, GENERATION_MODEL),
    "review": (REVIEW_LLM_PROVIDER, REVIEW_MODEL),
}


def _get_llm(chain_name: str = "generation"):
    provider, model_override = _CHAIN_CONFIG.get(chain_name, (GENERATION_LLM_PROVIDER, GENERATION_MODEL))

    return ChatGoogleGenerativeAI(
        model=model_override or GEMINI_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=0,
        request_timeout=60,
    )


def screening_chain(full_context: str) -> ScreeningResult:
    llm = _get_llm("screening")
    prompt = ChatPromptTemplate.from_template(SCREENING_PROMPT)

    chain = prompt | llm | _json_parser()

    try:
        result = chain.invoke({"email_content": full_context})
        return ScreeningResult(**result)
    except Exception as e:
        logger.error("Screening chain failed: %s", e)
        return ScreeningResult(
            is_related=False,
            reason=f"screening error: {str(e)}",
        )


def tagging_chain(full_context: str) -> TaggingResult:
    llm = _get_llm()
    prompt = ChatPromptTemplate.from_template(TAGGING_PROMPT)

    chain = prompt | llm | _json_parser()

    try:
        result = chain.invoke({"email_content": full_context})
        return TaggingResult(**result)
    except Exception as e:
        logger.error("Tagging chain failed: %s", e)
        return TaggingResult(
            tags=[],
            product_query="",
            confidence="low",
        )


RAG_EVAL_PROMPT = """ประเมินว่า Knowledge Base ที่ดึงมานั้นเพียงพอในการตอบคำถามลูกค้าหรือไม่

ตอบเป็น JSON เท่านั้น ห้ามมีข้อความอื่น:
{{
  "sufficient": true/false,
  "refined_query": "query ใหม่ที่เจาะจงกว่าเดิม (ถ้า sufficient=false มิฉะนั้นใส่ string ว่าง)"
}}

เกณฑ์:
- sufficient: true ถ้า Knowledge Base ตอบคำถามลูกค้าได้ครบถ้วน
- sufficient: false ถ้าข้อมูลไม่ครบ ไม่ตรงประเด็น หรือขาดรายละเอียดสำคัญ
- refined_query ต้องเป็นภาษาไทย และเจาะจงกว่า query เดิม

อีเมลลูกค้า:
{email_content}

Query ที่ใช้ค้นหา:
{query}

Knowledge Base ที่ดึงมา:
{rag_context}"""


def llm_tagging_chain(full_context: str) -> dict:
    llm = _get_llm()
    prompt = ChatPromptTemplate.from_template(TAGGING_PROMPT)
    chain = prompt | llm | _json_parser()
    try:
        result = chain.invoke({"email_content": full_context})
        return TaggingResult(**result).__dict__
    except Exception as e:
        logger.error("llm_tagging_chain failed: %s", e)
        return {"tags": [], "product_query": "", "confidence": "low"}


def rag_eval_chain(email_content: str, query: str, rag_context: str) -> dict:
    llm = _get_llm("rag_eval")
    prompt = ChatPromptTemplate.from_template(RAG_EVAL_PROMPT)

    chain = prompt | llm | _json_parser()

    try:
        result = chain.invoke({
            "email_content": email_content,
            "query": query,
            "rag_context": rag_context,
        })
        return {
            "sufficient": bool(result.get("sufficient", True)),
            "refined_query": str(result.get("refined_query", "")),
        }
    except Exception as e:
        logger.error("rag_eval_chain failed: %s", e)
        return {"sufficient": True, "refined_query": ""}


GENERATION_PROMPT = """คุณเป็นพนักงานตอบอีเมลของร้านฟ้าใหม่ (FahMai) ร้านขายอิเล็กทรอนิกส์ออนไลน์ของไทย

กฎเด็ดขาด:
- ใช้เฉพาะข้อมูลที่ปรากฏใน Knowledge Base เท่านั้น
- ถ้า Knowledge Base มีข้อมูลของสินค้านั้นโดยตรง ให้ใช้ข้อมูลนั้น
- ถ้า Knowledge Base มีข้อมูลของสินค้ารุ่นอื่นในซีรีส์เดียวกัน ให้ใช้ข้อมูลนั้นได้ แต่ต้องระบุว่าอ้างอิงจากรุ่นใด
- ถ้า Knowledge Base ว่างเปล่าจริงๆ ไม่มีข้อมูลที่เกี่ยวข้องเลย ให้แจ้งลูกค้าตรงๆ ว่าไม่พบข้อมูลในระบบ และแนะนำให้ติดต่อทีมงานโดยตรง
- ห้ามเดา ห้ามแต่งข้อมูลที่ไม่มีใน Knowledge Base โดยเด็ดขาด

ตอบเป็น JSON เท่านั้น ห้ามมีข้อความอื่น:
{{
  "subject": "Re: <หัวข้ออีเมลที่เหมาะสม>",
  "body": "<เนื้อหาอีเมลตอบกลับ>"
}}

อีเมลลูกค้า:
{email_content}

หมวดหมู่: {tags}

Knowledge Base (แต่ละ chunk ขึ้นต้นด้วย [ชื่อไฟล์] และ # ชื่อสินค้า เพื่อระบุว่าข้อมูลนั้นเป็นของสินค้าชิ้นนั้นโดยเฉพาะ):
{rag_context}"""


REVIEW_PROMPT = """คุณเป็น Agent QA ของร้านฟ้าใหม่ ตรวจร่างอีเมลแล้วตัดสินใจว่าจะทำอะไรต่อไป

ตอบเป็น JSON เท่านั้น ห้ามมีข้อความอื่น:
{{
  "action": "<เลือกหนึ่งใน  4 ตัวเลือก>",
  "reason": "เหตุผลที่เลือก action นี้",
  "refined_query": "เฉพาะถ้า action=retry_rag ให้ใส่ query ใหม่ ไม่งั้น null"
}}

4 ตัวเลือก:
- "send"           → ร่างดี ข้อมูลครบถูกต้อง ส่งได้เลย
- "retry_generate" → KB มีข้อมูลอยู่แต่ร่างไม่ได้ใช้ หรือภาษา/โทนไม่ถูก → ให้ generate ใหม่
- "retry_rag"      → KB ที่ดึงมาไม่ตรงประเด็นหรือขาดรายละเอียด → ให้ retrieve ใหม่ด้วย refined_query
- "human_review"   → ร้องเรียน/คืนเงิน/ขู่ฟ้อง หรือ KB ว่างเปล่าจริงๆ ไม่มีข้อมูลที่เกี่ยวข้องเลย → ให้คนดู

เกณฑ์:
- ถ้า Knowledge Base มีข้อมูลที่เกี่ยวข้องอยู่ แต่ร่างระบุว่า 'ไม่พบข้อมูล' ให้เลือก retry_generate ไม่ใช่ human_review
- ตรวจว่าร่างตอบครบทุกคำถามโดยใช้ข้อมูลจาก KB
- ข้อมูลตรงกับ Knowledge Base ไม่แต่งเพิ่ม
- ภาษาสุภาพ เหมาะสมการบริการลูกค้า

อีเมลลูกค้า:
{email_content}

ร่างอีเมลตอบกลับ:
{draft_reply}

Knowledge Base ที่ใช้:
{rag_context}"""


def generation_chain(
    full_context: str,
    rag_context: str,
    tags: list[str],
) -> GeneratedReply:
    llm = _get_llm("generation")
    prompt = ChatPromptTemplate.from_template(GENERATION_PROMPT)

    chain = prompt | llm | _json_parser()

    try:
        result = chain.invoke({
            "email_content": full_context,
            "rag_context": rag_context,
            "tags": ", ".join(tags),
        })
        return GeneratedReply(**result)
    except Exception as e:
        logger.error("Generation chain failed: %s", e)
        return GeneratedReply(
            subject="Re: คำถามของคุณ",
            body=f"ขออภัย ระบบไม่สามารถสร้างข้อความตอบกลับได้ในขณะนี้ กรุณาติดต่อทีมงานโดยตรง",
        )


def review_chain(
    full_context: str,
    draft_reply: str,
    rag_context: str,
) -> ReviewResult:
    llm = _get_llm("review")
    prompt = ChatPromptTemplate.from_template(REVIEW_PROMPT)

    chain = prompt | llm | _json_parser()

    try:
        result = chain.invoke({
            "email_content": full_context,
            "draft_reply": draft_reply,
            "rag_context": rag_context,
        })
        return ReviewResult(**result)
    except Exception as e:
        logger.error("Review chain failed: %s", e)
        return ReviewResult(action="human_review", reason=f"review chain error: {e}")
