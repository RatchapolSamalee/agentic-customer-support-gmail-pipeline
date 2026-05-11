import logging
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from src.config import KNOWLEDGE_BASE_PATH, CHROMA_COLLECTION_NAME, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP
from src.rag import get_chroma_client, get_collection

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CATEGORY_MAP = {
    "policies": "policy",
    "products": "product",
    "store_info": "store_info",
}


def load_documents() -> list[dict]:
    docs = []
    for folder, category in CATEGORY_MAP.items():
        folder_path = os.path.join(KNOWLEDGE_BASE_PATH, folder)
        if not os.path.exists(folder_path):
            logger.warning("Folder not found: %s", folder_path)
            continue

        for filename in os.listdir(folder_path):
            if not filename.endswith(".md"):
                continue
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if content:
                docs.append({
                    "content": content,
                    "category": category,
                    "filename": filename,
                    "filepath": filepath,
                })

    logger.info("Loaded %d documents", len(docs))
    return docs


def _extract_doc_title(content: str) -> str:
    match = re.search(r'^#\s+(.+)', content, re.MULTILINE)
    return match.group(1).strip() if match else ""


def _nearest_section(content: str, char_pos: int) -> str:
    headers = [(m.start(), m.group(1).strip()) for m in re.finditer(r'^#{2,}\s+(.+)', content, re.MULTILINE)]
    section = ""
    for pos, title in headers:
        if pos <= char_pos:
            section = title
        else:
            break
    return section


def split_documents(docs: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAG_CHUNK_SIZE,
        chunk_overlap=RAG_CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = []
    for doc in docs:
        content = doc["content"]
        doc_title = _extract_doc_title(content)
        parts = splitter.split_text(content)
        char_pos = 0
        for i, part in enumerate(parts):
            char_pos = content.find(part, char_pos)
            section = _nearest_section(content, char_pos)
            if doc_title and section:
                header = f"[{doc_title} > {section}]\n"
            elif doc_title:
                header = f"[{doc_title}]\n"
            else:
                header = ""
            chunks.append({
                "text": part,
                "embed_text": header + part,
                "category": doc["category"],
                "filename": doc["filename"],
                "chunk_index": i,
            })
            if char_pos != -1:
                char_pos += len(part)

    logger.info("Split into %d chunks", len(chunks))
    return chunks


def ingest(chunks: list[dict]) -> None:
    logger.info("Loading BGE-M3 embedding model")
    embed_model = SentenceTransformer("BAAI/bge-m3", device="cpu")

    client = get_chroma_client()
    collection = get_collection(client)

    # clear existing data before reingest
    existing = collection.count()
    if existing > 0:
        collection.delete(where={"category": {"$in": ["policy", "product", "store_info"]}})
        logger.info("Cleared %d existing chunks", existing)

    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]
        embed_texts = [c["embed_text"] for c in batch]
        ids = [f"{c['filename']}__chunk_{c['chunk_index']}" for c in batch]
        metadatas = [{"category": c["category"], "filename": c["filename"]} for c in batch]

        vectors = embed_model.encode(embed_texts, normalize_embeddings=True).tolist()

        collection.add(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info("Ingested batch %d-%d", i, i + len(batch))

    logger.info("Done. Total chunks in collection: %d", collection.count())


if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    ingest(chunks)
