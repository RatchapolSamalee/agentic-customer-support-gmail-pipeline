import logging
from typing import Optional

import chromadb
from rank_bm25 import BM25Okapi
from rapidfuzz import process as fuzz_process
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.config import (
    CHROMA_DB_PATH,
    CHROMA_COLLECTION_NAME,
    RAG_BM25_TOP_K,
    RAG_DENSE_TOP_K,
    RAG_RERANK_TOP_K,
    RERANKER_MODEL,
)

logger = logging.getLogger(__name__)

RRF_K = 60
FUZZY_FILENAME_THRESHOLD = 60

_chroma_client: Optional[chromadb.ClientAPI] = None
_embed_model: Optional[SentenceTransformer] = None
_reranker: Optional[CrossEncoder] = None


def get_chroma_client() -> chromadb.ClientAPI:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return _chroma_client


def get_collection(client: chromadb.ClientAPI) -> chromadb.Collection:
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        logger.info("Loading BGE embedding model")
        _embed_model = SentenceTransformer("BAAI/bge-m3", device="cpu")
    return _embed_model


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        logger.info("Loading BGE reranker model")
        _reranker = CrossEncoder(RERANKER_MODEL, device="cpu")
    return _reranker


def _bm25_search(corpus: list[dict], query: str, top_k: int) -> list[dict]:
    if not corpus:
        return []
    tokenized_corpus = [doc["text"].split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.split())
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [corpus[i] for i in top_indices]


def _get_product_filenames(collection: chromadb.Collection) -> list[str]:
    response = collection.get(
        where={"category": "product"},
        include=["metadatas"],
    )
    seen = set()
    filenames = []
    for meta in response["metadatas"]:
        fn = meta.get("filename", "")
        if fn and fn not in seen:
            seen.add(fn)
            filenames.append(fn)
    return filenames


def _fuzzy_match_filenames(query: str, filenames: list[str], threshold: int) -> list[str]:
    if not query or not filenames:
        return []
    matches = fuzz_process.extract(query, filenames, limit=3)
    return [fn for fn, score, _ in matches if score >= threshold]


def _dense_search(
    collection: chromadb.Collection,
    query: str,
    tag: str,
    top_k: int,
    filename_filter: Optional[list[str]] = None,
) -> list[dict]:
    embed = _get_embed_model()
    query_vector = embed.encode(query, normalize_embeddings=True).tolist()

    if filename_filter:
        where = {"$and": [{"category": tag}, {"filename": {"$in": filename_filter}}]}
    else:
        where = {"category": tag}

    response = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas"],
    )
    results = []
    for doc_id, text, meta in zip(
        response["ids"][0], response["documents"][0], response["metadatas"][0]
    ):
        results.append({"id": doc_id, "text": text, "metadata": meta})
    return results


def _rrf_merge(ranked_lists: list[list[dict]], k: int = RRF_K) -> list[dict]:
    """Reciprocal Rank Fusion across multiple ranked lists."""
    scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            doc_map[doc_id] = doc

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[doc_id] for doc_id in sorted_ids]


def _rerank_with_scores(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """Rerank candidates and attach reranker_score to each doc."""
    if not candidates:
        return []
    reranker = _get_reranker()
    pairs = [[query, doc["text"]] for doc in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(
        zip(scores, candidates),
        key=lambda x: x[0],
        reverse=True,
    )
    results = []
    for score, doc in ranked[:top_k]:
        results.append({**doc, "reranker_score": float(score)})
    return results



def retrieve_documents(
    query: str,
    tags: list[str],
    product_query: str = "",
) -> list[dict]:
    client = get_chroma_client()
    collection = get_collection(client)
    results = []
    seen_ids: set = set()

    for tag in tags:
        if not query:
            continue

        # fuzzy match product filenames if product_query provided
        filename_filter: list[str] = []
        if tag == "product" and product_query:
            all_filenames = _get_product_filenames(collection)
            filename_filter = _fuzzy_match_filenames(product_query, all_filenames, FUZZY_FILENAME_THRESHOLD)
            if filename_filter:
                logger.info("Fuzzy filename filter: %s", filename_filter)

        # dense + BM25 candidates
        dense_hits = _dense_search(collection, query, tag, RAG_DENSE_TOP_K, filename_filter or None)
        bm25_hits = _bm25_search(dense_hits, query, RAG_BM25_TOP_K)

        # RRF merge
        rrf_merged = _rrf_merge([dense_hits, bm25_hits])

        # rerank
        reranked = _rerank_with_scores(query, rrf_merged, RAG_RERANK_TOP_K)

        for doc in reranked:
            if doc["id"] not in seen_ids:
                seen_ids.add(doc["id"])
                results.append(doc)

    logger.info("Retrieved %d chunks for tags %s", len(results), tags)
    return results


def format_context(documents: list[dict]) -> str:
    if not documents:
        return ""
    parts = []
    for doc in documents:
        filename = doc["metadata"].get("filename", "unknown")
        parts.append(f"[{filename}]\n{doc['text']}")
    return "\n\n".join(parts)
