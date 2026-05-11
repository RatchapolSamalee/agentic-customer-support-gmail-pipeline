import os
from dotenv import load_dotenv

load_dotenv()

# LLM Provider
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")

# Per-chain LLM provider override (ถ้าไม่ตั้งค่า จะใช้ LLM_PROVIDER)
SCREENING_LLM_PROVIDER: str = os.getenv("SCREENING_LLM_PROVIDER", LLM_PROVIDER)
RAG_EVAL_LLM_PROVIDER: str = os.getenv("RAG_EVAL_LLM_PROVIDER", LLM_PROVIDER)
GENERATION_LLM_PROVIDER: str = os.getenv("GENERATION_LLM_PROVIDER", LLM_PROVIDER)
REVIEW_LLM_PROVIDER: str = os.getenv("REVIEW_LLM_PROVIDER", LLM_PROVIDER)

# Per-chain model override (ถ้าไม่ตั้งค่า จะใช้โมเดล default ของ provider นั้น)
SCREENING_MODEL: str = os.getenv("SCREENING_MODEL", "")
RAG_EVAL_MODEL: str = os.getenv("RAG_EVAL_MODEL", "")
GENERATION_MODEL: str = os.getenv("GENERATION_MODEL", "")
REVIEW_MODEL: str = os.getenv("REVIEW_MODEL", "")

# Gemini API
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_VISION_MODEL: str = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")


# Gmail API
GMAIL_CREDENTIALS_FILE: str = os.getenv("GMAIL_CREDENTIALS_FILE", "credentials.json")
GMAIL_TOKEN_FILE: str = os.getenv("GMAIL_TOKEN_FILE", "token.json")
GMAIL_SCOPES: list[str] = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.compose",
]

# PostgreSQL database
DB_HOST: str = os.getenv("DB_HOST", "localhost")
DB_PORT: int = int(os.getenv("DB_PORT", "5432"))
DB_NAME: str = os.getenv("DB_NAME", "autogmail")
DB_USER: str = os.getenv("DB_USER", "postgres")
DB_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "")
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
)

# ChromaDB vector store
CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./chroma_db")
CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "fahmai_knowledge_base")

# Knowledge base
KNOWLEDGE_BASE_PATH: str = os.getenv("KNOWLEDGE_BASE_PATH", "./store data/knowledge_base")

# Attachments
ATTACHMENTS_DIR: str = os.getenv("ATTACHMENTS_DIR", "./attachments")

# Pipeline
MAX_REVIEW_RETRIES: int = int(os.getenv("MAX_REVIEW_RETRIES", "2"))
GMAIL_FETCH_MAX_RESULTS: int = int(os.getenv("GMAIL_FETCH_MAX_RESULTS", "50"))

# RAG
RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "5"))
RAG_CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", "800"))
RAG_CHUNK_OVERLAP: int = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))
RAG_BM25_TOP_K: int = int(os.getenv("RAG_BM25_TOP_K", "20"))
RAG_DENSE_TOP_K: int = int(os.getenv("RAG_DENSE_TOP_K", "30"))
RAG_RERANK_TOP_K: int = int(os.getenv("RAG_RERANK_TOP_K", "8"))
RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

# BERT Classifier (V2)
BERT_MODEL_PATH: str = os.getenv("BERT_MODEL_PATH", "models/bert-email-tagger")
BERT_MODEL_NAME: str = os.getenv("BERT_MODEL_NAME", "bert-email-tagger")
BERT_THRESHOLD: float = float(os.getenv("BERT_THRESHOLD", "0.5"))

# MLflow (V2-09)
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_REGISTRY_URI: str = os.getenv("MLFLOW_REGISTRY_URI", "http://localhost:5000")

# BERT API (V2-11)
BERT_API_URL: str = os.getenv("BERT_API_URL", "http://localhost:8002")

# Gmail Pub/Sub
PUBSUB_TOPIC: str = os.getenv("PUBSUB_TOPIC", "")

# Logging
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
