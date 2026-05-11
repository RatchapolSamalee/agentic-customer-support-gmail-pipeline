import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.classifier import BERTTagger

logger = logging.getLogger(__name__)

app = FastAPI(title="BERT Email Tagger API", version="1.0")

tagger: Optional[BERTTagger] = None
loaded_at: Optional[str] = None


class PredictRequest(BaseModel):
    text: str


@app.on_event("startup")
def startup() -> None:
    global tagger, loaded_at
    try:
        tagger = BERTTagger().load_from_mlflow(stage="Production")
    except Exception:
        logger.warning("MLflow load failed, falling back to local path")
        tagger = BERTTagger().load_from_path()
    loaded_at = datetime.utcnow().isoformat()
    logger.info("BERT model loaded: %s", tagger.model_version)


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    if tagger is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return tagger.predict(req.text)


@app.post("/reload")
def reload() -> dict:
    global tagger, loaded_at
    old_version = tagger.model_version if tagger else None
    try:
        tagger = BERTTagger().load_from_mlflow(stage="Production")
    except Exception:
        tagger = BERTTagger().load_from_path()
    loaded_at = datetime.utcnow().isoformat()
    return {"status": "reloaded", "old_version": old_version, "new_version": tagger.model_version}


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_version": tagger.model_version if tagger else None,
        "loaded_at": loaded_at,
    }
