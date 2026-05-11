import logging

import httpx

from src.config import BERT_API_URL

logger = logging.getLogger(__name__)


def predict_tags(text: str, timeout: float = 15.0) -> dict:
    for attempt in range(2):
        try:
            resp = httpx.post(f"{BERT_API_URL}/tag", json={"text": text}, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except (httpx.TimeoutException, httpx.HTTPError) as e:
            if attempt == 1:
                raise
            logger.warning("BERT API attempt %d failed: %s, retrying...", attempt + 1, e)
