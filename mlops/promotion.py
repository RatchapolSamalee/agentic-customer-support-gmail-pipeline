import logging

logger = logging.getLogger(__name__)


def should_promote(new_metrics: dict, prod_metrics: dict | None, criteria: dict | None = None) -> bool:
    min_f1 = (criteria or {}).get("min_f1", 0.80)
    max_drop = (criteria or {}).get("max_label_drop", 0.03)

    if new_metrics.get("macro_f1", 0) < min_f1:
        logger.info("Promotion denied: macro_f1 %.3f < %.3f", new_metrics.get("macro_f1", 0), min_f1)
        return False

    if prod_metrics:
        for label in ("is_policy", "is_product", "is_store_info"):
            new_f1 = new_metrics.get(f"f1_{label}", 0)
            prod_f1 = prod_metrics.get(f"f1_{label}", 0)
            if prod_f1 - new_f1 > max_drop:
                logger.info("Promotion denied: %s dropped %.3f -> %.3f", label, prod_f1, new_f1)
                return False

    logger.info("Promotion approved: macro_f1=%.3f", new_metrics.get("macro_f1", 0))
    return True
