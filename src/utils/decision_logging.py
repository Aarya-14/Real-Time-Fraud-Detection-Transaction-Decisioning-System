# src/utils/decision_logging.py

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.api.schemas import TransactionRequest
from src.core.decision import DecisionResult
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Logs folder: project_root/logs/decisions.jsonl
DEFAULT_LOG_PATH = Path(__file__).resolve().parents[2] / "logs" / "decisions.jsonl"


def log_decision(
    tx: TransactionRequest,
    result: DecisionResult,
    model_version: str = "v2",
    latency_ms: Optional[float] = None, # <--- FIX: Added latency_ms argument
    log_path: Optional[Path] = None,
):
    """Append a fraud decision event to logs/decisions.jsonl"""

    if log_path is None:
        log_path = DEFAULT_LOG_PATH

    log_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "model_version": model_version,
        "latency_ms": latency_ms,  # <--- FIX: Added latency to the log record

        # Request fields
        "user_id": tx.user_id,
        "card_id": tx.card_id,
        "merchant_id": tx.merchant_id,
        "amount": tx.amount,
        "ts": tx.timestamp,

        # Model output
        "score": float(result.score),
        "decision": result.decision,
        "reason": result.reason,

        # Thresholds at time of decision
        "thresholds": result.thresholds,
    }

    try:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        # Enhanced console logging to include latency
        latency_info = f", latency={latency_ms:.2f}ms" if latency_ms is not None else ""
        logger.info(
            f"Logged decision → {record['decision']} (score={record['score']:.3f}){latency_info}"
        )
    except Exception as e:
        logger.error(f"Failed to write decision log: {e}")