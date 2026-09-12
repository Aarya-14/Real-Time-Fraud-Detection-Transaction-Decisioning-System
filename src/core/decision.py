## src/core/decision.py

# src/core/decision.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import joblib
import numpy as np

from src.api.schemas import TransactionRequest
from src.core.config import MODEL_DIR, PROJECT_ROOT
from src.ml.feature_eng import FeatureEngine
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# -------------------------------------------------------------------
# Decision logic
# -------------------------------------------------------------------


@dataclass
class DecisionResult:
    score: float
    decision: str     # "ALLOW", "PEND", "REJECT"
    reason: str
    thresholds: Dict[str, float]


class DecisionEngine:
    """
    Applies business thresholds on top of model scores.

    Thresholds are loaded from config/thresholds.json, for example:

    {
      "pend": 0.2,
      "reject": 0.6
    }
    """

    def __init__(self, thresholds: Dict[str, float]):
        self.thresholds = thresholds
        self.pend_thr = thresholds.get("pend", 0.2)
        self.reject_thr = thresholds.get("reject", 0.6)

        logger.info(
            f"DecisionEngine initialized with thresholds: "
            f"pend>={self.pend_thr}, reject>={self.reject_thr}"
        )

    @classmethod
    def load_from_config(
        cls,
        project_root: Path | None = None,
        config_path: Path | None = None,
    ) -> "DecisionEngine":
        import json

        if project_root is None:
            project_root = PROJECT_ROOT
        if config_path is None:
            config_path = project_root / "config" / "thresholds.json"

        logger.info(f"Loading decision thresholds from: {config_path}")
        with open(config_path, "r") as f:
            thresholds = json.load(f)

        return cls(thresholds=thresholds)

    def decide(self, score: float) -> DecisionResult:
        """
        Map score → ALLOW / PEND / REJECT + human-readable reason.
        """
        if score >= self.reject_thr:
            decision = "REJECT"
            reason = (
                f"Score {score:.3f} >= reject threshold {self.reject_thr:.2f} "
                "(high fraud risk)."
            )
        elif score >= self.pend_thr:
            decision = "PEND"
            reason = (
                f"Score {score:.3f} between pend threshold {self.pend_thr:.2f} "
                f"and reject threshold {self.reject_thr:.2f}."
            )
        else:
            decision = "ALLOW"
            reason = (
                f"Score {score:.3f} < pend threshold {self.pend_thr:.2f} "
                "(low fraud risk)."
            )

        logger.info(f"Decision={decision} | score={score:.4f}")
        return DecisionResult(
            score=score,
            decision=decision,
            reason=reason,
            thresholds=self.thresholds,
        )


# -------------------------------------------------------------------
# Fraud model wrapper used by the API
# -------------------------------------------------------------------


class FraudModel:
    """
    Thin wrapper around the trained LightGBM model + FeatureEngine.

    Used by the FastAPI app for scoring single transactions.
    """

    def __init__(self, model, feature_engine: FeatureEngine):
        self.model = model
        self.feature_engine = feature_engine

    @classmethod
    def load_from_disk(
        cls,
        project_root: Path | None = None,
        model_name: str = "baseline_lgbm_v2.pkl",
        fe_dir_name: str = "feature_engine_v2",
    ) -> "FraudModel":
        """
        Load trained model + FeatureEngine artifacts from the models/ directory.
        """

        if project_root is None:
            project_root = PROJECT_ROOT

        model_path = MODEL_DIR / model_name
        fe_dir = MODEL_DIR / fe_dir_name

        logger.info(f"Loading trained model from: {model_path}")
        model = joblib.load(model_path)

        logger.info(f"Loading FeatureEngine from: {fe_dir}")
        feature_engine = FeatureEngine.load(fe_dir)

        logger.info("FraudModel successfully loaded.")
        return cls(model=model, feature_engine=feature_engine)

    def score_transaction(self, tx: TransactionRequest) -> float:
        """
        Transform TransactionRequest → feature vector → fraud probability.
        """
        # FeatureEngine handles transforming a single transaction into a vector
        features = self.feature_engine.transform_single(tx)
        X = np.array(features, dtype=float).reshape(1, -1)

        prob_1 = float(self.model.predict_proba(X)[:, 1][0])
        logger.info(f"Scored transaction with fraud probability={prob_1:.6f}")
        return prob_1
