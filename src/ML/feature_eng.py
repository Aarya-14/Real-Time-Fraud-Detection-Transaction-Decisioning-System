# src/ml/feature_eng.py

# src/ml/feature_eng.py

import json
from pathlib import Path
from typing import List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd

from src.api.schemas import TransactionRequest
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class FeatureEngine:
    """
    Production-grade feature engineering engine.

    - Used during training to define the full feature set + ordering.
    - Used during inference (API) to transform a TransactionRequest into a
      model-ready feature vector with the exact same ordering.

    V2: Supports optional velocity features (user/card 1h/24h windows) if present
    in the input dataframe (e.g., all_merged_fs_v2.parquet).
    """

    HIGH_RISK_MCC_CODES = {
        "5311", "5310", "5300", "4829", "5814", "5912"
    }

    LOW_CARDINALITY_COLS = ["use_chip", "card_type", "card_brand", "gender"]
    FREQ_ENCODE_COLS = ["merchant_category_code", "mcc_description", "merchant_state"]

    # Velocity feature names (only used if columns are present in df)
    VELOCITY_COLS = [
        # user windows
        "user_txn_count_1h",
        "user_amount_sum_1h",
        "user_txn_count_24h",
        "user_amount_sum_24h",
        # card windows
        "card_txn_count_1h",
        "card_amount_sum_1h",
        "card_txn_count_24h",
        "card_amount_sum_24h",
    ]

    def __init__(self) -> None:
        # Learned mappings & state
        self.freq_maps: dict[str, pd.Series] = {}
        self.batch_feature_order: List[str] = []  # final feature ordering from training

    # ---------------------------------------------------------------------
    #                           TRAINING PHASE
    # ---------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "FeatureEngine":
        """
        Fit frequency encoders & discover feature ordering from a training dataframe.

        This should be called on the *training* data (or full data) before training
        the model. The resulting FeatureEngine is then saved and reused for inference.
        """
        logger.info("Fitting FeatureEngine on dataset...")

        if "timestamp" not in df.columns:
            logger.error("Missing 'timestamp' column during FeatureEngine.fit!")
            raise KeyError("timestamp must be present in dataframe.")

        if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
            logger.warning("timestamp column not datetime; attempting conversion.")
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        logger.info("Fitting frequency encoders...")
        self._fit_frequency_encoders(df)

        # Generate ordering from training dataset
        logger.info("Generating batch feature order from training set...")
        X, _ = self._transform_batch(df, fit_mode=True)
        self.batch_feature_order = list(X.columns)

        logger.info(
            "FeatureEngine.fit complete. Total features: %d",
            len(self.batch_feature_order),
        )
        return self

    # ---------------------------------------------------------------------
    #                           TRANSFORM (BATCH)
    # ---------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Transform batch dataframe into model-ready feature matrix.

        Ensures:
        - same columns as discovered during fit
        - same ordering
        """
        logger.info("Transforming batch dataset using FeatureEngine...")

        X, y = self._transform_batch(df, fit_mode=False)

        if not self.batch_feature_order:
            logger.warning(
                "batch_feature_order is empty during transform; did you forget to call fit()?"
            )

        # Ensure feature order consistency
        X = X.reindex(columns=self.batch_feature_order, fill_value=0.0)

        logger.info("Batch transformation complete. Shape: %s", (X.shape,))
        return X, y

    # ---------------------------------------------------------------------
    #                           TRANSFORM SINGLE
    # ---------------------------------------------------------------------

    def transform_single(self, tx: TransactionRequest) -> pd.DataFrame:
        """
        Transform a single transaction into a model-ready feature vector (pd.DataFrame).

        This is used by the API. It assumes:
        - FeatureEngine was fitted & loaded from disk
        - batch_feature_order is populated
        """
        logger.info("Transforming single transaction for inference.")

        df = pd.DataFrame([tx])

        # X is now a DataFrame with the features derived from df
        X, _ = self._transform_batch(df, fit_mode=False)
        
        # This reindexes X to match the exact feature order from training
        X = X.reindex(columns=self.batch_feature_order, fill_value=0.0)

        logger.debug(
            "Single transaction transformed into feature vector of shape %s",
            (X.shape,),
        )
        # Return the DataFrame (X)
        return X

    # ---------------------------------------------------------------------
    #                           SAVE / LOAD
    # ---------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """
        Save frequency encoders + feature order to disk.

        These artifacts are needed for consistent inference.
        """
        path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.freq_maps, path / "freq_maps.pkl")

        with open(path / "feature_order.json", "w") as f:
            json.dump(self.batch_feature_order, f, indent=2)

        logger.info("FeatureEngine saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "FeatureEngine":
        """
        Load FeatureEngine state from disk.
        """
        fe = cls()
        try:
            fe.freq_maps = joblib.load(path / "freq_maps.pkl")
            with open(path / "feature_order.json", "r") as f:
                fe.batch_feature_order = json.load(f)

            logger.info("FeatureEngine loaded from %s", path)
        except Exception as e:
            logger.error("Error loading FeatureEngine: %s", e)
            raise e

        return fe

    # ---------------------------------------------------------------------
    #                           HELPERS
    # ---------------------------------------------------------------------

    def _fit_frequency_encoders(self, df: pd.DataFrame) -> None:
        for col in self.FREQ_ENCODE_COLS:
            if col in df.columns:
                freq = df[col].value_counts(normalize=True)
                self.freq_maps[col] = freq
                logger.info(
                    "Fitted frequency encoding for '%s' with %d categories.",
                    col,
                    len(freq),
                )
            else:
                logger.warning(
                    "Column '%s' missing — skipping frequency encoding.",
                    col,
                )

    def _frequency_encode(self, df: pd.DataFrame, col: str) -> pd.Series:
        if col not in self.freq_maps:
            logger.warning("No freq_map for '%s' — defaulting to 0.", col)
            return pd.Series([0.0] * len(df), index=df.index)
        return df[col].map(self.freq_maps[col]).fillna(0.0)

    # ---------------------------------------------------------------------
    #         BATCH TRANSFORMATION LOGIC (TRAIN + INFERENCE)
    # ---------------------------------------------------------------------

    def _transform_batch(
        self, df: pd.DataFrame, fit_mode: bool
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Core transformation logic shared by:
        - fit()  (with fit_mode=True, for discovering columns)
        - transform()/transform_single() (fit_mode=False)

        NOTE:
        - Velocity features (user_* / card_*) are assumed to be precomputed and
          present in the dataframe (e.g. all_merged_fs_v2.parquet) during training.
        - For online scoring, if those columns are missing they will simply be
          filled with 0.0 via reindex().
        """
        df = df.copy()

        # Timestamp
        if "timestamp" not in df.columns:
            logger.warning("No 'timestamp' column found during transform.")
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            if df["timestamp"].isna().any():
                logger.warning("Some timestamps could not be parsed.")

            df["transaction_hour"] = df["timestamp"].dt.hour
            df["transaction_dayofweek"] = df["timestamp"].dt.dayofweek
            df["transaction_month"] = df["timestamp"].dt.month

        # Numeric features
        raw_numeric = [
            "amount",
            "yearly_income",
            "total_debt",
            "current_age",
            "credit_score",
            "num_credit_cards",
        ]

        for col in raw_numeric:
            if col not in df.columns:
                logger.warning("Missing numeric column '%s' in dataframe.", col)
                df[col] = 0.0

        eps = 1e-6
        df["amount_to_income"] = df["amount"] / (df["yearly_income"] + eps)
        df["debt_to_income"] = df["total_debt"] / (df["yearly_income"] + eps)

        # Buckets
        df["amount_bin"] = df["amount"].apply(self._amount_bucket)
        df["hour_bin"] = df.get("transaction_hour", 0).apply(self._hour_bucket)
        df["income_bin"] = df["yearly_income"].apply(self._income_bucket)

        # Channel one-hot (use_chip)
        if "use_chip" in df.columns:
            df["use_chip_lower"] = df["use_chip"].astype(str).str.lower()
        else:
            logger.warning("Column 'use_chip' missing; creating default.")
            df["use_chip_lower"] = ""

        df["is_online"] = (df["use_chip_lower"] == "online transaction").astype(int)
        df["is_chip"] = (df["use_chip_lower"] == "chip transaction").astype(int)
        df["is_swipe"] = (df["use_chip_lower"] == "swipe transaction").astype(int)

        # Card type one-hot
        if "card_type" not in df.columns:
            logger.warning("Column 'card_type' missing; creating default.")
            df["card_type"] = ""

        df["is_prepaid"] = (df["card_type"] == "Debit (Prepaid)").astype(int)
        df["is_credit"] = (df["card_type"] == "Credit").astype(int)
        df["is_debit"] = (df["card_type"] == "Debit").astype(int)

        # Card brand one-hot
        if "card_brand" not in df.columns:
            logger.warning("Column 'card_brand' missing; creating default.")
            df["card_brand"] = ""

        for brand in ["Mastercard", "Visa", "Amex", "Discover"]:
            df[f"brand_{brand.lower()}"] = (df["card_brand"] == brand).astype(int)

        # MCC flags
        if "merchant_category_code" not in df.columns:
            logger.warning(
                "Column 'merchant_category_code' missing; creating default."
            )
            df["merchant_category_code"] = ""

        df["high_risk_mcc"] = (
            df["merchant_category_code"].astype(str).isin(self.HIGH_RISK_MCC_CODES)
        ).astype(int)

        # Simple flags
        df["test_amount_flag"] = (
            (df["amount"] > 0) & (df["amount"] <= 10)
        ).astype(int)
        df["high_amount_flag"] = (df["amount"] >= 200).astype(int)

        # Frequency encodings
        for col in self.FREQ_ENCODE_COLS:
            if col in df.columns:
                df[f"{col}_freq"] = self._frequency_encode(df, col)
            else:
                logger.warning(
                    "Column '%s' missing; freq feature %s_freq will be 0.",
                    col,
                    col,
                )
                df[f"{col}_freq"] = 0.0

        # Label
        y = df["target"] if "target" in df.columns else None

        # -----------------------------------------------------------------
        #   Assemble final feature list
        #   - core numeric / ratios / time / one-hot / flags / freq encodings
        #   - PLUS any velocity features that are present in df
        # -----------------------------------------------------------------

        base_feature_cols = [
            *raw_numeric,
            "amount_to_income",
            "debt_to_income",
            "transaction_hour",
            "transaction_dayofweek",
            "transaction_month",
            "amount_bin",
            "hour_bin",
            "income_bin",
            "is_online",
            "is_chip",
            "is_swipe",
            "is_prepaid",
            "is_credit",
            "is_debit",
            "brand_mastercard",
            "brand_visa",
            "brand_amex",
            "brand_discover",
            "high_risk_mcc",
            "test_amount_flag",
            "high_amount_flag",
            *[f"{c}_freq" for c in self.FREQ_ENCODE_COLS],
        ]

        # Only use velocity columns that are actually present
        velocity_cols_present = [c for c in self.VELOCITY_COLS if c in df.columns]
        missing_velocity = [c for c in self.VELOCITY_COLS if c not in df.columns]

        if velocity_cols_present:
            logger.info(
                "Using %d velocity features: %s",
                len(velocity_cols_present),
                velocity_cols_present,
            )
        if missing_velocity:
            logger.info(
                "Velocity cols not found in dataframe (will be 0 if in feature_order): %s",
                missing_velocity,
            )

        feature_cols = base_feature_cols + velocity_cols_present

        # Ensure all feature columns exist (create with 0 if missing)
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0

        X = df[feature_cols]

        logger.debug(
            "Batch transform complete. Feature columns: %d",
            len(feature_cols),
        )
        return X, y

    # ---------------------------------------------------------------------
    #         BUCKET FUNCTIONS (shared train/inference)
    # ---------------------------------------------------------------------

    @staticmethod
    def _amount_bucket(amount: float) -> int:
        if amount <= 0:
            return 0
        if amount <= 10:
            return 1
        if amount <= 50:
            return 2
        if amount <= 100:
            return 3
        if amount <= 200:
            return 4
        if amount <= 500:
            return 5
        if amount <= 1000:
            return 6
        return 7

    @staticmethod
    def _hour_bucket(hour: float) -> int:
        # hour may be NaN or float; coerce to int-like
        try:
            h = int(hour)
        except Exception:
            h = 0
        if h < 6:
            return 0
        if h < 12:
            return 1
        if h < 18:
            return 2
        return 3

    @staticmethod
    def _income_bucket(income: float) -> int:
        if income <= 20000:
            return 0
        if income <= 40000:
            return 1
        if income <= 60000:
            return 2
        if income <= 80000:
            return 3
        if income <= 120000:
            return 4
        return 5


# ----------------------------------------------------------
# BATCH FEATURE ENGINEERING FOR TRAINING (for train.py)
# ----------------------------------------------------------

def build_feature_dataframe(
    df: pd.DataFrame,
    label_col: str = "target",
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Helper for train.py:

    - Instantiates FeatureEngine
    - Fits it on the given dataframe
    - Returns (X, y) with consistent feature ordering

    NOTE:
    - The caller is responsible for saving the fitted FeatureEngine via .save()
      if it will be used later in the API.
    """
    fe = FeatureEngine()
    fe.fit(df)
    X, y = fe.transform(df)

    # y is taken from df['target'] inside _transform_batch.
    # label_col is kept here for future extensibility if needed.
    return X, y
