# src/ml/train.py

import os
from pathlib import Path
from typing import Dict, Tuple

import joblib
import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from src.ml.feature_eng import FeatureEngine
from src.utils.logging_utils import get_logger
from src.core.config import PROJECT_ROOT, MODEL_DIR

logger = get_logger(__name__)

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "all_merged_fs_v2.parquet"
FEATURE_ENGINE_DIR = MODEL_DIR / "feature_engine_v2"
MODEL_PATH = MODEL_DIR / "baseline_lgbm_v2.pkl"

MLFLOW_EXPERIMENT_NAME = "fraud_baseline_lgbm_v2"


# -------------------------------------------------------------------
# DATA LOADING + SPLITTING
# -------------------------------------------------------------------

def load_data(path: Path) -> pd.DataFrame:
    logger.info(f"Loading data from: {path}")
    df = pd.read_parquet(path)
    logger.info(
        "Loaded dataframe with shape %s and columns: %s",
        df.shape,
        list(df.columns),
    )
    return df


def time_based_split(
    df: pd.DataFrame,
    ts_col: str = "timestamp",
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Time-based split to mimic production: train on older data, test on newer.

    - train_frac: portion of earliest data used for training
    - val_frac:   portion after train used for validation
    - remainder:  test
    """
    logger.info("Sorting data by timestamp for time-based split...")
    df = df.sort_values(ts_col)

    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    logger.info(
        "Split sizes -> train: %d, val: %d, test: %d",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    logger.info(
        "Train range: %s to %s",
        train_df[ts_col].min(),
        train_df[ts_col].max(),
    )
    logger.info(
        "Val range:   %s to %s",
        val_df[ts_col].min(),
        val_df[ts_col].max(),
    )
    logger.info(
        "Test range:  %s to %s",
        test_df[ts_col].min(),
        test_df[ts_col].max(),
    )

    return train_df, val_df, test_df


# -------------------------------------------------------------------
# METRICS
# -------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    label: str,
    fpr_targets=(0.005, 0.01, 0.02),
) -> Dict[str, float]:
    """
    Compute core fraud metrics:
    - PR AUC
    - ROC AUC
    - Recall at given FPR levels (approximate, via score threshold grid).
    """
    metrics: Dict[str, float] = {}

    pr_auc = average_precision_score(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)

    metrics[f"{label}_pr_auc"] = pr_auc
    metrics[f"{label}_roc_auc"] = roc_auc

    # Approximate recall@FPR using a score grid
    thresholds = np.quantile(y_proba, np.linspace(0.0, 1.0, 500))

    y_true = np.array(y_true)
    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()

    for fpr_target in fpr_targets:
        best_recall = 0.0
        best_thr = None

        for thr in thresholds:
            y_pred = (y_proba >= thr).astype(int)

            fp = ((y_pred == 1) & (y_true == 0)).sum()
            tp = ((y_pred == 1) & (y_true == 1)).sum()

            fpr = fp / max(n_neg, 1)
            recall = tp / max(n_pos, 1)

            if fpr <= fpr_target and recall > best_recall:
                best_recall = recall
                best_thr = thr

        metrics[f"{label}_recall_at_{int(fpr_target*1000)/10:.1f}pct_fpr"] = best_recall

        logger.info(
            "%s: recall@FPR<=%.2f%% = %.4f (thr=%s)",
            label,
            fpr_target * 100,
            best_recall,
            None if best_thr is None else f"{best_thr:.4f}",
        )

    return metrics


def log_confusion_and_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["Legitimate", "Fraud"])

    logger.info("Confusion Matrix (%s):\n%s", label, cm)
    logger.info("Classification Report (%s):\n%s", label, report)


# -------------------------------------------------------------------
# TRAINING
# -------------------------------------------------------------------

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> LGBMClassifier:
    """
    Train a baseline LGBM model with class weighting for imbalance.
    """
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_pos_weight = n_neg / max(n_pos, 1)

    logger.info(
        "Class distribution (train): pos=%d, neg=%d, scale_pos_weight=%.2f",
        n_pos,
        n_neg,
        scale_pos_weight,
    )

    clf = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
    )

    logger.info("Training model...")
    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
    )

    logger.info("Model training completed.")
    return clf


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main() -> None:
    # Ensure model dir exists
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_ENGINE_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------
    # 1. Load data
    # ---------------------------
    df = load_data(DATA_PATH)

    # ---------------------------
    # 2. Time-based split
    # ---------------------------
    train_df, val_df, test_df = time_based_split(df, ts_col="timestamp")

    # ---------------------------
    # 3. Fit FeatureEngine on TRAINING DATA ONLY
    # ---------------------------
    logger.info("Fitting FeatureEngine on TRAINING dataset for V2...")
    fe = FeatureEngine()
    fe.fit(train_df)

    # save FeatureEngine to disk (for API)
    fe.save(FEATURE_ENGINE_DIR)
    logger.info("FeatureEngine V2 saved to: %s", FEATURE_ENGINE_DIR)

    # ---------------------------
    # 4. Transform train/val/test
    # ---------------------------
    logger.info("Transforming train/val/test with FeatureEngine V2...")

    X_train, y_train = fe.transform(train_df)
    X_val, y_val = fe.transform(val_df)
    X_test, y_test = fe.transform(test_df)

    logger.info(
        "Feature matrix shapes -> "
        "X_train=%s, X_val=%s, X_test=%s",
        X_train.shape,
        X_val.shape,
        X_test.shape,
    )
    logger.info(
        "Label shapes -> "
        "y_train=%s, y_val=%s, y_test=%s",
        y_train.shape,
        y_val.shape,
        y_test.shape,
    )

    # ---------------------------
    # 5. MLflow setup
    # ---------------------------
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="baseline_lgbm_v2_with_fs_v2"):

        # Log basic params
        mlflow.log_param("model_type", "LGBMClassifier")
        mlflow.log_param("feature_engine_version", "v2")
        mlflow.log_param("data_path", str(DATA_PATH))
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_val", len(X_val))
        mlflow.log_param("n_test", len(X_test))

        # -------------------
        # 6. Train model
        # -------------------
        clf = train_model(X_train, y_train, X_val, y_val)

        # -------------------
        # 7. Evaluate
        # -------------------
        logger.info("Model training completed. Generating predictions...")

        y_val_proba = clf.predict_proba(X_val)[:, 1]
        y_test_proba = clf.predict_proba(X_test)[:, 1]

        # Default decision threshold = 0.5
        y_val_pred = (y_val_proba >= 0.5).astype(int)
        y_test_pred = (y_test_proba >= 0.5).astype(int)

        # Metrics
        val_metrics = compute_metrics(y_val.values, y_val_proba, label="val")
        test_metrics = compute_metrics(y_test.values, y_test_proba, label="test")

        # Log metrics
        logger.info("Validation metrics:")
        for k, v in val_metrics.items():
            logger.info("  %s: %.4f", k, v)
            mlflow.log_metric(k, v)

        logger.info("Test metrics:")
        for k, v in test_metrics.items():
            logger.info("  %s: %.4f", k, v)
            mlflow.log_metric(k, v)

        # Confusion matrices (threshold 0.5)
        log_confusion_and_report(y_val.values, y_val_pred, label="val")
        log_confusion_and_report(y_test.values, y_test_pred, label="test")

        # -------------------
        # 8. Save model locally
        # -------------------
        joblib.dump(clf, MODEL_PATH)
        logger.info("Saved model to: %s", MODEL_PATH)

        # -------------------
        # 9. Log artifacts to MLflow
        # -------------------
        # Log model
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
            registered_model_name=None,
        )

        # Log FeatureEngine artifacts (freq_maps + feature_order)
        mlflow.log_artifacts(str(FEATURE_ENGINE_DIR), artifact_path="feature_engine")

        logger.info(
            "FeatureEngine artifacts logged to MLflow under 'feature_engine/'."
        )


if __name__ == "__main__":
    main()









