# src/ml/feature_store.py

"""
Offline feature store builder for fraud detection.

This module takes the base merged table (all_merged.parquet),
computes behavior / velocity features for user_id and card_id,
merges them back, and saves an enriched parquet:

    data/processed/all_merged_fs_v2.parquet

You can run it as:
    PYTHONPATH=. python -m src.ml.feature_store
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Core velocity feature function (moved from notebook)
# ---------------------------------------------------------------------


def add_entity_velocity(
    df: pd.DataFrame,
    entity_col: str,
    prefix: str,
    windows: Tuple[str, str] = ("1h", "24h"),
    time_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Add rolling transaction count / amount features per entity (user or card).

    For each window (e.g. "1h", "24h") we compute:
      - {prefix}_txn_count_{win}
      - {prefix}_amount_sum_{win}

    Requirements:
      - df contains: [entity_col, time_col, "amount"]
      - time_col is datetime (will be converted if needed)
    """
    logger.info(
        f"Computing velocity features for {entity_col} with windows={windows}..."
    )

    work = df[[entity_col, time_col, "amount"]].copy()

    # Ensure datetime & sorted
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work = work.sort_values([entity_col, time_col])

    # Set multi-index for efficient rolling
    work = work.set_index([entity_col, time_col]).sort_index()

    for win in windows:
        win_lower = win.lower()  # avoid 'H' FutureWarning

        count_col = f"{prefix}_txn_count_{win_lower}"
        sum_col = f"{prefix}_amount_sum_{win_lower}"

        # Rolling over time index per entity
        roll = (
            work
            .groupby(level=0)["amount"]
            .rolling(win_lower)          # time-based window, e.g. '1h', '24h'
        )

        work[count_col] = roll.count().values.astype("float32")
        work[sum_col] = roll.sum().values.astype("float32")

    # Bring index back as columns
    work = work.reset_index()

    # Merge back to original df
    df = df.merge(
        work[
            [entity_col, time_col]
            + [c for c in work.columns if c.startswith(f"{prefix}_")]
        ],
        on=[entity_col, time_col],
        how="left",
    )

    return df


# ---------------------------------------------------------------------
# Feature store builder
# ---------------------------------------------------------------------


def build_feature_store_v2(
    project_root: Path | None = None,
    base_rel_path: str = "data/processed/all_merged.parquet",
    output_rel_path: str = "data/processed/all_merged_fs_v2.parquet",
) -> Path:
    """
    Load base merged dataset, compute velocity features for user_id and card_id,
    merge them back, and save to a new parquet.

    Returns the output path.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]

    base_path = project_root / base_rel_path
    output_path = project_root / output_rel_path

    logger.info(f"Loading base merged data from: {base_path}")
    df = pd.read_parquet(base_path)
    logger.info(f"Base dataframe shape: {df.shape}")

    # We only need a subset for velocity computation
    vel_cols = ["user_id", "card_id", "timestamp", "amount"]
    for col in vel_cols:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' is missing from base dataframe.")

    df_vel = df[vel_cols].copy()

    # Compute per-user velocity
    df_vel = add_entity_velocity(
        df_vel,
        entity_col="user_id",
        prefix="user",
        windows=("1h", "24h"),
        time_col="timestamp",
    )

    # Compute per-card velocity
    df_vel = add_entity_velocity(
        df_vel,
        entity_col="card_id",
        prefix="card",
        windows=("1h", "24h"),
        time_col="timestamp",
    )

    logger.info("Velocity feature sample:")
    logger.info(df_vel.head().to_string())

    # Avoid duplicating amount (already present in df)
    df_vel = df_vel.drop(columns=["amount"])

    logger.info("Merging velocity features back into full table...")
    df_enriched = df.merge(
        df_vel,
        on=["user_id", "card_id", "timestamp"],
        how="left",
    )

    logger.info(f"Enriched dataframe shape: {df_enriched.shape}")

    # Ensure target is still present
    if "target" not in df_enriched.columns:
        raise KeyError("target column missing after merge!")

    # Save to parquet (create folder if needed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_enriched.to_parquet(output_path, index=False)
    logger.info(f"Saved enriched feature store to: {output_path}")

    return output_path


# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------


if __name__ == "__main__":
    """
    Allow running as:

        PYTHONPATH=. python -m src.ml.feature_store
    """
    try:
        out = build_feature_store_v2()
        logger.info(f"Feature store V2 build complete. Output: {out}")
    except Exception as e:
        logger.exception(f"Failed to build feature store V2: {e}")
        raise
