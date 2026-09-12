# src/core/config.py

"""
Central config module.

Thresholds for fraud decisions live in config/thresholds.json
and are loaded dynamically by DecisionEngine.
"""

from pathlib import Path

# Project root (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Model artifacts folder
MODEL_DIR = PROJECT_ROOT / "models"

# Data folder
DATA_DIR = PROJECT_ROOT / "data"

# Config folder
CONFIG_DIR = PROJECT_ROOT / "config"

