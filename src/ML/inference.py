# src/ml/inference.py

from src.api.schemas import TransactionRequest
from src.ml.feature_eng import transform_single


def predict_proba(transaction: TransactionRequest) -> float:
    """
    Dummy prediction function.

    For Phase 1, we ignore real ML and just:
    - call transform_single (so the pipeline shape is correct)
    - return a fixed probability (e.g. 0.1)

    Later, you will:
    - load a trained model
    - compute features
    - return model.predict_proba(features)
    """
    features = transform_single(transaction)
    # You can print features for debugging if you like:
    # print("Features:", features)

    # Dummy: always return 0.1 fraud probability
    return 0.1
