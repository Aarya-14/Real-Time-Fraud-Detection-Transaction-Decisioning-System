# src/api/schemas.py

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

# --- Input Schema ---

class TransactionRequest(BaseModel):
    """Schema for the incoming transaction request."""
    user_id: str = Field(..., example="e2e_user_123")
    card_id: str = Field(..., example="e2e_card_456")
    amount: float = Field(..., gt=0)
    merchant_id: str = Field(..., example="19752") # Changed example to match your data format
    timestamp: str = Field(..., example="2025-12-07T15:21:45.000")
    
    # CRITICAL ADDITIONS TO SATISFY FEATURE ENGINE (using values from your data):
    use_chip: str = Field(..., example="swipe transaction", description="Method of transaction (swipe, chip, online)")
    card_type: str = Field(..., example="Credit", description="Credit or Debit")
    card_brand: str = Field(..., example="Visa", description="Visa, Mastercard, Amex, Discover")
    merchant_category_code: str = Field(..., example="5411", description="4-digit MCC (e.g., 5411 for Groceries)")
    merchant_state: str = Field(..., example="CA", description="2-letter state code, or ONLINE")
    mcc_description: str = Field(..., example="Grocery Stores, Supermarkets", description="Description of the MCC")



class DecisionResponse(BaseModel):
    """A simpler schema for model output (Score, Decision)."""
    probability: float
    decision: str

# --- Output Schema (Required for the /score endpoint) ---

class FraudDecisionResponse(BaseModel):
    """
    Schema for the complete response returned by the /score endpoint,
    including audit and performance data.
    """
    score: float
    decision: str
    reason: str
    thresholds: Dict[str, float]
    latency_ms: float
    velocity_features: Dict[str, Any] # Use Any as feature values can be float or int