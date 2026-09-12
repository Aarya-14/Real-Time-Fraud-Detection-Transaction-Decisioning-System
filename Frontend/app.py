# frontend/app.py

import os
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional

import requests
import streamlit as st
from pytz import timezone # Required for robust timezone handling

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

# Default backend URL (FastAPI)
# Note: We rely on the FastAPI server to load its own config from .env
BACKEND_URL = os.getenv("FRAUD_API_URL", "http://127.0.0.1:8000/score")

# Define key velocity features to extract and display separately
KEY_VELOCITY_FEATURES = [
    'user_txn_count_1h', 
    'user_amount_sum_1h', 
    'card_txn_count_24h'
]

# -------------------------------------------------------------------
# Small helpers
# -------------------------------------------------------------------

def call_fraud_api(
    amount: float,
    merchant_id: str | None,
    timestamp: str,
    card_id: str | None,
    user_id: str | None,
    use_chip: str,
    card_type: str,
    card_brand: str,
    merchant_category_code: str,
    merchant_state: str,
    mcc_description: str,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Call the FastAPI /score endpoint and return JSON response or error.
    """
    payload = {
        "amount": amount,
        "merchant_id": merchant_id or None,
        "timestamp": timestamp,
        "card_id": card_id or None,
        "user_id": user_id or None,
        "use_chip": use_chip,
        "card_type": card_type,
        "card_brand": card_brand,
        "merchant_category_code": merchant_category_code,
        "merchant_state": merchant_state,
        "mcc_description": mcc_description,
        # --- END PAYLOAD ADDITIONS ---
    }
    

    try:
        # Time out set to 5 seconds (Industry best practice)
        resp = requests.post(BACKEND_URL, json=payload, timeout=5)
    except requests.exceptions.RequestException as e:
        return None, f"Request error: Could not reach API endpoint. Is Uvicorn running? Details: {e}"

    if resp.status_code != 200:
        try:
            err_body = resp.json()
        except Exception:
            err_body = resp.text
        return None, f"API returned {resp.status_code}: {err_body}"

    try:
        data = resp.json()
    except Exception as e:
        return None, f"Failed to parse JSON response: {e}"

    return data, None


def format_reason_box(response: dict) -> str:
    """
    Build a short explanation text for the decision box, enhancing transparency.
    """
    score = response.get("score", 0.0)
    decision = response.get("decision", "UNKNOWN")
    thresholds = response.get("thresholds", {})

    pend = thresholds.get("pend", None)
    reject = thresholds.get("reject", None)

    lines = [f"**Decision:** `{decision}`"]
    lines.append(f"**Model score:** `{score:.4f}`")

    if pend is not None and reject is not None:
        lines.append(
            f"**Thresholds:** pend ≥ `{pend:.2f}`, reject ≥ `{reject:.2f}`"
        )

    # Simple, human-friendly interpretation
    if decision == "ALLOW":
        lines.append(
            "This transaction is considered **low risk** based on historical patterns and thresholds."
        )
    elif decision == "PEND":
        lines.append(
            "This transaction falls into a **grey zone**. It should be **reviewed manually** or "
            "require **step-up authentication** (e.g., 3D Secure)."
        )
    elif decision == "REJECT":
        lines.append(
            "This transaction looks **HIGH RISK** and should be **blocked** "
            "to prevent potential financial loss."
        )
    else:
        lines.append("Decision status is unknown. Please check FastAPI logs.")

    return "\n\n".join(lines)


# -------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------

st.set_page_config(
    page_title="Fraud Intelligence – Transaction Scoring",
    page_icon="💳",
    layout="wide", # Use wide layout for better data display
)

st.title("💳 SecurePay – Real-Time Transaction Scoring")

st.markdown(
    """
This UI simulates a payment initiation and sends the data to the fraud decision API. 
Use it to test **low latency**, **decision logic**, and the effectiveness of **velocity features** across multiple submissions of the same User/Card ID.
"""
)

# Sidebar – backend config
st.sidebar.header("Backend Settings")

backend_url_input = st.sidebar.text_input(
    "Fraud API URL",
    value=BACKEND_URL,
    help="FastAPI /score endpoint. Uses FRAUD_API_URL environment variable.",
)

if backend_url_input != BACKEND_URL:
    BACKEND_URL = backend_url_input # update runtime value

st.sidebar.markdown("---")
st.sidebar.caption(
    "Tip: keep the FastAPI server running with:\n\n"
    "`uvicorn src.api.app:app --reload`"
)

# -------------------------------------------------------------------
# Main form – transaction input
# -------------------------------------------------------------------

st.subheader(" Transaction Details")

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input(
        "Amount (USD)",
        min_value=0.01, # Enforce minimum value
        step=5.0,
        value=120.5,
        format="%.2f",
        help="Transaction amount to authorize.",
    )

    merchant_id = st.text_input(
        "Merchant ID",
        value="5300",
        help="Merchant identifier or MCC code (e.g., '5300' for Wholesale Clubs).",
    )

with col2:
    # Use unique IDs for testing velocity features in a fresh state
    user_id = st.text_input(
        "User ID",
        value="test_user_999",
        help="Customer / account identifier. Change this value to reset velocity history.",
    )

    card_id = st.text_input(
        "Card ID",
        value="card_888",
        help="Tokenized or internal card identifier. Change this value to reset card velocity.",
    )
st.markdown("#### Transaction Context Features")
col3, col4, col5 = st.columns(3)

with col3:
    use_chip = st.selectbox(
        "Transaction Type (use_chip)",
        options=["online transaction", "chip transaction", "swipe transaction"],
        index=0,
    )
    card_brand = st.selectbox(
        "Card Brand",
        options=["Visa", "Mastercard", "Amex", "Discover"],
        index=0,
    )

with col4:
    card_type = st.selectbox(
        "Card Type",
        options=["Credit", "Debit", "Debit (Prepaid)"],
        index=0,
    )
    merchant_category_code = st.text_input(
        "Merchant Category Code (MCC)",
        value="5300",
        help="4-digit MCC. Try '5814' (Fast Food) or '5300' (Wholesale)",
    )

with col5:
    merchant_state = st.text_input(
        "Merchant State",
        value="CA",
        max_chars=2,
        help="2-letter state code or 'ONLINE'.",
    )
    mcc_description = st.text_input(
        "MCC Description (Optional/Mock)",
        value="Wholesale Clubs",
        help="Description (used for frequency encoding).",
    )

# Timestamp Input with Current Time Default
# Timezone is crucial for velocity features
ts = st.datetime_input(
    "Transaction Time (UTC)",
    value=datetime.now(), # Start with current time
    help="Timestamp the transaction is initiated. Must be UTC for accurate velocity lookups.",
)

# Convert to timezone-aware UTC and then ISO 8601 string for the API
ts_utc = ts.astimezone(timezone('UTC')) 
timestamp_str = ts_utc.isoformat()

# Submit button
st.markdown("### ")
submit = st.button("Score Transaction", type="primary")

# -------------------------------------------------------------------
# Handle submission
# -------------------------------------------------------------------

if submit:
    # --- Client-Side Validation (Suggestion #2) ---
    if not user_id.strip() or not card_id.strip() or amount <= 0.0:
        st.error(" Error: Please ensure User ID, Card ID, and Amount (> 0.0) are provided.")
        st.stop()
    # -----------------------------------------------

    with st.spinner("Calling fraud API..."):
        response, error = call_fraud_api(
            amount=amount,
            merchant_id=merchant_id.strip() or None,
            timestamp=timestamp_str,
            card_id=card_id.strip() or None,
            user_id=user_id.strip() or None,
            use_chip=use_chip, 
            card_type=card_type, 
            card_brand=card_brand, 
            merchant_category_code=merchant_category_code, 
            merchant_state=merchant_state, 
            mcc_description=mcc_description, 
        )

    if error:
        st.error(f" Failed to score transaction:\n\n`{error}`")
    else:
        st.success(" Decision received from fraud engine")

        score = float(response.get("score", 0.0))
        decision = response.get("decision", "UNKNOWN")
        latency_ms = response.get("latency_ms", 0.0)

        # -------------------------------------------------------------------
        # 1. Top-level metric summary (Score, Decision, Thresholds, Latency)
        # -------------------------------------------------------------------
        st.markdown("####  Real-Time Metrics")
        col_a, col_b, col_c, col_d = st.columns(4)
        thr = response.get("thresholds", {})
        
        with col_a:
            st.metric("Decision", decision)
        with col_b:
            st.metric("Fraud Score", f"{score:.4f}")
        with col_c:
            st.metric(
                "Thresholds (P/R)",
                f"{thr.get('pend', 0.0):.2f} / {thr.get('reject', 0.0):.2f}",
                help="Pend Score / Reject Score"
            )
        with col_d:
            # Display latency as a key metric
            st.metric("API Latency", f"{latency_ms:.2f} ms")


        # -------------------------------------------------------------------
        # 2. Key Velocity Features (Suggestion #3)
        # -------------------------------------------------------------------
        st.markdown("####  Key Velocity Features (State before scoring)")
        
        velocity_cols = st.columns(len(KEY_VELOCITY_FEATURES))
        velocity_data = response.get('velocity_features', {})

        for i, feature_key in enumerate(KEY_VELOCITY_FEATURES):
            value = velocity_data.get(feature_key, 0)
            
            # Format display value for currency/count
            display_value = f"${value:.2f}" if 'amount' in feature_key else str(value)

            with velocity_cols[i]:
                st.metric(feature_key.replace('_', ' ').title(), display_value)
                

        # -------------------------------------------------------------------
        # 3. Explanation and Raw Response
        # -------------------------------------------------------------------
        st.markdown("####  Reasoning")
        st.markdown(format_reason_box(response))

        with st.expander(" Raw API Response (for Debugging)"):
            st.code(json.dumps(response, indent=2), language="json")

else:
    st.info("Fill the form and click **“ Score Transaction”** to simulate a payment.")