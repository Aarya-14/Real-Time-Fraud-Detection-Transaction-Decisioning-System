
# tests/test_e2e.py


import requests
import time
import logging
import uuid
from dotenv import load_dotenv
import redis # <-- NEW IMPORT
import os # To access environment variables
from typing import Dict, Any

# --- Load Environment Variables ---
load_dotenv()

# --- 1. Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ...
TEST_SUFFIX = str(uuid.uuid4())[:8]
TEST_USER_ID = f"e2e_user_{TEST_SUFFIX}"
TEST_CARD_ID = f"e2e_card_{TEST_SUFFIX}"

# --- 2. Configuration & Redis Client ---
API_URL = "http://localhost:8000"
# Use environment variable (REDIS_URL) for connecting to Redis for cleanup
REDIS_URL = os.environ.get("AIVEN_REDIS_URI") 

# Check for safety
if not REDIS_URL:
    raise EnvironmentError("AIVEN_REDIS_URI not found. Please check your .env file.")

redis_client = redis.from_url(REDIS_URL)

# --- 3. Helper Function ---
def log_and_assert_response(response: requests.Response, expected_status: int = 200) -> Dict[str, Any]:
    """Helper to check status, log details, and return JSON."""
    assert response.status_code == expected_status, f" Test Failed: Expected status {expected_status}, got {response.status_code}"
    result = response.json()
    
    logger.info(f"Status: {response.status_code}")
    logger.info(f"Decision: {result.get('decision')}")
    logger.info(f"Score: {result.get('score'):.4f}")
    logger.info(f"Latency: {result.get('latency_ms', 0):.2f}ms")
    
    return result

# --- 4. Main Test Function ---
def test_full_flow():
    """Test complete fraud detection flow: Scoring, Velocity Update, and Logging."""
    
    # --- Define Test IDs ---
    #TEST_USER_ID = "test_user_1556" # Use unique ID to avoid collision with live data
    #TEST_CARD_ID = "test_card_2972"
    
    # --- GLOBAL SETUP/TEARDOWN: CLEAN REDIS ---
    logger.info(" Cleaning up Redis keys before running tests...")
    
    # The keys used by RedisFeatureStore are typically f'user:{ID}' and f'card:{ID}'
    redis_client.delete(f"user:{TEST_USER_ID}", f"card:{TEST_CARD_ID}")
    # redis_client.delete(f"user:{TEST_USER_ID}", f"card:{TEST_CARD_ID}")
    
    logger.info(" Testing End-to-End Flow")
    logger.info("-" * 35)
    
    # Base transaction payload
    # REQUIRED FIELDS (Aligned with training data):
    txn_base = {
        "user_id": TEST_USER_ID,
        "card_id": TEST_CARD_ID,
        "amount": 50.00,
        "timestamp": "2025-12-07T10:30:00",
        "merchant_id": "19752", 
        "use_chip": "swipe transaction", 
        "card_type": "Credit",
        "card_brand": "Visa",
        "merchant_category_code": "5411",
        "merchant_state": "MS",
        "mcc_description": "Grocery Stores, Supermarkets",
    }
    
    
    # --- Test 1: Legitimate transaction (Initial state) ---
    logger.info("--- Test 1: Initial Transaction (Velocity Check Before Update) ---")
    response = requests.post(f"{API_URL}/score", json=txn_base)
    result1 = log_and_assert_response(response)
    
    logger.info(f"Velocity Features: {result1['velocity_features']}")
    
    # FIX: The feature update happens AFTER the response is generated.
    # Therefore, the initial transaction should show a count of 0 (the state BEFORE the txn).
    assert result1['velocity_features']['user_txn_count_1h'] == 0 
    logger.info(" Initial velocity check correct (count = 0 before update).")
    
    logger.info("-" * 35)
    
    # --- Test 2: Rapid second transaction (should show velocity update) ---
    logger.info("--- Test 2: Rapid Second Transaction (Velocity Check After 1st Txn) ---")
    
    # This second transaction will read features that were updated by the first transaction.
    txn2 = {**txn_base, "amount": 75.00, "timestamp": "2025-12-07T10:30:01"}
    time.sleep(0.5) 
    
    response = requests.post(f"{API_URL}/score", json=txn2)
    result2 = log_and_assert_response(response)
    
    logger.info(f"Velocity Features: {result2['velocity_features']}")
    # The count should now be 1, as the first transaction updated the features.
    assert result2['velocity_features']['user_txn_count_1h'] == 1
    logger.info(" Velocity count updated correctly to 1!")
    
    logger.info("-" * 35)
    
    # --- Test 3: Check recent decisions (PostgreSQL Audit Check) ---
    logger.info("--- Test 3: Recent Decisions (PostgreSQL Audit Check) ---")
    
    # Give Postgres a moment to guarantee the second log is committed
    time.sleep(1) 
    response = requests.get(f"{API_URL}/recent_decisions")
    assert response.status_code == 200, " Failed to retrieve recent decisions."
    
    data = response.json()
    logger.info(f"Found {data['count']} recent decisions in DB.")
    
    # We expect at least 2 logs from the two test transactions
    assert data['count'] >= 2, f" Expected at least 2 logs, found {data['count']}"
    
    # Check that the latest log record matches the latest transaction (txn2: 75.00)
    latest_log = data['decisions'][0]
    assert latest_log['amount'] == 75.00, " Latest log amount mismatch."
    
    logger.info(" PostgreSQL logging validated successfully.")
    
    logger.info("-" * 35)
    logger.info("All End-to-End tests passed successfully!")

if __name__ == "__main__":
    test_full_flow()