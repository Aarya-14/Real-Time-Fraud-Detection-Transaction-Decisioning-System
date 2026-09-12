import logging
# Assuming your logging utility has a function/method to get a configured logger
# You might need to adjust the import path based on your actual file location (e.g., src.utils.logging_utils)
from src.utils.logging_utils import get_logger 
from src.features.redis_feature_store import RedisFeatureStore

# --- 1. Get Configured Logger ---
# Get a logger instance, named for this specific module
logger = get_logger(__name__) 
# -----------------------------------

def test_redis_connection():
    """Test Redis connection"""
    logger.info("Starting Redis connection test...")
    try:
        # Assuming RedisFeatureStore initialization handles connection status checks internally
        store = RedisFeatureStore()
        stats = store.get_stats()
        
        logger.info("✅ Redis connected successfully.")
        logger.info("Stats: %s", stats)
        
    except Exception as e:
        logger.error("❌ Failed to connect to Redis. Error: %s", e)
        # In a real test file, you might want to raise the exception here
        
def test_velocity_features():
    """Test velocity feature storage and retrieval"""
    logger.info("Starting velocity feature tests...")
    store = RedisFeatureStore()
    
    # Clear existing data
    store.clear_all()
    logger.info("Cleared existing Redis data.")
    
    # Simulate 3 transactions
    user_id = 1556
    card_id = 2972
    
    # --- Transaction 1 ---
    logger.info("--- Transaction 1 ---")
    store.update_velocity(user_id, card_id, 50.0)
    features = store.get_velocity_features(user_id, card_id)
    logger.info("Features after txn 1: %s", features)
    
    try:
        assert features.get('user_txn_count_1h') == 1
        assert features.get('user_amount_sum_1h') == 50.0
    except AssertionError:
        logger.error("Assertion failed for Transaction 1 features!")
        raise
    
    # --- Transaction 2 ---
    logger.info("--- Transaction 2 ---")
    store.update_velocity(user_id, card_id, 75.0)
    features = store.get_velocity_features(user_id, card_id)
    logger.info("Features after txn 2: %s", features)
    
    try:
        assert features.get('user_txn_count_1h') == 2
        assert features.get('user_amount_sum_1h') == 125.0
    except AssertionError:
        logger.error("Assertion failed for Transaction 2 features!")
        raise
    
    # --- Transaction 3 ---
    logger.info("--- Transaction 3 ---")
    store.update_velocity(user_id, card_id, 100.0)
    features = store.get_velocity_features(user_id, card_id)
    logger.info("Features after txn 3: %s", features)
    
    try:
        assert features.get('user_txn_count_1h') == 3
        assert features.get('user_amount_sum_1h') == 225.0
    except AssertionError:
        logger.error("Assertion failed for Transaction 3 features!")
        raise
    
    logger.info("\n✅ All velocity feature tests passed successfully!")

if __name__ == "__main__":
    test_redis_connection()
    test_velocity_features()