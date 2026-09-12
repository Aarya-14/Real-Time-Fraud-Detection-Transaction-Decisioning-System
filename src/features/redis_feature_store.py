"""
Real-time feature store using Aiven Redis
Replaces static parquet lookups with live velocity computation
"""


import redis.asyncio as redis 
import os
import logging
from typing import Dict, Optional, List
from dotenv import load_dotenv

# Set up logging for this module
logger = logging.getLogger("fraud-system.redis-store")
logger.setLevel(logging.INFO)

# Load environment variables, useful for local testing
load_dotenv()


class RedisFeatureStore:
    """
    Real-time feature store for velocity features, utilizing asynchronous I/O (redis.asyncio).
    Handles connection via a full URI or decomposed Cloud Run environment variables.
    This component is crucial for the synchronous, low-latency scoring path.
    """
    
    # Define time constants for clarity
    TTL_1H = 3600    # 1 hour in seconds
    TTL_24H = 86400  # 24 hours in seconds

    # Define all feature keys for cleaner access
    FEATURE_KEYS = {
        'user': ['txn:1h', 'txn:24h', 'amt:1h', 'amt:24h'],
        'card': ['txn:1h', 'txn:24h', 'amt:1h', 'amt:24h']
    }
    
    def __init__(self, redis_uri: Optional[str] = None):
        """
        Initialize asynchronous Redis connection.
        """
        
        # 1. Try pre-built URI (Aiven or user-supplied)
        final_redis_uri = redis_uri or os.getenv("AIVEN_REDIS_URI")
        
        # 2. If URI is not found, try constructing it from decomposed Cloud Run/GCP variables
        if not final_redis_uri:
            host = os.getenv("REDIS_HOST")
            port = os.getenv("REDIS_PORT")
            
            if host and port:
                # Assuming basic setup with optional password
                password = os.getenv("REDIS_PASSWORD")
                auth = f":{password}@" if password else ""
                final_redis_uri = f"redis://{auth}{host}:{port}/0"
                logger.info("Redis URI constructed successfully from decomposed environment variables.")
            else:
                # 3. Final failure if connection details are missing
                logger.critical("FATAL: Redis connection parameters (URI or decomposed HOST/PORT) not set.")
                raise ValueError("Redis connection details not configured.")
        
        # Initialize the asynchronous Redis client
        self.client = redis.from_url(
            final_redis_uri,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5
        )
        logger.info(f"Redis client initialized. Host: {final_redis_uri.split('@')[-1].split('/')[0]}")
    
    async def __aenter__(self):
        """Allows connection testing during application startup (FastAPI lifespan)."""
        try:
            await self.client.ping()
            logger.info("Redis connection successful.")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            raise
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close connection on shutdown."""
        await self.client.close()
    
    
    async def get_velocity_features(self, user_id: str, card_id: str) -> Dict[str, float]:
        """
        Get current velocity features for a user and card using non-blocking I/O (MGET).
        """
        
        user_id_str, card_id_str = str(user_id), str(card_id)

        # 1. Build list of keys to fetch 
        keys_to_fetch = []
        for time_window in self.FEATURE_KEYS['user']:
            keys_to_fetch.append(f"user:{user_id_str}:{time_window}")
        for time_window in self.FEATURE_KEYS['card']:
            keys_to_fetch.append(f"card:{card_id_str}:{time_window}")

        # 2. Use MGET (single round-trip I/O)
        values = await self.client.mget(keys_to_fetch)
        
        features = {}
        
        # 3. Process results and map to model's expected feature names
        for i, key in enumerate(keys_to_fetch):
            value = values[i]
            
            parts = key.split(':') 
            entity = parts[0]
            metric_abbr = parts[2]
            window = parts[3]

            if metric_abbr == 'txn':
                # Model expects: user_txn_count_1h
                feature_name = f"{entity}_txn_count_{window}"
                # Safely convert to integer, defaulting to 0
                features[feature_name] = int(value) if value else 0
            else: # metric_abbr == 'amt'
                # Model expects: user_amount_sum_1h
                feature_name = f"{entity}_amount_sum_{window}"
                # Safely convert to float, defaulting to 0.0
                features[feature_name] = float(value) if value else 0.0
            
        return features
    
    async def update_velocity(self, user_id: str, card_id: str, amount: float):
        """
        Update velocity counters using non-blocking Redis Pipelining.
        """
        user_id_str, card_id_str = str(user_id), str(card_id)
        
        # Use a Pipeline to send all commands in one atomic round trip
        pipe = self.client.pipeline()
        
        # --- Define all 8 commands for user and card across 1h and 24h windows ---
        
        # User Updates (1h & 24h)
        pipe.incr(f"user:{user_id_str}:txn:1h")
        pipe.incrbyfloat(f"user:{user_id_str}:amt:1h", amount)
        pipe.expire(f"user:{user_id_str}:txn:1h", self.TTL_1H)
        pipe.expire(f"user:{user_id_str}:amt:1h", self.TTL_1H)
        
        pipe.incr(f"user:{user_id_str}:txn:24h")
        pipe.incrbyfloat(f"user:{user_id_str}:amt:24h", amount)
        pipe.expire(f"user:{user_id_str}:txn:24h", self.TTL_24H)
        pipe.expire(f"user:{user_id_str}:amt:24h", self.TTL_24H)

        # Card Updates (1h & 24h)
        pipe.incr(f"card:{card_id_str}:txn:1h")
        pipe.incrbyfloat(f"card:{card_id_str}:amt:1h", amount)
        pipe.expire(f"card:{card_id_str}:txn:1h", self.TTL_1H)
        pipe.expire(f"card:{card_id_str}:amt:1h", self.TTL_1H)
        
        pipe.incr(f"card:{card_id_str}:txn:24h")
        pipe.incrbyfloat(f"card:{card_id_str}:amt:24h", amount)
        pipe.expire(f"card:{card_id_str}:txn:24h", self.TTL_24H)
        pipe.expire(f"card:{card_id_str}:amt:24h", self.TTL_24H)

        # Execute the pipeline asynchronously
        await pipe.execute() 
    
    async def clear_all(self):
        """Clear all velocity data (for testing) using asynchronous commands."""
        keys_to_delete: List[str] = []

        # Use SCAN_ITER to avoid blocking the server if the dataset is large
        async for key in self.client.scan_iter("user:*"):
            keys_to_delete.append(key)
        
        async for key in self.client.scan_iter("card:*"):
            keys_to_delete.append(key)

        if keys_to_delete:
            await self.client.delete(*keys_to_delete)
            logger.info(f"Redis cleared {len(keys_to_delete)} keys.")
        else:
            logger.info("No Redis keys found to clear.")

    async def get_stats(self) -> Dict:
        """Get Redis statistics asynchronously."""
        try:
            info = await self.client.info()
            dbsize = await self.client.dbsize()
            return {
                "total_keys": dbsize,
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "unknown")
            }
        except Exception as e:
            logger.error(f"Failed to fetch Redis stats: {e}")
            return {"error": str(e)}