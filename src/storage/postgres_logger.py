# src/storage/postgres_logger.py

"""
PostgreSQL audit logger using Aiven PostgreSQL
Replaces JSONL file logging with database storage
"""

import asyncpg
import os
import time
from datetime import datetime
from typing import Dict, Optional
from dotenv import load_dotenv
from src.utils.logging_utils import get_logger 

load_dotenv()
logger = get_logger(__name__)


class PostgresAuditLogger:
    """
    Async PostgreSQL logger for fraud decisions, adapted for Cloud Run environment variables.
    Stores every decision in PostgreSQL for audit trail.
    """
    
    def __init__(self, postgres_uri: Optional[str] = None):
        """
        Initialize PostgreSQL connection settings, prioritizing decomposed
        Cloud Run/GCP variables if a full URI is not provided.
        """
        
        # 1. Try pre-built URIs (Aiven or standard DATABASE_URL)
        self.postgres_uri = postgres_uri or os.getenv("AIVEN_POSTGRES_URI") or os.getenv("DATABASE_URL")
        
        # 2. If URI is not found, try constructing it from Cloud Run/GCP decomposed variables
        if not self.postgres_uri:
            user = os.getenv("POSTGRES_USER")
            password = os.getenv("POSTGRES_PASSWORD")
            host = os.getenv("POSTGRES_HOST")
            database = os.getenv("POSTGRES_DB", "fraud_db") # Default to 'fraud_db'
            
            if all([user, password, host]):
                self.postgres_uri = f"postgresql://{user}:{password}@{host}/{database}"
                logger.info("PostgreSQL URI constructed successfully from decomposed environment variables.")
            else:
                # 3. Final failure if neither URI nor decomposed parts are available
                logger.critical("FATAL: PostgreSQL connection parameters (URI or decomposed parts) not set in environment.")
                raise ValueError("PostgreSQL connection URI not configured.")
        
        self.pool = None
    
    async def init_pool(self):
        """Initialize connection pool if it doesn't exist."""
        if not self.pool:
            try:
                self.pool = await asyncpg.create_pool(
                    self.postgres_uri,
                    min_size=1,
                    max_size=10,
                    command_timeout=60
                )
                logger.info("PostgreSQL connection pool initialized.")
                
                # OPTIONAL: You may want to call _create_table_if_not_exists() here
                # to ensure the table exists immediately after pool initialization.
                
            except Exception as e:
                logger.error(f" Failed to initialize PostgreSQL pool: {e}", exc_info=True)
                raise ConnectionError("Database pool creation failed.")


    async def log_decision(
        self,
        user_id: str,
        card_id: str,
        amount: float,
        merchant_id: str,
        transaction_ts: str, # The timestamp from the request
        model_version: str,
        fraud_score: float,
        decision: str,
        reason: str,
        latency_ms: float, # This is the API response time
        velocity_features: Dict
    ):
        """
        Log a fraud decision to PostgreSQL using the connection pool.
        """
        
        db_start_time = time.perf_counter()

        #  Ensure transaction_ts is a timezone-naive datetime
        # Cloud Run/GCP often prefers naive timestamps.

        try:
            # First, parse the ISO string into a datetime object
            transaction_dt = datetime.fromisoformat(transaction_ts)
            # Then, strip the timezone info to make it naive
            transaction_dt = transaction_dt.replace(tzinfo=None)
        except Exception as e:
            logger.error(f"Failed to parse or strip timezone from transaction_ts '{transaction_ts}': {e}")
            # Use current time as a last resort if parsing fails
            transaction_dt = datetime.utcnow().replace(tzinfo=None)
        
        # Pool initialization check (fallback)
        if not self.pool:
            logger.warning("PostgreSQL pool not initialized, attempting dynamic initialization.")
            try:
                await self.init_pool()
            except Exception:
                logger.error("Failed to initialize PostgreSQL pool dynamically. Aborting log.")
                return 
        
        # SQL INSERT Statement 
        sql = """
            INSERT INTO fraud_decisions (
                user_id, card_id, merchant_id, amount, transaction_ts,
                model_version, fraud_score, decision, reason, latency_ms, 
                velocity_features
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        """
        
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    sql,
                    user_id, card_id, merchant_id, amount, transaction_dt, # Use the parsed datetime
                    model_version, fraud_score, decision, reason, latency_ms,
                    velocity_features 
                )
            
            db_end_time = time.perf_counter()
            db_latency_ms = (db_end_time - db_start_time) * 1000
            
            logger.info(
                f"AUDIT SUCCESS: Decision={decision}, User={user_id}. "
                f"API Latency recorded: {latency_ms:.2f}ms. "
                f"DB Write Time: {db_latency_ms:.2f}ms"
            )
            
        except Exception as e:
            logger.error(f"AUDIT FAILURE: Failed to log decision for user={user_id} to PostgreSQL: {e}", exc_info=True)


    async def get_recent_decisions(self, limit: int = 10) -> list:
        # ... (Method remains unchanged)
        if not self.pool:
            logger.warning("PostgreSQL pool not initialized, attempting dynamic initialization.")
            try:
                await self.init_pool()
            except Exception:
                logger.error("Failed to initialize PostgreSQL pool dynamically. Aborting fetch.")
                return []
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT user_id, amount, decision, fraud_score, created_at, latency_ms
                FROM fraud_decisions
                ORDER BY created_at DESC
                LIMIT $1
            ''', limit)
            
            return [dict(row) for row in rows]
    
    async def close(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL connection pool closed.")