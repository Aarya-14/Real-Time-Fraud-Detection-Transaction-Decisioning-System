# src/api/app.py

import time
from datetime import datetime
import os
import json
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager # Required for modern FastAPI lifecycle management

from fastapi import FastAPI, HTTPException 
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from src.api.schemas import TransactionRequest, FraudDecisionResponse # Added FraudDecisionResponse
from src.core.decision import FraudModel, DecisionEngine
from src.streaming.kafka_consumer import FraudKafkaConsumer
from src.utils.logging_utils import get_logger
import asyncio


from src.features.redis_feature_store import RedisFeatureStore
from src.storage.postgres_logger import PostgresAuditLogger 


load_dotenv()
logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 1. Global Service Instances (Must be managed by lifespan) 
# Define instances as Optional and initialize them in the lifespan function.
fraud_model: Optional[FraudModel] = None
decision_engine: Optional[DecisionEngine] = None
redis_store: Optional[RedisFeatureStore] = None
postgres_logger: Optional[PostgresAuditLogger] = None

# ASYNCHRONOUS STATIC PROFILE LOOKUP SIMULATION 
async def get_user_profile(user_id: str) -> dict:
    """
    Simulates fetching static user profile data required by the model.
    """
    # Simulate network latency (20ms non-blocking wait)
    await asyncio.sleep(0.02) 

    # Profile for E2E Test User
    if "e2e_user" in user_id:
        return {
            # REQUIRED STATIC FEATURES from feature_order.json:
            "yearly_income": 85000.0,
            "total_debt": 22000.0,
            "current_age": 42,
            "credit_score": 760,
            "num_credit_cards": 3
        }
    
    # Default/fallback profile for manual testing
    return {
        "yearly_income": 60000.0,
        "total_debt": 15000.0,
        "current_age": 35,
        "credit_score": 700,
        "num_credit_cards": 2
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes services before startup and cleans up on shutdown."""
    global fraud_model, decision_engine, redis_store, postgres_logger
    
    logger.info("Starting API Service Lifespan...")
    
    # --- STARTUP ---

    # 1. Initialize Postgres Audit Logger (First, as it's critical for audit)
    logger.info("Initializing PostgreSQL Audit Logger...")
    try:
        postgres_logger = PostgresAuditLogger()
        await postgres_logger.init_pool()
    except Exception as e:
        logger.error(f"FATAL: Could not initialize Postgres Logger pool: {e}", exc_info=True)
        # Note: API will start without audit logging if DB fails, allowing fail-safe operation.
        
    # 2. Initialize Redis Feature Store
    logger.info("Initializing Redis Feature Store...")
    redis_store = RedisFeatureStore() # Initialization is synchronous here
    
    # 3. Load Model and Decision Engine
    logger.info("Loading FraudModel and DecisionEngine...")
    fraud_model = FraudModel.load_from_disk(project_root=PROJECT_ROOT)
    decision_engine = DecisionEngine.load_from_config(project_root=PROJECT_ROOT)

    # 4. Initialize Kafka Consumer  
    kafka_consumer = None
    try:
        logger.info("Initializing Kafka Consumer...")
        kafka_consumer = FraudKafkaConsumer(
            fraud_model=fraud_model,
            decision_engine=decision_engine,
            redis_store=redis_store,
            postgres_logger=postgres_logger
        )
        
        # Start consumer in background
        asyncio.create_task(kafka_consumer.consume_loop())
        logger.info("Kafka consumer started in background")
        
    except Exception as e:
        logger.warning(f" Kafka consumer failed to start: {e}")
        kafka_consumer = None

    logger.info("All core services initialized.")
    yield # Application serves requests
    
    # --- SHUTDOWN ---
    logger.info("Shutting down services...")
    
    # Stop Kafka consumer
    if kafka_consumer:
        kafka_consumer.stop()

    #  Close Postgres Logger Pool
    if postgres_logger:
        await postgres_logger.close()
        
    # No explicit async shutdown needed for RedisFeatureStore or Model
    logger.info("Services shut down complete.")


# --- 2. FastAPI Application Definition ---
app = FastAPI(
    title="Fraud Detection API",
    version="0.1.0",
    description="Real-time fraud scoring and decisioning service.",
    lifespan=lifespan # NEW: Use the modern lifespan context manager
)

# CORS Middleware 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    """Simple health check to verify service is running."""
    # Check if critical services are loaded/initialized
    status = "ok" if fraud_model and decision_engine and redis_store else "degraded"
    return {"status": status}


@app.post("/score", response_model=FraudDecisionResponse)
async def score_transaction(request: TransactionRequest):
    """Score a transaction for fraud, incorporating real-time velocity features."""
    start_time = time.time()
    
    # -------------------------------------------------------------
    # 0. System Health Check
    # -------------------------------------------------------------
    if not fraud_model or not decision_engine or not redis_store:
        logger.error("Service not fully initialized. Returning PEND.")
        raise HTTPException(
            status_code=503, 
            detail="System is currently unavailable (core services failed to load)."
        )

    logger.info(
        f"Received request: user={request.user_id}, card={request.card_id}, amount={request.amount}"
    )

    try:
        # -------------------------------------------------------------
        # 1. CONCURRENTLY GET ALL REQUIRED FEATURES
        # -------------------------------------------------------------
        # Start both the static profile lookup (simulated I/O) and 
        # the velocity feature lookup (Redis I/O) simultaneously.
        profile_task = get_user_profile(request.user_id)
        velocity_task = redis_store.get_velocity_features(
            request.user_id,
            request.card_id
        )

        # Wait for both tasks to complete concurrently
        user_profile_features, velocity_features = await asyncio.gather(
            profile_task, velocity_task
        )

        
        # 2. Build complete feature set for model (CRITICAL MERGE STEP)
        
        request_dict = request.model_dump()
        
        # Merge static profile features (yearly_income, credit_score, etc.)
        request_dict.update(user_profile_features) 
        
        # Merge velocity features (txn_count_1h, amt_sum_24h, etc.)
        request_dict.update(velocity_features)

        # 3. Score transaction (CPU-bound, no await needed here)
        logger.info("Transforming transaction and scoring with full feature set.")
        score = fraud_model.score_transaction(request_dict)

        # 4. Get decision
        decision_result = decision_engine.decide(score)
        
        # -------------------------------------------------------------
        # 5. CRITICAL: Update Redis velocity (MUST be awaited)
        # -------------------------------------------------------------
        await redis_store.update_velocity( # <- ADDED await
            request.user_id,
            request.card_id,
            request.amount
        )

        # -------------------------------------------------------------
        # 6. Calculate FINAL Latency (This is the time reported to the client)
        # -------------------------------------------------------------
        latency_ms = (time.time() - start_time) * 1000
        
        # -------------------------------------------------------------
        # 7. DECOUPLE AUDIT LOGGING 
        # -------------------------------------------------------------
        if postgres_logger:
            try:
                # Prepare arguments for the background task
                # Convert timestamp string to datetime object
                #transaction_dt = datetime.fromisoformat(request.timestamp)
                transaction_dt_aware = datetime.fromisoformat(request.timestamp)
                transaction_dt = transaction_dt_aware.replace(tzinfo=None)

                velocity_features_json = json.dumps(velocity_features)
                
                # Initiate the logging task in the background without 'await'
                # This prevents the slow DB write from blocking the HTTP response.
                asyncio.create_task(
                    postgres_logger.log_decision( 
                        user_id=request.user_id,
                        card_id=request.card_id,
                        amount=request.amount,
                        merchant_id=request.merchant_id,
                        transaction_ts=transaction_dt, 
                        model_version="v2_lgbm_baseline", 
                        fraud_score=decision_result.score,
                        decision=decision_result.decision,
                        reason=decision_result.reason,
                        latency_ms=latency_ms,
                        velocity_features=velocity_features_json
                    )
                )
                logger.info(
                    f"Audit logging initiated for user={request.user_id} as background task."
                )
            except Exception as e:
                # Log the failure but allow the transaction to proceed (fail-safe)
                logger.error(
                    f"Failed to create background logging task: {e}", exc_info=True
                )
        else:
            logger.warning("PostgreSQL logger is not initialized. Decision not audited to DB.")

        # -------------------------------------------------------------
        # 8. Build and return response IMMEDIATELY
        # -------------------------------------------------------------
        response_data = {
            "score": float(score),
            "decision": decision_result.decision,
            "reason": decision_result.reason,
            "thresholds": decision_result.thresholds,
            "latency_ms": round(latency_ms, 2),
            "velocity_features": velocity_features
        }
        
        logger.info(
            f"/score response -> decision={decision_result.decision}, score={score:.4f}, "
            f"latency={latency_ms:.2f}ms"
        )
        
        return FraudDecisionResponse(**response_data)

    except Exception as e:
        logger.error(f"Critical error scoring transaction: {e}", exc_info=True)
        # On scoring failure, return a PEND with error reason.
        return FraudDecisionResponse(
            decision="PEND", 
            reason="SYSTEM_ERROR", 
            score=0.5, # Default to a mid-range score on failure
            thresholds=decision_engine.thresholds if decision_engine else {},
            latency_ms=(time.time() - start_time) * 1000,
            velocity_features={}
        )


@app.get("/recent_decisions")
async def get_recent_decisions(limit: int = 10):
    """Get recent fraud decisions from PostgreSQL"""
    try:
        decisions = await postgres_logger.get_recent_decisions(limit)
        return {"decisions": decisions, "count": len(decisions)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
