# KAFKA CONSUMER - src/streaming/kafka_consumer.py
# Consumes transactions from Kafka and scores them


import os
import json
import time
import asyncio
import logging
from typing import Optional
from kafka import KafkaConsumer
from kafka.errors import KafkaError

logger = logging.getLogger(__name__)


class FraudKafkaConsumer:
    """Kafka consumer that reads transactions and scores them."""
    
    def __init__(self, fraud_model, decision_engine, redis_store, postgres_logger):
        """Initialize consumer with fraud detection components."""
        self.fraud_model = fraud_model
        self.decision_engine = decision_engine
        self.redis_store = redis_store
        self.postgres_logger = postgres_logger
        self.consumer: Optional[KafkaConsumer] = None
        self.running = False
        
    def create_consumer(self) -> KafkaConsumer:
        """Create Kafka consumer with SSL configuration."""
        
        brokers = os.getenv("KAFKA_BROKERS")
        topic = os.getenv("KAFKA_TOPIC", "raw_transactions")
        group_id = os.getenv("KAFKA_GROUP_ID", "fraud-scoring-service")
        
        if not brokers:
            raise ValueError("KAFKA_BROKERS environment variable not set")
        
        # Check for mounted secret files
        mounted_ca = "/kafka-ca/secret"
        mounted_cert = "/kafka-cert/secret"
        mounted_key = "/kafka-key/secret"
        
        if not all(os.path.exists(p) for p in [mounted_ca, mounted_cert, mounted_key]):
            raise ValueError("Kafka SSL certificates not found at /kafka-ca/, /kafka-cert/, /kafka-key/")
        
        logger.info(f"Creating Kafka consumer for topic: {topic}, group: {group_id}")
        
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=brokers.split(','),
            group_id=group_id,
            security_protocol='SSL',
            ssl_cafile=mounted_ca,
            ssl_certfile=mounted_cert,
            ssl_keyfile=mounted_key,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='earliest',  # Start from beginning if no offset
            enable_auto_commit=True,
            auto_commit_interval_ms=1000,
            max_poll_records=10,  # Process 10 messages at a time
            session_timeout_ms=30000,
        )
        
        logger.info("Kafka consumer created successfully")
        return consumer
    
    async def process_transaction(self, message):
        """Process a single transaction from Kafka."""
        start_time = time.time()
        
        try:
            # Extract transaction data
            transaction = message.value
            transaction_id = transaction.get('transaction_id', 'unknown')
            user_id = transaction.get('user_id')
            card_id = transaction.get('card_id')
            amount = transaction.get('amount')
            
            logger.info(f" Processing transaction: {transaction_id} | User: {user_id} | Amount: ${amount}")
            
            # Get velocity features from Redis
            velocity_features = await self.redis_store.get_velocity_features(user_id, card_id)
            
            # Merge transaction with velocity features
            transaction_with_features = {**transaction, **velocity_features}
            
            # Get static user profile (simulated - in production, fetch from DB)
            user_profile = await self._get_user_profile(user_id)
            transaction_with_features.update(user_profile)
            
            # Score transaction
            score = self.fraud_model.score_transaction(transaction_with_features)
            
            # Get decision
            decision_result = self.decision_engine.decide(score)
            
            # Update Redis velocity
            await self.redis_store.update_velocity(user_id, card_id, amount)
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Log to PostgreSQL (async, non-blocking)
            if self.postgres_logger:
                from datetime import datetime
                transaction_ts = datetime.fromisoformat(transaction['timestamp'].replace('Z', ''))
                
                asyncio.create_task(
                    self.postgres_logger.log_decision(
                        user_id=user_id,
                        card_id=card_id,
                        amount=amount,
                        merchant_id=transaction.get('merchant_id', 'unknown'),
                        transaction_ts=transaction_ts,
                        model_version="v2_lgbm_kafka",
                        fraud_score=score,
                        decision=decision_result.decision,
                        reason=decision_result.reason,
                        latency_ms=latency_ms,
                        velocity_features=json.dumps(velocity_features)
                    )
                )
            
            logger.info(
                f"Scored: {transaction_id} | Decision: {decision_result.decision} | "
                f"Score: {score:.4f} | Latency: {latency_ms:.2f}ms"
            )
            
            return {
                "transaction_id": transaction_id,
                "score": float(score),
                "decision": decision_result.decision,
                "latency_ms": latency_ms
            }
            
        except Exception as e:
            logger.error(f" Error processing transaction: {e}", exc_info=True)
            return None
    
    async def _get_user_profile(self, user_id: str) -> dict:
        """Simulate fetching user profile (replace with actual DB call in production)."""
        await asyncio.sleep(0.01)  # Simulate DB lookup
        
        # Default profile
        return {
            "yearly_income": 60000.0,
            "total_debt": 15000.0,
            "current_age": 35,
            "credit_score": 700,
            "num_credit_cards": 2
        }
    
    async def consume_loop(self):
        """Main consumer loop - runs in background."""
        logger.info(" Starting Kafka consumer loop...")
        self.running = True
        
        try:
            self.consumer = self.create_consumer()
            
            while self.running:
                # Poll for messages (timeout 1 second)
                messages = self.consumer.poll(timeout_ms=1000)
                
                if not messages:
                    await asyncio.sleep(0.1)  # No messages, brief sleep
                    continue
                
                # Process messages
                for topic_partition, records in messages.items():
                    for message in records:
                        try:
                            await self.process_transaction(message)
                        except Exception as e:
                            logger.error(f" Error in message processing: {e}")
                            # Continue processing other messages
                
        except Exception as e:
            logger.error(f" Consumer loop error: {e}", exc_info=True)
        finally:
            if self.consumer:
                self.consumer.close()
                logger.info(" Kafka consumer closed")
    
    def stop(self):
        """Stop the consumer loop."""
        logger.info("Stopping Kafka consumer...")
        self.running = False
