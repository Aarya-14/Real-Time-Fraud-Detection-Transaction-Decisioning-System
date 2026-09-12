# Real-Time-Fraud-Detection-Transaction-Decisioning-System
Fraud Detection System built for real-time payment decisioning.
The system consists of:

- An **offline ML pipeline** that:
  - builds an enriched **feature store** (lifetime + velocity features),
  - trains a **LightGBM** fraud model,
  - tracks experiments with **MLflow**,
  - exports the model and feature engine as artifacts.

- An **online real-time pipeline** that:
  - receives payment transactions via a **FastAPI** service,
  - applies the same **FeatureEngine** used in training,
  - runs the ML model to get a fraud probability,
  - maps it to a decision: **ALLOW / PEND / REJECT**,
  - logs all decisions to a **JSONL audit log**,
  - exposes a simple **Streamlit frontend** as a checkout-like UI.

---

## 🧱 Architecture Overview

### Offline pipeline (training & feature store)

**Goal:** produce a production-ready model and feature engine based on historical data.

1. **Raw data (Kaggle dataset)**
   - `data/raw/transactions_data.csv`
   - `data/raw/users_data.csv`
   - `data/raw/cards_data.csv`
   - `data/raw/mcc_codes.json`
   - `data/raw/train_fraud_labels.json`

2. **Preprocessing & merge**
   - Notebook: `notebooks/01_Data_preprocessing.ipynb`
   - Output: `data/processed/all_merged.parquet`
   - Tasks:
     - join transaction, user, card, and label tables
     - clean data (ages, amounts, dates)
     - add basic features (hour, day of week, etc.)

3. **EDA**
   - `notebooks/02_EDA.ipynb`
   - Includes numerical + categorical EDA and fraud patterns:
     - fraud by amount, hour, MCC, channel, etc.
   - Figures saved under `reports/figures/`.

4. **Feature Store v2 (lifetime + velocity features)**
   - `src/ml/feature_store.py` (and exploratory notebook)
   - Input: `all_merged.parquet`
   - Computes:
     - **lifetime features**:
       - user-level: `user_txn_count_lifetime`, `user_avg_amount_lifetime`, etc.
       - card-level: `card_txn_count_lifetime`, `card_max_amount_lifetime`, etc.
       - merchant-level: `merchant_txn_count_lifetime`, `merchant_fraud_rate_lifetime`, etc.
     - **velocity features**:
       - `user_txn_count_1h`, `user_amount_sum_1h`
       - `user_txn_count_24h`, `user_amount_sum_24h`
       - `card_txn_count_1h`, `card_amount_sum_1h`
       - `card_txn_count_24h`, `card_amount_sum_24h`
   - Output:
     - `data/processed/all_merged_fs_v2.parquet`

5. **FeatureEngine (shared between training & inference)**
   - `src/ml/feature_eng.py`
   - Class: `FeatureEngine`
   - Responsibilities:
     - numeric features:
       - `amount`, `yearly_income`, `total_debt`,
       - `current_age`, `credit_score`, `num_credit_cards`, etc.
     - ratios:
       - `amount_to_income`, `debt_to_income`
     - time features:
       - `transaction_hour`, `transaction_dayofweek`, `transaction_month`
     - buckets:
       - `amount_bin`, `hour_bin`, `income_bin`
     - one-hot encoding:
       - `use_chip` (online / chip / swipe),
       - `card_type` (debit, credit, prepaid),
       - `card_brand` (Mastercard, Visa, etc.)
     - risk flags:
       - `high_risk_mcc`,
       - amount flags (`test_amount_flag`, `high_amount_flag`)
     - frequency encodings:
       - `merchant_category_code_freq`
       - `mcc_description_freq`
       - `merchant_state_freq`
     - velocity features (when present in the dataframe):
       - `user_txn_count_1h`, `user_amount_sum_1h`, ...
   - Artifacts saved to:
     - `models/feature_engine_v2/`:
       - `freq_maps.pkl`
       - `feature_order.json`

6. **Model training with MLflow**
   - `src/ml/train.py`
   - Steps:
     1. Load `data/processed/all_merged_fs_v2.parquet`.
     2. Time-based split into **train / val / test**.
     3. Fit `FeatureEngine` on train only.
     4. Transform train/val/test → feature matrices.
     5. Train **LightGBM** classifier (handles heavy class imbalance).
     6. Evaluate:
        - ROC-AUC, PR-AUC,
        - Recall @ fixed FPR (0.5%, 1%, 2%),
        - confusion matrix & classification report.
     7. Log params/metrics/artifacts using **MLflow**.
     8. Save model & feature engine locally:
        - `models/baseline_lgbm_v2.pkl`
        - `models/feature_engine_v2/`

7. **Decision thresholds**
   - `config/thresholds.json`:
     ```json
     {
       "pend": 0.4,
       "reject": 0.8
     }
     ```
   - Interpretation:
     - `score < 0.4` → **ALLOW**
     - `0.4 ≤ score < 0.8` → **PEND** (manual review)
     - `score ≥ 0.8` → **REJECT**

---

### Online pipeline (real-time scoring)

**Goal:** given a new transaction, return:

- Fraud probability (model score),
- Decision: `ALLOW` / `PEND` / `REJECT`,
- Human-readable reason + thresholds.

1. **API service (FastAPI)**
   - `src/api/app.py`
   - Schemas: `src/api/schemas.py`
     - `TransactionRequest`
       - `amount`
       - `merchant_id`
       - `timestamp`
       - `card_id`
       - `user_id`
     - `DecisionResponse` (optional if used).

2. **Model wrapper + Decision engine**
   - `src/core/decision.py`
   - `FraudModel`:
     - loads:
       - `models/baseline_lgbm_v2.pkl`
       - `models/feature_engine_v2/`
     - `score_transaction(request: TransactionRequest) -> float`
       - builds a DataFrame from request,
       - applies `FeatureEngine.transform_single`,
       - calls `model.predict_proba` and returns fraud probability.
   - `DecisionEngine`:
     - loads thresholds from `config/thresholds.json`,
     - `decide(score) -> DecisionResult`:
       - decision: `"ALLOW"` / `"PEND"` / `"REJECT"`,
       - reason: string,
       - thresholds: dict.

3. **/score endpoint flow**
   - Endpoint: `POST /score`
   - Request body:
     ```json
     {
       "amount": 250.0,
       "merchant_id": "5300",
       "timestamp": "2019-05-10T14:30:00",
       "card_id": "4639",
       "user_id": "0"
     }
     ```
   - Response:
     ```json
     {
       "score": 1.0,
       "decision": "REJECT",
       "reason": "Score 1.000 >= reject threshold 0.80 (high fraud risk).",
       "thresholds": {
         "pend": 0.4,
         "reject": 0.8
       }
     }
     ```

4. **Decision logging**
   - Path: `logs/decisions.jsonl`
   - Each call to `/score` appends a JSON line:
     - timestamp
     - request payload
     - model score
     - final decision
     - thresholds
     - model version
   - Used for:
     - auditability,
     - error analysis,
     - future performance dashboards.

5. **Frontend – Fraud Scoring UI**
   - `frontend/app.py` (Streamlit)
   - Features:
     - form to enter: amount, merchant_id, user_id, card_id, timestamp
     - calls the FastAPI `/score` endpoint
     - displays:
       - **Decision** (`ALLOW` / `PEND` / `REJECT`)
       - **Fraud Score**
       - **Thresholds (pend / reject)**
       - **Explanation text**
       - **Raw API JSON**

---

## 🛠️ Tech Stack

- **Language:** Python 3.11  
- **Modeling:** LightGBM, scikit-learn, pandas, numpy  
- **Experiment tracking:** MLflow  
- **API:** FastAPI, Uvicorn  
- **Frontend:** Streamlit  
- **Serialization:** joblib, parquet (PyArrow)  
- **Logging:** standard logging + JSONL audit logs  

---

## 🚀 How to Run (Local)

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare data

Place Kaggle files under data/raw/:

- transactions_data.csv

- users_data.csv

- cards_data.csv

- mcc_codes.json

- train_fraud_labels.json

Run preprocessing (either via notebook or script) to generate:
```bash
data/processed/all_merged.parquet
```
Then build the FeatureStore V2 and save:
```bash
data/processed/all_merged_fs_v2.parquet
```

### 3. Train the model (offline)
```bash
export PYTHONPATH="."
python -m src.ml.train
```
This will:

- Fit the FeatureEngine V2

- Train the LightGBM model

- Save:

 - models/baseline_lgbm_v2.pkl

 - models/feature_engine_v2/

- Log to MLflow under ./mlruns

Run MLflow UI:
```bash
mlflow ui
```
Open http://127.0.0.1:5000 to view runs.

### 4. Start the API (online scoring)
```bash
export PYTHONPATH="."
uvicorn src.api.app:app --reload
```

Open docs at: http://127.0.0.1:8000/docs

Test POST /score with a JSON body.

### 5. Start the Streamlit UI
```bash
cd frontend
streamlit run app.py
```

Use the UI as a checkout-like transaction simulator.


### Future Extensions

Add real online feature store using Redis or a database.

Add automated monitoring dashboard based on logs/decisions.jsonl.

Add scheduled retraining based on new labeled data.

Expose batch scoring for backfills.
