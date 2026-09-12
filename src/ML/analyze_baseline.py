"""
Baseline Model Analysis Script
Run this to generate all the metrics and insights needed for iteration planning.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import (
    classification_report, 
    f1_score, 
    precision_score, 
    recall_score,
    roc_auc_score,
    confusion_matrix
)
import json

# ============================================
# CONFIGURATION - Update these paths
# ============================================

MODEL_PATH = "models/baseline_lgbm_v1.pkl"
DATA_PATH = "data/processed/all_merged.parquet"
FEATURE_ENGINE_PATH = "models/feature_engine_v1"
OUTPUT_PATH = "results/baseline_analysis.txt"

# ============================================
# 1. LOAD MODEL AND DATA
# ============================================

print("="*80)
print("BASELINE MODEL ANALYSIS")
print("="*80)

print("\n1. Loading model and data...")

# Load model
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# Load feature engine
try:
    import sys
    sys.path.append('.')
    from src.ml.feature_eng import FeatureEngine
    
    feature_engine = FeatureEngine()
    feature_engine = joblib.load(FEATURE_ENGINE_PATH)
    print(f"✅ Feature engine loaded: {FEATURE_ENGINE_PATH}")
except Exception as e:
    print(f"❌ Error loading feature engine: {e}")
    exit(1)

# Load data
try:
    df = pd.read_parquet(DATA_PATH)
    print(f"✅ Data loaded: {DATA_PATH}")
    print(f"   Total rows: {len(df):,}")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit(1)

# ============================================
# 2. PREPARE TEST DATA (Time-based split)
# ============================================

print("\n2. Preparing test data (time-based split)...")

# Sort by timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')

# Use last 20% as test set (simulates real-world deployment)
split_point = int(len(df) * 0.8)
train_df = df.iloc[:split_point]
test_df = df.iloc[split_point:]

print(f"   Training set: {len(train_df):,} rows ({train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
print(f"   Test set: {len(test_df):,} rows ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})")

# Transform features
X_test, y_test = feature_engine.transform(test_df)

print(f"   Features: {X_test.shape[1]}")
print(f"   Test samples: {len(X_test):,}")

# ============================================
# 3. FRAUD RATE ANALYSIS
# ============================================

print("\n" + "="*80)
print("📊 FRAUD RATE ANALYSIS")
print("="*80)

train_fraud_rate = train_df['target'].mean()
test_fraud_rate = test_df['target'].mean()
overall_fraud_rate = df['target'].mean()

print(f"\nOverall fraud rate: {overall_fraud_rate:.2%} ({df['target'].sum():,} frauds out of {len(df):,} transactions)")
print(f"Training fraud rate: {train_fraud_rate:.2%}")
print(f"Test fraud rate: {test_fraud_rate:.2%}")

fraud_counts = df['target'].value_counts()
print(f"\nClass distribution:")
print(f"  Legitimate (0): {fraud_counts[0]:,} ({fraud_counts[0]/len(df)*100:.1f}%)")
print(f"  Fraud (1): {fraud_counts[1]:,} ({fraud_counts[1]/len(df)*100:.1f}%)")
print(f"  Imbalance ratio: 1:{fraud_counts[0]/fraud_counts[1]:.1f}")

# ============================================
# 4. MODEL PREDICTIONS
# ============================================

print("\n" + "="*80)
print("🎯 MODEL PERFORMANCE")
print("="*80)

print("\n3. Making predictions on test set...")

# Get predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Calculate metrics
baseline_f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"\n{'='*80}")
print(f"🏆 BASELINE F1 SCORE: {baseline_f1:.4f}")
print(f"{'='*80}")

print(f"\nDetailed Metrics:")
print(f"  Precision: {precision:.4f} (of flagged transactions, {precision*100:.1f}% are actually fraud)")
print(f"  Recall: {recall:.4f} (catches {recall*100:.1f}% of all fraud)")
print(f"  ROC-AUC: {roc_auc:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\nConfusion Matrix:")
print(f"  True Negatives (Correct ALLOW): {tn:,}")
print(f"  False Positives (Wrong REJECT): {fp:,}")
print(f"  False Negatives (Missed Fraud): {fn:,}")
print(f"  True Positives (Caught Fraud): {tp:,}")

print(f"\nError Analysis:")
print(f"  False Positive Rate: {fp/(fp+tn)*100:.2f}% (legitimate flagged as fraud)")
print(f"  False Negative Rate: {fn/(fn+tp)*100:.2f}% (fraud missed)")

# Classification report
print(f"\n{classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud'])}")

# ============================================
# 5. FEATURE IMPORTANCE
# ============================================

print("\n" + "="*80)
print("🔍 TOP 5 MOST IMPORTANT FEATURES")
print("="*80)

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': X_test.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 5 Features:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"  {idx+1}. {row['feature']}: {row['importance']:.4f}")

print(f"\nTop 10 Features:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"  {idx+1}. {row['feature']}: {row['importance']:.4f}")

# ============================================
# 6. ERROR PATTERN ANALYSIS
# ============================================

print("\n" + "="*80)
print("🔎 ERROR PATTERN ANALYSIS")
print("="*80)

# Create prediction dataframe
test_df_copy = test_df.copy().reset_index(drop=True)
test_df_copy['prediction'] = y_pred
test_df_copy['fraud_score'] = y_proba
test_df_copy['is_correct'] = (test_df_copy['target'] == test_df_copy['prediction'])

# False Negatives (Missed Fraud)
false_negatives = test_df_copy[(test_df_copy['target'] == 1) & (test_df_copy['prediction'] == 0)]
print(f"\n❌ FALSE NEGATIVES (Missed Fraud): {len(false_negatives):,}")

if len(false_negatives) > 0:
    print(f"\nCharacteristics of missed fraud:")
    print(f"  Average amount: ${false_negatives['amount'].mean():.2f} (vs ${test_df_copy[test_df_copy['target']==1]['amount'].mean():.2f} for all fraud)")
    print(f"  Median amount: ${false_negatives['amount'].median():.2f}")
    print(f"  Average fraud score: {false_negatives['fraud_score'].mean():.4f} (just below threshold)")
    
    print(f"\n  Most common hours:")
    for hour, count in false_negatives['transaction_hour'].value_counts().head(3).items():
        print(f"    Hour {hour}: {count} transactions ({count/len(false_negatives)*100:.1f}%)")
    
    print(f"\n  Most common MCC codes:")
    for mcc, count in false_negatives['merchant_category_code'].value_counts().head(3).items():
        print(f"    MCC {mcc}: {count} transactions ({count/len(false_negatives)*100:.1f}%)")
    
    print(f"\n  Most common card types:")
    for card_type, count in false_negatives['card_type'].value_counts().head(3).items():
        print(f"    {card_type}: {count} transactions ({count/len(false_negatives)*100:.1f}%)")
    
    print(f"\n  Most common channels:")
    for channel, count in false_negatives['use_chip'].value_counts().head(3).items():
        print(f"    {channel}: {count} transactions ({count/len(false_negatives)*100:.1f}%)")

# False Positives (Wrong Flags)
false_positives = test_df_copy[(test_df_copy['target'] == 0) & (test_df_copy['prediction'] == 1)]
print(f"\n❌ FALSE POSITIVES (Legitimate Flagged as Fraud): {len(false_positives):,}")

if len(false_positives) > 0:
    print(f"\nCharacteristics of false alarms:")
    print(f"  Average amount: ${false_positives['amount'].mean():.2f} (vs ${test_df_copy[test_df_copy['target']==0]['amount'].mean():.2f} for all legit)")
    print(f"  Median amount: ${false_positives['amount'].median():.2f}")
    print(f"  Average fraud score: {false_positives['fraud_score'].mean():.4f} (above threshold)")
    
    print(f"\n  Most common hours:")
    for hour, count in false_positives['transaction_hour'].value_counts().head(3).items():
        print(f"    Hour {hour}: {count} transactions ({count/len(false_positives)*100:.1f}%)")
    
    print(f"\n  Most common MCC codes:")
    for mcc, count in false_positives['merchant_category_code'].value_counts().head(3).items():
        print(f"    MCC {mcc}: {count} transactions ({count/len(false_positives)*100:.1f}%)")

# ============================================
# 7. FRAUD SCORE DISTRIBUTION
# ============================================

print("\n" + "="*80)
print("📈 FRAUD SCORE DISTRIBUTION")
print("="*80)

# Analyze score distribution
legit_scores = y_proba[y_test == 0]
fraud_scores = y_proba[y_test == 1]

print(f"\nLegitimate transactions:")
print(f"  Mean fraud score: {legit_scores.mean():.4f}")
print(f"  Median fraud score: {np.median(legit_scores):.4f}")
print(f"  95th percentile: {np.percentile(legit_scores, 95):.4f}")

print(f"\nFraudulent transactions:")
print(f"  Mean fraud score: {fraud_scores.mean():.4f}")
print(f"  Median fraud score: {np.median(fraud_scores):.4f}")
print(f"  5th percentile: {np.percentile(fraud_scores, 5):.4f}")

print(f"\nScore separation:")
print(f"  Score overlap zone: {np.percentile(legit_scores, 95):.4f} to {np.percentile(fraud_scores, 5):.4f}")

# ============================================
# 8. KEY INSIGHTS & RECOMMENDATIONS
# ============================================

print("\n" + "="*80)
print("💡 KEY INSIGHTS & RECOMMENDATIONS")
print("="*80)

print(f"\n1. Model Performance:")
if baseline_f1 > 0.70:
    print(f"   ✅ Strong baseline! F1={baseline_f1:.4f} is excellent for fraud detection.")
elif baseline_f1 > 0.60:
    print(f"   ✅ Good baseline! F1={baseline_f1:.4f} is solid. Room for improvement with behavioral features.")
elif baseline_f1 > 0.50:
    print(f"   ⚠️  Moderate baseline. F1={baseline_f1:.4f} shows signal exists, but needs behavioral features.")
else:
    print(f"   ⚠️  Weak baseline. F1={baseline_f1:.4f} suggests current features have limited predictive power.")

print(f"\n2. Class Imbalance:")
if overall_fraud_rate < 0.01:
    print(f"   ⚠️  Severe imbalance ({overall_fraud_rate:.2%} fraud). Consider SMOTE or adjusted class weights.")
elif overall_fraud_rate < 0.05:
    print(f"   ⚠️  Moderate imbalance ({overall_fraud_rate:.2%} fraud). Class weights are helping.")
else:
    print(f"   ✅ Manageable imbalance ({overall_fraud_rate:.2%} fraud).")

print(f"\n3. Top Static Features Working:")
for idx, row in feature_importance.head(3).iterrows():
    print(f"   • {row['feature']} (importance: {row['importance']:.4f})")

print(f"\n4. Next Iteration Recommendations:")
print(f"   Based on error analysis, add these features:")

# Recommendations based on false negatives
if len(false_negatives) > 0:
    avg_fn_amount = false_negatives['amount'].mean()
    avg_all_fraud = test_df_copy[test_df_copy['target']==1]['amount'].mean()
    
    if avg_fn_amount < avg_all_fraud * 0.8:
        print(f"   📊 ADD: Amount anomaly features (missed frauds have unusual amounts)")
    
    fn_hours = false_negatives['transaction_hour'].value_counts()
    if fn_hours.index[0] in [0,1,2,3,4,5,22,23]:
        print(f"   🌙 ADD: Time-based features (missed frauds happen at night/early morning)")
    
    if 'Online Transaction' in false_negatives['use_chip'].values:
        print(f"   💻 ADD: Channel risk features (online transactions are being missed)")

print(f"\n   🚀 PRIORITY: Add velocity features (txn_count_1h, txn_count_24h)")
print(f"   📈 EXPECTED IMPACT: +10-15% F1 improvement")

# ============================================
# 9. SAVE RESULTS
# ============================================

print(f"\n" + "="*80)
print(f"💾 SAVING RESULTS")
print(f"="*80)

# Save to file
Path("results").mkdir(exist_ok=True)

output = {
    "baseline_f1": float(baseline_f1),
    "precision": float(precision),
    "recall": float(recall),
    "roc_auc": float(roc_auc),
    "fraud_rate": float(overall_fraud_rate),
    "test_samples": int(len(X_test)),
    "false_negatives": int(len(false_negatives)),
    "false_positives": int(len(false_positives)),
    "top_features": feature_importance.head(10).to_dict('records'),
    "timestamp": pd.Timestamp.now().isoformat()
}

with open("results/baseline_metrics.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"✅ Metrics saved to: results/baseline_metrics.json")

# Save detailed report
with open(OUTPUT_PATH, "w") as f:
    f.write("BASELINE MODEL ANALYSIS REPORT\n")
    f.write("="*80 + "\n\n")
    f.write(f"F1 Score: {baseline_f1:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"ROC-AUC: {roc_auc:.4f}\n\n")
    f.write(f"Fraud Rate: {overall_fraud_rate:.2%}\n\n")
    f.write("Top 10 Features:\n")
    for idx, row in feature_importance.head(10).iterrows():
        f.write(f"{idx+1}. {row['feature']}: {row['importance']:.4f}\n")

print(f"✅ Report saved to: {OUTPUT_PATH}")

print(f"\n{'='*80}")
print(f"✅ ANALYSIS COMPLETE!")
print(f"{'='*80}")
print(f"\nShare these results for iteration planning:")
print(f"  1. Baseline F1 Score: {baseline_f1:.4f}")
print(f"  2. Top 5 features: {', '.join(feature_importance.head(5)['feature'].tolist())}")
print(f"  3. Fraud rate: {overall_fraud_rate:.2%}")
print(f"  4. Main error patterns: Check output above")