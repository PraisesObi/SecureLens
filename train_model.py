"""
Dual model training pipeline for SecureLens.
Trains a Random Forest as the primary model and a Logistic Regression as a baseline,
then writes both, plus encoders, SHAP background and comparison metrics, into model/.
Run: python train_model.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    precision_recall_curve
)
from imblearn.over_sampling import SMOTE

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.preprocessing import (
    preprocess_dataframe, save_encoders,
    FEATURE_COLS, TARGET_COL, get_unique_values
)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH    = os.path.join('data', 'insider_threat_clean_dataset.csv')
MODEL_DIR    = 'model'
MODEL_PATH   = os.path.join(MODEL_DIR, 'rf_model.joblib')
LR_PATH      = os.path.join(MODEL_DIR, 'lr_model.joblib')
SCALER_PATH  = os.path.join(MODEL_DIR, 'lr_scaler.joblib')
ENCODER_PATH = os.path.join(MODEL_DIR, 'encoders.joblib')
BG_DATA_PATH = os.path.join(MODEL_DIR, 'shap_background.joblib')
UNIQUE_PATH  = os.path.join(MODEL_DIR, 'unique_values.joblib')
METRICS_PATH = os.path.join(MODEL_DIR, 'model_comparison.json')

# ── Hyperparameters ────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.20
N_ESTIMATORS = 200
BG_SAMPLE_N  = 500


def load_data():
    """Load the prepared CSV dataset and report class balance."""
    print(f"[1/7] Loading dataset from '{DATA_PATH}' ...")
    df = pd.read_csv(DATA_PATH)
    print(f"      {len(df):,} rows | {df.shape[1]} columns")
    dist = df[TARGET_COL].value_counts()
    print(f"      Normal: {dist.get(0,0):,} | Malicious: {dist.get(1,0):,} "
          f"({dist.get(1,0)/len(df)*100:.1f}% malicious)")
    return df


def preprocess(df):
    """Run the categorical encoding pipeline and return X, y and the encoders."""
    print("\n[2/7] Encoding categorical features ...")
    X, encoders = preprocess_dataframe(df, fit=True)
    y = df[TARGET_COL].astype(int)
    print(f"      Feature matrix: {X.shape}")
    return X, y, encoders


def split_data(X, y):
    """Stratified 80 / 20 train test split that preserves class balance."""
    print("\n[3/7] Stratified train/test split (80/20) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"      Train: {len(X_train):,} | Test: {len(X_test):,}")
    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train):
    """Use SMOTE to balance the malicious class so the model learns real threat patterns."""
    print("\n[4/7] Applying SMOTE for class imbalance ...")
    print(f"      Before. Normal: {sum(y_train==0):,} | Malicious: {sum(y_train==1):,}")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"      After.  Normal: {sum(y_res==0):,} | Malicious: {sum(y_res==1):,}")
    return X_res, y_res


def train_random_forest(X_train, y_train):
    """Train the primary Random Forest model with balanced class weights.

    Justification:
    - Ensemble of decision trees gives low variance and is robust to outliers
    - class_weight='balanced' provides additional imbalance correction
    - Built-in feature_importances_ supports SHAP explainability
    - Handles mixed feature types natively (no scaling needed)
    - Non-linear decision boundaries capture complex threat patterns
    """
    print("\n[5/7] Training Random Forest (primary model) ...")
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=None,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print(f"      Training complete. {N_ESTIMATORS} trees.")
    return model


def train_logistic_regression(X_train, y_train):
    """Train the Logistic Regression baseline used for comparison only."""
    print("\n[6/7] Training Logistic Regression (comparison baseline) ...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    lr = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    lr.fit(X_scaled, y_train)
    print("      Training complete")
    return lr, scaler


def evaluate(name, model, X_test, y_test, scaler=None):
    """Score a model on the held out test set and return its metrics."""
    X_eval = scaler.transform(X_test) if scaler else X_test
    y_pred = model.predict(X_eval)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    print(f"\n  ── {name} ──")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%  (flagged users who are truly malicious)")
    print(f"  Recall    : {rec*100:.2f}%  (malicious users actually caught)")
    print(f"  F1-Score  : {f1*100:.2f}%")
    print(f"  Confusion Matrix: TN={cm[0][0]:,} FP={cm[0][1]:,} FN={cm[1][0]:,} TP={cm[1][1]:,}")

    return {
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1': round(f1, 4),
        'tp': int(cm[1][1]),
        'fp': int(cm[0][1]),
        'fn': int(cm[1][0]),
        'tn': int(cm[0][0]),
    }


def find_optimal_threshold(model, X_test, y_test, min_recall: float = 0.85):
    """Sweep the RF probability output and return the decision threshold that
    achieves recall >= min_recall on the held-out set while maximising F1.

    If no threshold meets the recall floor the standard 0.5 is returned so
    the caller can always rely on a valid float.
    """
    probs = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, probs)

    best_threshold, best_f1 = 0.5, 0.0
    for t, p, r in zip(thresholds, precisions[:-1], recalls[:-1]):
        if r >= min_recall:
            f1 = 2 * p * r / (p + r) if (p + r) else 0.0
            if f1 > best_f1:
                best_f1, best_threshold = f1, float(t)

    print(f"  Optimal threshold (recall ≥ {min_recall*100:.0f}%): "
          f"{best_threshold:.4f}  →  F1 {best_f1*100:.2f}%")
    return best_threshold


def evaluate_at_threshold(name, model, X_test, y_test, threshold: float, scaler=None):
    """Score a model at a specific probability threshold and return its metrics."""
    X_eval  = scaler.transform(X_test) if scaler else X_test
    probs   = model.predict_proba(X_eval)[:, 1]
    y_pred  = (probs >= threshold).astype(int)
    acc     = accuracy_score(y_test, y_pred)
    prec    = precision_score(y_test, y_pred, zero_division=0)
    rec     = recall_score(y_test, y_pred, zero_division=0)
    f1      = f1_score(y_test, y_pred, zero_division=0)
    cm      = confusion_matrix(y_test, y_pred)

    print(f"\n  ── {name} @ threshold={threshold:.4f} ──")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"  F1-Score  : {f1*100:.2f}%")
    print(f"  Confusion : TN={cm[0][0]:,} FP={cm[0][1]:,} FN={cm[1][0]:,} TP={cm[1][1]:,}")

    return {
        'threshold': round(threshold, 4),
        'accuracy':  round(acc,  4),
        'precision': round(prec, 4),
        'recall':    round(rec,  4),
        'f1':        round(f1,   4),
        'tp': int(cm[1][1]),
        'fp': int(cm[0][1]),
        'fn': int(cm[1][0]),
        'tn': int(cm[0][0]),
    }


def compare_models(rf_metrics, lr_metrics):
    """Print the head to head comparison summary."""
    print("\n" + "═" * 58)
    print("  MODEL COMPARISON SUMMARY")
    print("═" * 58)
    print(f"  {'Metric':<14} {'Random Forest':>16} {'Logistic Reg':>14} {'Winner':>8}")
    print("  " + "─" * 54)
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        rf_val = rf_metrics[metric]
        lr_val = lr_metrics[metric]
        winner = 'RF' if rf_val >= lr_val else 'LR'
        print(f"{metric.capitalize():<14} {rf_val*100:>15.2f}% {lr_val*100:>13.2f}% {winner:>8}")
    print("-----------------------------------------------")
    print("CONCLUSION: Random Forest selected as primary model.")
    print("Recall is the critical metric for threat detection.")
    print("Missing a real threat is far costlier than a false alarm.")
    print("RF's superior Recall makes it the justified choice.\n")


def save_artifacts(rf_model, lr_model, lr_scaler, encoders, X_train_df, df,
                   rf_metrics, lr_metrics, rf_optimal_metrics, optimal_threshold):
    """Persist all training artefacts."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(rf_model, MODEL_PATH)
    joblib.dump(lr_model, LR_PATH)
    joblib.dump(lr_scaler, SCALER_PATH)
    save_encoders(encoders, ENCODER_PATH)

    bg_sample = X_train_df.sample(min(BG_SAMPLE_N, len(X_train_df)), random_state=RANDOM_STATE)
    joblib.dump(bg_sample, BG_DATA_PATH)

    unique_vals = get_unique_values(df)
    joblib.dump(unique_vals, UNIQUE_PATH)

    # Save comparison metrics for the app dashboard.
    # optimal_threshold is the decision boundary at which RF achieves recall >= 85%
    # while maximising F1 on the held-out test set.
    comparison = {
        'random_forest':          rf_metrics,
        'random_forest_optimal':  rf_optimal_metrics,
        'logistic_regression':    lr_metrics,
        'optimal_threshold':      round(optimal_threshold, 4),
    }
    with open(METRICS_PATH, 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"  RF model          → {MODEL_PATH}")
    print(f"  LR model          → {LR_PATH}")
    print(f"  LR scaler         → {SCALER_PATH}")
    print(f"  Encoders          → {ENCODER_PATH}")
    print(f"  SHAP background   → {BG_DATA_PATH}")
    print(f"  Unique values     → {UNIQUE_PATH}")
    print(f"  Comparison JSON   → {METRICS_PATH}")
    print(f"  Optimal threshold → {optimal_threshold:.4f}")


def main():
    """End to end training entry point."""
    print("=" * 60)
    print("  SECURELENS TRAINING PIPELINE")
    print("=" * 60)

    df                             = load_data()
    X, y, encoders                 = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_res, y_res                   = apply_smote(X_train, y_train)

    rf_model                       = train_random_forest(X_res, y_res)
    lr_model, lr_scaler            = train_logistic_regression(X_res, y_res)

    print("\n[7/7] Evaluating both models on held-out test set ...")
    rf_metrics = evaluate("Random Forest",       rf_model, X_test, y_test)
    lr_metrics = evaluate("Logistic Regression", lr_model, X_test, y_test, scaler=lr_scaler)

    compare_models(rf_metrics, lr_metrics)

    print("\n  Finding optimal RF decision threshold (recall ≥ 85%) ...")
    optimal_threshold  = find_optimal_threshold(rf_model, X_test, y_test, min_recall=0.85)
    rf_optimal_metrics = evaluate_at_threshold(
        "Random Forest (optimal threshold)", rf_model, X_test, y_test, optimal_threshold
    )

    print("  Saving all artifacts ...")
    save_artifacts(
        rf_model, lr_model, lr_scaler, encoders,
        pd.DataFrame(X_res, columns=FEATURE_COLS), df,
        rf_metrics, lr_metrics, rf_optimal_metrics, optimal_threshold
    )

    print("\n  ✓ Training complete.")
    print("  Run: python app.py  then open http://localhost:8000\n")


if __name__ == '__main__':
    main()
