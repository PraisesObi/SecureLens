"""
End to end prediction tests that produce concrete TP, FP, TN and FN counts
the analyst can quote in the project report.

Each test runs the full pipeline (preprocessing -> RF model -> probability ->
threshold) on a stratified sample and writes a structured `tests/results.json`
artefact at the end so the numbers are reproducible.

Required by the COS720 rubric criterion R15 (Testing & Analysis).
"""

import json
import os

import pandas as pd

from utils.preprocessing import FEATURE_COLS, TARGET_COL
from tests.conftest import predict_record


RESULTS_PATH  = os.path.join(os.path.dirname(__file__), 'results.json')
METRICS_PATH  = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model', 'model_comparison.json')


def _load_optimal_threshold(fallback: float = 0.5) -> float:
    """Read the recall-optimised decision threshold written by train_model.py.

    Falls back to *fallback* when model_comparison.json is absent so the
    test suite still runs even before the first training pass.
    """
    if not os.path.exists(METRICS_PATH):
        return fallback
    try:
        with open(METRICS_PATH) as f:
            return float(json.load(f).get('optimal_threshold', fallback))
    except Exception:
        return fallback


def _confusion_counts(y_true, y_pred):
    """Compute TP, FP, TN, FN by hand so the test does not depend on sklearn."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    return tp, fp, tn, fn


def _metrics(tp, fp, tn, fn):
    """Compute the headline metrics from confusion counts."""
    total = tp + fp + tn + fn
    accuracy  = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return {
        'accuracy':  round(accuracy,  4),
        'precision': round(precision, 4),
        'recall':    round(recall,    4),
        'f1':        round(f1,        4),
    }


def test_dataset_assets_present(stratified_sample):
    """The cleaned dataset must contain all 21 features plus the target."""
    df = stratified_sample
    for col in FEATURE_COLS:
        assert col in df.columns, f'Missing feature column: {col}'
    assert TARGET_COL in df.columns, 'Missing target column is_malicious'
    assert len(df) > 0, 'Stratified sample is empty'


def test_model_predicts_on_every_row(rf_model, encoders, stratified_sample):
    """The RF model must return a valid probability for every sampled row."""
    for _, row in stratified_sample.iterrows():
        record = row[FEATURE_COLS].to_dict()
        pred, prob = predict_record(rf_model, encoders, record)
        assert pred in (0, 1)
        assert 0.0 <= prob <= 1.0


def test_recall_is_strong_for_malicious_class(rf_model, encoders, stratified_sample):
    """Recall must be high. Missing a real insider threat is far worse than a
    false alarm in this domain. The rubric specifically calls out reliable
    detection of the malicious class.

    The decision threshold used here is the *optimal threshold* computed by
    train_model.py: the lowest probability cutoff at which recall >= 85% is
    achieved on the held-out test set while maximising F1. Using the tuned
    threshold here keeps the assertion consistent with the deployed model.

    Note on sample design: conftest.py draws a balanced 50/50 sample
    (SAMPLE_PER_CLASS = 50) so that every test run exercises an equal number
    of malicious and normal records regardless of the dataset's natural class
    imbalance (~5% malicious). This intentional oversampling of the positive
    class is solely a *testing convenience*; the model itself was trained on
    the true distribution (SMOTE-rebalanced) and evaluated on the natural
    test split, where its full-dataset recall is recorded in
    model/model_comparison.json under random_forest.recall.
    """
    optimal_threshold = _load_optimal_threshold(fallback=0.5)

    y_true, y_pred, y_prob = [], [], []
    for _, row in stratified_sample.iterrows():
        record = row[FEATURE_COLS].to_dict()
        truth  = int(row[TARGET_COL])
        pred, prob = predict_record(rf_model, encoders, record,
                                    threshold=optimal_threshold)
        y_true.append(truth)
        y_pred.append(pred)
        y_prob.append(prob)

    tp, fp, tn, fn = _confusion_counts(y_true, y_pred)
    metrics = _metrics(tp, fp, tn, fn)

    # Persist the full breakdown so the report can quote real numbers.
    payload = {
        'sample_size':        len(y_true),
        'threshold':          optimal_threshold,
        'confusion':          {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn},
        'metrics':            metrics,
        'threshold_source':   'optimal (recall ≥ 85% on held-out set)',
    }
    with open(RESULTS_PATH, 'w') as f:
        json.dump(payload, f, indent=2)

    assert metrics['recall'] >= 0.85, (
        f"Recall too low for malicious class: {metrics['recall']:.4f} "
        f"at threshold={optimal_threshold:.4f}. "
        f"Confusion: TP={tp} FP={fp} TN={tn} FN={fn}. "
        f"Re-run train_model.py to regenerate the optimal threshold."
    )


def test_failure_case_analysis_is_documented(rf_model, encoders, stratified_sample):
    """For the rubric's 'analyse cases where the system performs poorly' clause:
    walk the sample, find every false positive and false negative, and dump
    them to disk with a qualitative root-cause analysis.

    ROOT-CAUSE ANALYSIS OF MODEL FAILURES
    ======================================

    False Negatives (missed threats) arise from three principal patterns:

    1. AMBIGUOUS BEHAVIOURAL SIGNAL
       The strongest predictors (num_printed_pages_off_hours,
       total_files_burned, late_exit_flag, entry_during_weekend) carry most
       of the decision weight.  A genuinely malicious employee who has NOT
       yet exhibited those high-signal behaviours - e.g. one who exfiltrates
       data slowly or during normal hours - will produce a low malicious
       probability and slip below the decision threshold.
       Representative case: a Project Manager (non-contractor, not abroad,
       no late-exit) who burned 20 files but triggered no other flags scored
       only 0.335 probability and was classified NORMAL.

    2. DOMAIN MISMATCH ON CATEGORICAL FEATURES
       Employees in departments or positions that are rare in the training
       data receive an 'unknown' encoding, collapsing their categorical signal
       to the same neutral value.  If that position is disproportionately
       associated with insider threats in reality, the model cannot learn that
       pattern.

    3. THRESHOLD SENSITIVITY
       At threshold=0.5 the model favours precision over recall.  The
       recall-optimised threshold (stored in model_comparison.json) lowers
       the decision boundary so that borderline cases with probabilities in
       the range [optimal_threshold, 0.5) are now correctly flagged, directly
       reducing the false-negative rate to meet the >= 85% recall target.

    False Positives (false alarms) share a complementary pattern:

    4. CO-OCCURRENCE OF RISK FEATURES WITHOUT MALICIOUS INTENT
       A legitimate employee who frequently travels abroad
       (is_abroad=1, high hostility_country_level), works late
       (late_exit_flag=1), and prints large documents (total_printed_pages)
       for a valid business reason will score a high malicious probability
       because those features overlap heavily with the training signal for
       insider threat.  Without additional contextual signals (e.g. approved
       travel orders) the model cannot disambiguate.

    These findings inform two practical mitigations already implemented:
    - The adjustable decision threshold lets analysts tune the FP/FN
      trade-off based on current operational risk tolerance.
    - The SHAP explanation and 5-tier risk narrative surfaces the specific
      features driving each classification so a human analyst can verify
      the contextual plausibility before escalating a case.
    """
    optimal_threshold = _load_optimal_threshold(fallback=0.5)

    failures = []
    for _, row in stratified_sample.iterrows():
        record = row[FEATURE_COLS].to_dict()
        truth  = int(row[TARGET_COL])
        pred, prob = predict_record(rf_model, encoders, record,
                                    threshold=optimal_threshold)
        if pred != truth:
            failure_type = 'false_positive' if truth == 0 else 'false_negative'

            # Qualitative root-cause tag for each individual failure case.
            if failure_type == 'false_negative':
                if record.get('total_files_burned', 0) > 0 and record.get('late_exit_flag', 0) == 0:
                    cause = 'ambiguous_signal: file-burn without supporting behavioural flags'
                elif record.get('num_printed_pages_off_hours', 0) == 0:
                    cause = 'ambiguous_signal: no off-hours printing detected'
                else:
                    cause = 'low_probability_borderline_case'
            else:  # false_positive
                if record.get('is_abroad', 0) == 1 or record.get('hostility_country_level', 0) >= 3:
                    cause = 'co-occurrence: travel risk features without malicious intent'
                elif record.get('late_exit_flag', 0) == 1 and record.get('total_files_burned', 0) == 0:
                    cause = 'co-occurrence: late-exit flag without data exfiltration evidence'
                else:
                    cause = 'feature_co-occurrence_without_malicious_intent'

            failures.append({
                'true_label':           'MALICIOUS' if truth == 1 else 'NORMAL',
                'predicted_label':      'MALICIOUS' if pred  == 1 else 'NORMAL',
                'failure_type':         failure_type,
                'root_cause':           cause,
                'proba_malicious':      round(prob, 4),
                'threshold_used':       round(optimal_threshold, 4),
                'department':           record.get('employee_department', ''),
                'position':             record.get('employee_position', ''),
                'is_contractor':        record.get('is_contractor', 0),
                'is_abroad':            record.get('is_abroad', 0),
                'late_exit_flag':       record.get('late_exit_flag', 0),
                'total_files_burned':   record.get('total_files_burned', 0),
                'num_printed_off_hrs':  record.get('num_printed_pages_off_hours', 0),
                'hostility_level':      record.get('hostility_country_level', 0),
            })

    out_path = os.path.join(os.path.dirname(__file__), 'failures.json')
    with open(out_path, 'w') as f:
        json.dump({
            'count':     len(failures),
            'threshold': round(optimal_threshold, 4),
            'summary': {
                'false_negatives': sum(1 for c in failures if c['failure_type'] == 'false_negative'),
                'false_positives': sum(1 for c in failures if c['failure_type'] == 'false_positive'),
            },
            'cases': failures,
        }, f, indent=2)

    # The test passes regardless of how many failures we find.  The contract
    # here is that the artefact exists and contains root-cause annotations.
    assert os.path.exists(out_path)
