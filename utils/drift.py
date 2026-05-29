"""
Data drift detection for SecureLens.

Uses the Population Stability Index (PSI) which is the industry standard for
tabular feature drift. PSI compares the distribution of a feature in the
training data (the reference) to a more recent sample of incoming data and
returns a single score per feature that has well known operational thresholds:

    PSI < 0.10        no significant change
    0.10 < PSI < 0.25 minor shift, monitor
    PSI > 0.25        significant shift, retrain recommended

This is more robust than raw KL divergence because PSI is symmetric and uses
deciles which means it tolerates zero density bins gracefully.
"""

import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from utils.preprocessing import FEATURE_COLS

EPS = 1e-6


def _bucket_edges(reference: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Return n+1 bin edges covering the reference distribution.

    Falls back to evenly spaced edges if the reference has too few unique values
    (for example a binary feature like is_contractor).
    """
    unique_count = np.unique(reference).size
    if unique_count <= 1:
        # Can't meaningfully bin a constant feature.
        return np.array([reference.min() - EPS, reference.max() + EPS])

    if unique_count < n_bins:
        return np.unique(np.concatenate([
            [reference.min() - EPS],
            np.unique(reference),
            [reference.max() + EPS],
        ]))

    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(reference, quantiles)
    edges[0] = reference.min() - EPS
    edges[-1] = reference.max() + EPS
    # Guarantee strictly monotonic edges, otherwise np.histogram raises.
    edges = np.unique(edges)
    return edges


def population_stability_index(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Compute PSI between a reference and a current 1D distribution."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    if reference.size == 0 or current.size == 0:
        return 0.0

    edges = _bucket_edges(reference, n_bins)
    if edges.size < 2:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)

    # Replace zeros with EPS to avoid log(0) and divide by zero.
    ref_pct = np.where(ref_pct == 0, EPS, ref_pct)
    cur_pct = np.where(cur_pct == 0, EPS, cur_pct)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def severity(psi: float) -> str:
    """Map a PSI value to a human readable severity."""
    if psi < 0.10:
        return 'stable'
    if psi < 0.25:
        return 'minor'
    return 'significant'


def compute_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: Optional[list] = None,
) -> dict:
    """Compute PSI per feature and a summary verdict.

    Both DataFrames must already be encoded (numeric) using the same encoders
    used at training time.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    rows = []
    for col in feature_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        psi = population_stability_index(
            reference_df[col].values,
            current_df[col].values,
        )
        rows.append({
            'feature':  col,
            'psi':      round(psi, 4),
            'severity': severity(psi),
        })

    rows.sort(key=lambda r: r['psi'], reverse=True)
    max_psi = max((r['psi'] for r in rows), default=0.0)
    return {
        'features':       rows,
        'max_psi':        round(max_psi, 4),
        'overall_status': severity(max_psi),
        'reference_n':    int(len(reference_df)),
        'current_n':      int(len(current_df)),
    }


def load_reference_background(path: str) -> Optional[pd.DataFrame]:
    """Load the SHAP background sample as the drift reference distribution."""
    if not os.path.exists(path):
        return None
    df = joblib.load(path)
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame(df, columns=FEATURE_COLS)
    return df
