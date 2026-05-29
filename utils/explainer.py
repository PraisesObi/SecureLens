"""
Explainability module for the Insider Threat Detection System.

Provides:
- SHAP-based feature importance for individual predictions
- Human-readable risk indicator summaries
- Risk score breakdown tailored for non-technical managers

Aligned with the AI Explainability requirement in the COS720 project spec
and the DFR-BUST paper's emphasis on interpretable analytic outputs.
"""

import numpy as np
import pandas as pd
import shap
from utils.preprocessing import FEATURE_DISPLAY_NAMES, FEATURE_RISK_DESCRIPTIONS, FEATURE_COLS


def build_explainer(model, X_background):
    """
    Build a SHAP TreeExplainer using a background dataset sample.
    TreeExplainer is optimised for Random Forest / tree-based models.

    Args:
        model: Trained sklearn Random Forest
        X_background: numpy array or DataFrame used as background for SHAP

    Returns:
        shap.TreeExplainer instance
    """
    explainer = shap.TreeExplainer(model, data=X_background, feature_perturbation='interventional')
    return explainer


def get_shap_values(explainer, X_instance):
    """
    Compute SHAP values for a single prediction instance.

    Returns:
        shap_vals: 1D array of SHAP values for the malicious class (class 1)
        base_value: expected model output (base rate)
    """
    shap_values = explainer.shap_values(X_instance)

    # Handle different SHAP output shapes across versions:
    # - List of arrays [class_0_array, class_1_array]  → shape (n_samples, n_features) each
    # - Single 3D array of shape (n_samples, n_features, n_classes)
    if isinstance(shap_values, list):
        # Older shap: list[class_idx] → shape (1, n_features)
        shap_vals = shap_values[1][0]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # Newer shap: shape (n_samples, n_features, n_classes)
        shap_vals = shap_values[0, :, 1]
    else:
        shap_vals = shap_values[0]

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(np.array(base_value)[1])

    return shap_vals, base_value


def get_feature_importance_df(shap_vals, feature_names=None):
    """
    Build a DataFrame of feature contributions sorted by absolute impact.

    Args:
        shap_vals: 1D array of SHAP values
        feature_names: list of feature names (defaults to FEATURE_COLS)

    Returns:
        DataFrame with columns: feature, display_name, shap_value, abs_impact, direction
    """
    if feature_names is None:
        feature_names = FEATURE_COLS

    df = pd.DataFrame({
        'feature': feature_names,
        'shap_value': shap_vals
    })
    df['display_name'] = df['feature'].map(FEATURE_DISPLAY_NAMES)
    df['abs_impact'] = df['shap_value'].abs()
    df['direction'] = df['shap_value'].apply(lambda v: 'risk' if v > 0 else 'safe')
    df = df.sort_values('abs_impact', ascending=False).reset_index(drop=True)
    return df


def build_human_readable_explanation(
    prediction_label,
    confidence,
    feature_importance_df,
    record_dict,
    top_n=5
):
    """
    Generate a plain-English explanation of a prediction
    suitable for a non-technical manager.

    Args:
        prediction_label: 'MALICIOUS' or 'NORMAL'
        confidence: float, e.g. 0.87
        feature_importance_df: output of get_feature_importance_df()
        record_dict: the original input record as a dict
        top_n: number of top factors to include in the explanation

    Returns:
        str: human-readable explanation paragraph
    """
    top_risk = feature_importance_df[feature_importance_df['direction'] == 'risk'].head(top_n)
    top_safe = feature_importance_df[feature_importance_df['direction'] == 'safe'].head(2)

    conf_pct = int(confidence * 100)

    if prediction_label == 'MALICIOUS':
        intro = (
            f"This employee's behaviour has been flagged as potentially malicious "
            f"with {conf_pct}% confidence. "
        )

        if len(top_risk) > 0:
            risk_phrases = []
            for _, row in top_risk.iterrows():
                feat = row['feature']
                desc = FEATURE_RISK_DESCRIPTIONS.get(feat, row['display_name'])
                risk_phrases.append(desc)

            if len(risk_phrases) == 1:
                factors_str = risk_phrases[0]
            elif len(risk_phrases) == 2:
                factors_str = f"{risk_phrases[0]} and {risk_phrases[1]}"
            else:
                factors_str = ", ".join(risk_phrases[:-1]) + f", and {risk_phrases[-1]}"

            intro += f"The primary risk indicators driving this classification include: {factors_str}. "

        if len(top_safe) > 0:
            safe_phrases = [FEATURE_RISK_DESCRIPTIONS.get(r['feature'], r['display_name'])
                            for _, r in top_safe.iterrows()]
            safe_str = " and ".join(safe_phrases)
            intro += f"Mitigating factors include: {safe_str}. "

        intro += (
            "It is recommended that this employee's activity be reviewed by a security "
            "officer and monitored closely in the coming period."
        )
    else:
        intro = (
            f"This employee's behaviour appears normal with {conf_pct}% confidence. "
            "No significant behavioural anomalies were detected. "
            "Continued routine monitoring is recommended as per organisational policy."
        )

    return intro


def get_risk_level(confidence, prediction):
    """
    Return a risk level label and colour based on prediction and confidence.

    The 5-tier scheme follows industry convention (CVSS v3, NIST SP 800-30,
    OWASP Risk Rating). The CRITICAL tier is carved off the top of HIGH at
    confidence >= 0.90, mirroring CVSS v3's 9.0-10.0 Critical band introduced
    in 2015 (FIRST). All other thresholds are unchanged from the previous
    scheme so only the most extreme malicious predictions are promoted.

    Returns:
        tuple: (risk_label, colour_hex)
    """
    if prediction == 0:
        if confidence >= 0.85:
            return "LOW RISK", "#27AE60"
        else:
            return "LOW-MODERATE RISK", "#F39C12"
    else:
        if confidence >= 0.90:
            return "CRITICAL RISK", "#B91C1C"
        elif confidence >= 0.80:
            return "HIGH RISK", "#E74C3C"
        elif confidence >= 0.60:
            return "MODERATE-HIGH RISK", "#E67E22"
        else:
            return "MODERATE RISK", "#F39C12"
