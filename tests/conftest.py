"""
Shared pytest fixtures for the SecureLens test suite.

Loads the trained Random Forest, encoders, SHAP explainer and a stratified
sample of the cleaned dataset exactly once per test session so individual
tests stay fast.
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

# Make the project root importable so utils.* resolves.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.preprocessing import (
    preprocess_single_record,
    load_encoders,
    FEATURE_COLS,
    TARGET_COL,
)
from utils.explainer import build_explainer

DATA_PATH    = os.path.join(ROOT, 'data', 'insider_threat_clean_dataset.csv')
MODEL_PATH   = os.path.join(ROOT, 'model', 'rf_model.joblib')
ENCODER_PATH = os.path.join(ROOT, 'model', 'encoders.joblib')
BG_PATH      = os.path.join(ROOT, 'model', 'shap_background.joblib')

RANDOM_SEED = 42
SAMPLE_PER_CLASS = 50


@pytest.fixture(scope='session')
def rf_model():
    """Trained Random Forest classifier."""
    assert os.path.exists(MODEL_PATH), \
        'rf_model.joblib not found. Run python train_model.py first.'
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope='session')
def encoders():
    """Categorical encoders saved during training."""
    assert os.path.exists(ENCODER_PATH), \
        'encoders.joblib not found. Run python train_model.py first.'
    return load_encoders(ENCODER_PATH)


@pytest.fixture(scope='session')
def explainer(rf_model):
    """SHAP TreeExplainer using the saved background sample."""
    bg = joblib.load(BG_PATH)
    return build_explainer(rf_model, bg)


@pytest.fixture(scope='session')
def stratified_sample():
    """A balanced sample of malicious and normal records from the dataset.

    Uses a fixed random seed so test results are reproducible across runs.
    """
    assert os.path.exists(DATA_PATH), \
        'Dataset CSV not found at data/insider_threat_clean_dataset.csv'
    df = pd.read_csv(DATA_PATH)

    rng = np.random.default_rng(RANDOM_SEED)
    malicious = df[df[TARGET_COL] == 1]
    normal    = df[df[TARGET_COL] == 0]

    n_mal  = min(SAMPLE_PER_CLASS, len(malicious))
    n_norm = min(SAMPLE_PER_CLASS, len(normal))

    mal_idx  = rng.choice(malicious.index, size=n_mal,  replace=False)
    norm_idx = rng.choice(normal.index,    size=n_norm, replace=False)

    sampled = pd.concat([df.loc[mal_idx], df.loc[norm_idx]]).reset_index(drop=True)
    return sampled


def predict_record(rf_model, encoders, record_dict, threshold=0.5):
    """Run a single record through the prediction pipeline.

    Helper used by tests so the assertion logic stays readable.
    """
    X = preprocess_single_record(record_dict, encoders)
    proba = rf_model.predict_proba(X)[0]
    pred = 1 if proba[1] >= threshold else 0
    return pred, float(proba[1])
