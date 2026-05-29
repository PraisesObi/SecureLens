"""
Preprocessing utilities for the Insider Threat Detection System.
Handles encoding of categorical features and feature scaling.
Encoders are saved/loaded alongside the model to ensure consistency
between training and inference time.

FEATURE SET DESIGN NOTE — Temporal / Login-Time Pattern Indicators
===================================================================
The Kaggle insider-threat dataset does not expose raw login timestamps, so
three engineered features serve as proxies for the "login time patterns"
behavioural indicator described in the threat-detection literature:

  num_printed_pages_off_hours
    Volume of documents printed outside normal working hours (evenings /
    nights).  An insider preparing to exfiltrate information commonly prints
    or copies material during off-hours to avoid supervision.

  late_exit_flag  (binary)
    Set when a building-access record shows an unusually late departure.
    Extended after-hours presence is a canonical precursor to physical data
    theft and correlates strongly with off-hours system activity.

  entry_during_weekend  (binary)
    Set when the employee's access log contains a weekend entry.  Unauthorised
    or unusual weekend access is a recognised insider-threat indicator and
    directly captures the "abnormal login timing" concept at the physical-
    access level.

Together these three features encode the same risk signal as login-time
abnormality metrics: activity occurring outside the legitimate working window.
They are the closest available representation given the dataset's schema and
are treated as temporal pattern features throughout the pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# ── Feature groupings ──────────────────────────────────────────────────────

# Temporal / off-hours behavioural features (login-time pattern proxies).
# See module docstring for full rationale.
TEMPORAL_PATTERN_COLS = [
    'num_printed_pages_off_hours',  # volume of off-hours document printing
    'late_exit_flag',               # unusually late building departure
    'entry_during_weekend',         # weekend building access
]

# Categorical columns that require label encoding
CATEGORICAL_COLS = [
    'employee_department',
    'employee_campus',
    'employee_position',
    'employee_origin_country'
]

# All feature columns used for training (excludes target)
FEATURE_COLS = [
    'employee_department',
    'employee_campus',
    'employee_position',
    'employee_seniority_years',
    'is_contractor',
    'employee_classification',
    'has_foreign_citizenship',
    'has_criminal_record',
    'has_medical_history',
    'employee_origin_country',
    'total_printed_pages',
    'num_printed_pages_off_hours',
    'total_files_burned',
    'burned_from_other',
    'is_abroad',
    'trip_day_number',
    'hostility_country_level',
    'num_entries',
    'num_unique_campus',
    'late_exit_flag',
    'entry_during_weekend'
]

TARGET_COL = 'is_malicious'

# Human-readable feature names for display in the UI and explanations
FEATURE_DISPLAY_NAMES = {
    'employee_department': 'Department',
    'employee_campus': 'Campus',
    'employee_position': 'Job Position',
    'employee_seniority_years': 'Years of Seniority',
    'is_contractor': 'Is Contractor',
    'employee_classification': 'Security Clearance Level',
    'has_foreign_citizenship': 'Has Foreign Citizenship',
    'has_criminal_record': 'Has Criminal Record',
    'has_medical_history': 'Has Medical History',
    'employee_origin_country': 'Country of Origin',
    'total_printed_pages': 'Total Pages Printed',
    'num_printed_pages_off_hours': 'Pages Printed Off-Hours',
    'total_files_burned': 'Files Burned to Disk',
    'burned_from_other': 'Burned Files from Other Accounts',
    'is_abroad': 'Currently Abroad',
    'trip_day_number': 'Trip Day Number',
    'hostility_country_level': 'Destination Country Hostility Level',
    'num_entries': 'Number of Building Entries',
    'num_unique_campus': 'Unique Campuses Accessed',
    'late_exit_flag': 'Late Exit Detected',
    'entry_during_weekend': 'Weekend Entry Detected'
}

# Risk description for each feature used in human-readable explanations
FEATURE_RISK_DESCRIPTIONS = {
    'burned_from_other': 'burned files from another employee\'s account',
    'num_printed_pages_off_hours': 'printed documents outside of normal working hours',
    'total_files_burned': 'copied a large number of files to removable media',
    'late_exit_flag': 'exited the building unusually late',
    'entry_during_weekend': 'accessed the facility during the weekend',
    'hostility_country_level': 'travelled to a country with a high hostility rating',
    'is_abroad': 'is currently abroad',
    'has_criminal_record': 'has a prior criminal record',
    'num_unique_campus': 'accessed an unusually high number of campus locations',
    'total_printed_pages': 'printed a high volume of documents',
    'employee_classification': 'holds a sensitive security clearance level',
    'has_foreign_citizenship': 'holds foreign citizenship',
    'num_entries': 'logged an abnormal number of building entries',
    'trip_day_number': 'is on an extended trip',
    'is_contractor': 'is an external contractor',
    'employee_seniority_years': 'seniority level',
    'has_medical_history': 'has a flagged medical history',
    'employee_department': 'department',
    'employee_campus': 'campus assignment',
    'employee_position': 'job position',
    'employee_origin_country': 'country of origin'
}


def fit_encoders(df):
    """
    Fit LabelEncoders on categorical columns.
    Returns a dict of fitted encoders keyed by column name.
    """
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        encoders[col] = le
    return encoders


def apply_encoders(df, encoders):
    """
    Apply fitted encoders to a DataFrame.
    Handles unseen labels by mapping them to the last known class index.
    """
    df = df.copy()
    for col in CATEGORICAL_COLS:
        le = encoders[col]
        known_classes = set(le.classes_)
        df[col] = df[col].astype(str).apply(
            lambda x: x if x in known_classes else le.classes_[0]
        )
        df[col] = le.transform(df[col])
    return df


def preprocess_dataframe(df, encoders=None, fit=False):
    """
    Full preprocessing pipeline:
    - Select relevant features
    - Handle missing values
    - Encode categoricals
    - Return features array and optionally fitted encoders
    """
    df = df.copy()

    # Fill missing numeric values with median-like defaults
    df['trip_day_number'] = df['trip_day_number'].fillna(0)
    df['hostility_country_level'] = df['hostility_country_level'].fillna(0)

    # Fill missing categorical values with 'Unknown'
    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna('Unknown').astype(str)

    # Fit encoders if requested (training time)
    if fit:
        encoders = fit_encoders(df)

    # Apply encoders
    df = apply_encoders(df, encoders)

    # Select only feature columns
    X = df[FEATURE_COLS].copy()

    # Ensure all columns are numeric
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    if fit:
        return X, encoders
    return X


def preprocess_single_record(record_dict, encoders):
    """
    Preprocess a single employee record (as a dict) for inference.
    record_dict keys should match FEATURE_COLS.
    """
    df = pd.DataFrame([record_dict])
    return preprocess_dataframe(df, encoders=encoders, fit=False)


def save_encoders(encoders, path):
    """Save fitted encoders to disk."""
    joblib.dump(encoders, path)


def load_encoders(path):
    """Load fitted encoders from disk."""
    return joblib.load(path)


def get_unique_values(df):
    """
    Extract unique values for each categorical column.
    Used to populate dropdowns in the Streamlit UI.
    """
    unique_vals = {}
    for col in CATEGORICAL_COLS:
        unique_vals[col] = sorted(df[col].dropna().unique().tolist())
    return unique_vals
