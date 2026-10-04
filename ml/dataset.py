"""
EDUPREDICT - ML Dataset Preparation Module
Handles dataset loading, feature matrix (X) extraction, target vector (y) derivation,
and mandatory Target Leakage prevention checks.
"""

import os
import pandas as pd

FEATURE_COLUMNS = [
    'attendance',
    'study_hours',
    'previous_semester_percentage',
    'assignment_average',
    'internal_assessment',
    'quiz_average',
    'completed_assignments',
    'total_assignments',
    'backlog_count'
]

REGRESSION_TARGET = 'final_score'

def derive_risk_category(score):
    """
    Project-defined demonstration thresholds for risk classification:
    - HIGH RISK: score < 60.0%
    - MEDIUM RISK: 60.0% <= score < 75.0%
    - LOW RISK: score >= 75.0%
    """
    if score < 60.0:
        return "HIGH"
    elif score < 75.0:
        return "MEDIUM"
    else:
        return "LOW"

def prepare_ml_dataset(filepath="data/students.csv"):
    """
    Load dataset and prepare X, y_reg, y_cls datasets.
    Performs mandatory target leakage check.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")

    df = pd.read_csv(filepath, comment="#")

    # Target Leakage Audit
    assert REGRESSION_TARGET not in FEATURE_COLUMNS, "TARGET LEAKAGE ERROR: final_score cannot be in feature matrix X!"

    # Feature Matrix X
    X = df[FEATURE_COLUMNS].copy()

    # Regression Target y_reg
    y_reg = df[REGRESSION_TARGET].copy()

    # Classification Target y_cls (Project-defined thresholds)
    y_cls = y_reg.apply(derive_risk_category)

    print(f"✅ [ML Dataset] Prepared X: {X.shape}, y_reg: {y_reg.shape}, y_cls: {y_cls.shape}")
    print(f"🔒 [Target Leakage Audit] Verified: '{REGRESSION_TARGET}' is NOT present in feature matrix X.")
    print(f"📊 [Class Distribution] {y_cls.value_counts().to_dict()}")

    return X, y_reg, y_cls
