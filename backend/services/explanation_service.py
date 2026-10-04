"""
EDUPREDICT - Individual Prediction Explanation Service
Calculates feature-level model contributions for individual student predictions using fitted
Scikit-Learn StandardScaler parameters and Logistic/Linear Regression coefficients.
"""

import os
import sys
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.predict import load_trained_models
from ml.dataset import FEATURE_COLUMNS

FEATURE_LABELS = {
    "attendance": "Lecture Attendance Rate",
    "study_hours": "Weekly Self-Study Hours",
    "previous_semester_percentage": "Previous Semester Percentage",
    "assignment_average": "Assignment Average Score",
    "internal_assessment": "Internal Assessment Score",
    "quiz_average": "Quiz Average Score",
    "completed_assignments": "Completed Assignments Count",
    "total_assignments": "Total Assignments Count",
    "backlog_count": "Active Backlog Count"
}

def explain_individual_prediction(input_data):
    """
    Compute individual prediction explanation for classification and regression models.

    Parameters:
    - input_data (dict): Student attribute dictionary.

    Returns:
    - dict: Structured explanation payload.
    """
    classifier_pipeline, regressor_pipeline = load_trained_models()
    if classifier_pipeline is None or regressor_pipeline is None:
        return {
            "error": "Trained model pipelines unavailable for explanation."
        }

    # 1. Build DataFrame matching feature order
    input_row = {}
    for col in FEATURE_COLUMNS:
        val = input_data.get(col)
        if val is None:
            val = 75.0 if 'percentage' in col or 'score' in col or 'average' in col or 'attendance' in col else 10.0
        input_row[col] = [float(val)]

    X_input = pd.DataFrame(input_row)

    # 2. Extract Classifier Pipeline Components
    clf_scaler = classifier_pipeline.named_steps['scaler']
    clf_model = classifier_pipeline.named_steps['classifier']

    X_scaled_clf = clf_scaler.transform(X_input)[0]  # shape (9,)
    predicted_class = str(classifier_pipeline.predict(X_input)[0])

    classes = list(clf_model.classes_)
    if predicted_class in classes:
        class_idx = classes.index(predicted_class)
    else:
        class_idx = 0

    # Retrieve class-specific coefficients (shape: n_classes x n_features)
    if hasattr(clf_model, "coef_"):
        if clf_model.coef_.ndim == 2:
            coef_vec = clf_model.coef_[class_idx]
        else:
            coef_vec = clf_model.coef_
    else:
        coef_vec = np.zeros(len(FEATURE_COLUMNS))

    # 3. Calculate Feature Contributions for Classification Decision Function
    clf_contributions = []
    positive_signals = []
    negative_signals = []

    for idx, col in enumerate(FEATURE_COLUMNS):
        raw_val = float(X_input[col].iloc[0])
        z_val = float(X_scaled_clf[idx])
        weight = float(coef_vec[idx])
        contrib = float(weight * z_val)

        label_name = FEATURE_LABELS.get(col, col)

        item = {
            "feature": col,
            "name": label_name,
            "raw_value": round(raw_val, 2),
            "standardized_value": round(z_val, 3),
            "weight": round(weight, 3),
            "contribution": round(contrib, 3)
        }

        clf_contributions.append(item)

        if contrib > 0.05:
            positive_signals.append({
                "feature": col,
                "name": label_name,
                "raw_value": round(raw_val, 2),
                "contribution": round(contrib, 3),
                "signal_type": "Positive Signal",
                "summary": f"Strong positive signal: {label_name} ({raw_val:.1f}) supports {predicted_class} risk classification."
            })
        elif contrib < -0.05:
            negative_signals.append({
                "feature": col,
                "name": label_name,
                "raw_value": raw_val,
                "contribution": round(contrib, 3),
                "signal_type": "Negative Signal",
                "summary": f"Downward pull signal: {label_name} ({raw_val:.1f}) reduces score trajectory."
            })

    # Sort positive signals (largest contribution first) and negative signals (most negative first)
    positive_signals.sort(key=lambda s: -s["contribution"])
    negative_signals.sort(key=lambda s: s["contribution"])

    # 4. Extract Regressor Pipeline Components (Linear Regression)
    reg_scaler = regressor_pipeline.named_steps['scaler']
    reg_model = regressor_pipeline.named_steps['regressor']

    X_scaled_reg = reg_scaler.transform(X_input)[0]
    predicted_score = float(regressor_pipeline.predict(X_input)[0])

    reg_coefs = reg_model.coef_ if hasattr(reg_model, "coef_") else np.zeros(len(FEATURE_COLUMNS))
    reg_intercept = float(reg_model.intercept_) if hasattr(reg_model, "intercept_") else 0.0

    reg_contributions = []
    sum_reg_contribs = 0.0

    for idx, col in enumerate(FEATURE_COLUMNS):
        raw_val = float(X_input[col].iloc[0])
        z_val = float(X_scaled_reg[idx])
        w = float(reg_coefs[idx])
        contrib = float(w * z_val)
        sum_reg_contribs += contrib

        reg_contributions.append({
            "feature": col,
            "name": FEATURE_LABELS.get(col, col),
            "raw_value": round(raw_val, 2),
            "standardized_value": round(z_val, 3),
            "coefficient": round(w, 3),
            "contribution": round(contrib, 3)
        })

    # Sort feature contributions by absolute influence
    clf_contributions.sort(key=lambda c: -abs(c["contribution"]))
    reg_contributions.sort(key=lambda c: -abs(c["contribution"]))

    return {
        "predicted_class": predicted_class,
        "positive_signals": positive_signals,
        "negative_signals": negative_signals,
        "feature_contributions": clf_contributions,
        "regression_explanation": {
            "intercept": round(reg_intercept, 2),
            "predicted_score": round(predicted_score, 1),
            "sum_contributions": round(sum_reg_contribs, 2),
            "reconstructed_score": round(reg_intercept + sum_reg_contribs, 1),
            "contributions": reg_contributions
        },
        "limitation_note": "These model contributions describe how the trained model responds to the supplied inputs. They do not establish causality."
    }
