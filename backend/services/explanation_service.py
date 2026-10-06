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

def get_study_hours_context(hours_val):
    """
    Contextual calibration for self-study hours (expected range 0-6 hrs/day or 0-42 hrs/wk).
    Distinguishes student study guidance from genuine model-derived XAI contributions.
    """
    # Normalize to daily average if passed as weekly total
    daily = hours_val if hours_val <= 6.0 else (hours_val / 7.0)

    if daily < 2.0:
        return "Strong Downward (Below 2h/day threshold)"
    elif daily < 3.0:
        return "Downward (2-3h/day)"
    elif daily < 4.0:
        return "Neutral / Watch (3-4h/day)"
    elif daily < 5.0:
        return "Positive (4-5h/day)"
    else:
        return "Strong Positive (5-6h+/day)"

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
    X_scaled_clf = clf_scaler.transform(X_input)[0]
    predicted_class = str(classifier_pipeline.predict(X_input)[0])

    classes = list(clf_model.classes_)
    class_idx = classes.index(predicted_class) if predicted_class in classes else 0

    coef_vec = clf_model.coef_[class_idx] if hasattr(clf_model, "coef_") and clf_model.coef_.ndim == 2 else np.zeros(len(FEATURE_COLUMNS))

    # 3. Extract Regressor Pipeline Components (Linear Regression Score Impact)
    reg_scaler = regressor_pipeline.named_steps['scaler']
    reg_model = regressor_pipeline.named_steps['regressor']
    X_scaled_reg = reg_scaler.transform(X_input)[0]
    predicted_score = float(regressor_pipeline.predict(X_input)[0])

    reg_coefs = reg_model.coef_ if hasattr(reg_model, "coef_") else np.zeros(len(FEATURE_COLUMNS))
    reg_intercept = float(reg_model.intercept_) if hasattr(reg_model, "intercept_") else 0.0

    # 4. Feature Contributions & Signal Classification
    feature_contributions = []
    positive_signals = []
    negative_signals = []
    sum_reg_contribs = 0.0

    for idx, col in enumerate(FEATURE_COLUMNS):
        raw_val = float(X_input[col].iloc[0])
        z_reg = float(X_scaled_reg[idx])
        w_reg = float(reg_coefs[idx])
        contrib_reg = float(w_reg * z_reg)
        sum_reg_contribs += contrib_reg

        z_clf = float(X_scaled_clf[idx])
        w_clf = float(coef_vec[idx])

        label_name = FEATURE_LABELS.get(col, col)

        # Model impact classification based strictly on actual score contribution
        if contrib_reg > 0.01:
            direction = "Upward Pull"
            explanation_text = f"This feature is pulling the predicted score upward by +{contrib_reg:.2f}% relative to the model baseline."
        elif contrib_reg < -0.01:
            direction = "Downward Pull"
            explanation_text = f"This feature is pulling the predicted score downward by {contrib_reg:.2f}% relative to the model baseline."
        else:
            direction = "Neutral"
            explanation_text = f"This feature has a near-zero impact ({contrib_reg:.2f}%) on the predicted score."

        # Contextual guidance statement
        contextual_signal = get_study_hours_context(raw_val) if col == "study_hours" else direction

        item = {
            "feature": col,
            "name": label_name,
            "raw_value": round(raw_val, 2),
            "standardized_value": round(z_reg, 3),
            "weight": round(w_reg, 3),
            "coefficient": round(w_reg, 3),
            "contribution": round(contrib_reg, 3),
            "direction": direction,
            "explanation": explanation_text,
            "contextual_signal": contextual_signal,
            "classifier_weight": round(w_clf, 3),
            "classifier_z": round(z_clf, 3)
        }

        feature_contributions.append(item)

        if contrib_reg > 0.01:
            positive_signals.append({
                "feature": col,
                "name": label_name,
                "raw_value": round(raw_val, 2),
                "contribution": round(contrib_reg, 3),
                "direction": "Upward Pull",
                "signal_type": "Positive Signal",
                "summary": f"Upward signal: {label_name} ({raw_val:.1f}) improves score trajectory by +{contrib_reg:.2f}%."
            })
        elif contrib_reg < -0.01:
            negative_signals.append({
                "feature": col,
                "name": label_name,
                "raw_value": round(raw_val, 2),
                "contribution": round(contrib_reg, 3),
                "direction": "Downward Pull",
                "signal_type": "Negative Signal",
                "summary": f"Downward pull signal: {label_name} ({raw_val:.1f}) reduces score trajectory by {contrib_reg:.2f}%."
            })

    # Sort positive signals (largest positive contribution first) and negative signals (most negative first)
    positive_signals.sort(key=lambda s: -s["contribution"])
    negative_signals.sort(key=lambda s: s["contribution"])
    feature_contributions.sort(key=lambda c: -abs(c["contribution"]))

    return {
        "predicted_class": predicted_class,
        "predicted_score": round(predicted_score, 1),
        "positive_signals": positive_signals,
        "negative_signals": negative_signals,
        "feature_contributions": feature_contributions,
        "regression_explanation": {
            "intercept": round(reg_intercept, 2),
            "predicted_score": round(predicted_score, 1),
            "sum_contributions": round(sum_reg_contribs, 2),
            "reconstructed_score": round(reg_intercept + sum_reg_contribs, 1),
            "contributions": feature_contributions
        },
        "limitation_note": "These model contributions describe how the trained model responds to the supplied inputs. They do not establish causality."
    }
