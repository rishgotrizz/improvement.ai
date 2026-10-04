"""
EDUPREDICT - ML Inference Service Module
Loads saved Scikit-Learn model pipelines (models/*.pkl) and executes real model inference.
"""

import os
import sys
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

_CLASSIFIER_PIPELINE = None
_REGRESSOR_PIPELINE = None

def load_trained_models():
    """
    Load persisted model pipelines into RAM for fast inference.
    Returns (classifier, regressor).
    """
    global _CLASSIFIER_PIPELINE, _REGRESSOR_PIPELINE

    if _CLASSIFIER_PIPELINE is not None and _REGRESSOR_PIPELINE is not None:
        return _CLASSIFIER_PIPELINE, _REGRESSOR_PIPELINE

    clf_path = os.path.join(MODELS_DIR, "risk_classifier.pkl")
    reg_path = os.path.join(MODELS_DIR, "score_regressor.pkl")

    if not os.path.exists(clf_path) or not os.path.exists(reg_path):
        return None, None

    try:
        _CLASSIFIER_PIPELINE = joblib.load(clf_path)
        _REGRESSOR_PIPELINE = joblib.load(reg_path)
        print("✅ [ML Inference] Successfully loaded model pipelines from disk.")
        return _CLASSIFIER_PIPELINE, _REGRESSOR_PIPELINE
    except Exception as e:
        print(f"❌ [ML Inference] Failed to load model pipelines: {e}")
        return None, None

def predict_academic_profile(feature_dict):
    """
    Perform inference using trained classification and regression pipelines.
    
    Returns dictionary with:
    - predicted_score (Float)
    - risk_category (String: LOW, MEDIUM, HIGH)
    - risk_probability (Float: probability of assigned class or high risk)
    - class_probabilities (Dict)
    """
    classifier, regressor = load_trained_models()
    if classifier is None or regressor is None:
        return None, "Prediction models are not loaded or models directory is missing."

    from ml.dataset import FEATURE_COLUMNS

    # Build DataFrame matching feature order
    input_row = {}
    for col in FEATURE_COLUMNS:
        val = feature_dict.get(col)
        if val is None:
            # Defaults if missing
            val = 75.0 if 'percentage' in col or 'score' in col or 'average' in col or 'attendance' in col else 10.0
        input_row[col] = [float(val)]

    X_input = pd.DataFrame(input_row)

    # Execute predictions
    predicted_score = float(regressor.predict(X_input)[0])
    predicted_score = round(min(99.5, max(20.0, predicted_score)), 1)

    risk_cat = str(classifier.predict(X_input)[0])

    # Probability prediction from Logistic Regression predict_proba()
    prob_dict = {}
    if hasattr(classifier, "predict_proba"):
        classes = classifier.classes_
        probs = classifier.predict_proba(X_input)[0]
        for cls_name, p in zip(classes, probs):
            prob_dict[str(cls_name)] = round(float(p), 3)

    # Risk probability for predicted category or HIGH risk
    risk_prob = prob_dict.get("HIGH", prob_dict.get(risk_cat, 0.15))

    return {
        "predicted_score": predicted_score,
        "risk_category": risk_cat,
        "risk_probability": round(float(risk_prob), 3),
        "class_probabilities": prob_dict,
        "isRealModelOutput": True
    }, None
