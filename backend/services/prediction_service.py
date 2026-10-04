"""
EDUPREDICT - Prediction Service Layer
Invokes trained Scikit-Learn pipelines via ml.predict and formats response payloads.
"""

import os
import json
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.predict import predict_academic_profile
try:
    from backend.services.explanation_service import explain_individual_prediction
    from backend.services.recommendation_service import fetch_recommendations_for_input
except ImportError:
    from services.explanation_service import explain_individual_prediction
    from services.recommendation_service import fetch_recommendations_for_input

METRICS_FILE = os.path.join(PROJECT_ROOT, "models/metrics.json")

def execute_ml_prediction(input_data):
    """
    Validate input data and run machine learning prediction with Phase 5 explainability & recommendations.
    Returns (success: bool, result_dict: dict, status_code: int).
    """
    if not input_data or not isinstance(input_data, dict):
        return False, {
            "success": False,
            "error": "Invalid request payload. Expected JSON dictionary."
        }, 400

    result, err = predict_academic_profile(input_data)
    if err:
        return False, {
            "success": False,
            "error": err,
            "status": "503 Prediction model unavailable."
        }, 503

    # Phase 5: Individual Prediction Explanation
    explanation = explain_individual_prediction(input_data)

    # Phase 5: Deterministic Recommendation Engine
    student_subjects = input_data.get("studentSubjects")
    recommendations = fetch_recommendations_for_input(input_data, student_subjects=student_subjects)

    return True, {
        "success": True,
        "predicted_score": result["predicted_score"],
        "risk_category": result["risk_category"],
        "risk_probability": result["risk_probability"],
        "class_probabilities": result.get("class_probabilities", {}),
        "explanation": explanation,
        "recommendations": recommendations,
        "isRealModelOutput": True,
        "disclaimer": "Model results are based on a synthetic demonstration dataset and should not be interpreted as real-world academic performance benchmarks."
    }, 200

def fetch_trained_model_metrics():
    """
    Retrieve stored metrics from models/metrics.json.
    """
    if not os.path.exists(METRICS_FILE):
        return {
            "status": "AI prediction pending model training (Phase 4)",
            "isDemo": True
        }

    try:
        with open(METRICS_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": f"Failed to read model metrics file: {e}"}
