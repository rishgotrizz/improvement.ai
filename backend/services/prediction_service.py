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

def sanitize_prediction_input(data):
    """
    Sanitize and validate academic feature input dictionary.
    Returns (cleaned_dict, error_str).
    """
    if not isinstance(data, dict):
        return None, "Invalid request payload. Expected JSON dictionary."

    cleaned = {}

    def parse_float(key, default, min_val, max_val, label):
        raw = data.get(key)
        if raw is None or raw == "":
            return default, None
        try:
            val = float(raw)
        except (ValueError, TypeError):
            return None, f"Invalid value for {label} ('{raw}'). Must be a valid number."
        if val < min_val or val > max_val:
            return None, f"{label} must be between {min_val} and {max_val}."
        return val, None

    def parse_int(key, default, min_val, max_val, label):
        raw = data.get(key)
        if raw is None or raw == "":
            return default, None
        try:
            val = int(raw)
        except (ValueError, TypeError):
            return None, f"Invalid integer value for {label} ('{raw}')."
        if val < min_val or val > max_val:
            return None, f"{label} must be between {min_val} and {max_val}."
        return val, None

    # Parse and validate each feature
    att, err = parse_float("attendance", 75.0, 0.0, 100.0, "Attendance")
    if err: return None, err
    cleaned["attendance"] = att

    study, err = parse_float("study_hours", 10.0, 0.0, 168.0, "Study hours")
    if err: return None, err
    cleaned["study_hours"] = study

    prev, err = parse_float("previous_semester_percentage", 75.0, 0.0, 100.0, "Previous semester percentage")
    if err: return None, err
    cleaned["previous_semester_percentage"] = prev

    asgn, err = parse_float("assignment_average", 75.0, 0.0, 100.0, "Assignment average")
    if err: return None, err
    cleaned["assignment_average"] = asgn

    internal, err = parse_float("internal_assessment", 75.0, 0.0, 100.0, "Internal assessment")
    if err: return None, err
    cleaned["internal_assessment"] = internal

    quiz, err = parse_float("quiz_average", 75.0, 0.0, 100.0, "Quiz average")
    if err: return None, err
    cleaned["quiz_average"] = quiz

    tot, err = parse_int("total_assignments", 15, 1, 100, "Total assignments")
    if err: return None, err
    cleaned["total_assignments"] = tot

    comp, err = parse_int("completed_assignments", 14, 0, 100, "Completed assignments")
    if err: return None, err
    if comp > tot:
        return None, f"Completed assignments ({comp}) cannot exceed total assignments ({tot})."
    cleaned["completed_assignments"] = comp

    backlogs, err = parse_int("backlog_count", 0, 0, 30, "Backlog count")
    if err: return None, err
    cleaned["backlog_count"] = backlogs

    # Preserve subject records if supplied
    if "studentSubjects" in data:
        cleaned["studentSubjects"] = data["studentSubjects"]

    return cleaned, None

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

    cleaned_input, val_err = sanitize_prediction_input(input_data)
    if val_err:
        return False, {
            "success": False,
            "error": val_err
        }, 400

    try:
        result, err = predict_academic_profile(cleaned_input)
        if err:
            return False, {
                "success": False,
                "error": err,
                "status": "503 Prediction model unavailable."
            }, 503

        # Phase 5: Individual Prediction Explanation
        explanation = explain_individual_prediction(cleaned_input)

        # Phase 5: Deterministic Recommendation Engine
        student_subjects = cleaned_input.get("studentSubjects")
        recommendations = fetch_recommendations_for_input(cleaned_input, student_subjects=student_subjects)

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
    except Exception as e:
        return False, {
            "success": False,
            "error": f"Internal prediction engine error: {str(e)}"
        }, 500

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
