"""
EDUPREDICT - What-If Scenario Service Layer
Executes baseline vs scenario model comparisons using existing ML pipelines without model retraining.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from backend.services.prediction_service import execute_ml_prediction
    from backend.services.recommendation_service import fetch_recommendations_for_input
except ImportError:
    from services.prediction_service import execute_ml_prediction
    from services.recommendation_service import fetch_recommendations_for_input

FEATURE_LABELS = {
    "attendance": "Attendance (%)",
    "study_hours": "Weekly Study Hours",
    "previous_semester_percentage": "Previous Semester (%)",
    "assignment_average": "Assignment Average (%)",
    "internal_assessment": "Internal Assessment (%)",
    "quiz_average": "Quiz Average (%)",
    "completed_assignments": "Completed Assignments",
    "total_assignments": "Total Assignments",
    "backlog_count": "Active Backlogs"
}

REQUIRED_FEATURES = [
    "attendance",
    "study_hours",
    "previous_semester_percentage",
    "assignment_average",
    "internal_assessment",
    "quiz_average",
    "completed_assignments",
    "total_assignments",
    "backlog_count"
]

def validate_academic_input(data, context="input"):
    """
    Validate student academic profile dictionary.
    Returns (is_valid: bool, error_message: str, cleaned_data: dict).
    """
    if not isinstance(data, dict):
        return False, f"Invalid {context} payload. Expected JSON dictionary.", None

    cleaned = {}
    missing = [f for f in REQUIRED_FEATURES if f not in data or data[f] is None]
    if missing:
        return False, f"Missing required feature(s) in {context}: {', '.join(missing)}.", None

    try:
        attendance = float(data["attendance"])
        study_hours = float(data["study_hours"])
        prev_sem = float(data["previous_semester_percentage"])
        assgn_avg = float(data["assignment_average"])
        internal = float(data["internal_assessment"])
        quiz_avg = float(data["quiz_average"])
        completed_assgn = int(data["completed_assignments"])
        total_assgn = int(data["total_assignments"])
        backlog_cnt = int(data["backlog_count"])
    except (ValueError, TypeError) as e:
        return False, f"Invalid numeric data format in {context}: {str(e)}.", None

    # Enforce Range Constraints
    if not (0 <= attendance <= 100):
        return False, f"Attendance in {context} must be between 0 and 100%. Got {attendance}.", None
    if not (0 <= study_hours <= 168):
        return False, f"Study hours in {context} must be between 0 and 168 hours. Got {study_hours}.", None
    if not (0 <= prev_sem <= 100):
        return False, f"Previous semester percentage in {context} must be between 0 and 100%. Got {prev_sem}.", None
    if not (0 <= assgn_avg <= 100):
        return False, f"Assignment average in {context} must be between 0 and 100%. Got {assgn_avg}.", None
    if not (0 <= internal <= 100):
        return False, f"Internal assessment in {context} must be between 0 and 100%. Got {internal}.", None
    if not (0 <= quiz_avg <= 100):
        return False, f"Quiz average in {context} must be between 0 and 100%. Got {quiz_avg}.", None

    if total_assgn <= 0:
        return False, f"Total assignments in {context} must be a positive integer greater than 0. Got {total_assgn}.", None
    if completed_assgn < 0:
        return False, f"Completed assignments in {context} cannot be negative. Got {completed_assgn}.", None
    if completed_assgn > total_assgn:
        return False, f"Completed assignments ({completed_assgn}) cannot exceed total assignments ({total_assgn}) in {context}.", None

    if backlog_cnt < 0:
        return False, f"Backlog count in {context} cannot be negative. Got {backlog_cnt}.", None

    cleaned = {
        "attendance": attendance,
        "study_hours": study_hours,
        "previous_semester_percentage": prev_sem,
        "assignment_average": assgn_avg,
        "internal_assessment": internal,
        "quiz_average": quiz_avg,
        "completed_assignments": completed_assgn,
        "total_assignments": total_assgn,
        "backlog_count": backlog_cnt
    }
    return True, "", cleaned

def run_what_if_analysis(baseline_raw, scenario_raw):
    """
    Executes what-if scenario comparison between baseline and hypothetical profile.
    Returns (success: bool, result_or_error: dict, status_code: int).
    """
    # 1. Validate baseline input
    valid_b, err_b, baseline = validate_academic_input(baseline_raw, context="baseline profile")
    if not valid_b:
        return False, {"success": False, "error": err_b}, 400

    # 2. Validate scenario input
    valid_s, err_s, scenario = validate_academic_input(scenario_raw, context="what-if scenario")
    if not valid_s:
        return False, {"success": False, "error": err_s}, 400

    # 3. Execute Baseline Prediction using existing ML service
    succ_b, res_b, status_b = execute_ml_prediction(baseline)
    if not succ_b:
        return False, {"success": False, "error": res_b.get("error", "Baseline prediction failed.")}, status_b

    # 4. Execute Scenario Prediction using existing ML service
    succ_s, res_s, status_s = execute_ml_prediction(scenario)
    if not succ_s:
        return False, {"success": False, "error": res_s.get("error", "Scenario prediction failed.")}, status_s

    # 5. Compute Score Change (Floating-point precision math, rounded for display)
    base_score = float(res_b["predicted_score"])
    scen_score = float(res_s["predicted_score"])
    raw_delta = scen_score - base_score
    score_change = round(raw_delta, 1)

    if score_change > 0:
        formatted_score_change = f"+{score_change:.1f} percentage points"
    elif score_change < 0:
        formatted_score_change = f"{score_change:.1f} percentage points"
    else:
        formatted_score_change = "0.0 percentage points"

    # 6. Compare Risk Category Transition
    base_risk = str(res_b["risk_category"])
    scen_risk = str(res_s["risk_category"])
    risk_changed = (base_risk != scen_risk)

    if risk_changed:
        risk_transition = f"{base_risk} → {scen_risk}"
        risk_summary = f"The model estimates a risk category transition from {base_risk} to {scen_risk}."
    else:
        risk_transition = f"{base_risk} (Unchanged)"
        risk_summary = f"The model estimates the risk category remains {base_risk}."

    # 7. Compare Class Probabilities
    base_probs = res_b.get("class_probabilities", {})
    scen_probs = res_s.get("class_probabilities", {})
    prob_comparison = {}

    all_classes = sorted(list(set(list(base_probs.keys()) + list(scen_probs.keys()))))
    for cls in all_classes:
        p_base = float(base_probs.get(cls, 0.0))
        p_scen = float(scen_probs.get(cls, 0.0))
        p_delta = round(p_scen - p_base, 3)
        delta_pct_points = round(p_delta * 100.0, 1)

        if delta_pct_points > 0:
            formatted_p_delta = f"+{delta_pct_points:.1f} percentage points"
        elif delta_pct_points < 0:
            formatted_p_delta = f"{delta_pct_points:.1f} percentage points"
        else:
            formatted_p_delta = "0.0 percentage points"

        prob_comparison[cls] = {
            "baseline_probability": round(p_base * 100.0, 1),
            "scenario_probability": round(p_scen * 100.0, 1),
            "delta_probability_points": delta_pct_points,
            "formatted_delta": formatted_p_delta
        }

    # 8. Detect Changed Factors Iteratively
    changed_factors = []
    for feat in REQUIRED_FEATURES:
        val_b = baseline[feat]
        val_s = scenario[feat]
        if val_b != val_s:
            diff = val_s - val_b
            if isinstance(val_b, float):
                diff = round(diff, 1)
                unit = "%" if "percentage" in feat or "average" in feat or "assessment" in feat or feat == "attendance" else " hrs"
                val_b_fmt = f"{val_b:.1f}{unit}"
                val_s_fmt = f"{val_s:.1f}{unit}"
                diff_fmt = f"+{diff:.1f}{unit}" if diff > 0 else f"{diff:.1f}{unit}"
            else:
                unit = ""
                val_b_fmt = str(val_b)
                val_s_fmt = str(val_s)
                diff_fmt = f"+{diff}" if diff > 0 else str(diff)

            changed_factors.append({
                "feature": feat,
                "label": FEATURE_LABELS.get(feat, feat),
                "baseline_value": val_b,
                "scenario_value": val_s,
                "baseline_formatted": val_b_fmt,
                "scenario_formatted": val_s_fmt,
                "difference": diff,
                "difference_formatted": diff_fmt
            })

    # 9. Reuse Phase 5 Recommendation Engine on Scenario Input
    student_subjects = scenario_raw.get("studentSubjects") or baseline_raw.get("studentSubjects")
    scenario_recommendations = fetch_recommendations_for_input(scenario, student_subjects=student_subjects)

    # 10. Construct Non-Causal Explanation Statement & Disclaimers
    simulation_summary = f"In this hypothetical scenario, the model's estimated score changes by {formatted_score_change}."

    return True, {
        "success": True,
        "baseline": {
            "predicted_score": base_score,
            "risk_category": base_risk,
            "risk_probability": res_b.get("risk_probability", 0.0),
            "class_probabilities": base_probs
        },
        "scenario": {
            "predicted_score": scen_score,
            "risk_category": scen_risk,
            "risk_probability": res_s.get("risk_probability", 0.0),
            "class_probabilities": scen_probs
        },
        "difference": {
            "score_change": score_change,
            "formatted_score_change": formatted_score_change,
            "risk_changed": risk_changed,
            "risk_transition": risk_transition,
            "risk_summary": risk_summary,
            "probability_comparison": prob_comparison
        },
        "changed_factors": changed_factors,
        "changed_factors_count": len(changed_factors),
        "recommendations": scenario_recommendations,
        "simulation_summary": simulation_summary,
        "disclaimer": "What-if analysis runs the existing trained model on hypothetical inputs. It does not retrain the model or establish causal effects. Results are based on synthetic demonstration data.",
        "isRealModelOutput": True
    }, 200
