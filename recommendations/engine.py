"""
EDUPREDICT - Deterministic Recommendation Engine Module
Evaluates student academic attributes against demonstration thresholds and produces
prioritized actionable recommendations.
"""

from .rules import DEMO_THRESHOLDS, evaluate_factor_status, get_rule_recommendation

def compute_severity_score(factor_key, val):
    """
    Compute a deterministic severity score measuring distance from the healthy threshold.
    Higher score indicates greater academic distance from healthy status.
    """
    if factor_key not in DEMO_THRESHOLDS:
        return 0.0

    t = DEMO_THRESHOLDS[factor_key]
    val = float(val)

    if factor_key == "backlog_count":
        return val * 10.0

    healthy_val = float(t["healthy"])
    if val >= healthy_val:
        return 0.0

    return max(0.0, (healthy_val - val) / healthy_val * 100.0)

def generate_academic_recommendations(feature_dict, student_subjects=None):
    """
    Generate prioritized, deterministic academic recommendations from student inputs.

    Parameters:
    - feature_dict (dict): Feature dictionary (attendance, study_hours, etc.)
    - student_subjects (list of dict, optional): List of subject performance dicts

    Returns:
    - list of dict: Prioritized recommendation items.
    """
    raw_recommendations = []

    # 1. Attribute-level evaluations
    for factor_key, meta in DEMO_THRESHOLDS.items():
        val = feature_dict.get(factor_key)
        if val is None:
            continue

        val = float(val)
        status = evaluate_factor_status(factor_key, val)

        if status == "Healthy":
            continue

        priority = 1 if status == "Needs Attention" else 2
        rec_text, reason_text = get_rule_recommendation(factor_key, val, status)
        severity = compute_severity_score(factor_key, val)

        raw_recommendations.append({
            "priority_tier": priority,
            "severity_score": round(severity, 3),
            "factor_key": factor_key,
            "factor": meta["name"],
            "status": status,
            "student_value": val,
            "recommendation": rec_text,
            "reason": reason_text
        })

    # 2. Subject-level evaluations (if subject data is provided)
    if student_subjects and isinstance(student_subjects, list):
        for sub in student_subjects:
            code = sub.get("code", "Course")
            name = sub.get("name", code)
            score = sub.get("score")
            att = sub.get("attendance")
            prio = sub.get("priority", "LOW")

            if score is not None and float(score) < 70.0:
                raw_recommendations.append({
                    "priority_tier": 1 if float(score) < 60.0 else 2,
                    "severity_score": round((70.0 - float(score)), 3),
                    "factor_key": f"subject_{code}",
                    "factor": f"Course: {code} ({name})",
                    "status": "Needs Attention" if float(score) < 60.0 else "Monitor",
                    "student_value": float(score),
                    "recommendation": f"Allocate extra revision time for {name} ({code}) and practice problem sets.",
                    "reason": f"Current subject percentage ({score:.1f}%) is below the 70.0% target score."
                })
            elif att is not None and float(att) < 75.0:
                raw_recommendations.append({
                    "priority_tier": 1,
                    "severity_score": round((75.0 - float(att)), 3),
                    "factor_key": f"subject_att_{code}",
                    "factor": f"Course Attendance: {code}",
                    "status": "Needs Attention",
                    "student_value": float(att),
                    "recommendation": f"Improve lecture attendance in {name} ({code}) to avoid eligibility shortage.",
                    "reason": f"Course attendance ({att:.1f}%) is below the required 75.0% institutional threshold."
                })

    # 3. Deterministic Sorting: Priority Tier ASC, Severity Score DESC, Factor Key ASC
    raw_recommendations.sort(key=lambda r: (r["priority_tier"], -r["severity_score"], r["factor_key"]))

    # 4. Final Formatting & Priority Re-indexing
    formatted_recommendations = []
    for idx, item in enumerate(raw_recommendations, start=1):
        formatted_recommendations.append({
            "priority": idx,
            "factor": item["factor"],
            "status": item["status"],
            "recommendation": item["recommendation"],
            "reason": item["reason"],
            "student_value": item["student_value"]
        })

    return formatted_recommendations
