"""
EDUPREDICT - Academic Demonstration Thresholds & Recommendation Rules
Centralized transparent rules and demonstration thresholds for academic factor evaluation.

NOTE: These thresholds are project-defined demonstration rules for academic risk analysis.
They are NOT official institutional academic policies.
"""

# Centralized Project-Defined Demonstration Thresholds
DEMO_THRESHOLDS = {
    "attendance": {
        "name": "Lecture Attendance Rate",
        "unit": "%",
        "healthy": 85.0,
        "monitor": 75.0,
        "min_possible": 0.0,
        "max_possible": 100.0
    },
    "study_hours": {
        "name": "Weekly Self-Study Hours",
        "unit": "hrs/wk",
        "healthy": 15.0,
        "monitor": 10.0,
        "min_possible": 0.0,
        "max_possible": 70.0
    },
    "previous_semester_percentage": {
        "name": "Previous Semester Percentage",
        "unit": "%",
        "healthy": 75.0,
        "monitor": 60.0,
        "min_possible": 0.0,
        "max_possible": 100.0
    },
    "assignment_average": {
        "name": "Assignment Average Score",
        "unit": "%",
        "healthy": 75.0,
        "monitor": 60.0,
        "min_possible": 0.0,
        "max_possible": 100.0
    },
    "internal_assessment": {
        "name": "Internal Assessment Score",
        "unit": "%",
        "healthy": 75.0,
        "monitor": 60.0,
        "min_possible": 0.0,
        "max_possible": 100.0
    },
    "quiz_average": {
        "name": "Quiz Average Score",
        "unit": "%",
        "healthy": 75.0,
        "monitor": 60.0,
        "min_possible": 0.0,
        "max_possible": 100.0
    },
    "completed_assignments": {
        "name": "Completed Assignments Count",
        "unit": "submitted",
        "healthy": 13,
        "monitor": 10,
        "min_possible": 0,
        "max_possible": 15
    },
    "backlog_count": {
        "name": "Active Backlog Count",
        "unit": "failed subjects",
        "healthy": 0,
        "monitor": 1,
        "min_possible": 0,
        "max_possible": 10
    }
}

def evaluate_factor_status(factor_key, val):
    """
    Evaluate academic status of a factor value against demonstration thresholds.
    Returns: 'Healthy', 'Monitor', or 'Needs Attention'.
    """
    if factor_key not in DEMO_THRESHOLDS:
        return "Healthy"

    t = DEMO_THRESHOLDS[factor_key]
    val = float(val)

    if factor_key == "backlog_count":
        # Backlogs: 0 is healthy, 1 is monitor, >= 2 needs attention
        if val <= 0:
            return "Healthy"
        elif val == 1:
            return "Monitor"
        else:
            return "Needs Attention"

    # Standard metrics where higher is better
    if val >= t["healthy"]:
        return "Healthy"
    elif val >= t["monitor"]:
        return "Monitor"
    else:
        return "Needs Attention"

def get_rule_recommendation(factor_key, val, status):
    """
    Retrieve deterministic recommendation string and reason for a given factor and status.
    """
    val_str = f"{val:.1f}" if isinstance(val, float) else str(val)

    if factor_key == "attendance":
        if status == "Needs Attention":
            return (
                "Prioritize attending all upcoming lectures and systematically review missed course material.",
                f"Lecture attendance ({val_str}%) is below the project's 75.0% demonstration threshold."
            )
        elif status == "Monitor":
            return (
                "Maintain steady attendance and avoid unexcused lecture absences.",
                f"Lecture attendance ({val_str}%) is in the monitoring range (75-85%)."
            )

    elif factor_key == "study_hours":
        if status == "Needs Attention":
            return (
                "Establish a structured daily study schedule adding at least 1-2 hours of revision per day.",
                f"Weekly study hours ({val_str} hrs/wk) are below the project's 10.0 hrs/wk demonstration threshold."
            )
        elif status == "Monitor":
            return (
                "Slightly increase weekly self-study time during exam preparation weeks.",
                f"Weekly study hours ({val_str} hrs/wk) are in the monitoring range (10-15 hrs/wk)."
            )

    elif factor_key == "assignment_average":
        if status == "Needs Attention":
            return (
                "Seek immediate coursework assistance and review assignment grading feedback.",
                f"Assignment average ({val_str}%) is below the project's 60.0% demonstration threshold."
            )
        elif status == "Monitor":
            return (
                "Allocate extra time to review assignment rubrics before submission.",
                f"Assignment average ({val_str}%) is in the monitoring range (60-75%)."
            )

    elif factor_key == "internal_assessment":
        if status == "Needs Attention":
            return (
                "Schedule mid-term revision sessions and solve prior year examination problems.",
                f"Internal assessment score ({val_str}%) is below the 60.0% demonstration threshold."
            )
        elif status == "Monitor":
            return (
                "Focus on core theoretical concepts to improve internal test scores.",
                f"Internal assessment score ({val_str}%) is in the monitoring range (60-75%)."
            )

    elif factor_key == "quiz_average":
        if status == "Needs Attention":
            return (
                "Conduct short daily self-quizzes to improve recall on weekly tests.",
                f"Quiz average ({val_str}%) is below the 60.0% demonstration threshold."
            )
        elif status == "Monitor":
            return (
                "Review weekly lecture summaries before quiz sessions.",
                f"Quiz average ({val_str}%) is in the monitoring range (60-75%)."
            )

    elif factor_key == "completed_assignments":
        if status == "Needs Attention":
            return (
                "Complete and submit all outstanding coursework assignments immediately.",
                f"Completed assignments ({val_str}/15) is below the minimum 10 assignment threshold."
            )
        elif status == "Monitor":
            return (
                "Ensure remaining assignments are submitted before deadline dates.",
                f"Completed assignments ({val_str}/15) is in the monitoring range (10-12)."
            )

    elif factor_key == "backlog_count":
        if status == "Needs Attention":
            return (
                "Create a dedicated backlog clearance plan and register for remedial classes.",
                f"Active backlog count ({val_str}) indicates 2 or more uncleared subjects."
            )
        elif status == "Monitor":
            return (
                "Prepare for the upcoming backlog re-examination alongside current term subjects.",
                f"Active backlog count ({val_str}) indicates 1 uncleared subject."
            )

    elif factor_key == "previous_semester_percentage":
        if status == "Needs Attention":
            return (
                "Focus on building foundational prerequisite knowledge across core subjects.",
                f"Previous semester score ({val_str}%) is below the 60.0% demonstration threshold."
            )
        elif status == "Monitor":
            return (
                "Maintain consistent academic effort to improve upon past semester performance.",
                f"Previous semester score ({val_str}%) is in the monitoring range (60-75%)."
            )

    return (
        "Maintain current academic performance levels.",
        f"Attribute ({val_str}) meets or exceeds healthy demonstration thresholds."
    )
