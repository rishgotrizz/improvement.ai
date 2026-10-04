"""
EDUPREDICT - Dynamic Insight Generation Module
Generates analytical findings strictly based on actual calculated correlation coefficients and descriptive statistics.
"""

def generate_eda_insights(stats, correlations):
    """
    Generate dynamic textual findings from calculated statistics and correlation matrix.
    """
    insights = []

    if not correlations or "final_score" not in correlations:
        return ["Exploratory analysis pending dataset statistics computation."]

    final_corrs = correlations.get("final_score", {})

    # 1. Attendance Insight
    att_r = final_corrs.get("attendance")
    if att_r is not None:
        if att_r >= 0.6:
            insights.append(f"Lecture attendance displays a strong positive linear correlation (r = {att_r}) with final academic score in this dataset.")
        elif att_r >= 0.3:
            insights.append(f"Lecture attendance displays a moderate positive correlation (r = {att_r}) with final score in this dataset.")
        else:
            insights.append(f"Lecture attendance shows a weak relationship (r = {att_r}) with final score in this sample cohort.")

    # 2. Study Hours Insight
    hours_r = final_corrs.get("study_hours")
    if hours_r is not None:
        if hours_r >= 0.6:
            insights.append(f"Weekly study hours shows a strong positive correlation (r = {hours_r}) with overall semester score.")
        elif hours_r >= 0.3:
            insights.append(f"Weekly study hours shows a moderate positive relationship (r = {hours_r}) with academic performance.")

    # 3. Assignment Score Insight
    ass_r = final_corrs.get("assignment_average")
    if ass_r is not None:
        if ass_r >= 0.5:
            insights.append(f"Assignment completion score is positively correlated (r = {ass_r}) with end-of-semester marks.")

    # 4. Backlog Count Insight
    backlog_r = final_corrs.get("backlog_count")
    if backlog_r is not None:
        if backlog_r <= -0.3:
            insights.append(f"Active backlog count exhibits a negative correlation (r = {backlog_r}) with overall student score.")

    # 5. Methodological Disclaimer
    insights.append("Methodological Note: All observed correlations indicate statistical association within this sample dataset and do not imply direct causation.")

    return insights
