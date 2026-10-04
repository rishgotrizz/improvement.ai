"""
EDUPREDICT - EDA Analysis Orchestrator Module
Runs full EDA pipeline: Data Loading -> Cleaning -> Statistics -> Outliers -> Visualizations -> Insights.
"""

from eda.loader import load_student_dataset
from eda.cleaning import clean_dataset
from eda.statistics import compute_descriptive_statistics, compute_pearson_correlations
from eda.outliers import analyze_outliers_iqr
from eda.visualizations import compute_eda_visualizations_data
from eda.insights import generate_eda_insights

# Global in-memory cached analysis payload to prevent recalculation overhead
_CACHED_EDA_RESULT = None

def run_full_eda_analysis(filepath="data/students.csv", force_reload=False):
    """
    Run full EDA analysis pipeline on student dataset.
    Caches result in RAM to optimize backend response time (O(1) retrieval).
    """
    global _CACHED_EDA_RESULT

    if _CACHED_EDA_RESULT is not None and not force_reload:
        return _CACHED_EDA_RESULT

    df, err = load_student_dataset(filepath)
    if df is None:
        return {"error": err or "Dataset failed to load."}

    cleaned_df = clean_dataset(df)
    
    stats = compute_descriptive_statistics(cleaned_df)
    correlations = compute_pearson_correlations(cleaned_df)
    outliers = analyze_outliers_iqr(cleaned_df)
    visuals = compute_eda_visualizations_data(cleaned_df)
    insights = generate_eda_insights(stats, correlations)

    analysis_payload = {
        "dataset": {
            "records": len(cleaned_df),
            "features": len(cleaned_df.columns),
            "targetVariable": "final_score"
        },
        "statistics": stats,
        "correlations": correlations,
        "distributions": visuals,
        "outliers": outliers,
        "insights": insights,
        "status": "SUCCESS"
    }

    _CACHED_EDA_RESULT = analysis_payload
    print(f"📊 [EDA Pipeline] Completed full analysis on {len(cleaned_df)} student records.")
    return analysis_payload
