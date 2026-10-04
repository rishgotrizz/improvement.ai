"""
EDUPREDICT - EDA Visualization Data Generation Module
Computes binned histogram distributions and scatter plot coordinate datasets for UI rendering.
"""

import numpy as np
import pandas as pd

def generate_distribution_bins(series, num_bins=10):
    """
    Compute histogram frequency bins for a numerical Pandas Series.
    Returns dictionary with bin labels and frequency counts.
    """
    clean_series = series.dropna()
    if len(clean_series) == 0:
        return {"bins": [], "counts": [], "totalCount": 0}

    counts, bin_edges = np.histogram(clean_series, bins=num_bins)
    
    bin_labels = []
    for i in range(len(counts)):
        label = f"{round(bin_edges[i], 1)}-{round(bin_edges[i+1], 1)}"
        bin_labels.append(label)

    return {
        "bins": bin_labels,
        "counts": [int(c) for c in counts],
        "binEdges": [round(float(b), 1) for b in bin_edges],
        "totalCount": int(sum(counts))
    }

def generate_scatter_points(df, x_col, y_col, max_samples=500):
    """
    Extract sample coordinate pairs (x, y) for UI scatter plots from actual dataset.
    """
    if df is None or x_col not in df.columns or y_col not in df.columns:
        return []

    valid_df = df[[x_col, y_col]].dropna()
    if len(valid_df) > max_samples:
        valid_df = valid_df.sample(n=max_samples, random_state=42)

    points = []
    for _, row in valid_df.iterrows():
        points.append({
            "x": round(float(row[x_col]), 1),
            "y": round(float(row[y_col]), 1)
        })
    return points

def compute_eda_visualizations_data(df):
    """
    Generate complete visualization payload for frontend dashboard.
    """
    if df is None or len(df) == 0:
        return {}

    return {
        "scoreDistribution": generate_distribution_bins(df.get("final_score", pd.Series()), num_bins=10),
        "attendanceDistribution": generate_distribution_bins(df.get("attendance", pd.Series()), num_bins=10),
        "studyHoursDistribution": generate_distribution_bins(df.get("study_hours", pd.Series()), num_bins=10),
        "scatterAttendanceVsScore": generate_scatter_points(df, "attendance", "final_score"),
        "scatterStudyHoursVsScore": generate_scatter_points(df, "study_hours", "final_score"),
        "scatterAssignmentVsScore": generate_scatter_points(df, "assignment_average", "final_score")
    }
