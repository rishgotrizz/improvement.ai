"""
EDUPREDICT - EDA Outlier Analysis Module
Implements Interquartile Range (IQR) method to identify potential statistical outliers.
Formula: Lower = Q1 - 1.5*IQR, Upper = Q3 + 1.5*IQR
"""

import numpy as np
import pandas as pd

def analyze_outliers_iqr(df):
    """
    Perform IQR outlier detection across all numerical features.
    Reports lower/upper bounds and outlier counts without modifying or deleting dataset rows.
    """
    if df is None or len(df) == 0:
        return {}

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_report = {}

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        outliers = series[(series < lower_bound) | (series > upper_bound)]
        outlier_count = len(outliers)

        outlier_report[col] = {
            "q1": round(float(q1), 2),
            "q3": round(float(q3), 2),
            "iqr": round(float(iqr), 2),
            "lowerBound": round(float(lower_bound), 2),
            "upperBound": round(float(upper_bound), 2),
            "outlierCount": int(outlier_count),
            "outlierPercentage": round((outlier_count / len(series)) * 100.0, 1),
            "classification": "potential statistical outlier"
        }

    return outlier_report
