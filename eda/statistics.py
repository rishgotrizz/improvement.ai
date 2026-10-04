"""
EDUPREDICT - EDA Statistical Analysis Module
Calculates descriptive statistics (Mean, Median, Mode, Min, Max, Sample Variance, Sample Std Dev)
and Pearson Correlation Coefficients across all numerical student dataset features.

Statistical Convention:
Sample Variance (s^2) uses Bessel's correction with degrees of freedom ddof = 1:
    s^2 = \sum_{i=1}^{n} (x_i - \bar{x})^2 / (n - 1)
Sample Standard Deviation (s) is the square root of Sample Variance:
    s = \sqrt{s^2}
"""

import numpy as np
import pandas as pd

def _clean_val(val):
    """Ensure value is standard JSON-serializable float or None if NaN/Inf."""
    if pd.isna(val) or np.isinf(val):
        return None
    return float(val)

def compute_descriptive_statistics(df):
    """
    Compute sample descriptive statistics (ddof=1) for all numerical features.
    """
    if df is None or len(df) == 0:
        return {}

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats_dict = {}

    for col in numeric_cols:
        series = df[col].dropna()
        n = len(series)
        if n == 0:
            continue

        mean_val = series.mean()
        # Sample variance with ddof=1 (Bessel's correction)
        var_val = series.var(ddof=1) if n > 1 else 0.0
        # Sample standard deviation
        std_val = series.std(ddof=1) if n > 1 else 0.0
        
        median_val = series.median()
        min_val = series.min()
        max_val = series.max()
        q1_val = series.quantile(0.25)
        q3_val = series.quantile(0.75)
        iqr_val = q3_val - q1_val

        # Mode calculation
        mode_series = series.mode()
        mode_val = mode_series.iloc[0] if len(mode_series) > 0 else mean_val

        stats_dict[col] = {
            "count": int(n),
            "mean": _clean_val(round(mean_val, 2)),
            "median": _clean_val(round(median_val, 2)),
            "mode": _clean_val(round(mode_val, 2)),
            "sampleVariance": _clean_val(round(var_val, 2)), # s^2 (ddof=1)
            "sampleStdDev": _clean_val(round(std_val, 2)),   # s (ddof=1)
            "std": _clean_val(round(std_val, 2)),             # alias for UI
            "variance": _clean_val(round(var_val, 2)),        # alias for UI
            "min": _clean_val(round(min_val, 2)),
            "max": _clean_val(round(max_val, 2)),
            "q1": _clean_val(round(q1_val, 2)),
            "q3": _clean_val(round(q3_val, 2)),
            "iqr": _clean_val(round(iqr_val, 2)),
            "varianceConvention": "Sample Variance (ddof=1)"
        }

    return stats_dict

def compute_pearson_correlations(df):
    """
    Compute Pearson Correlation Matrix (-1.0 <= r <= 1.0) across all numerical features.
    Formula: r = \sum (x_i - \bar{x})(y_i - \bar{y}) / \sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}
    """
    if df is None or len(df) == 0:
        return {}

    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr(method="pearson")

    result = {}
    for col1 in corr_matrix.columns:
        result[col1] = {}
        for col2 in corr_matrix.columns:
            r_val = corr_matrix.loc[col1, col2]
            result[col1][col2] = _clean_val(round(r_val, 3))

    return result
