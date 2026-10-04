"""
EDUPREDICT - Dataset Quality Report Module
Analyzes CSV dataset integrity, records count, missing values, duplicates, and column types.
"""

import os
import pandas as pd

def generate_quality_report(filepath="data/students.csv"):
    """
    Generate comprehensive dataset quality report for EDA pipeline.
    """
    if not os.path.exists(filepath):
        return {"error": f"Dataset file not found at {filepath}"}

    # Load skipping comment header if present
    df = pd.read_csv(filepath, comment="#")

    total_records = len(df)
    missing_dict = df.isnull().sum().to_dict()
    total_missing = sum(missing_dict.values())
    duplicate_rows = df.duplicated().sum()

    # Invalid value checks (e.g. attendance out of 0-100 bounds)
    invalid_records = 0
    if "attendance" in df.columns:
        invalid_records += ((df["attendance"] < 0) | (df["attendance"] > 100)).sum()
    if "study_hours" in df.columns:
        invalid_records += (df["study_hours"] < 0).sum()

    column_types = {col: str(dtype) for col, dtype in df.dtypes.items()}

    report = {
        "datasetPath": filepath,
        "totalRecords": total_records,
        "totalColumns": len(df.columns),
        "columns": list(df.columns),
        "columnTypes": column_types,
        "totalMissingValues": int(total_missing),
        "missingPerColumn": {k: int(v) for k, v in missing_dict.items()},
        "duplicateRows": int(duplicate_rows),
        "invalidRecordsCount": int(invalid_records),
        "status": "HEALTHY" if (total_missing == 0 and invalid_records == 0) else "NEEDS_CLEANING",
        "isDemoDataset": True
    }
    return report
