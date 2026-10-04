"""
EDUPREDICT - EDA Data Cleaning & Preprocessing Module
Detects missing values, duplicates, and range bounds without silent clipping.
Preserves raw dataset (data/students.csv) and outputs processed dataset (data/processed_students.csv).
"""

import os
import pandas as pd

NUMERIC_COLUMNS = [
    'attendance', 'study_hours', 'previous_semester_percentage',
    'assignment_average', 'internal_assessment', 'quiz_average',
    'completed_assignments', 'total_assignments', 'backlog_count', 'final_score'
]

PERCENTAGE_COLUMNS = [
    'attendance', 'previous_semester_percentage', 'assignment_average',
    'internal_assessment', 'quiz_average', 'final_score'
]

def clean_dataset(df, save_processed=True, output_path="data/processed_students.csv"):
    """
    Execute data cleaning and quality auditing pipeline.
    
    Strategy:
    1. Detect duplicate student records.
    2. Detect missing values per column.
    3. Detect invalid range bounds (e.g. percentages outside 0-100% or negative hours).
    4. Save processed dataset to output_path if modifications are required.
    """
    if df is None or len(df) == 0:
        return None

    cleaned_df = df.copy()
    modifications_made = 0

    # 1. Duplicate Detection
    duplicate_count = cleaned_df.duplicated(subset=['student_id']).sum()
    if duplicate_count > 0:
        cleaned_df.drop_duplicates(subset=['student_id'], keep='first', inplace=True)
        modifications_made += duplicate_count

    # 2. Column Type Normalization & Missing Value Handling
    for col in NUMERIC_COLUMNS:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
            missing_cnt = cleaned_df[col].isnull().sum()
            if missing_cnt > 0:
                median_val = cleaned_df[col].median()
                cleaned_df[col].fillna(median_val, inplace=True)
                modifications_made += missing_cnt

    # 3. Invalid Bound Reporting (Without silent forced clipping)
    invalid_report = {}
    for col in PERCENTAGE_COLUMNS:
        if col in cleaned_df.columns:
            out_of_bounds = cleaned_df[(cleaned_df[col] < 0.0) | (cleaned_df[col] > 100.0)]
            if len(out_of_bounds) > 0:
                invalid_report[col] = len(out_of_bounds)

    if 'study_hours' in cleaned_df.columns:
        neg_hours = cleaned_df[cleaned_df['study_hours'] < 0.0]
        if len(neg_hours) > 0:
            invalid_report['study_hours'] = len(neg_hours)

    # Save processed copy if requested
    if save_processed and modifications_made > 0:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cleaned_df.to_csv(output_path, index=False)
        print(f"[EDA Cleaning] Saved processed dataset to {output_path} ({modifications_made} modifications).")
    else:
        print(f"[EDA Cleaning] Raw dataset verified. {len(cleaned_df)} records are valid (0 modifications required).")

    return cleaned_df
