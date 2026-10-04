"""
EDUPREDICT - EDA Data Loader Module
Handles CSV dataset reading, header verification, and Pandas DataFrame initialization.
"""

import os
import pandas as pd

REQUIRED_COLUMNS = [
    'student_id', 'attendance', 'study_hours', 'previous_semester_percentage',
    'assignment_average', 'internal_assessment', 'quiz_average',
    'completed_assignments', 'total_assignments', 'backlog_count', 'final_score'
]

def load_student_dataset(filepath="data/students.csv"):
    """
    Load raw student academic dataset into Pandas DataFrame.
    Verifies required columns and returns (df, error_message).
    """
    if not os.path.exists(filepath):
        return None, f"Dataset file not found at {filepath}"
        
    try:
        df = pd.read_csv(filepath, comment="#")
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            return df, f"Warning: Dataset is missing required columns: {missing_cols}"
            
        print(f"[EDA Loader] Successfully loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.")
        return df, None
    except Exception as e:
        return None, f"Error parsing dataset CSV: {str(e)}"
