"""
EDUPREDICT - ML Preprocessing Pipeline Module
Constructs feature scaling transformer fitted strictly on training data (preventing data leakage).
"""

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def build_preprocessing_pipeline():
    """
    Construct Scikit-Learn preprocessing pipeline (StandardScaler).
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])
    return pipeline
