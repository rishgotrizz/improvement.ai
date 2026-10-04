"""
EDUPREDICT - ML Classification Models Module
Trains Logistic Regression (Primary) and Random Forest (Comparison Baseline) risk classifiers.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

def train_logistic_regression(X_train, y_train):
    """
    Train primary Logistic Regression classifier pipeline.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def train_random_forest_classifier(X_train, y_train):
    """
    Train comparison Random Forest classifier pipeline.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def train_dummy_classifier(X_train, y_train):
    """
    Train baseline DummyClassifier (predicting most frequent class).
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', DummyClassifier(strategy="most_frequent"))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline
