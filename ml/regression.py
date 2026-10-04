"""
EDUPREDICT - ML Regression Models Module
Trains Linear Regression (Primary) and Random Forest (Comparison Baseline) grade regressors.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor

def train_linear_regression(X_train, y_train):
    """
    Train primary Linear Regression regressor pipeline.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def train_random_forest_regressor(X_train, y_train):
    """
    Train comparison Random Forest regressor pipeline.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def train_dummy_regressor(X_train, y_train):
    """
    Train baseline DummyRegressor (predicting mean training target value).
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', DummyRegressor(strategy="mean"))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline
