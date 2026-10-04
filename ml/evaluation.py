"""
EDUPREDICT - ML Evaluation Metrics Module
Computes held-out test set evaluation metrics for Classification (Accuracy, Precision, Recall, F1)
and Regression (MAE, RMSE, R2 Score).
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)

def evaluate_classifier(pipeline, X_test, y_test, model_name="Logistic Regression"):
    """
    Evaluate classification model on test set using weighted multi-class averaging and confusion matrix.
    """
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    labels = ["HIGH", "MEDIUM", "LOW"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    return {
        "modelName": model_name,
        "averagingMethod": "weighted",
        "accuracy": round(float(acc), 3),
        "precision": round(float(prec), 3),
        "recall": round(float(rec), 3),
        "f1Score": round(float(f1), 3),
        "confusionMatrix": {
            "labels": labels,
            "matrix": cm.tolist()
        }
    }

def evaluate_regressor(pipeline, X_test, y_test, model_name="Linear Regression"):
    """
    Evaluate regression model on test set.
    """
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return {
        "modelName": model_name,
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "r2Score": round(float(r2), 3)
    }
