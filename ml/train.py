"""
EDUPREDICT - ML Model Training Orchestrator Script
Executes 80/20 train/test split, trains Scikit-Learn pipelines, evaluates held-out metrics,
persists model artifacts (models/*.pkl), and saves structured evaluation results (models/metrics.json).

Usage:
    python3 -m ml.train
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split

from ml.dataset import prepare_ml_dataset, FEATURE_COLUMNS, REGRESSION_TARGET
from ml.classification import train_logistic_regression, train_random_forest_classifier, train_dummy_classifier
from ml.regression import train_linear_regression, train_random_forest_regressor, train_dummy_regressor
from ml.evaluation import evaluate_classifier, evaluate_regressor

def train_and_evaluate():
    """
    Main training execution function.
    """
    print("🚀 Starting EDUPREDICT ML Training Pipeline...")

    # 1. Prepare X and y
    X, y_reg, y_cls = prepare_ml_dataset()

    # 2. Target Leakage Verification
    assert REGRESSION_TARGET not in X.columns, "CRITICAL ERROR: Target leakage detected!"

    # 3. Train / Test Split (80% Train, 20% Test)
    X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = train_test_split(
        X, y_cls, y_reg,
        test_size=0.20,
        random_state=42,
        stratify=y_cls
    )

    print(f"✂️ Train/Test Split: {len(X_train)} training records (80%), {len(X_test)} testing records (20%).")

    # 4. Train Classification Models
    print("🤖 Training Classification Models...")
    clf_logreg = train_logistic_regression(X_train, y_cls_train)
    clf_rf = train_random_forest_classifier(X_train, y_cls_train)
    clf_dummy = train_dummy_classifier(X_train, y_cls_train)

    metrics_logreg = evaluate_classifier(clf_logreg, X_test, y_cls_test, "Logistic Regression")
    metrics_rf_cls = evaluate_classifier(clf_rf, X_test, y_cls_test, "Random Forest Classifier")
    metrics_dummy_cls = evaluate_classifier(clf_dummy, X_test, y_cls_test, "Dummy Classifier (Most Frequent)")

    print(f"  - Logistic Regression Accuracy: {metrics_logreg['accuracy']} (F1: {metrics_logreg['f1Score']})")
    print(f"  - Random Forest Classifier Accuracy: {metrics_rf_cls['accuracy']} (F1: {metrics_rf_cls['f1Score']})")
    print(f"  - Dummy Classifier Accuracy: {metrics_dummy_cls['accuracy']} (F1: {metrics_dummy_cls['f1Score']})")

    # 5. Train Regression Models
    print("📈 Training Regression Models...")
    reg_linreg = train_linear_regression(X_train, y_reg_train)
    reg_rf = train_random_forest_regressor(X_train, y_reg_train)
    reg_dummy = train_dummy_regressor(X_train, y_reg_train)

    metrics_linreg = evaluate_regressor(reg_linreg, X_test, y_reg_test, "Linear Regression")
    metrics_rf_reg = evaluate_regressor(reg_rf, X_test, y_reg_test, "Random Forest Regressor")
    metrics_dummy_reg = evaluate_regressor(reg_dummy, X_test, y_reg_test, "Dummy Regressor (Training Mean)")

    print(f"  - Linear Regression MAE: {metrics_linreg['mae']}%, RMSE: {metrics_linreg['rmse']}%, R2: {metrics_linreg['r2Score']}")
    print(f"  - Random Forest Regressor MAE: {metrics_rf_reg['mae']}%, RMSE: {metrics_rf_reg['rmse']}%, R2: {metrics_rf_reg['r2Score']}")
    print(f"  - Dummy Regressor MAE: {metrics_dummy_reg['mae']}%, RMSE: {metrics_dummy_reg['rmse']}%, R2: {metrics_dummy_reg['r2Score']}")

    # 6. Save Model Artifacts
    models_dir = os.path.abspath(os.path.join(PROJECT_ROOT, "models"))
    os.makedirs(models_dir, exist_ok=True)

    clf_path = os.path.join(models_dir, "risk_classifier.pkl")
    reg_path = os.path.join(models_dir, "score_regressor.pkl")

    joblib.dump(clf_logreg, clf_path)
    joblib.dump(reg_linreg, reg_path)
    print(f"💾 Persisted model pipelines: {clf_path}, {reg_path}")

    # 7. Save Metrics JSON File
    metrics_payload = {
        "datasetInfo": {
            "totalRecords": len(X),
            "trainRecords": len(X_train),
            "testRecords": len(X_test),
            "trainRatio": 0.80,
            "testRatio": 0.20,
            "randomState": 42,
            "isStratified": True,
            "features": FEATURE_COLUMNS,
            "regressionTarget": REGRESSION_TARGET,
            "classificationTarget": "academic_risk_category",
            "isSyntheticDataset": True
        },
        "classification": {
            "primary": metrics_logreg,
            "comparison": metrics_rf_cls,
            "dummyBaseline": metrics_dummy_cls
        },
        "regression": {
            "primary": metrics_linreg,
            "comparison": metrics_rf_reg,
            "dummyBaseline": metrics_dummy_reg
        },
        "trainedAt": datetime.now().isoformat(),
        "disclaimer": "Model results are based on a synthetic demonstration dataset and should not be interpreted as real-world academic performance benchmarks."
    }

    metrics_json_path = os.path.join(models_dir, "metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"📝 Saved metrics JSON to {metrics_json_path}")
    print("✅ ML Training & Evaluation complete!")
    return metrics_payload

if __name__ == "__main__":
    train_and_evaluate()
