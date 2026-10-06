# EDUPREDICT — Comprehensive Experimental Evaluation & Empirical Benchmark Report

This document contains the complete, unedited experimental evaluation results of the **EduPredict** project, generated directly from code execution on the 500-record synthetic academic dataset (`data/students.csv`).

---

## 1. Experimental Setup
- **OS / Runtime**: macOS / Python 3.13
- **Primary Libraries**: Scikit-Learn 1.6+, Pandas 2.2+, NumPy 2.2+
- **Train/Test Partitioning**: 80/20 Stratified Held-out Evaluation ($N=400$ Training, $N=100$ Testing, `random_state=42`)
- **Cross-Validation**: 5-Fold Stratified CV (Classification) and 5-Fold CV (Regression) using `sklearn.pipeline.make_pipeline(StandardScaler(), ...)`

---

## 2. Dataset Overview
- **File**: `data/students.csv`
- **Total Records ($N$)**: 500
- **Total Features**: 9 Model Features (11 Columns Total)
- **Missing Values**: 0
- **Duplicate Records**: 0
- **Risk Category Class Distribution**:
  - `MEDIUM` Risk: 193 (38.6%)
  - `HIGH` Risk: 81 (16.2%)
  - `LOW` Risk: 226 (45.2%)

---

## 3. Feature Set ($X \in \mathbb{R}^{N 	imes 9}$)
1. `attendance` (Lecture Attendance %)
2. `study_hours` (Weekly Self-Study Hours)
3. `previous_semester_percentage` (Prior Semester %)
4. `assignment_average` (Coursework Assignment Average %)
5. `internal_assessment` (Internal Exam Score %)
6. `quiz_average` (Short Quiz Mean %)
7. `completed_assignments` (Submitted Coursework Count)
8. `total_assignments` (Total Coursework Count)
9. `backlog_count` (Active Failed Subjects)

---

## 4. Train/Test Split
- **Training Set**: 400 records (80%)
- **Test Set**: 100 records (20%, Stratified by `academic_risk`)

---

## 5. Classification Results (Held-Out $N=100$ Test Set)
- **Model**: `LogisticRegression(max_iter=1000, random_state=42)` inside `StandardScaler` Pipeline
- **Accuracy**: `0.9300` (93.00%)
- **Precision (Weighted)**: `0.9321` (93.21%)
- **Recall (Weighted)**: `0.9300` (93.00%)
- **F1-Score (Weighted)**: `0.9301` (93.01%)
- **Correct Predictions**: `93 / 100`
- **Incorrect Predictions**: `7 / 100`
- **Confusion Matrix** (Rows=Actual `[HIGH, MEDIUM, LOW]`, Cols=Predicted `[HIGH, MEDIUM, LOW]`):
```text
[[16  0  0]
 [ 0 37  2]
 [ 0  5 40]]
```

---

## 6. Classification 5-Fold Cross-Validation
- **Mean Accuracy**: `0.9260 ± 0.0273`
- **Mean Precision**: `0.9307 ± 0.0258`
- **Mean Recall**: `0.9260 ± 0.0273`
- **Mean F1-Score**: `0.9260 ± 0.0274`

---

## 7. Regression Results (Held-Out $N=100$ Test Set)
- **Model**: `LinearRegression()` inside `StandardScaler` Pipeline
- **$R^2$ Score**: `0.9837`
- **Mean Absolute Error (MAE)**: `1.6082%`
- **Root Mean Squared Error (RMSE)**: `1.8430%`
- **Min Absolute Error**: `0.0255%`
- **Max Absolute Error**: `3.4945%`

---

## 8. Regression 5-Fold Cross-Validation
- **Mean $R^2$ Score**: `0.9846 ± 0.0020`
- **Mean MAE**: `1.5105 ± 0.0608`
- **Mean RMSE**: `1.7420 ± 0.0363`

---

## 9. Explainability Results
- **Model Formula**: $\hat{y} = eta_0 + \sum_{i=1}^9 eta_i \cdot z_i$
- **Evaluated Profiles Count**: `5`
- **Mean Score Reconstruction Error**: `0.00000`
- **Max Score Reconstruction Error**: `0.00000`

---

## 10. What-If Analysis Results
- **Evaluated Scenarios**:
  - Scenario A (Attendance 65% $ightarrow$ 80%): `+4.3 percentage points` | `HIGH (Unchanged)`
  - Scenario B (Study Hours 8h $ightarrow$ 14h): `+5.1 percentage points` | `HIGH (Unchanged)`
  - Scenario C (Assignment Avg 60% $ightarrow$ 75%): `+2.9 percentage points` | `HIGH (Unchanged)`
  - Scenario D (Backlog Count 2 $ightarrow$ 0): `+5.4 percentage points` | `HIGH (Unchanged)`
  - Scenario E (Combined Improvement): `+21.7 percentage points` | `HIGH → MEDIUM`
- **Model Stability Verification**: `VERIFIED IMMUTABLE (PASS)` (Zero retraining, SHA-256 hashes & weights match 100%).

---

## 11. Recommendation Validation
- **Total Test Cases**: 8
- **Rule Verification Status**: 100% PASS across threshold alerts, combined profiles, normal condition, and priority sorting.

---

## 12. System Testing
- **Automated Test Suite**: 33 Total Tests (33 Passed, 0 Failed, 0 Skipped)
- **Pass Rate**: `100.0%`

---

## 13. Dataset Validation
- **Records Count**: 500
- **Features Count**: 9
- **Duplicates**: 0
- **Missing Values**: 0

---

## 14. Data Leakage Check
- **Target Matrix Isolation**: VERIFIED ABSENT (`final_score` not in $X$)
- **Scaler Isolation**: VERIFIED (Scaler fitted strictly on training data)
- **Cross-Validation Leakage Prevention**: VERIFIED (Pipeline wrapper enforced)

---

## 15. Limitations
- **Synthetic Data**: Dataset is formulaically generated for first-year academic demonstration.
- **Model Class**: Linear estimators do not capture non-linear feature interactions.
- **Non-Causal Representation**: Model outputs represent standardized statistical projections rather than real-world causal guarantees.

---

## 📊 Summary Performance Benchmark Table

| Metric | Model | Result | Dataset/Evaluation |
| :--- | :--- | :--- | :--- |
| **Classification Accuracy** | Logistic Regression | `93.00%` | Held-Out Test Set ($N=100$) |
| **Classification Precision (Weighted)** | Logistic Regression | `93.21%` | Held-Out Test Set ($N=100$) |
| **Classification Recall (Weighted)** | Logistic Regression | `93.00%` | Held-Out Test Set ($N=100$) |
| **Classification F1-Score (Weighted)** | Logistic Regression | `93.01%` | Held-Out Test Set ($N=100$) |
| **Regression $R^2$ Score** | Linear Regression | `0.9837` | Held-Out Test Set ($N=100$) |
| **Regression MAE** | Linear Regression | `1.6082%` | Held-Out Test Set ($N=100$) |
| **Regression RMSE** | Linear Regression | `1.8430%` | Held-Out Test Set ($N=100$) |
| **Classification CV Mean Accuracy** | Logistic Regression | `92.60% ± 2.73%` | 5-Fold Stratified CV ($N=500$) |
| **Classification CV Mean F1** | Logistic Regression | `92.60% ± 2.74%` | 5-Fold Stratified CV ($N=500$) |
| **Regression CV Mean $R^2$** | Linear Regression | `0.9846 ± 0.0020` | 5-Fold CV ($N=500$) |
| **Regression CV Mean MAE** | Linear Regression | `1.5105% ± 0.0608%` | 5-Fold CV ($N=500$) |
| **Explainability Reconstruction Error** | Standardized Math ($eta_i z_i$) | `0.00000` | Test Profiles Sample ($N=5$) |
| **Automated Test Pass Rate** | PyTest / Unittest | `100.0%` | 33 Unit & Integration Tests |
