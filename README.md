<<<<<<< HEAD
# improvement.ai
=======
# 🎓 EDUPREDICT: AI-Powered Student Performance & Academic Risk Analysis System

**EduPredict** is an end-to-end Explainable Learning Analytics & Early Warning System (EWS) designed for higher education student performance prediction, academic risk intervention, model contribution explainability, and counterfactual scenario simulation.

---

## 🌟 Key Features

- **Multi-Class Risk Classification**: Classifies student academic risk into `HIGH`, `MEDIUM`, and `LOW` tiers using Scikit-Learn `LogisticRegression` ($93.0\%$ test accuracy).
- **Continuous Score Regression**: Predicts end-of-semester numerical percentage scores ($20.0\% - 99.5\%$) using `LinearRegression` ($R^2 = 0.984$, MAE $1.61\%$).
- **Model Contribution Explainability**: Derives standardized feature-level contributions ($\text{contrib}_i = \beta_i \times z_i$) proving exact linear score reconstruction ($\hat{y} = \beta_0 + \sum \text{contrib}_i$).
- **Deterministic Recommendation Engine**: Evaluates student attributes against explicit academic thresholds and prioritizes action items by urgency tier rank and severity.
- **What-If Academic Scenario Simulator**: Allows students to simulate hypothetical feature modifications ($\Delta \hat{y} = \hat{y}_{\text{scenario}} - \hat{y}_{\text{baseline}}$) without model retraining.
- **First-Year Engineering Syllabus Alignment**: Connects directly to six core subjects: Applied Mathematics-I (AM-I), Computer Programming & Algorithms (CPA), Computer Organization & Architecture (COA), Intro to Artificial Intelligence (IAI), Exploratory Data Analysis (EDA), and Ethics in AI (EAI).

---

## 🏛️ Project Structure

```text
edupredict/
├── backend/
│   ├── app.py                     # Flask REST API Server & Route Handlers
│   ├── routes/                    # API Route Blueprints (Students, Assessments)
│   └── services/                  # Prediction, Explanation, What-If, Recommendation Services
├── data/
│   ├── edupredict.db              # SQLite Relational Persistence Store
│   └── students.csv               # 500-Record Academic Synthetic Dataset
├── eda/                           # Exploratory Data Analysis & Quality Reporting Modules
├── frontend/                      # User Interface (HTML5 / Modern CSS / Vanilla JS)
│   ├── analytics.html             # Real Distribution Charts & Heatmap Dashboard
│   ├── assessment.html            # Student Profile Entry & Database Persistence
│   ├── model.html                 # Held-Out 80/20 Test Set Metrics Dashboard
│   ├── prediction.html            # Risk Prediction & Feature Signals Panel
│   ├── what-if.html               # Scenario Simulator & Score Difference Comparison
│   └── mathematics.html           # Syllabus Math & Formulation Reference
├── ml/                            # ML Pipeline, Feature Matrices, & Inference Modules
├── models/                        # Saved Model Artifacts (.pkl) & Evaluation Metrics
├── recommendations/               # Centralized Threshold Rules & Deterministic Engine
└── tests/                         # Automated Unit & Integration Test Suites
```

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.9+
- Flask, Pandas, NumPy, Scikit-Learn, Joblib

### 1. Installation
```bash
git clone https://github.com/rishgotrizz/improvement.ai.git
cd improvement.ai
```

### 2. Start the Backend API Server
```bash
python3 backend/app.py
```
The server will start locally on `http://localhost:5001`.

### 3. Key Web Interfaces
- **Homepage / Overview**: `http://localhost:5001/`
- **Risk Prediction & Signals**: `http://localhost:5001/prediction`
- **What-If Scenario Simulator**: `http://localhost:5001/what-if`
- **EDA Analytics Dashboard**: `http://localhost:5001/analytics`
- **AI Model Metrics**: `http://localhost:5001/model`

---

## 🧪 Running Automated Tests

Run the test suite covering prediction pipelines, recommendation rules, and what-if scenario validation:

```bash
PYTHONPATH=. python3 tests/test_phase5.py
PYTHONPATH=. python3 tests/test_phase6.py
```

---

## 📄 License & Ethical Disclaimer

> *Model results and what-if simulations represent linear model projections based on synthetic demonstration data. They do not establish causal guarantees or real-world deterministic promises.*
>>>>>>> b16d957 (Initial commit: EduPredict AI-Powered Academic Risk & Scenario Simulator System)
