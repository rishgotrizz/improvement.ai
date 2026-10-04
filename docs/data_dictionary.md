# EduPredict — Dataset Data Dictionary

This document details the feature attributes, data types, physical bounds, and target variable definitions for the 500-record synthetic student dataset (`data/students.csv`).

---

## Attribute Specifications

| Field Name | Data Type | Range / Constraints | Description & Role |
| :--- | :--- | :--- | :--- |
| `student_id` | String / Text | `STU-2026-001` to `STU-2026-500` | Anonymous student identifier. Protects student privacy by avoiding personal PII. |
| `attendance` | Float | $0.0 \le x \le 100.0\%$ | Percentage of mandatory lectures attended by the student. |
| `study_hours` | Float | $x \ge 0.0$ (hrs/week) | Average self-study hours spent per week outside class. |
| `previous_semester_percentage` | Float | $0.0 \le x \le 100.0\%$ | Cumulative overall percentage score from the preceding academic term. |
| `assignment_average` | Float | $0.0 \le x \le 100.0\%$ | Mean percentage marks achieved across submitted coursework assignments. |
| `internal_assessment` | Float | $0.0 \le x \le 100.0\%$ | Mid-term or internal evaluation percentage score. |
| `quiz_average` | Float | $0.0 \le x \le 100.0\%$ | Mean percentage score achieved in short class quizzes. |
| `completed_assignments` | Integer | $0 \le x \le \text{total\_assignments}$ | Number of completed coursework assignments submitted on time. |
| `total_assignments` | Integer | $x \ge 1$ (Default: 15) | Total number of assigned coursework assignments. |
| `backlog_count` | Integer | $x \ge 0$ | Number of active uncleared failed subjects. |
| `final_score` | Float | $0.0 \le x \le 100.0\%$ | **TARGET VARIABLE**: Total end-of-semester overall percentage score. |

---

## Target Leakage Prevention

> [!IMPORTANT]
> **Prediction Target Designation**:
> `final_score` is the continuous **dependent target variable** ($y$) for regression modeling in **Phase 4**.
> To prevent **target leakage**, `final_score` must NEVER be included as an input feature ($x_i$) when training regression models or generating risk predictions.
