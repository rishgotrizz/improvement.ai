"""
EDUPREDICT - Phase 5 Unit & Integration Test Suite
Verifies Explainability Engine, Multiclass Logistic Feature Contributions,
Linear Regression Score Reconstruction, Recommendation Rules, Prioritization Sorting,
and API Endpoints.
"""

import unittest
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.services.explanation_service import explain_individual_prediction
from backend.services.prediction_service import execute_ml_prediction
from recommendations.engine import generate_academic_recommendations, compute_severity_score
from recommendations.rules import evaluate_factor_status, get_rule_recommendation

class TestPhase5(unittest.TestCase):

    def setUp(self):
        self.profile_a_strong = {
            "attendance": 87.0,
            "study_hours": 18.0,
            "previous_semester_percentage": 79.0,
            "assignment_average": 82.5,
            "internal_assessment": 76.0,
            "quiz_average": 80.0,
            "completed_assignments": 14,
            "total_assignments": 15,
            "backlog_count": 0
        }

        self.profile_b_at_risk = {
            "attendance": 45.0,
            "study_hours": 4.0,
            "previous_semester_percentage": 50.0,
            "assignment_average": 40.0,
            "internal_assessment": 42.0,
            "quiz_average": 45.0,
            "completed_assignments": 5,
            "total_assignments": 15,
            "backlog_count": 3
        }

        self.profile_c_mixed = {
            "attendance": 88.0,
            "study_hours": 8.0,
            "previous_semester_percentage": 72.0,
            "assignment_average": 55.0,
            "internal_assessment": 78.0,
            "quiz_average": 72.0,
            "completed_assignments": 14,
            "total_assignments": 15,
            "backlog_count": 1
        }

    # 1. Explanation output exists
    def test_explanation_output_exists(self):
        exp = explain_individual_prediction(self.profile_a_strong)
        self.assertIsNotNone(exp)
        self.assertIn("predicted_class", exp)
        self.assertIn("positive_signals", exp)
        self.assertIn("negative_signals", exp)
        self.assertIn("feature_contributions", exp)
        self.assertIn("regression_explanation", exp)

    # 2. Feature contributions contain all expected features
    def test_feature_contributions_contain_all_features(self):
        exp = explain_individual_prediction(self.profile_a_strong)
        features_in_exp = [c["feature"] for c in exp["feature_contributions"]]
        self.assertEqual(len(features_in_exp), 9)
        self.assertIn("attendance", features_in_exp)
        self.assertIn("study_hours", features_in_exp)
        self.assertIn("backlog_count", features_in_exp)

    # 3 & 4. Multiclass predicted-class coefficients and scaled values used correctly
    def test_multiclass_coefficient_selection(self):
        exp = explain_individual_prediction(self.profile_b_at_risk)
        self.assertEqual(exp["predicted_class"], "HIGH")
        # Attendance 45% reduces semester score (contribution < 0) while contributing positively to HIGH risk log-odds
        att_contrib = next(c for c in exp["feature_contributions"] if c["feature"] == "attendance")
        self.assertLess(att_contrib["contribution"], 0)
        self.assertGreater(att_contrib["classifier_weight"] * att_contrib["classifier_z"], 0)

    # 5. Regression contribution calculation is numerically consistent
    def test_regression_contribution_reconstruction(self):
        exp = explain_individual_prediction(self.profile_a_strong)
        reg_exp = exp["regression_explanation"]
        intercept = reg_exp["intercept"]
        sum_contribs = reg_exp["sum_contributions"]
        reconstructed = intercept + sum_contribs
        self.assertAlmostEqual(reconstructed, reg_exp["reconstructed_score"], places=1)

    # 6. Positive/negative signal extraction
    def test_signal_extraction(self):
        exp = explain_individual_prediction(self.profile_a_strong)
        self.assertGreater(len(exp["positive_signals"]), 0)

    # 7. Recommendation engine handles healthy inputs
    def test_recommendations_healthy_profile(self):
        recs = generate_academic_recommendations(self.profile_a_strong)
        self.assertEqual(len(recs), 0)

    # 8 & 9. Recommendation engine handles weak inputs and multiple weak factors
    def test_recommendations_at_risk_profile(self):
        recs = generate_academic_recommendations(self.profile_b_at_risk)
        self.assertGreaterEqual(len(recs), 7)
        factors = [r["factor"] for r in recs]
        self.assertIn("Weekly Self-Study Hours", factors)
        self.assertIn("Lecture Attendance Rate", factors)
        self.assertIn("Active Backlog Count", factors)

    # 10. Recommendations are deterministically ordered
    def test_deterministic_recommendation_ordering(self):
        recs_run1 = generate_academic_recommendations(self.profile_c_mixed)
        recs_run2 = generate_academic_recommendations(self.profile_c_mixed)
        self.assertEqual(recs_run1, recs_run2)
        priorities = [r["priority"] for r in recs_run1]
        self.assertEqual(priorities, list(range(1, len(recs_run1) + 1)))

    # 11. Missing subject data does not crash recommendation engine or API
    def test_missing_subject_data(self):
        recs = generate_academic_recommendations(self.profile_c_mixed, student_subjects=None)
        self.assertIsNotNone(recs)

    # 12. Invalid prediction input still returns validation errors
    def test_invalid_input(self):
        success, result, status = execute_ml_prediction(None)
        self.assertFalse(success)
        self.assertEqual(status, 400)

    # 13. Existing Phase 4 prediction output remains functional
    def test_phase4_prediction_output_functional(self):
        success, result, status = execute_ml_prediction(self.profile_a_strong)
        self.assertTrue(success)
        self.assertEqual(status, 200)
        self.assertIn("predicted_score", result)
        self.assertIn("risk_category", result)
        self.assertIn("risk_probability", result)
        self.assertIn("explanation", result)
        self.assertIn("recommendations", result)

    # 14. No recommendation logic changes the ML prediction itself
    def test_recommendation_does_not_mutate_prediction(self):
        success, res1, _ = execute_ml_prediction(self.profile_c_mixed)
        score1 = res1["predicted_score"]
        cat1 = res1["risk_category"]
        success, res2, _ = execute_ml_prediction(self.profile_c_mixed)
        self.assertEqual(score1, res2["predicted_score"])
        self.assertEqual(cat1, res2["risk_category"])

    # 15. No Phase 6 functionality introduced
    def test_no_phase6_what_if_in_response(self):
        success, result, _ = execute_ml_prediction(self.profile_a_strong)
        self.assertNotIn("simulated_score", result)
        self.assertNotIn("what_if", result)

if __name__ == "__main__":
    unittest.main()
