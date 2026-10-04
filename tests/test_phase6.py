"""
EDUPREDICT - Phase 6 Unit & Integration Test Suite
Validates What-If analysis, scenario comparison, input validation, non-causal outputs, and model immutability.
"""

import os
import sys
import unittest
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.services.what_if_service import validate_academic_input, run_what_if_analysis
from backend.services.prediction_service import execute_ml_prediction
from backend.services.recommendation_service import fetch_recommendations_for_input

DEFAULT_BASELINE = {
    "attendance": 65.0,
    "study_hours": 8.0,
    "previous_semester_percentage": 65.0,
    "assignment_average": 60.0,
    "internal_assessment": 58.0,
    "quiz_average": 55.0,
    "completed_assignments": 6,
    "total_assignments": 10,
    "backlog_count": 2
}

class TestPhase6WhatIf(unittest.TestCase):

    def test_01_valid_baseline_scenario_request(self):
        scenario = dict(DEFAULT_BASELINE, attendance=80.0, study_hours=12.0)
        succ, res, status = run_what_if_analysis(DEFAULT_BASELINE, scenario)
        self.assertTrue(succ)
        self.assertEqual(status, 200)
        self.assertTrue(res["success"])
        self.assertIn("baseline", res)
        self.assertIn("scenario", res)
        self.assertIn("difference", res)

    def test_02_identical_baseline_and_scenario(self):
        succ, res, status = run_what_if_analysis(DEFAULT_BASELINE, DEFAULT_BASELINE)
        self.assertTrue(succ)
        self.assertEqual(res["difference"]["score_change"], 0.0)
        self.assertFalse(res["difference"]["risk_changed"])
        self.assertEqual(res["changed_factors_count"], 0)

    def test_03_attendance_only_change(self):
        scenario = dict(DEFAULT_BASELINE, attendance=85.0)
        succ, res, status = run_what_if_analysis(DEFAULT_BASELINE, scenario)
        self.assertTrue(succ)
        self.assertEqual(res["changed_factors_count"], 1)
        self.assertEqual(res["changed_factors"][0]["feature"], "attendance")
        self.assertEqual(res["changed_factors"][0]["baseline_value"], 65.0)
        self.assertEqual(res["changed_factors"][0]["scenario_value"], 85.0)

    def test_04_study_hours_only_change(self):
        scenario = dict(DEFAULT_BASELINE, study_hours=14.0)
        succ, res, status = run_what_if_analysis(DEFAULT_BASELINE, scenario)
        self.assertTrue(succ)
        self.assertEqual(res["changed_factors_count"], 1)
        self.assertEqual(res["changed_factors"][0]["feature"], "study_hours")
        self.assertEqual(res["changed_factors"][0]["difference"], 6.0)

    def test_05_multiple_factor_change(self):
        scenario = dict(DEFAULT_BASELINE, attendance=80.0, study_hours=14.0, assignment_average=80.0, backlog_count=0)
        succ, res, status = run_what_if_analysis(DEFAULT_BASELINE, scenario)
        self.assertTrue(succ)
        self.assertEqual(res["changed_factors_count"], 4)

    def test_06_backlog_reduction(self):
        scenario = dict(DEFAULT_BASELINE, backlog_count=0)
        succ, res, status = run_what_if_analysis(DEFAULT_BASELINE, scenario)
        self.assertTrue(succ)
        self.assertEqual(res["changed_factors"][0]["feature"], "backlog_count")
        self.assertEqual(res["changed_factors"][0]["difference"], -2)

    def test_07_invalid_attendance(self):
        scenario = dict(DEFAULT_BASELINE, attendance=150.0)
        succ, res, status = run_what_if_analysis(DEFAULT_BASELINE, scenario)
        self.assertFalse(succ)
        self.assertEqual(status, 400)
        self.assertIn("Attendance", res["error"])

    def test_08_invalid_study_hours(self):
        scenario = dict(DEFAULT_BASELINE, study_hours=200.0)
        succ, res, status = run_what_if_analysis(DEFAULT_BASELINE, scenario)
        self.assertFalse(succ)
        self.assertEqual(status, 400)
        self.assertIn("Study hours", res["error"])

    def test_09_negative_backlog(self):
        scenario = dict(DEFAULT_BASELINE, backlog_count=-1)
        succ, res, status = run_what_if_analysis(DEFAULT_BASELINE, scenario)
        self.assertFalse(succ)
        self.assertEqual(status, 400)
        self.assertIn("Backlog count", res["error"])

    def test_10_completed_greater_than_total_assignments(self):
        scenario = dict(DEFAULT_BASELINE, completed_assignments=12, total_assignments=10)
        succ, res, status = run_what_if_analysis(DEFAULT_BASELINE, scenario)
        self.assertFalse(succ)
        self.assertEqual(status, 400)
        self.assertIn("cannot exceed total assignments", res["error"])

    def test_11_missing_required_field(self):
        invalid_base = dict(DEFAULT_BASELINE)
        del invalid_base["attendance"]
        succ, res, status = run_what_if_analysis(invalid_base, DEFAULT_BASELINE)
        self.assertFalse(succ)
        self.assertEqual(status, 400)
        self.assertIn("Missing required feature", res["error"])

    def test_12_probability_comparison(self):
        scenario = dict(DEFAULT_BASELINE, attendance=90.0, study_hours=18.0)
        succ, res, status = run_what_if_analysis(DEFAULT_BASELINE, scenario)
        self.assertTrue(succ)
        probs = res["difference"]["probability_comparison"]
        self.assertIn("LOW", probs)
        self.assertIn("MEDIUM", probs)
        self.assertIn("HIGH", probs)
        self.assertIn("formatted_delta", probs["LOW"])

    def test_13_risk_category_comparison(self):
        scenario = dict(DEFAULT_BASELINE, attendance=90.0, study_hours=18.0, backlog_count=0)
        succ, res, status = run_what_if_analysis(DEFAULT_BASELINE, scenario)
        self.assertTrue(succ)
        self.assertIn("risk_transition", res["difference"])
        self.assertIn("risk_summary", res["difference"])

    def test_14_changed_factor_detection(self):
        scenario = dict(DEFAULT_BASELINE, quiz_average=85.0)
        succ, res, status = run_what_if_analysis(DEFAULT_BASELINE, scenario)
        self.assertTrue(succ)
        self.assertEqual(len(res["changed_factors"]), 1)
        self.assertEqual(res["changed_factors"][0]["label"], "Quiz Average (%)")

    def test_15_scenario_recommendations_generated_from_scenario_values(self):
        scenario = dict(DEFAULT_BASELINE, attendance=90.0, study_hours=18.0, backlog_count=0, assignment_average=90.0, quiz_average=90.0, internal_assessment=90.0)
        succ, res, status = run_what_if_analysis(DEFAULT_BASELINE, scenario)
        self.assertTrue(succ)
        # Improved profile should have fewer or 0 recommendations compared to baseline
        base_recs = fetch_recommendations_for_input(DEFAULT_BASELINE)
        scen_recs = res["recommendations"]
        self.assertLessEqual(len(scen_recs), len(base_recs))

    def test_16_existing_phase4_prediction_still_works(self):
        succ, res, status = execute_ml_prediction(DEFAULT_BASELINE)
        self.assertTrue(succ)
        self.assertEqual(status, 200)
        self.assertIn("predicted_score", res)
        self.assertIn("risk_category", res)

    def test_17_existing_phase5_recommendations_still_work(self):
        recs = fetch_recommendations_for_input(DEFAULT_BASELINE)
        self.assertIsInstance(recs, list)

    def test_18_what_if_does_not_retrain_model(self):
        clf_path = os.path.join(PROJECT_ROOT, "models/risk_classifier.pkl")
        mtime_before = os.path.getmtime(clf_path) if os.path.exists(clf_path) else 0
        scenario = dict(DEFAULT_BASELINE, attendance=85.0)
        run_what_if_analysis(DEFAULT_BASELINE, scenario)
        mtime_after = os.path.getmtime(clf_path) if os.path.exists(clf_path) else 0
        self.assertEqual(mtime_before, mtime_after)

    def test_19_deterministic_output_for_identical_inputs(self):
        scenario = dict(DEFAULT_BASELINE, attendance=80.0, study_hours=12.0)
        succ1, res1, _ = run_what_if_analysis(DEFAULT_BASELINE, scenario)
        succ2, res2, _ = run_what_if_analysis(DEFAULT_BASELINE, scenario)
        self.assertEqual(res1["difference"]["score_change"], res2["difference"]["score_change"])
        self.assertEqual(res1["difference"]["risk_transition"], res2["difference"]["risk_transition"])

    def test_20_no_causal_language_in_output(self):
        scenario = dict(DEFAULT_BASELINE, attendance=85.0)
        succ, res, status = run_what_if_analysis(DEFAULT_BASELINE, scenario)
        res_str = json.dumps(res).lower()
        forbidden_assertions = ["will increase your", "will decrease your", "guarantees that", "causes your performance", "proves causality"]
        for term in forbidden_assertions:
            self.assertNotIn(term, res_str)

if __name__ == "__main__":
    unittest.main()
