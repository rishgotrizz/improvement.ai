"""
EDUPREDICT - XAI & What-If Service Integration Tests
Verifies end-to-end API payloads, explanation contribution calculations,
and What-If scenario delta calculations without model retraining.
"""

import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app import app
from backend.services.explanation_service import explain_individual_prediction
from backend.services.what_if_service import run_what_if_analysis

class TestXAIAndWhatIfDebug(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()
        self.sample_valid_input = {
            "attendance": 85.0,
            "study_hours": 14.0,
            "previous_semester_percentage": 78.0,
            "assignment_average": 82.0,
            "internal_assessment": 80.0,
            "quiz_average": 76.0,
            "completed_assignments": 9,
            "total_assignments": 10,
            "backlog_count": 0
        }

    def test_01_xai_explanation_structure(self):
        """Verify explain_individual_prediction returns required XAI fields."""
        exp = explain_individual_prediction(self.sample_valid_input)
        self.assertIn("predicted_class", exp)
        self.assertIn("positive_signals", exp)
        self.assertIn("negative_signals", exp)
        self.assertIn("feature_contributions", exp)
        self.assertIn("regression_explanation", exp)

        # Verify contribution math structure
        contribs = exp["feature_contributions"]
        self.assertEqual(len(contribs), 9)
        for item in contribs:
            self.assertIn("feature", item)
            self.assertIn("name", item)
            self.assertIn("raw_value", item)
            self.assertIn("standardized_value", item)
            self.assertIn("weight", item)
            self.assertIn("contribution", item)

    def test_02_predict_api_xai_integration(self):
        """Verify /api/predict returns top-level explanation and recommendations when authenticated."""
        with self.client.session_transaction() as sess:
            sess["username"] = "testuser"
            sess["role"] = "student"
        res = self.client.post("/api/predict", json=self.sample_valid_input)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertIn("predicted_score", data)
        self.assertIn("risk_category", data)
        self.assertIn("explanation", data)
        self.assertIn("recommendations", data)

        # Check positive/negative signals
        exp = data["explanation"]
        self.assertIn("feature_contributions", exp)
        self.assertEqual(len(exp["feature_contributions"]), 9)

    def test_03_what_if_analysis_scenario_delta(self):
        """Verify run_what_if_analysis calculates baseline vs scenario differences."""
        baseline = self.sample_valid_input
        scenario = dict(baseline)
        scenario["attendance"] = 95.0
        scenario["study_hours"] = 20.0

        succ, res, status = run_what_if_analysis(baseline, scenario)
        self.assertTrue(succ)
        self.assertEqual(status, 200)
        self.assertTrue(res["success"])
        self.assertIn("baseline", res)
        self.assertIn("scenario", res)
        self.assertIn("difference", res)
        self.assertIn("changed_factors", res)

        diff = res["difference"]
        self.assertIn("score_change", diff)
        self.assertIn("risk_transition", diff)
        self.assertEqual(len(res["changed_factors"]), 2)

    def test_04_what_if_api_endpoint(self):
        """Verify /api/what-if endpoint returns HTTP 200 and scenario comparisons when authenticated."""
        with self.client.session_transaction() as sess:
            sess["username"] = "testuser"
            sess["role"] = "student"
        baseline = self.sample_valid_input
        scenario = dict(baseline)
        scenario["backlog_count"] = 0

        res = self.client.post("/api/what-if", json={"baseline": baseline, "scenario": scenario})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertIn("baseline", data)
        self.assertIn("scenario", data)
        self.assertIn("difference", data)

    def test_05_what_if_invalid_inputs(self):
        """Verify invalid inputs return HTTP 400 error for authenticated session."""
        with self.client.session_transaction() as sess:
            sess["username"] = "testuser"
            sess["role"] = "student"
        invalid_payload = {"baseline": {"attendance": 150.0}, "scenario": self.sample_valid_input}
        res = self.client.post("/api/what-if", json=invalid_payload)
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertFalse(data["success"])
        self.assertIn("error", data)

    def test_06_unauthenticated_api_and_route_protection(self):
        """Verify unauthenticated requests to XAI and What-If endpoints return 401/redirect."""
        # Unauthenticated /api/xai returns 401
        xai_res = self.client.post("/api/xai", json=self.sample_valid_input)
        self.assertEqual(xai_res.status_code, 401)
        xai_data = xai_res.get_json()
        self.assertFalse(xai_data["success"])
        self.assertIn("Authentication required", xai_data["error"])

        # Unauthenticated /api/what-if returns 401
        whatif_res = self.client.post("/api/what-if", json={"baseline": self.sample_valid_input, "scenario": self.sample_valid_input})
        self.assertEqual(whatif_res.status_code, 401)
        whatif_data = whatif_res.get_json()
        self.assertFalse(whatif_data["success"])
        self.assertIn("Authentication required", whatif_data["error"])

        # Unauthenticated /api/predict succeeds with basic predictions but omits detailed XAI explanation
        pred_res = self.client.post("/api/predict", json=self.sample_valid_input)
        self.assertEqual(pred_res.status_code, 200)
        pred_data = pred_res.get_json()
        self.assertTrue(pred_data["success"])
        self.assertIsNone(pred_data.get("explanation"))
        self.assertIn("predicted_score", pred_data)
        self.assertIn("risk_category", pred_data)
        self.assertIn("recommendations", pred_data)

        # Unauthenticated page routes redirect to /login
        pred_page = self.client.get("/prediction")
        self.assertEqual(pred_page.status_code, 302)
        self.assertIn("/login", pred_page.headers.get("Location", ""))

        whatif_page = self.client.get("/what-if")
        self.assertEqual(whatif_page.status_code, 302)
        self.assertIn("/login", whatif_page.headers.get("Location", ""))

if __name__ == "__main__":
    unittest.main()
