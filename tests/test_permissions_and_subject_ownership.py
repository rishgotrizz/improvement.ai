"""
EDUPREDICT - Security, Permissions & Subject Faculty Ownership Tests
Tests for:
1. Enrolled student read-only permissions (403 Forbidden on official record mutation attempts).
2. Student MCQ quiz taking capability (backend grading, score calculation).
3. Subject Faculty A vs Subject Faculty B isolation (403 Forbidden on unauthorized subject modification).
4. What-If simulation non-mutating behavior on official academic records.
5. Seeded synthetic history verification for demo student STU-2026-001 vs clean empty state for single-assessment students.
"""

import unittest
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app import app
from backend.services.database import init_db, get_db_connection
from backend.services.student_service import get_student_history, get_student_profile
from backend.services.quiz_service import submit_quiz_attempt

class TestPermissionsAndSubjectOwnership(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()
        cls.client = app.test_client()

    def test_01_student_cannot_invoke_faculty_endpoints(self):
        """Verify enrolled student role receives 403 Forbidden on faculty endpoints."""
        headers = {"X-User-Role": "student", "X-User-Username": "student"}
        
        # 1. Create class attempt
        res1 = self.client.post("/api/faculty/classes", json={"className": "Hack Class"}, headers=headers)
        self.assertEqual(res1.status_code, 403)
        self.assertIn("Unauthorized", res1.get_json().get("error", ""))

        # 2. Add student attempt
        res2 = self.client.post("/api/faculty/classes/1/students", json={"name": "Hacker"}, headers=headers)
        self.assertEqual(res2.status_code, 403)

        # 3. Create quiz attempt
        res3 = self.client.post("/api/faculty/classes/1/quizzes", json={"title": "Hacked Quiz"}, headers=headers)
        self.assertEqual(res3.status_code, 403)

        # 4. Intervention attempt
        res4 = self.client.post("/api/faculty/interventions", json={"actionTitle": "Fake Intervention"}, headers=headers)
        self.assertEqual(res4.status_code, 403)

    def test_02_student_can_take_quiz_and_get_backend_graded_score(self):
        """Verify student can attempt quiz and score is computed strictly server-side."""
        # Submit attempt for MAT101 quiz (ID 1)
        answers = {"1": "B", "2": "B", "3": "A", "4": "A", "5": "A", "6": "A"} # All 6 correct
        res = self.client.post("/api/student/quizzes/1/attempt", json={
            "studentId": "STU-2026-001",
            "answers": answers
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["result"]["score"], 6)
        self.assertEqual(data["result"]["totalQuestions"], 6)
        self.assertEqual(data["result"]["scorePercentage"], 100.0)

    def test_03_subject_faculty_isolation(self):
        """Verify Subject Faculty A (prof_math) cannot modify Subject Faculty B's (prof_cs) subject records."""
        valid_questions = [
            {"questionText": "What is x?", "optionA": "1", "optionB": "2", "optionC": "3", "optionD": "4", "correctOption": "A", "topic": "Variables"}
        ]

        # prof_math attempting to create quiz for CS101 (owned by prof_cs)
        headers_math = {"X-User-Role": "faculty", "X-User-Username": "prof_math"}
        res = self.client.post("/api/faculty/classes/1/quizzes", json={
            "subjectCode": "CS101",
            "title": "Unauthorized Math Quiz on CS Course",
            "questions": valid_questions
        }, headers=headers_math)
        self.assertEqual(res.status_code, 403)
        self.assertIn("cannot modify records belonging to", res.get_json().get("error", ""))

        # prof_cs creating quiz for CS101 should succeed
        headers_cs = {"X-User-Role": "faculty", "X-User-Username": "prof_cs"}
        res_ok = self.client.post("/api/faculty/classes/1/quizzes", json={
            "subjectCode": "CS101",
            "title": "Temporary Test Quiz",
            "questions": valid_questions
        }, headers=headers_cs)
        self.assertEqual(res_ok.status_code, 201)
        created_qz = res_ok.get_json().get("quiz", {})
        if created_qz and created_qz.get("quizId"):
            from backend.services.database import get_db_connection
            conn = get_db_connection()
            conn.execute("DELETE FROM quiz_questions WHERE quiz_id = ?", (created_qz["quizId"],))
            conn.execute("DELETE FROM quizzes WHERE quiz_id = ?", (created_qz["quizId"],))
            conn.commit()
            conn.close()

    def test_04_what_if_does_not_mutate_official_academic_records(self):
        """Verify What-If scenario simulation returns hypothetical output without altering SQLite DB."""
        profile_before = get_student_profile("STU-2026-001")
        official_score_before = profile_before["academicScore"]

        # Log in test session for What-If simulation
        with self.client.session_transaction() as sess:
            sess["username"] = "STU-2026-001"
            sess["role"] = "student"

        # Run What-If simulation with boosted study hours
        baseline = {"attendance": 87.0, "study_hours": 18.0, "previous_semester_percentage": 79.0, "assignment_average": 82.5, "internal_assessment": 76.0, "quiz_average": 80.0, "completed_assignments": 14, "total_assignments": 15, "backlog_count": 0}
        scenario = {"attendance": 95.0, "study_hours": 25.0, "previous_semester_percentage": 79.0, "assignment_average": 90.0, "internal_assessment": 85.0, "quiz_average": 88.0, "completed_assignments": 15, "total_assignments": 15, "backlog_count": 0}
        
        res = self.client.post("/api/what-if", json={"baseline": baseline, "scenario": scenario})
        self.assertEqual(res.status_code, 200)
        
        # Profile after simulation MUST be identical
        profile_after = get_student_profile("STU-2026-001")
        self.assertEqual(profile_after["academicScore"], official_score_before)

    def test_05_demo_student_history_points(self):
        """Verify demo student STU-2026-001 has 5 seeded historical snapshots for the demo graphs."""
        history = get_student_history("STU-2026-001")
        self.assertEqual(len(history), 5)
        
        scores = [float(h["final_score"]) for h in sorted(history, key=lambda x: x["submitted_at"])]
        self.assertEqual(scores, [56.0, 62.0, 69.0, 75.0, 78.4])

    def test_06_non_demo_student_has_single_record(self):
        """Verify non-demo student STU-2026-002 has only 1 snapshot (shows missing-data notice when < 2)."""
        history = get_student_history("STU-2026-002")
        self.assertEqual(len(history), 1)

if __name__ == "__main__":
    unittest.main()
