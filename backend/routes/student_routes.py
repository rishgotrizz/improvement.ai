"""
EDUPREDICT - Student API Blueprint
Routes for student profiles, assessment history, course mark breakdowns, class join codes, MCQ quiz taking, and topic weakness analysis.
"""

import os
import sys
from flask import Blueprint, request, jsonify, session

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from backend.services.student_service import (
        get_student_profile,
        get_student_history,
        get_student_subjects,
        join_class_by_code,
        get_student_classes
    )
    from backend.services.quiz_service import (
        get_class_quizzes,
        get_quiz_questions,
        submit_quiz_attempt,
        get_student_topic_analysis
    )
except ImportError:
    from services.student_service import (
        get_student_profile,
        get_student_history,
        get_student_subjects,
        join_class_by_code,
        get_student_classes
    )
    from services.quiz_service import (
        get_class_quizzes,
        get_quiz_questions,
        submit_quiz_attempt,
        get_student_topic_analysis
    )

student_bp = Blueprint("student_bp", __name__)

@student_bp.route("/api/student/<student_id>", methods=["GET"])
def get_student(student_id):
    """Retrieve student profile and latest assessment."""
    profile = get_student_profile(student_id)
    if not profile:
        return jsonify({
            "error": "Student profile not found.",
            "requestedId": student_id
        }), 404
    return jsonify(profile), 200

@student_bp.route("/api/student/<student_id>/history", methods=["GET"])
def get_history(student_id):
    """Retrieve student assessment history."""
    history = get_student_history(student_id)
    return jsonify({
        "studentId": student_id,
        "count": len(history),
        "history": history
    }), 200

@student_bp.route("/api/student/<student_id>/subjects", methods=["GET"])
def get_subjects(student_id):
    """Retrieve latest course mark breakdown for a student."""
    subjects = get_student_subjects(student_id)
    return jsonify({
        "studentId": student_id,
        "subjects": subjects
    }), 200

@student_bp.route("/api/student/join-class", methods=["POST"])
def post_join_class():
    """Allow a student to join a class using a join code."""
    data = request.get_json(silent=True) or {}
    class_code = data.get("classCode")
    student_id = data.get("studentId") or session.get("studentId") or "STU-2026-001"

    success, msg = join_class_by_code(student_id, class_code)
    if not success:
        return jsonify({"success": False, "error": msg}), 400

    return jsonify({"success": True, "message": msg}), 200

@student_bp.route("/api/student/<student_id>/classes", methods=["GET"])
def get_classes_for_student(student_id):
    """Retrieve list of classes joined by a student."""
    classes = get_student_classes(student_id)
    return jsonify({"success": True, "studentId": student_id, "classes": classes}), 200

@student_bp.route("/api/student/<student_id>/quizzes", methods=["GET"])
def get_quizzes(student_id):
    """Retrieve quizzes available for student's classrooms."""
    classes = get_student_classes(student_id)
    all_quizzes = []

    for c in classes:
        qz_list = get_class_quizzes(c["class_id"])
        all_quizzes.extend(qz_list)

    return jsonify({"success": True, "studentId": student_id, "quizzes": all_quizzes}), 200

@student_bp.route("/api/student/quizzes/<int:quiz_id>", methods=["GET"])
def get_quiz(quiz_id):
    """Retrieve quiz questions for student taking."""
    quiz_data = get_quiz_questions(quiz_id, include_correct=False)
    if not quiz_data:
        return jsonify({"error": "Quiz not found."}), 404
    return jsonify({"success": True, "quiz": quiz_data}), 200

@student_bp.route("/api/student/quizzes/<int:quiz_id>/attempt", methods=["POST"])
def post_quiz_attempt(quiz_id):
    """Submit student MCQ quiz answers and get instant score & topic breakdown."""
    data = request.get_json(silent=True) or {}
    student_id = data.get("studentId") or session.get("studentId") or "STU-2026-001"
    answers = data.get("answers") or {}

    success, result, msg = submit_quiz_attempt(student_id, quiz_id, answers)
    if not success:
        return jsonify({"success": False, "error": msg}), 400

    return jsonify({"success": True, "result": result, "message": msg}), 200

@student_bp.route("/api/student/<student_id>/topic-analysis", methods=["GET"])
def get_topic_analysis(student_id):
    """Retrieve student topic-level weakness analysis."""
    analysis = get_student_topic_analysis(student_id)
    return jsonify({"success": True, "studentId": student_id, "analysis": analysis}), 200
