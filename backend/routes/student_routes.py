"""
EDUPREDICT - Student API Blueprint
Routes for retrieving student profiles, assessment history, and course mark breakdowns.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from flask import Blueprint, jsonify
try:
    from backend.services.student_service import (
        get_student_profile,
        get_student_history,
        get_student_subjects
    )
except ImportError:
    from services.student_service import (
        get_student_profile,
        get_student_history,
        get_student_subjects
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
