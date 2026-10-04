"""
EDUPREDICT - Assessment API Blueprint
Routes for submitting new academic assessments and retrieving specific assessment records.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from flask import Blueprint, jsonify, request
try:
    from backend.services.assessment_service import (
        create_assessment,
        get_assessment_by_id
    )
except ImportError:
    from services.assessment_service import (
        create_assessment,
        get_assessment_by_id
    )

assessment_bp = Blueprint("assessment_bp", __name__)

@assessment_bp.route("/api/assessment", methods=["POST"])
def post_assessment():
    """Submit and validate a new academic assessment."""
    if not request.is_json:
        return jsonify({
            "success": False,
            "errors": {"payload": "Invalid JSON format in request body."}
        }), 400

    data = request.get_json()
    success, result, status_code = create_assessment(data)
    return jsonify(result), status_code

@assessment_bp.route("/api/assessment/<int:assessment_id>", methods=["GET"])
def get_assessment(assessment_id):
    """Retrieve specific assessment record by ID."""
    record = get_assessment_by_id(assessment_id)
    if not record:
        return jsonify({
            "error": "Assessment record not found.",
            "assessmentId": assessment_id
        }), 404
    return jsonify(record), 200
