"""
EDUPREDICT - Faculty Monitoring & Analytics API Blueprint
Routes for faculty classes, overview metrics, searchable student roster, detailed XAI audit, manual student entry, MCQ quiz builder with topic tagging, class topic analytics, and intervention tracking.
"""

import os
import sys
from flask import Blueprint, request, jsonify, session

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from backend.services.faculty_service import (
        get_faculty_classes,
        create_faculty_class,
        update_faculty_class_status,
        get_class_subjects,
        get_faculty_overview,
        get_faculty_student_list,
        add_student_to_class_manually,
        get_faculty_student_detail,
        add_faculty_intervention
    )
    from backend.services.quiz_service import (
        create_quiz,
        get_class_quizzes,
        get_class_topic_analytics
    )
except ImportError:
    from services.faculty_service import (
        get_faculty_classes,
        create_faculty_class,
        update_faculty_class_status,
        get_class_subjects,
        get_faculty_overview,
        get_faculty_student_list,
        add_student_to_class_manually,
        get_faculty_student_detail,
        add_faculty_intervention
    )
    from services.quiz_service import (
        create_quiz,
        get_class_quizzes,
        get_class_topic_analytics
    )

faculty_bp = Blueprint("faculty_bp", __name__)

def check_faculty_access():
    """Helper to verify if request is from an authorized faculty user."""
    req_role = request.headers.get("X-User-Role") or session.get("role")
    if req_role == "student":
        return False, "Unauthorized. Student role cannot modify official academic records."
    return True, None

def check_subject_faculty_access(class_id, subject_code=None):
    """Verify if requesting faculty owns the class or is assigned to the specific subject."""
    is_fac, msg = check_faculty_access()
    if not is_fac:
        return False, msg

    req_user = request.headers.get("X-User-Username") or session.get("username", "faculty")
    
    # Allow superuser / default demo admin 'faculty'
    if req_user == "faculty":
        return True, None

    try:
        from backend.services.database import get_db_connection
        conn = get_db_connection()
        cursor = conn.cursor()

        # Check if user is class owner
        cursor.execute("SELECT faculty_username FROM classes WHERE class_id = ?;", (class_id,))
        cls_row = cursor.fetchone()
        if cls_row and cls_row['faculty_username'] == req_user:
            conn.close()
            return True, None

        # Check subject assignment
        if subject_code:
            cursor.execute("SELECT faculty_username FROM subjects WHERE class_id = ? AND subject_code = ?;", (class_id, subject_code))
            sub_row = cursor.fetchone()
            if sub_row and sub_row['faculty_username']:
                if sub_row['faculty_username'] != req_user:
                    owner = sub_row['faculty_username']
                    conn.close()
                    return False, f"Unauthorized. Subject faculty '{req_user}' cannot modify records belonging to '{owner}'."

        conn.close()
    except Exception:
        pass

    return True, None

@faculty_bp.route("/api/faculty/classes", methods=["GET"])
def get_classes():
    """Retrieve list of classes managed by faculty."""
    ok, err = check_faculty_access()
    if not ok:
        return jsonify({"error": err}), 403

    faculty_username = request.headers.get("X-User-Username") or session.get("username", "faculty")
    classes = get_faculty_classes(faculty_username)
    return jsonify({"success": True, "count": len(classes), "classes": classes}), 200

@faculty_bp.route("/api/faculty/classes", methods=["POST"])
def post_class():
    """Create a new academic class with unique join code."""
    ok, err = check_faculty_access()
    if not ok:
        return jsonify({"error": err}), 403

    data = request.get_json(silent=True) or {}
    class_name = data.get("className")
    custom_code = data.get("classCode")
    course = data.get("course")
    semester = data.get("semester")
    section = data.get("section")
    faculty_username = request.headers.get("X-User-Username") or session.get("username", "faculty")

    success, res_data, msg = create_faculty_class(faculty_username, class_name, custom_code, course=course, semester=semester, section=section)
    if not success:
        return jsonify({"success": False, "error": msg}), 400

    return jsonify({"success": True, "class": res_data, "message": msg}), 201

@faculty_bp.route("/api/faculty/classes/<int:class_id>/status", methods=["PATCH"])
def patch_class_status(class_id):
    """Update active/archived status of a classroom."""
    ok, err = check_faculty_access()
    if not ok:
        return jsonify({"error": err}), 403

    data = request.get_json(silent=True) or {}
    status = data.get("status", "ACTIVE").upper()
    faculty_username = request.headers.get("X-User-Username") or session.get("username", "faculty")

    success, msg = update_faculty_class_status(class_id, faculty_username, status)
    if not success:
        return jsonify({"success": False, "error": msg}), 400

    return jsonify({"success": True, "message": msg, "classId": class_id, "status": status}), 200

@faculty_bp.route("/api/faculty/classes/<int:class_id>/subjects", methods=["GET"])
def get_classroom_subjects(class_id):
    """Retrieve list of subjects assigned to a specific class."""
    ok, err = check_faculty_access()
    if not ok:
        return jsonify({"error": err}), 403

    subjects = get_class_subjects(class_id)
    return jsonify({"success": True, "classId": class_id, "subjects": subjects}), 200

@faculty_bp.route("/api/faculty/overview", methods=["GET"])
def get_overview():
    """Retrieve aggregate faculty monitoring KPI cards metrics."""
    ok, err = check_faculty_access()
    if not ok:
        return jsonify({"error": err}), 403

    class_id = request.args.get("class_id")
    faculty_username = request.headers.get("X-User-Username") or session.get("username", "faculty")
    overview = get_faculty_overview(faculty_username, class_id=class_id)
    return jsonify({"success": True, "overview": overview}), 200

@faculty_bp.route("/api/faculty/students", methods=["GET"])
def get_students():
    """Retrieve searchable, filterable, and sortable student roster for faculty."""
    ok, err = check_faculty_access()
    if not ok:
        return jsonify({"error": err}), 403

    search = request.args.get("search", "")
    risk_filter = request.args.get("risk", "ALL")
    sort_by = request.args.get("sort", "risk")
    class_id = request.args.get("class_id")

    faculty_username = request.headers.get("X-User-Username") or session.get("username", "faculty")
    students = get_faculty_student_list(faculty_username=faculty_username, search=search, risk_filter=risk_filter, sort_by=sort_by, class_id=class_id)
    return jsonify({
        "success": True,
        "count": len(students),
        "students": students
    }), 200

@faculty_bp.route("/api/faculty/classes/<int:class_id>/students", methods=["POST"])
def post_manual_student(class_id):
    """Manually add a student and their academic metrics to a class."""
    ok, err = check_subject_faculty_access(class_id)
    if not ok:
        return jsonify({"error": err}), 403

    data = request.get_json(silent=True) or {}
    student_name = data.get("name")
    metrics = data.get("metrics", {})
    faculty_username = request.headers.get("X-User-Username") or session.get("username", "faculty")

    success, msg = add_student_to_class_manually(faculty_username, class_id, student_name, metrics)
    if not success:
        return jsonify({"success": False, "error": msg}), 400

    return jsonify({"success": True, "message": msg}), 201

@faculty_bp.route("/api/faculty/classes/<int:class_id>/quizzes", methods=["POST"])
def post_quiz(class_id):
    """Create an MCQ evaluation quiz with topic-tagged questions."""
    data = request.get_json(silent=True) or {}
    subject_code = data.get("subjectCode", "MAT101")
    
    ok, err = check_subject_faculty_access(class_id, subject_code)
    if not ok:
        return jsonify({"error": err}), 403

    title = data.get("title")
    questions = data.get("questions", [])

    success, res_data, msg = create_quiz(class_id, subject_code, title, questions)
    if not success:
        return jsonify({"success": False, "error": msg}), 400

    return jsonify({"success": True, "quiz": res_data, "message": msg}), 201

@faculty_bp.route("/api/faculty/classes/<int:class_id>/topic-analytics", methods=["GET"])
def get_class_topics(class_id):
    """Retrieve class-average topic performance analytics for faculty."""
    ok, err = check_faculty_access()
    if not ok:
        return jsonify({"error": err}), 403

    analytics = get_class_topic_analytics(class_id)
    return jsonify({"success": True, "classId": class_id, "analytics": analytics}), 200

@faculty_bp.route("/api/faculty/students/<student_id>", methods=["GET"])
def get_student_detail(student_id):
    """Retrieve full student audit details including ML predictions, XAI signals, recommendations, and interventions."""
    ok, err = check_faculty_access()
    if not ok:
        return jsonify({"error": err}), 403

    detail = get_faculty_student_detail(student_id)
    if not detail:
        return jsonify({"error": "Student profile not found.", "studentId": student_id}), 404

    return jsonify({"success": True, "student": detail}), 200

@faculty_bp.route("/api/faculty/interventions", methods=["POST"])
def post_intervention():
    """Record a new faculty intervention for a student."""
    ok, err = check_faculty_access()
    if not ok:
        return jsonify({"error": err}), 403

    data = request.get_json(silent=True) or {}
    student_id = data.get("studentId")
    action_title = data.get("actionTitle")
    notes = data.get("notes", "")
    status = data.get("status", "Assigned")
    follow_up_date = data.get("followUpDate", "")

    faculty_username = request.headers.get("X-User-Username") or session.get("username", "faculty")

    success, msg = add_faculty_intervention(student_id, faculty_username, action_title, notes, status, follow_up_date)
    if not success:
        return jsonify({"success": False, "error": msg}), 400

    return jsonify({"success": True, "message": msg}), 200
