"""
EDUPREDICT - Assessment Service Layer
Mandatory backend validation and SQLite database persistence for student academic assessments.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from backend.services.database import get_db_connection
except ImportError:
    from services.database import get_db_connection

def validate_assessment_data(data):
    """
    Mandatory backend validation rules.
    Returns (is_valid: bool, errors: dict).
    """
    errors = {}

    def check_pct(field_name, label):
        val = data.get(field_name)
        if val is None or val == "":
            errors[field_name] = f"{label} is required."
            return None
        try:
            val_f = float(val)
            if val_f < 0.0 or val_f > 100.0:
                errors[field_name] = f"{label} must be between 0 and 100."
                return None
            return val_f
        except (ValueError, TypeError):
            errors[field_name] = f"{label} must be a valid number."
            return None

    # Attendance
    att = check_pct("attendance", "Attendance rate")
    
    # Study Hours
    hours = data.get("study_hours")
    if hours is None or hours == "":
        errors["study_hours"] = "Study hours is required."
    else:
        try:
            hours_f = float(hours)
            if hours_f < 0.0:
                errors["study_hours"] = "Study hours must be non-negative."
        except (ValueError, TypeError):
            errors["study_hours"] = "Study hours must be a valid number."

    # Previous Score Percentage
    check_pct("previous_score", "Previous semester percentage")
    
    # Assignment Average
    check_pct("assignment_avg", "Assignment average")
    
    # Internal Assessment
    check_pct("internal_score", "Internal assessment score")
    
    # Quiz Average
    if "quiz_avg" in data and data["quiz_avg"] is not None and data["quiz_avg"] != "":
        check_pct("quiz_avg", "Quiz average")

    # Completed & Total Assignments
    comp = data.get("completed_assignments", 14)
    tot = data.get("total_assignments", 15)
    try:
        comp_i = int(comp)
        tot_i = int(tot)
        if comp_i < 0:
            errors["completed_assignments"] = "Completed assignments must be non-negative."
        if tot_i < comp_i:
            errors["total_assignments"] = "Total assignments must be greater than or equal to completed assignments."
    except (ValueError, TypeError):
        errors["completed_assignments"] = "Assignment counts must be valid integers."

    # Backlog Count
    backlogs = data.get("backlog_count", 0)
    try:
        if int(backlogs) < 0:
            errors["backlog_count"] = "Backlog count must be non-negative."
    except (ValueError, TypeError):
        errors["backlog_count"] = "Backlog count must be a valid integer."

    is_valid = len(errors) == 0
    return is_valid, errors

def create_assessment(data):
    """
    Validate and save a new student assessment record to SQLite.
    Returns (success: bool, response_dict: dict, status_code: int).
    """
    is_valid, errors = validate_assessment_data(data)
    if not is_valid:
        return False, {"success": False, "errors": errors}, 400

    student_id = data.get("student_id", "STU-2026-001")

    conn = get_db_connection()
    cursor = conn.cursor()

    # Ensure student exists in DB
    cursor.execute("SELECT student_id FROM students WHERE student_id = ?;", (student_id,))
    if not cursor.fetchone():
        cursor.execute("""
            INSERT INTO students (student_id, name, semester, branch)
            VALUES (?, 'Demo Student', 2, 'Computer Engineering');
        """, (student_id,))

    # Calculate weighted average as placeholder score (No ML prediction used)
    att = float(data.get("attendance"))
    hours = float(data.get("study_hours"))
    prev = float(data.get("previous_score"))
    ass = float(data.get("assignment_avg"))
    internal = float(data.get("internal_score", 76.0))
    quiz = float(data.get("quiz_avg", 80.0))
    comp = int(data.get("completed_assignments", 14))
    tot = int(data.get("total_assignments", 15))
    backlogs = int(data.get("backlog_count", 0))

    calc_score = (0.30 * att) + (0.30 * prev) + (0.20 * ass) + (0.20 * internal)
    calc_score = round(min(99.0, max(20.0, calc_score)), 1)

    # Insert into assessments
    cursor.execute("""
        INSERT INTO assessments (
            student_id, attendance, study_hours, previous_semester_percentage,
            assignment_average, internal_assessment, quiz_average,
            completed_assignments, total_assignments, backlog_count, final_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, (student_id, att, hours, prev, ass, internal, quiz, comp, tot, backlogs, calc_score))

    assessment_id = cursor.lastrowid

    # Insert course mark breakdowns
    default_courses = [
        ("CS102", "Data Structures", data.get("score_cs102", 74.0)),
        ("MATH101", "Mathematics", data.get("score_math101", 72.0)),
        ("PHY101", "Physics", data.get("score_phy101", 81.0)),
        ("CS101", "Computer Programming", data.get("score_cs101", 88.0)),
        ("EE101", "Basic Electrical Engineering", data.get("score_ee101", 84.0))
    ]

    for code, name, score in default_courses:
        score_f = float(score) if score is not None else 75.0
        cursor.execute("""
            INSERT INTO subject_performance (
                assessment_id, student_id, subject_code, subject_name, marks, attendance, assignment_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?);
        """, (assessment_id, student_id, code, name, score_f, att, ass))

    conn.commit()
    conn.close()

    return True, {
        "success": True,
        "message": "Academic assessment successfully saved to database.",
        "assessmentId": assessment_id,
        "studentId": student_id,
        "summary": {
            "attendance": att,
            "studyHours": hours,
            "calculatedAverageScore": calc_score
        }
    }, 201

def get_assessment_by_id(assessment_id):
    """Retrieve specific assessment record from SQLite."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM assessments WHERE assessment_id = ?;", (assessment_id,))
    row = cursor.fetchone()
    conn.close()

    if row:
        return dict(row)
    return None
