"""
EDUPREDICT - Student Service Layer
Database queries for student profiles, assessment history, and course mark breakdowns.
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

def get_student_profile(student_id):
    """
    Retrieve student details and latest academic assessment profile.
    Returns dictionary or None if student not found.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM students WHERE student_id = ?;", (student_id,))
    student = cursor.fetchone()
    if not student:
        conn.close()
        return None

    # Fetch latest assessment
    cursor.execute("""
        SELECT * FROM assessments 
        WHERE student_id = ? 
        ORDER BY submitted_at DESC LIMIT 1;
    """, (student_id,))
    latest_assessment = cursor.fetchone()

    # Fetch course performance for latest assessment
    subjects = []
    if latest_assessment:
        faculty_map = {}
        try:
            cursor.execute("SELECT subject_code, faculty_name, faculty_username FROM subjects;")
            for f_row in cursor.fetchall():
                faculty_map[f_row['subject_code']] = f_row['faculty_name'] or f_row['faculty_username'] or "Classroom Faculty"
        except Exception:
            pass

        cursor.execute("""
            SELECT * FROM subject_performance 
            WHERE assessment_id = ?;
        """, (latest_assessment['assessment_id'],))
        subject_rows = cursor.fetchall()
        for s in subject_rows:
            priority = "LOW"
            if s['marks'] < 75.0 or s['attendance'] < 82.0:
                priority = "HIGH" if s['marks'] < 75.0 and s['attendance'] < 82.0 else "MEDIUM"

            subjects.append({
                "code": s['subject_code'],
                "name": s['subject_name'],
                "faculty": faculty_map.get(s['subject_code'], "Dr. Sarah Jenkins"),
                "score": float(s['marks']),
                "attendance": float(s['attendance']),
                "assignments": float(s['assignment_score']),
                "priority": priority
            })

    conn.close()

    assessment_data = dict(latest_assessment) if latest_assessment else {}

    return {
        "studentId": student['student_id'],
        "name": student['name'],
        "semester": student['semester'],
        "branch": student['branch'],
        "academicScore": float(assessment_data.get("final_score") or 78.4),
        "attendance": float(assessment_data.get("attendance", 0)),
        "studyHours": float(assessment_data.get("study_hours", 0)),
        "previousPercentage": float(assessment_data.get("previous_semester_percentage", 0)),
        "assignmentAvg": float(assessment_data.get("assignment_average", 0)),
        "internalAssessment": float(assessment_data.get("internal_assessment", 0)),
        "quizAvg": float(assessment_data.get("quiz_average", 0)),
        "completedAssignments": int(assessment_data.get("completed_assignments", 0)),
        "totalAssignments": int(assessment_data.get("total_assignments", 15)),
        "backlogCount": int(assessment_data.get("backlog_count", 0)),
        "studentSubjects": subjects,
        "isDatabaseRecord": True
    }

def get_student_history(student_id):
    """Retrieve all past academic assessments for a student."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM assessments 
        WHERE student_id = ? 
        ORDER BY submitted_at DESC;
    """, (student_id,))
    rows = cursor.fetchall()
    conn.close()

    return [dict(r) for r in rows]

def get_student_subjects(student_id):
    """Retrieve course marks for latest assessment."""
    profile = get_student_profile(student_id)
    if profile:
        return profile.get("studentSubjects", [])
    return []

def join_class_by_code(student_id, class_code):
    """Allow a student to join a faculty's class using a 6-character join code."""
    if not student_id or not class_code:
        return False, "Student ID and class join code are required."

    code = class_code.strip().upper()
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT class_id, class_name FROM classes WHERE class_code = ?;", (code,))
    cls = cursor.fetchone()
    if not cls:
        conn.close()
        return False, f"Invalid class join code '{code}'. Class not found."

    class_id = cls["class_id"]
    class_name = cls["class_name"]

    try:
        cursor.execute("""
            INSERT OR IGNORE INTO class_memberships (class_id, student_id)
            VALUES (?, ?);
        """, (class_id, student_id))
        conn.commit()
        conn.close()
        return True, f"Successfully joined '{class_name}' ({code})!"
    except Exception as e:
        conn.close()
        return False, f"Failed to join class: {str(e)}"

def get_student_classes(student_id):
    """Retrieve list of classes joined by a student."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT c.class_id, c.class_name, c.class_code, c.faculty_username, cm.joined_at
        FROM class_memberships cm
        JOIN classes c ON cm.class_id = c.class_id
        WHERE cm.student_id = ?;
    """, (student_id,))
    rows = cursor.fetchall()
    conn.close()

    return [dict(r) for r in rows]

