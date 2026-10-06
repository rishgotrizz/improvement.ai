"""
EDUPREDICT - Faculty Analytics & Student Monitoring Service Layer
Handles class management, student cohort aggregation, ML risk classification & XAI audit, and intervention tracking.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from backend.services.database import get_db_connection, generate_class_code
    from backend.services.prediction_service import execute_ml_prediction
    from backend.services.explanation_service import explain_individual_prediction
    from backend.services.recommendation_service import fetch_recommendations_for_input
    from backend.services.student_service import get_student_profile, get_student_history
except ImportError:
    from services.database import get_db_connection, generate_class_code
    from services.prediction_service import execute_ml_prediction
    from services.explanation_service import explain_individual_prediction
    from services.recommendation_service import fetch_recommendations_for_input
    from services.student_service import get_student_profile, get_student_history

def get_faculty_classes(faculty_username):
    """Retrieve list of classes owned by a faculty member with student counts."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT c.class_id, c.class_name, c.class_code, c.created_at,
               COUNT(m.student_id) as student_count
        FROM classes c
        LEFT JOIN class_memberships m ON c.class_id = m.class_id
        WHERE c.faculty_username = ?
        GROUP BY c.class_id
        ORDER BY c.created_at ASC;
    """, (faculty_username,))
    rows = cursor.fetchall()
    conn.close()

    return [dict(r) for r in rows]

def create_faculty_class(faculty_username, class_name, custom_code=None):
    """Create a new academic class with unique join code."""
    if not class_name or not class_name.strip():
        return False, None, "Class name is required."

    class_code = custom_code.strip().upper() if custom_code else generate_class_code()

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            INSERT INTO classes (class_name, class_code, faculty_username)
            VALUES (?, ?, ?);
        """, (class_name.strip(), class_code, faculty_username))
        class_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return True, {"classId": class_id, "className": class_name, "classCode": class_code}, "Class created successfully."
    except Exception as e:
        conn.close()
        return False, None, f"Failed to create class: {str(e)}"

def get_faculty_overview(faculty_username=None):
    """
    Compute aggregate faculty monitoring statistics across all assigned student records.
    """
    students_list = get_faculty_student_list(faculty_username=faculty_username, search=None, risk_filter="ALL", sort_by="risk")

    total_students = len(students_list)
    high_risk = sum(1 for s in students_list if s["riskLevel"] == "HIGH")
    medium_risk = sum(1 for s in students_list if s["riskLevel"] == "MEDIUM")
    low_risk = sum(1 for s in students_list if s["riskLevel"] == "LOW")

    requiring_attention = sum(1 for s in students_list if s["riskLevel"] == "HIGH" or (s["riskLevel"] == "MEDIUM" and s["backlogCount"] > 0))

    improving_count = 0
    improving_supported = False

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT student_id, COUNT(assessment_id) as cnt 
        FROM assessments GROUP BY student_id HAVING cnt >= 2;
    """)
    multi_assessments = cursor.fetchall()

    if multi_assessments:
        improving_supported = True
        for row in multi_assessments:
            sid = row["student_id"]
            cursor.execute("""
                SELECT final_score FROM assessments 
                WHERE student_id = ? ORDER BY submitted_at DESC LIMIT 2;
            """, (sid,))
            scores = cursor.fetchall()
            if len(scores) == 2 and scores[0]["final_score"] > scores[1]["final_score"]:
                improving_count += 1
    conn.close()

    return {
        "totalStudents": total_students,
        "highRiskCount": high_risk,
        "mediumRiskCount": medium_risk,
        "lowRiskCount": low_risk,
        "requiringAttentionCount": requiring_attention,
        "improvingCount": improving_count,
        "historicalDataAvailable": improving_supported,
        "improvingNote": "Calculated from multi-assessment submission trends." if improving_supported else "Historical trend data unavailable for single-assessment cohort."
    }

def get_faculty_student_list(faculty_username=None, search=None, risk_filter="ALL", sort_by="risk", class_id=None):
    """
    Retrieve students belonging to faculty's classes with ML predictions, risk classification, and academic factors.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    query = """
        SELECT DISTINCT s.student_id, s.name, s.semester, s.branch,
               a.attendance, a.study_hours, a.previous_semester_percentage,
               a.assignment_average, a.internal_assessment, a.quiz_average,
               a.completed_assignments, a.total_assignments, a.backlog_count,
               a.final_score, a.submitted_at, c.class_name, c.class_id
        FROM students s
        LEFT JOIN (
            SELECT * FROM assessments
            WHERE assessment_id IN (
                SELECT MAX(assessment_id) FROM assessments GROUP BY student_id
            )
        ) a ON s.student_id = a.student_id
        LEFT JOIN class_memberships cm ON s.student_id = cm.student_id
        LEFT JOIN classes c ON cm.class_id = c.class_id
    """
    params = []

    if class_id:
        query += " WHERE c.class_id = ?"
        params.append(class_id)
    elif faculty_username:
        query += " WHERE c.faculty_username = ?"
        params.append(faculty_username)

    query += " ORDER BY s.student_id ASC;"

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    results = []
    for r in rows:
        sid = r["student_id"]
        name = r["name"]
        att = float(r["attendance"]) if r["attendance"] is not None else 75.0
        hrs = float(r["study_hours"]) if r["study_hours"] is not None else 10.0
        prev = float(r["previous_semester_percentage"]) if r["previous_semester_percentage"] is not None else 70.0
        ass_avg = float(r["assignment_average"]) if r["assignment_average"] is not None else 70.0
        int_ass = float(r["internal_assessment"]) if r["internal_assessment"] is not None else 70.0
        quiz_avg = float(r["quiz_average"]) if r["quiz_average"] is not None else 70.0
        comp = int(r["completed_assignments"]) if r["completed_assignments"] is not None else 10
        tot = int(r["total_assignments"]) if r["total_assignments"] is not None else 15
        backlog = int(r["backlog_count"]) if r["backlog_count"] is not None else 0

        input_data = {
            "attendance": att,
            "study_hours": hrs,
            "previous_semester_percentage": prev,
            "assignment_average": ass_avg,
            "internal_assessment": int_ass,
            "quiz_average": quiz_avg,
            "completed_assignments": comp,
            "total_assignments": tot,
            "backlog_count": backlog
        }

        # Run ML Prediction Service
        _, ml_res, _ = execute_ml_prediction(input_data)

        predicted_score = ml_res.get("predicted_score", r["final_score"] or 75.0)
        risk_level = ml_res.get("risk_category", "MEDIUM")

        item = {
            "studentId": sid,
            "name": name,
            "semester": r["semester"],
            "branch": r["branch"],
            "className": r["class_name"] or "General Cohort",
            "classId": r["class_id"],
            "predictedScore": round(float(predicted_score), 1),
            "riskLevel": risk_level,
            "attendance": att,
            "studyHours": hrs,
            "previousPercentage": prev,
            "assignmentAvg": ass_avg,
            "internalAssessment": int_ass,
            "quizAvg": quiz_avg,
            "completedAssignments": comp,
            "totalAssignments": tot,
            "backlogCount": backlog,
            "lastSubmitted": r["submitted_at"]
        }
        results.append(item)

    # 1. Apply Search Filter
    if search:
        s_query = search.strip().lower()
        results = [s for s in results if s_query in s["name"].lower() or s_query in s["studentId"].lower()]

    # 2. Apply Risk Category Filter
    if risk_filter and risk_filter.upper() in ["HIGH", "MEDIUM", "LOW"]:
        results = [s for s in results if s["riskLevel"] == risk_filter.upper()]

    # 3. Apply Sorting
    if sort_by == "risk":
        risk_rank = {"HIGH": 1, "MEDIUM": 2, "LOW": 3}
        results.sort(key=lambda s: (risk_rank.get(s["riskLevel"], 4), s["predictedScore"]))
    elif sort_by == "score_asc":
        results.sort(key=lambda s: s["predictedScore"])
    elif sort_by == "score_desc":
        results.sort(key=lambda s: -s["predictedScore"])
    elif sort_by == "attendance":
        results.sort(key=lambda s: s["attendance"])
    elif sort_by == "name":
        results.sort(key=lambda s: s["name"])

    return results

def add_student_to_class_manually(faculty_username, class_id, student_name, metrics):
    """
    Manually add a new student and their academic metrics to a faculty's class.
    """
    if not student_name or not metrics:
        return False, "Student name and academic metrics are required."

    conn = get_db_connection()
    cursor = conn.cursor()

    # Generate next student ID
    cursor.execute("SELECT COUNT(*) FROM students;")
    count = cursor.fetchone()[0] + 1
    student_id = f"STU-2026-{count:03d}"

    try:
        cursor.execute("""
            INSERT INTO students (student_id, name, semester, branch)
            VALUES (?, ?, 2, 'Computer Engineering');
        """, (student_id, student_name.strip()))

        att = float(metrics.get("attendance", 75.0))
        hrs = float(metrics.get("study_hours", 10.0))
        prev = float(metrics.get("previous_semester_percentage", 70.0))
        ass_avg = float(metrics.get("assignment_average", 70.0))
        int_ass = float(metrics.get("internal_assessment", 70.0))
        quiz_avg = float(metrics.get("quiz_average", 70.0))
        comp = int(metrics.get("completed_assignments", 10))
        tot = int(metrics.get("total_assignments", 15))
        backlog = int(metrics.get("backlog_count", 0))

        # Calculate initial score via ML
        input_data = {
            "attendance": att, "study_hours": hrs, "previous_semester_percentage": prev,
            "assignment_average": ass_avg, "internal_assessment": int_ass, "quiz_average": quiz_avg,
            "completed_assignments": comp, "total_assignments": tot, "backlog_count": backlog
        }
        _, ml_res, _ = execute_ml_prediction(input_data)
        final_score = ml_res.get("predicted_score", 75.0)

        cursor.execute("""
            INSERT INTO assessments (
                student_id, attendance, study_hours, previous_semester_percentage,
                assignment_average, internal_assessment, quiz_average,
                completed_assignments, total_assignments, backlog_count, final_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (student_id, att, hrs, prev, ass_avg, int_ass, quiz_avg, comp, tot, backlog, final_score))

        # Attach to Class
        if class_id:
            cursor.execute("""
                INSERT OR IGNORE INTO class_memberships (class_id, student_id)
                VALUES (?, ?);
            """, (class_id, student_id))

        conn.commit()
        conn.close()
        return True, f"Successfully added student '{student_name}' ({student_id}) to class."
    except Exception as e:
        conn.close()
        return False, f"Failed to add student: {str(e)}"

def get_faculty_student_detail(student_id):
    """Retrieve full student profile audit details for faculty."""
    profile = get_student_profile(student_id)
    if not profile:
        return None

    input_data = {
        "attendance": profile["attendance"],
        "study_hours": profile["studyHours"],
        "previous_semester_percentage": profile["previousPercentage"],
        "assignment_average": profile["assignmentAvg"],
        "internal_assessment": profile["internalAssessment"],
        "quiz_average": profile["quizAvg"],
        "completed_assignments": profile["completedAssignments"],
        "total_assignments": profile["totalAssignments"],
        "backlog_count": profile["backlogCount"]
    }

    _, ml_pred, _ = execute_ml_prediction(input_data)
    xai_explanation = explain_individual_prediction(input_data)
    recs = fetch_recommendations_for_input(input_data, student_subjects=profile.get("studentSubjects"))
    history = get_student_history(student_id)
    interventions = get_student_interventions(student_id)

    return {
        "studentId": profile["studentId"],
        "name": profile["name"],
        "semester": profile["semester"],
        "branch": profile["branch"],
        "metrics": input_data,
        "prediction": ml_pred,
        "explanation": xai_explanation,
        "recommendations": recs,
        "history": history,
        "interventions": interventions,
        "studentSubjects": profile.get("studentSubjects", [])
    }

def add_faculty_intervention(student_id, faculty_username, action_title, notes, status, follow_up_date):
    """Record a new faculty intervention action."""
    if not student_id or not action_title or not status:
        return False, "Student ID, action title, and status are required."

    if status not in ['Recommended', 'Assigned', 'In Progress', 'Completed']:
        status = 'Assigned'

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO interventions (student_id, faculty_username, action_title, notes, status, follow_up_date)
        VALUES (?, ?, ?, ?, ?, ?);
    """, (student_id, faculty_username, action_title.strip(), notes.strip() if notes else "", status, follow_up_date.strip() if follow_up_date else None))

    conn.commit()
    conn.close()
    return True, "Intervention recorded successfully."

def get_student_interventions(student_id):
    """Retrieve all recorded faculty interventions for a student."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM interventions
        WHERE student_id = ?
        ORDER BY created_at DESC;
    """, (student_id,))
    rows = cursor.fetchall()
    conn.close()

    return [dict(r) for r in rows]
