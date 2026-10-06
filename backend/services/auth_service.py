"""
EDUPREDICT - Authentication & Registration Service Manager
Handles user login, student/faculty registration, role verification, and credential validation.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from backend.services.database import get_db_connection, hash_password
except ImportError:
    from services.database import get_db_connection, hash_password

def register_user(username, password, role, name, student_id=None, department=None, roll_number=None):
    """
    Register a new user account (Student or Faculty).
    Returns (success: bool, user_data: dict, error_message: str).
    """
    if not username or not password or not role or not name:
        return False, None, "Username, password, role, and name are required."

    role = role.strip().lower()
    if role not in ["student", "faculty"]:
        return False, None, "Role must be 'student' or 'faculty'."

    username = username.strip().lower()
    conn = get_db_connection()
    cursor = conn.cursor()

    # Check if username exists
    cursor.execute("SELECT user_id FROM users WHERE username = ?;", (username,))
    if cursor.fetchone():
        conn.close()
        return False, None, f"Username '{username}' is already registered. Please login or choose another username."

    # For student role, generate or use student_id
    assigned_student_id = None
    if role == "student":
        if student_id and student_id.strip():
            assigned_student_id = student_id.strip().upper()
        else:
            cursor.execute("SELECT COUNT(*) FROM students;")
            count = cursor.fetchone()[0] + 1
            assigned_student_id = f"STU-2026-{count:03d}"

        # Ensure student record exists in students table
        cursor.execute("INSERT OR IGNORE INTO students (student_id, name, semester, branch) VALUES (?, ?, 2, 'Computer Engineering');", (assigned_student_id, name.strip()))

        # Seed initial assessment so student has working baseline ML metrics
        cursor.execute("SELECT COUNT(*) FROM assessments WHERE student_id = ?;", (assigned_student_id,))
        if cursor.fetchone()[0] == 0:
            cursor.execute("""
                INSERT INTO assessments (
                    student_id, attendance, study_hours, previous_semester_percentage,
                    assignment_average, internal_assessment, quiz_average,
                    completed_assignments, total_assignments, backlog_count, final_score
                ) VALUES (?, 75.0, 10.0, 70.0, 70.0, 70.0, 70.0, 10, 15, 0, 72.5);
            """, (assigned_student_id,))

    pwd_hash = hash_password(password.strip())
    try:
        cursor.execute("""
            INSERT INTO users (username, password_hash, role, name, student_id, department, roll_number)
            VALUES (?, ?, ?, ?, ?, ?, ?);
        """, (username, pwd_hash, role, name.strip(), assigned_student_id, department.strip() if department else None, roll_number.strip() if roll_number else None))
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()

        user_dict = {
            "userId": user_id,
            "username": username,
            "name": name.strip(),
            "role": role,
            "studentId": assigned_student_id
        }
        return True, user_dict, None
    except Exception as e:
        conn.close()
        return False, None, f"Registration failed: {str(e)}"

def authenticate_user(username, password):
    """
    Authenticate username and password against database.
    Returns (success: bool, user_data: dict, error_message: str).
    """
    if not username or not password:
        return False, None, "Username and password are required."

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users WHERE username = ?;", (username.strip().lower(),))
    user = cursor.fetchone()
    conn.close()

    if not user:
        return False, None, "Invalid username or password."

    pwd_hash = hash_password(password.strip())
    if user["password_hash"] != pwd_hash:
        return False, None, "Invalid username or password."

    user_dict = {
        "userId": user["user_id"],
        "username": user["username"],
        "name": user["name"],
        "role": user["role"],
        "studentId": user["student_id"]
    }
    return True, user_dict, None

def get_user_by_username(username):
    """Retrieve user object by username."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT user_id, username, role, name, student_id FROM users WHERE username = ?;", (username.strip().lower(),))
    user = cursor.fetchone()
    conn.close()

    if not user:
        return None

    return {
        "userId": user["user_id"],
        "username": user["username"],
        "name": user["name"],
        "role": user["role"],
        "studentId": user["student_id"]
    }
