"""
EDUPREDICT - SQLite Database Service Manager
Handles SQLite schema creation, database connection, user auth tables, classes, subjects, attendance, assessments, quizzes with topic tagging, attempts, topic analysis, interventions, and seeded student cohort.
"""

import os
import sqlite3
import hashlib
import random
import string
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../data/edupredict.db"))

def hash_password(password: str) -> str:
    """Hash password using SHA-256 for secure authentication."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def generate_class_code(prefix="EDU-"):
    """Generate unique 6-character uppercase non-predictable class join code."""
    chars = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    return f"{prefix}{chars}"

def get_db_connection():
    """Create and return a new SQLite database connection with row factory."""
    global DB_PATH
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn
    except (sqlite3.OperationalError, OSError, PermissionError):
        tmp_db = "/tmp/edupredict.db"
        if os.path.exists(DB_PATH) and not os.path.exists(tmp_db):
            try:
                import shutil
                shutil.copy2(DB_PATH, tmp_db)
            except Exception:
                pass
        DB_PATH = tmp_db
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

def init_db():
    """Initialize database tables and seed demo student cohort, classes, subjects, quizzes, and user accounts cleanly."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Students Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            student_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            semester INTEGER NOT NULL DEFAULT 2,
            branch TEXT NOT NULL DEFAULT 'Computer Engineering',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # 2. Assessments Table (Historical/ML Prediction Snapshots)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS assessments (
            assessment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            attendance REAL NOT NULL,
            study_hours REAL NOT NULL,
            previous_semester_percentage REAL NOT NULL,
            assignment_average REAL NOT NULL,
            internal_assessment REAL NOT NULL,
            quiz_average REAL NOT NULL,
            completed_assignments INTEGER NOT NULL,
            total_assignments INTEGER NOT NULL DEFAULT 15,
            backlog_count INTEGER NOT NULL DEFAULT 0,
            final_score REAL,
            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students (student_id) ON DELETE CASCADE
        );
    """)

    # 3. Subject Performance Table (Snapshot marks)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS subject_performance (
            record_id INTEGER PRIMARY KEY AUTOINCREMENT,
            assessment_id INTEGER NOT NULL,
            student_id TEXT NOT NULL,
            subject_code TEXT NOT NULL,
            subject_name TEXT NOT NULL,
            marks REAL NOT NULL,
            attendance REAL NOT NULL,
            assignment_score REAL NOT NULL,
            FOREIGN KEY (assessment_id) REFERENCES assessments (assessment_id) ON DELETE CASCADE,
            FOREIGN KEY (student_id) REFERENCES students (student_id) ON DELETE CASCADE
        );
    """)

    # 4. Users Table (Role-based Authentication)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('student', 'faculty')),
            name TEXT NOT NULL,
            student_id TEXT,
            department TEXT,
            roll_number TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students (student_id) ON DELETE SET NULL
        );
    """)

    # 5. Classes Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS classes (
            class_id INTEGER PRIMARY KEY AUTOINCREMENT,
            class_name TEXT NOT NULL,
            class_code TEXT UNIQUE NOT NULL,
            faculty_username TEXT NOT NULL,
            course TEXT DEFAULT 'B.Tech Engineering',
            semester INTEGER DEFAULT 2,
            section TEXT DEFAULT 'A',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (faculty_username) REFERENCES users (username) ON DELETE CASCADE
        );
    """)

    # 6. Class Memberships Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS class_memberships (
            membership_id INTEGER PRIMARY KEY AUTOINCREMENT,
            class_id INTEGER NOT NULL,
            student_id TEXT NOT NULL,
            joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (class_id) REFERENCES classes (class_id) ON DELETE CASCADE,
            FOREIGN KEY (student_id) REFERENCES students (student_id) ON DELETE CASCADE,
            UNIQUE(class_id, student_id)
        );
    """)

    # 7. Dynamic Subjects Table (Controlled by Faculty per Class or Independent Student)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS subjects (
            subject_id INTEGER PRIMARY KEY AUTOINCREMENT,
            class_id INTEGER,
            student_id TEXT,
            subject_code TEXT NOT NULL,
            subject_name TEXT NOT NULL,
            faculty_username TEXT,
            faculty_name TEXT,
            semester INTEGER DEFAULT 2,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (class_id) REFERENCES classes (class_id) ON DELETE CASCADE,
            FOREIGN KEY (student_id) REFERENCES students (student_id) ON DELETE CASCADE
        );
    """)

    # 8. Faculty Attendance Records
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance_records (
            attendance_id INTEGER PRIMARY KEY AUTOINCREMENT,
            class_id INTEGER NOT NULL,
            student_id TEXT NOT NULL,
            subject_code TEXT NOT NULL,
            attendance_percentage REAL NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (class_id) REFERENCES classes (class_id) ON DELETE CASCADE,
            FOREIGN KEY (student_id) REFERENCES students (student_id) ON DELETE CASCADE,
            UNIQUE(class_id, student_id, subject_code)
        );
    """)

    # 9. Faculty Classroom Assessments
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS classroom_assessments (
            assessment_item_id INTEGER PRIMARY KEY AUTOINCREMENT,
            class_id INTEGER NOT NULL,
            subject_code TEXT NOT NULL,
            title TEXT NOT NULL,
            max_marks REAL NOT NULL DEFAULT 30,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (class_id) REFERENCES classes (class_id) ON DELETE CASCADE
        );
    """)

    # 10. Assessment Marks
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS assessment_marks (
            mark_id INTEGER PRIMARY KEY AUTOINCREMENT,
            assessment_item_id INTEGER NOT NULL,
            student_id TEXT NOT NULL,
            obtained_marks REAL NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (assessment_item_id) REFERENCES classroom_assessments (assessment_item_id) ON DELETE CASCADE,
            FOREIGN KEY (student_id) REFERENCES students (student_id) ON DELETE CASCADE,
            UNIQUE(assessment_item_id, student_id)
        );
    """)

    # 11. MCQ Quizzes
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS quizzes (
            quiz_id INTEGER PRIMARY KEY AUTOINCREMENT,
            class_id INTEGER NOT NULL,
            subject_code TEXT NOT NULL,
            title TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (class_id) REFERENCES classes (class_id) ON DELETE CASCADE
        );
    """)

    # 12. MCQ Questions (With Topic Tagging)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS quiz_questions (
            question_id INTEGER PRIMARY KEY AUTOINCREMENT,
            quiz_id INTEGER NOT NULL,
            question_text TEXT NOT NULL,
            option_a TEXT NOT NULL,
            option_b TEXT NOT NULL,
            option_c TEXT NOT NULL,
            option_d TEXT NOT NULL,
            correct_option TEXT NOT NULL,
            topic TEXT NOT NULL,
            FOREIGN KEY (quiz_id) REFERENCES quizzes (quiz_id) ON DELETE CASCADE
        );
    """)

    # 13. Quiz Attempts & Topic Analytics
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS quiz_attempts (
            attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
            quiz_id INTEGER NOT NULL,
            student_id TEXT NOT NULL,
            score REAL NOT NULL,
            total_questions INTEGER NOT NULL,
            topic_breakdown TEXT,
            attempted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (quiz_id) REFERENCES quizzes (quiz_id) ON DELETE CASCADE,
            FOREIGN KEY (student_id) REFERENCES students (student_id) ON DELETE CASCADE
        );
    """)

    # 14. Interventions Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interventions (
            intervention_id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            faculty_username TEXT NOT NULL,
            action_title TEXT NOT NULL,
            notes TEXT,
            status TEXT NOT NULL CHECK(status IN ('Recommended', 'Assigned', 'In Progress', 'Completed')),
            follow_up_date TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students (student_id) ON DELETE CASCADE
        );
    """)

    conn.commit()

    # Seed Default Data if quizzes table is empty
    cursor.execute("SELECT COUNT(*) FROM quizzes;")
    if cursor.fetchone()[0] == 0:
        _seed_demo_data(cursor)

    conn.commit()
    conn.close()
    print(f"✅ [SQLite DB] Database initialized successfully with Quizzes, Topics, and Data Ownership schema at {DB_PATH}")

def _seed_demo_data(cursor):
    """Seed demo student cohort, classes, dynamic subjects, quizzes with topic tagging, multi-faculty ownership, and historical demo snapshots."""
    students_data = [
        ("STU-2026-001", "Aarav Sharma", 2, "Computer Engineering", 87.0, 18.0, 79.0, 82.5, 76.0, 80.0, 14, 15, 0, 78.4),
        ("STU-2026-002", "Ananya Verma", 2, "Computer Engineering", 48.0, 4.5, 54.0, 52.0, 49.0, 50.0, 5, 15, 3, 49.2),
        ("STU-2026-003", "Rohan Gupta", 2, "Computer Engineering", 64.0, 8.0, 68.0, 62.0, 60.0, 58.0, 9, 15, 1, 63.8),
        ("STU-2026-004", "Priya Patel", 2, "Computer Engineering", 94.0, 22.0, 88.0, 90.0, 86.0, 92.0, 15, 15, 0, 89.6),
        ("STU-2026-005", "Kabir Mehta", 2, "Computer Engineering", 52.0, 5.0, 58.0, 45.0, 50.0, 48.0, 6, 15, 2, 51.5),
        ("STU-2026-006", "Sanya Malhotra", 2, "Computer Engineering", 72.0, 12.0, 74.0, 75.0, 70.0, 73.0, 11, 15, 0, 71.8),
        ("STU-2026-007", "Devansh Joshi", 2, "Computer Engineering", 42.0, 3.0, 49.0, 40.0, 45.0, 42.0, 4, 15, 4, 43.1),
        ("STU-2026-008", "Isha Nair", 2, "Computer Engineering", 89.0, 16.0, 82.0, 85.0, 80.0, 84.0, 13, 15, 0, 81.9),
        ("STU-2026-009", "Aditya Rao", 2, "Computer Engineering", 68.0, 9.5, 65.0, 67.0, 64.0, 66.0, 10, 15, 1, 66.4),
        ("STU-2026-010", "Neha Kapoor", 2, "Computer Engineering", 96.0, 24.0, 91.0, 94.0, 92.0, 95.0, 15, 15, 0, 93.5)
    ]

    for sid, name, sem, branch, att, hrs, prev, ass_avg, int_ass, quiz_avg, comp, tot, back, score in students_data:
        cursor.execute("INSERT OR IGNORE INTO students (student_id, name, semester, branch) VALUES (?, ?, ?, ?);", (sid, name, sem, branch))
        
        # For STU-2026-001 (Demo Student), seed 5 deterministic historical snapshots for the demo graphs
        if sid == "STU-2026-001":
            demo_snapshots = [
                ("STU-2026-001", 68.0, 10.0, 65.0, 60.0, 55.0, 58.0, 8, 15, 2, 56.0, "2026-06-15 10:00:00"),
                ("STU-2026-001", 74.0, 12.0, 68.0, 68.0, 63.0, 65.0, 10, 15, 1, 62.0, "2026-07-15 10:00:00"),
                ("STU-2026-001", 80.0, 14.0, 72.0, 74.0, 70.0, 71.0, 12, 15, 1, 69.0, "2026-08-15 10:00:00"),
                ("STU-2026-001", 84.0, 16.0, 76.0, 80.0, 74.0, 78.0, 13, 15, 0, 75.0, "2026-09-15 10:00:00"),
                ("STU-2026-001", att, hrs, prev, ass_avg, int_ass, quiz_avg, comp, tot, back, score, "2026-10-06 10:00:00")
            ]
            for h_sid, h_att, h_hrs, h_prev, h_ass, h_int, h_qz, h_comp, h_tot, h_back, h_score, h_date in demo_snapshots:
                cursor.execute("""
                    INSERT INTO assessments (
                        student_id, attendance, study_hours, previous_semester_percentage,
                        assignment_average, internal_assessment, quiz_average,
                        completed_assignments, total_assignments, backlog_count, final_score, submitted_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (h_sid, h_att, h_hrs, h_prev, h_ass, h_int, h_qz, h_comp, h_tot, h_back, h_score, h_date))
                aid = cursor.lastrowid

                demo_subjects = [
                    ("CS102", "Data Structures", round(h_score - 3, 1), round(h_att - 2, 1), round(h_ass - 2, 1)),
                    ("MAT101", "Engineering Mathematics", round(h_score - 5, 1), round(h_att + 1, 1), round(h_ass, 1)),
                    ("PHY101", "Engineering Physics", round(h_score + 2, 1), round(h_att, 1), round(h_ass + 3, 1)),
                    ("CS101", "Programming Fundamentals", round(h_score + 4, 1), round(h_att + 3, 1), round(h_ass + 4, 1))
                ]
                for code, sub_name, marks, sub_att, ass_score in demo_subjects:
                    cursor.execute("""
                        INSERT INTO subject_performance (
                            assessment_id, student_id, subject_code, subject_name, marks, attendance, assignment_score
                        ) VALUES (?, ?, ?, ?, ?, ?, ?);
                    """, (aid, sid, code, sub_name, max(10.0, min(100.0, marks)), max(10.0, min(100.0, sub_att)), max(10.0, min(100.0, ass_score))))
        else:
            cursor.execute("""
                INSERT INTO assessments (
                    student_id, attendance, study_hours, previous_semester_percentage,
                    assignment_average, internal_assessment, quiz_average,
                    completed_assignments, total_assignments, backlog_count, final_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (sid, att, hrs, prev, ass_avg, int_ass, quiz_avg, comp, tot, back, score))
            aid = cursor.lastrowid

            demo_subjects = [
                ("CS102", "Data Structures", round(score - 3, 1), round(att - 2, 1), round(ass_avg - 2, 1)),
                ("MAT101", "Engineering Mathematics", round(score - 5, 1), round(att + 1, 1), round(ass_avg, 1)),
                ("PHY101", "Engineering Physics", round(score + 2, 1), round(att, 1), round(ass_avg + 3, 1)),
                ("CS101", "Programming Fundamentals", round(score + 4, 1), round(att + 3, 1), round(ass_avg + 4, 1))
            ]
            for code, sub_name, marks, sub_att, ass_score in demo_subjects:
                cursor.execute("""
                    INSERT INTO subject_performance (
                        assessment_id, student_id, subject_code, subject_name, marks, attendance, assignment_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?);
                """, (aid, sid, code, sub_name, max(10.0, min(100.0, marks)), max(10.0, min(100.0, sub_att)), max(10.0, min(100.0, ass_score))))

    # Seed Accounts
    cursor.execute("""
        INSERT OR IGNORE INTO users (username, password_hash, role, name, student_id, roll_number)
        VALUES ('student', ?, 'student', 'Aarav Sharma (Student)', 'STU-2026-001', 'CS2026-001');
    """, (hash_password('student123'),))

    cursor.execute("""
        INSERT OR IGNORE INTO users (username, password_hash, role, name, department)
        VALUES ('faculty', ?, 'faculty', 'Dr. Sarah Jenkins (Academic Advisor)', 'Computer Engineering');
    """, (hash_password('faculty123'),))

    cursor.execute("""
        INSERT OR IGNORE INTO users (username, password_hash, role, name, department)
        VALUES ('prof_math', ?, 'faculty', 'Prof. Ananya Roy (Mathematics)', 'Mathematics');
    """, (hash_password('faculty123'),))

    cursor.execute("""
        INSERT OR IGNORE INTO users (username, password_hash, role, name, department)
        VALUES ('prof_cs', ?, 'faculty', 'Prof. Rajesh Verma (Computer Science)', 'Computer Science');
    """, (hash_password('faculty123'),))

    cursor.execute("""
        INSERT OR IGNORE INTO users (username, password_hash, role, name, department)
        VALUES ('prof_phy', ?, 'faculty', 'Prof. Sunita Sharma (Physics)', 'Physics');
    """, (hash_password('faculty123'),))

    # Seed Default Classes
    cursor.execute("""
        INSERT OR IGNORE INTO classes (class_name, class_code, faculty_username, course, semester, section)
        VALUES ('FY Computer Engineering — Section A', 'EDU-A7K92P', 'faculty', 'B.Tech Computer Science', 2, 'A');
    """)
    class_a_id = cursor.lastrowid or 1

    cursor.execute("""
        INSERT OR IGNORE INTO classes (class_name, class_code, faculty_username, course, semester, section)
        VALUES ('FY Computer Engineering — Section B', 'EDU-B3M8X1', 'faculty', 'B.Tech Computer Science', 2, 'B');
    """)
    class_b_id = cursor.lastrowid or 2

    # Enroll Students in Classes
    for sid in ["STU-2026-001", "STU-2026-002", "STU-2026-003", "STU-2026-004", "STU-2026-005"]:
        cursor.execute("INSERT OR IGNORE INTO class_memberships (class_id, student_id) VALUES (?, ?);", (class_a_id, sid))

    for sid in ["STU-2026-006", "STU-2026-007", "STU-2026-008", "STU-2026-009", "STU-2026-010"]:
        cursor.execute("INSERT OR IGNORE INTO class_memberships (class_id, student_id) VALUES (?, ?);", (class_b_id, sid))

    # Seed Dynamic Classroom Subjects with Assigned Subject Faculty for Class A
    faculty_subjects = [
        ("MAT101", "Engineering Mathematics", "prof_math", "Prof. Ananya Roy"),
        ("CS101", "Programming Fundamentals", "prof_cs", "Prof. Rajesh Verma"),
        ("PHY101", "Engineering Physics", "prof_phy", "Prof. Sunita Sharma"),
        ("CS102", "Data Structures", "faculty", "Dr. Sarah Jenkins")
    ]
    for code, name, f_user, f_name in faculty_subjects:
        cursor.execute("INSERT OR IGNORE INTO subjects (class_id, subject_code, subject_name, faculty_username, faculty_name) VALUES (?, ?, ?, ?, ?);", (class_a_id, code, name, f_user, f_name))

    # Seed Demo MCQ Quiz with Topic Tagging
    cursor.execute("""
        INSERT INTO quizzes (class_id, subject_code, title)
        VALUES (?, 'MAT101', 'Matrices & Calculus Evaluation Quiz 1');
    """, (class_a_id,))
    quiz_id = cursor.lastrowid

    quiz_questions_data = [
        ("What is the determinant of a 2x2 identity matrix?", "0", "1", "2", "-1", "B", "Matrices"),
        ("If A is an orthogonal matrix, then det(A) equals:", "0", "±1", "2", "Undefined", "B", "Matrices"),
        ("What is the derivative of sin(x) with respect to x?", "cos(x)", "-cos(x)", "tan(x)", "sec^2(x)", "A", "Differentiation"),
        ("Find d/dx of x^3.", "3x^2", "x^2", "2x^3", "3x", "A", "Differentiation"),
        ("What is the indefinite integral ∫ x dx?", "(1/2)x^2 + C", "x^2 + C", "2x + C", "ln(x) + C", "A", "Integration"),
        ("What is ∫ cos(x) dx?", "sin(x) + C", "-sin(x) + C", "tan(x) + C", "cos(x) + C", "A", "Integration")
    ]

    for q_text, opt_a, opt_b, opt_c, opt_d, corr, topic in quiz_questions_data:
        cursor.execute("""
            INSERT INTO quiz_questions (quiz_id, question_text, option_a, option_b, option_c, option_d, correct_option, topic)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """, (quiz_id, q_text, opt_a, opt_b, opt_c, opt_d, corr, topic))

    # Seed Demo Attempt for STU-2026-001 (Aarav Sharma) showing Integration weakness
    topic_breakdown_demo = json.dumps({
        "Matrices": {"correct": 2, "total": 2, "percentage": 100.0},
        "Differentiation": {"correct": 2, "total": 2, "percentage": 100.0},
        "Integration": {"correct": 0, "total": 2, "percentage": 0.0}
    })
    cursor.execute("""
        INSERT INTO quiz_attempts (quiz_id, student_id, score, total_questions, topic_breakdown)
        VALUES (?, 'STU-2026-001', 4, 6, ?);
    """, (quiz_id, topic_breakdown_demo))

    # Seed Sample Faculty Intervention
    cursor.execute("""
        INSERT INTO interventions (student_id, faculty_username, action_title, notes, status, follow_up_date)
        VALUES ('STU-2026-002', 'faculty', 'Mandatory Attendance & Integration Remedial Session', 'Student attendance is 48% and Integration quiz score is 0%. Schedule urgent meeting.', 'In Progress', '2026-10-18');
    """)

    print("🌱 [SQLite DB] Seeded demo cohort, classes (EDU-A7K92P), MCQ quizzes with topics, attempts, and topic analytics.")
