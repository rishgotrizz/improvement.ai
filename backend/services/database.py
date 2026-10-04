"""
EDUPREDICT - SQLite Database Service Manager
Handles SQLite schema creation, database connection, and seeded demo student initialization.
"""

import os
import sqlite3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../data/edupredict.db"))

def get_db_connection():
    """Create and return a new SQLite database connection with row factory."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # Enable foreign keys
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    """Initialize database tables and seed demo student records cleanly."""
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

    # 2. Assessments Table
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

    # 3. Subject Performance Table
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

    conn.commit()

    # Seed Demo Student STU-2026-001 if missing
    cursor.execute("SELECT student_id FROM students WHERE student_id = 'STU-2026-001';")
    if not cursor.fetchone():
        _seed_demo_student(cursor)

    conn.commit()
    conn.close()
    print(f"✅ [SQLite DB] Database initialized successfully at {DB_PATH}")

def _seed_demo_student(cursor):
    """Seed STU-2026-001 demo record with initial academic assessment & course performance."""
    cursor.execute("""
        INSERT INTO students (student_id, name, semester, branch)
        VALUES ('STU-2026-001', 'Demo Student (First Year)', 2, 'Computer Engineering');
    """)

    cursor.execute("""
        INSERT INTO assessments (
            student_id, attendance, study_hours, previous_semester_percentage,
            assignment_average, internal_assessment, quiz_average,
            completed_assignments, total_assignments, backlog_count, final_score
        ) VALUES (
            'STU-2026-001', 87.0, 18.0, 79.0,
            82.5, 76.0, 80.0,
            14, 15, 0, 78.4
        );
    """)
    assessment_id = cursor.lastrowid

    # Seed Student Academic Performance Subjects
    demo_subjects = [
        ("CS102", "Data Structures", 74.0, 80.0, 75.0),
        ("MATH101", "Mathematics", 72.0, 85.0, 78.0),
        ("PHY101", "Physics", 81.0, 88.0, 85.0),
        ("CS101", "Computer Programming", 88.0, 92.0, 90.0),
        ("EE101", "Basic Electrical Engineering", 84.0, 90.0, 88.0)
    ]

    for code, name, marks, att, ass in demo_subjects:
        cursor.execute("""
            INSERT INTO subject_performance (
                assessment_id, student_id, subject_code, subject_name, marks, attendance, assignment_score
            ) VALUES (?, 'STU-2026-001', ?, ?, ?, ?, ?);
        """, (assessment_id, code, name, marks, att, ass))

    print("🌱 [SQLite DB] Seeded demo student 'STU-2026-001' with academic records.")
