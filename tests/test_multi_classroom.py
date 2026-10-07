"""
EDUPREDICT - Multi-Classroom Management Test Suite
Tests multi-classroom creation, join code uniqueness, classroom switching & data isolation, status archiving, and student join restrictions for archived classrooms.
"""

import os
import sys
import pytest

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app import app
from backend.services.database import init_db, get_db_connection, hash_password
from backend.services.faculty_service import (
    create_faculty_class,
    get_faculty_classes,
    update_faculty_class_status,
    get_faculty_overview,
    get_faculty_student_list
)
from backend.services.student_service import join_class_by_code

@pytest.fixture(autouse=True)
def setup_database():
    """Initialize test database before tests and seed dummy test faculty accounts."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    for f in ["prof_test_multi", "prof_test_archive", "prof_test_isolation"]:
        cursor.execute("""
            INSERT OR IGNORE INTO users (username, password_hash, role, name, department)
            VALUES (?, ?, 'faculty', 'Test Faculty', 'Computer Science');
        """, (f, hash_password('faculty123')))
    conn.commit()
    conn.close()

@pytest.fixture
def client():
    """Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_faculty_can_create_multiple_classrooms():
    """Test faculty creating multiple classrooms with distinct details and unique join codes."""
    fac = "prof_test_multi"

    # Create Classroom 1
    ok1, class1, msg1 = create_faculty_class(fac, "Data Structures - Section A", course="B.Tech CS", semester=3, section="A")
    assert ok1 is True
    assert class1["classCode"].startswith("EDU-")
    assert class1["className"] == "Data Structures - Section A"
    assert class1["status"] == "ACTIVE"

    # Create Classroom 2
    ok2, class2, msg2 = create_faculty_class(fac, "Data Structures - Section B", course="B.Tech CS", semester=3, section="B")
    assert ok2 is True
    assert class2["classCode"].startswith("EDU-")
    assert class2["classCode"] != class1["classCode"]

    # Verify both classrooms exist for faculty
    classes = get_faculty_classes(fac)
    class_codes = [c["class_code"] for c in classes]
    assert class1["classCode"] in class_codes
    assert class2["classCode"] in class_codes

def test_classroom_status_archiving_and_student_join_restriction():
    """Test archiving a classroom and verifying students cannot join archived classrooms."""
    fac = "prof_test_archive"
    ok, cls, _ = create_faculty_class(fac, "Physics Lab - Section X")
    assert ok is True
    class_id = cls["classId"]
    code = cls["classCode"]

    # Archive Classroom
    upd_ok, upd_msg = update_faculty_class_status(class_id, fac, "ARCHIVED")
    assert upd_ok is True

    # Verify status is ARCHIVED in database
    classes = get_faculty_classes(fac)
    target = next((c for c in classes if c["class_id"] == class_id), None)
    assert target is not None
    assert target["status"] == "ARCHIVED"

    # Attempt student join on ARCHIVED classroom
    join_ok, join_msg = join_class_by_code("STU-2026-001", code)
    assert join_ok is False
    assert "archived" in join_msg.lower()

    # Unarchive classroom and verify student CAN join
    upd_ok2, _ = update_faculty_class_status(class_id, fac, "ACTIVE")
    assert upd_ok2 is True

    join_ok2, join_msg2 = join_class_by_code("STU-2026-001", code)
    assert join_ok2 is True
    assert "Successfully joined" in join_msg2

def test_classroom_data_isolation_overview_and_roster():
    """Test that overview metrics and student roster are properly isolated by class_id."""
    fac = "prof_test_isolation"
    
    # Create two classrooms
    ok1, c1, _ = create_faculty_class(fac, "Class 1", course="B.Tech", semester=2, section="A")
    assert ok1 is True
    ok2, c2, _ = create_faculty_class(fac, "Class 2", course="B.Tech", semester=2, section="B")
    assert ok2 is True

    c1_id = c1["classId"]
    c2_id = c2["classId"]

    # Enroll STU-2026-001 in C1, STU-2026-006 in C2
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO class_memberships (class_id, student_id) VALUES (?, ?);", (c1_id, "STU-2026-001"))
    cursor.execute("INSERT OR IGNORE INTO class_memberships (class_id, student_id) VALUES (?, ?);", (c2_id, "STU-2026-006"))
    conn.commit()
    conn.close()

    # Roster for Class 1 should only contain STU-2026-001
    roster1 = get_faculty_student_list(faculty_username=fac, class_id=c1_id)
    r1_sids = [s["studentId"] for s in roster1]
    assert "STU-2026-001" in r1_sids
    assert "STU-2026-006" not in r1_sids

    # Roster for Class 2 should only contain STU-2026-006
    roster2 = get_faculty_student_list(faculty_username=fac, class_id=c2_id)
    r2_sids = [s["studentId"] for s in roster2]
    assert "STU-2026-006" in r2_sids
    assert "STU-2026-001" not in r2_sids

    # Overview KPI for Class 1 should report totalStudents == 1
    ov1 = get_faculty_overview(faculty_username=fac, class_id=c1_id)
    assert ov1["totalStudents"] == 1

def test_api_patch_class_status_route(client):
    """Test API endpoint PATCH /api/faculty/classes/<class_id>/status."""
    fac = "faculty"
    ok, cls, _ = create_faculty_class(fac, "API Test Class")
    class_id = cls["classId"]

    # Call API to archive
    res = client.patch(
        f"/api/faculty/classes/{class_id}/status",
        headers={"X-User-Role": "faculty", "X-User-Username": fac},
        json={"status": "ARCHIVED"}
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["status"] == "ARCHIVED"

    # Call API to activate again
    res2 = client.patch(
        f"/api/faculty/classes/{class_id}/status",
        headers={"X-User-Role": "faculty", "X-User-Username": fac},
        json={"status": "ACTIVE"}
    )
    assert res2.status_code == 200
    data2 = res2.get_json()
    assert data2["success"] is True
    assert data2["status"] == "ACTIVE"

def test_api_get_classroom_subjects(client):
    """Test API endpoint GET /api/faculty/classes/<class_id>/subjects."""
    # Class 1 (seeded EDU-A7K92P) has class_id=1
    res = client.get(
        "/api/faculty/classes/1/subjects",
        headers={"X-User-Role": "faculty", "X-User-Username": "faculty"}
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert isinstance(data["subjects"], list)
    assert len(data["subjects"]) >= 1
