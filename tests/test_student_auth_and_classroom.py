"""
EDUPREDICT - Student Auth & Classroom Integration Tests
Tests student registration, login, joining class by code, and ensuring faculty endpoints are removed.
"""

import os
import sys
import pytest

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app import app
from backend.services.database import init_db, get_db_connection

@pytest.fixture(scope="module")
def client():
    app.config["TESTING"] = True
    init_db()
    with app.test_client() as test_client:
        yield test_client

def test_health_check(client):
    """Test health check API is alive and responsive."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "healthy"
    assert "version" in data

def test_student_login(client):
    """Test demo student login with student / student123."""
    res = client.post("/api/auth/login", json={
        "username": "student",
        "password": "student123"
    })
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["user"]["role"] == "student"
    assert data["user"]["username"] == "student"

import uuid

def test_student_registration(client):
    """Test registering a new student account."""
    unique_user = f"stu_{uuid.uuid4().hex[:8]}"
    res = client.post("/api/auth/register", json={
        "username": unique_user,
        "password": "password123",
        "name": "Test Student User",
        "rollNumber": "TEST-2026-99"
    })
    assert res.status_code == 201
    data = res.get_json()
    assert data["success"] is True
    assert data["user"]["username"] == unique_user
    assert data["user"]["role"] == "student"

def test_student_join_class(client):
    """Test joining a class using demo join code EDU-A7K92P."""
    # Authenticate as student
    with client.session_transaction() as sess:
        sess["username"] = "student"
        sess["role"] = "student"
        sess["studentId"] = "STU-2026-001"

    res = client.post("/api/student/join-class", json={
        "classCode": "EDU-A7K92P"
    })
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True

def test_faculty_endpoints_removed(client):
    """Verify all faculty endpoints return 404 Not Found."""
    res1 = client.get("/api/faculty/classes")
    assert res1.status_code == 404

    res2 = client.get("/api/faculty/overview")
    assert res2.status_code == 404

    res3 = client.get("/api/faculty/students")
    assert res3.status_code == 404

    res4 = client.get("/faculty")
    assert res4.status_code == 404

    res5 = client.get("/login/faculty")
    assert res5.status_code == 404

    res6 = client.get("/register/faculty")
    assert res6.status_code == 404
