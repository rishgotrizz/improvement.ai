"""
EDUPREDICT - Authentication & Registration API Blueprint
Routes for user registration (Student & Faculty), login, logout, and active session check.
"""

import os
import sys
from flask import Blueprint, request, jsonify, session

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from backend.services.auth_service import authenticate_user, register_user, get_user_by_username
except ImportError:
    from services.auth_service import authenticate_user, register_user, get_user_by_username

auth_bp = Blueprint("auth_bp", __name__)

@auth_bp.route("/api/auth/register", methods=["POST"])
def register():
    """Register a new student or faculty user account."""
    data = request.get_json(silent=True) or {}
    username = data.get("username")
    password = data.get("password")
    role = data.get("role")
    name = data.get("name")
    student_id = data.get("studentId")
    department = data.get("department")
    roll_number = data.get("rollNumber")

    success, user, error_msg = register_user(
        username=username,
        password=password,
        role=role,
        name=name,
        student_id=student_id,
        department=department,
        roll_number=roll_number
    )

    if not success:
        return jsonify({"success": False, "error": error_msg}), 400

    # Auto-login session on registration
    session["username"] = user["username"]
    session["role"] = user["role"]
    session["name"] = user["name"]
    session["studentId"] = user["studentId"]

    return jsonify({
        "success": True,
        "user": user,
        "message": f"Account created successfully for {user['name']}!"
    }), 201

@auth_bp.route("/api/auth/login", methods=["POST"])
def login():
    """Authenticate user credentials and initiate session."""
    data = request.get_json(silent=True) or {}
    username = data.get("username")
    password = data.get("password")

    success, user, error_msg = authenticate_user(username, password)
    if not success:
        return jsonify({"success": False, "error": error_msg}), 401

    session["username"] = user["username"]
    session["role"] = user["role"]
    session["name"] = user["name"]
    session["studentId"] = user["studentId"]

    return jsonify({
        "success": True,
        "user": user,
        "message": f"Welcome back, {user['name']}!"
    }), 200

@auth_bp.route("/api/auth/logout", methods=["POST"])
def logout():
    """Clear active user session."""
    session.clear()
    return jsonify({"success": True, "message": "Logged out successfully."}), 200

@auth_bp.route("/api/auth/me", methods=["GET"])
def get_current_user():
    """Retrieve active session user information."""
    username = session.get("username")
    if not username:
        return jsonify({"authenticated": False, "user": None}), 200

    user = get_user_by_username(username)
    if not user:
        session.clear()
        return jsonify({"authenticated": False, "user": None}), 200

    return jsonify({"authenticated": True, "user": user}), 200
