"""
EDUPREDICT - Flask Backend API Server
First-Year Engineering & AI Academic Project
Phase 3: Real EDA & Statistical Analysis Pipeline Integration
"""

import os
import sys

# Ensure project root directory is at position 0 of sys.path for robust package imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

candidate_roots = [
    os.path.abspath(os.path.join(BASE_DIR, "..")),
    BASE_DIR,
    os.getcwd(),
    "/var/task"
]

PROJECT_ROOT = None
for cand in candidate_roots:
    if os.path.exists(os.path.join(cand, "backend")) or os.path.exists(os.path.join(cand, "frontend")):
        PROJECT_ROOT = cand
        break

if not PROJECT_ROOT:
    PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

for p in [PROJECT_ROOT, BASE_DIR, os.path.join(PROJECT_ROOT, "backend"), "/var/task"]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

import types
if "backend" not in sys.modules:
    try:
        import backend
    except ImportError:
        target_dir = None
        for cand in [BASE_DIR, os.path.join(PROJECT_ROOT, "backend"), PROJECT_ROOT, "/var/task"]:
            if os.path.exists(os.path.join(cand, "services")):
                target_dir = cand
                break
        if target_dir:
            backend_mod = types.ModuleType("backend")
            backend_mod.__path__ = [target_dir]
            backend_mod.__file__ = os.path.join(target_dir, "__init__.py")
            sys.modules["backend"] = backend_mod

from flask import Flask, jsonify, send_from_directory, request, session, redirect

# Import Database, Blueprint services, and EDA modules
from backend.services.database import init_db
from backend.routes.student_routes import student_bp
from backend.routes.assessment_routes import assessment_bp
from backend.routes.auth_routes import auth_bp
from backend.routes.faculty_routes import faculty_bp
from backend.services.prediction_service import fetch_trained_model_metrics, execute_ml_prediction
from backend.services.explanation_service import explain_individual_prediction
from backend.services.recommendation_service import fetch_recommendations_for_input
from backend.services.what_if_service import run_what_if_analysis

from eda.quality import generate_quality_report
from eda.analysis import run_full_eda_analysis

FRONTEND_DIR = None
for cand in [os.path.join(PROJECT_ROOT, "frontend"), os.path.join(BASE_DIR, "frontend"), os.path.join(BASE_DIR, "../frontend"), "/var/task/frontend"]:
    if os.path.exists(cand):
        FRONTEND_DIR = os.path.abspath(cand)
        break
if not FRONTEND_DIR:
    FRONTEND_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "frontend"))

# Initialize Flask application
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="/static")
app.secret_key = os.environ.get("SECRET_KEY", "edupredict-secret-key-2026-pbl")

# Register API Blueprints
app.register_blueprint(student_bp)
app.register_blueprint(assessment_bp)
app.register_blueprint(auth_bp)
app.register_blueprint(faculty_bp)

# ==============================================================================
# BASE REST API ENDPOINTS (/api/...)
# ==============================================================================

@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint verifying API service, database, EDA, ML, Recommendation, and What-If availability."""
    return jsonify({
        "status": "healthy",
        "system": "EDUPREDICT Academic Risk Analysis API",
        "version": "6.0.0",
        "phase": 6,
        "database": "SQLite (data/edupredict.db)",
        "edaPipeline": "Active (Pandas / NumPy / Scikit-Learn)",
        "mlPipeline": "Active (Logistic Regression & Linear Regression)",
        "explanationEngine": "Active (Standardized Model Contributions)",
        "recommendationEngine": "Active (Deterministic Threshold Rules)",
        "whatIfSimulator": "Active (Academic Scenario Simulator)",
        "mode": "Phase 6 - What-If Analysis / Academic Scenario Simulator"
    })

@app.route("/api/data/quality", methods=["GET"])
def get_dataset_quality():
    """Dataset EDA quality report endpoint."""
    report = generate_quality_report()
    return jsonify(report), 200

@app.route("/api/analytics", methods=["GET"])
def get_analytics_summary():
    """Dataset real EDA statistics, correlations, distributions, and insights endpoint."""
    eda_data = run_full_eda_analysis()
    return jsonify(eda_data), 200

@app.route("/api/model/metrics", methods=["GET"])
def get_model_metrics():
    """AI Model evaluation metrics endpoint (Phase 4 trained metrics)."""
    metrics = fetch_trained_model_metrics()
    return jsonify(metrics), 200

@app.route("/api/predict", methods=["POST"])
def predict():
    """AI ML prediction endpoint executing classification, score regression, and recommendations. Unauthenticated calls omit XAI signals."""
    data = request.get_json(silent=True) or {}
    success, result, status_code = execute_ml_prediction(data)
    if success and not session.get("username"):
        # Unauthenticated public prediction returns basic predictions & recommendations, withholding detailed XAI
        result["explanation"] = None
    return jsonify(result), status_code

@app.route("/api/xai", methods=["POST"])
@app.route("/api/explain", methods=["POST"])
def get_xai_explanation():
    """Authenticated API endpoint retrieving feature-level XAI model contributions."""
    if not session.get("username"):
        return jsonify({
            "success": False,
            "error": "Authentication required. Please log in to view detailed XAI explanations."
        }), 401

    data = request.get_json(silent=True) or {}
    explanation = explain_individual_prediction(data)
    return jsonify({
        "success": True,
        "explanation": explanation
    }), 200

@app.route("/api/recommendations", methods=["POST"])
def get_recommendations():
    """Standalone API endpoint generating deterministic academic recommendations."""
    data = request.get_json(silent=True) or {}
    student_subjects = data.get("studentSubjects")
    recs = fetch_recommendations_for_input(data, student_subjects=student_subjects)
    return jsonify({
        "success": True,
        "recommendations": recs,
        "count": len(recs)
    }), 200

@app.route("/api/what-if", methods=["POST"])
def run_what_if():
    """What-If scenario simulation API comparing baseline vs hypothetical profile (Requires Login)."""
    if not session.get("username"):
        return jsonify({
            "success": False,
            "error": "Authentication required. Please log in to access the What-If simulator."
        }), 401

    data = request.get_json(silent=True) or {}
    baseline_raw = data.get("baseline")
    scenario_raw = data.get("scenario")
    success, result, status_code = run_what_if_analysis(baseline_raw, scenario_raw)
    return jsonify(result), status_code

# ==============================================================================
# USER-FACING CLEAN PAGE ROUTING (/...)
# ==============================================================================

@app.route("/")
def serve_index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/register")
def serve_register():
    return send_from_directory(FRONTEND_DIR, "register.html")

@app.route("/register/student")
def serve_register_student():
    return send_from_directory(FRONTEND_DIR, "register_student.html")

@app.route("/register/faculty")
def serve_register_faculty():
    return send_from_directory(FRONTEND_DIR, "register_faculty.html")

@app.route("/login")
def serve_login():
    return send_from_directory(FRONTEND_DIR, "login.html")

@app.route("/login/student")
def serve_login_student():
    return send_from_directory(FRONTEND_DIR, "login_student.html")

@app.route("/login/faculty")
def serve_login_faculty():
    return send_from_directory(FRONTEND_DIR, "login_faculty.html")

@app.route("/dashboard")
def serve_dashboard():
    return send_from_directory(FRONTEND_DIR, "dashboard.html")

@app.route("/faculty")
def serve_faculty():
    return send_from_directory(FRONTEND_DIR, "faculty.html")

@app.route("/assessment")
def serve_assessment():
    return send_from_directory(FRONTEND_DIR, "assessment.html")

@app.route("/prediction")
@app.route("/xai")
def serve_prediction():
    if not session.get("username"):
        return redirect("/login?next=/prediction")
    return send_from_directory(FRONTEND_DIR, "prediction.html")

@app.route("/what-if")
def serve_what_if():
    if not session.get("username"):
        return redirect("/login?next=/what-if")
    return send_from_directory(FRONTEND_DIR, "what-if.html")

@app.route("/planner")
def serve_planner():
    return send_from_directory(FRONTEND_DIR, "planner.html")

@app.route("/analytics")
def serve_analytics():
    return send_from_directory(FRONTEND_DIR, "analytics.html")

@app.route("/model")
def serve_model():
    return send_from_directory(FRONTEND_DIR, "model.html")

@app.route("/mathematics")
def serve_mathematics():
    return send_from_directory(FRONTEND_DIR, "mathematics.html")

@app.route("/algorithms")
def serve_algorithms():
    return send_from_directory(FRONTEND_DIR, "algorithms.html")

@app.route("/responsible-ai")
def serve_responsible_ai():
    return send_from_directory(FRONTEND_DIR, "responsible-ai.html")

@app.route("/about")
def serve_about():
    return send_from_directory(FRONTEND_DIR, "about.html")

# Static assets fallback (css, js, images)
@app.route("/css/<path:path>")
def serve_css(path):
    return send_from_directory(os.path.join(FRONTEND_DIR, "css"), path)

@app.route("/js/<path:path>")
def serve_js(path):
    return send_from_directory(os.path.join(FRONTEND_DIR, "js"), path)

@app.route("/images/<path:path>")
def serve_images(path):
    return send_from_directory(os.path.join(FRONTEND_DIR, "images"), path)

# Support direct /frontend/<path> access (e.g. from Vercel rewrites or relative paths)
@app.route("/frontend/<path:path>")
def serve_frontend_assets(path):
    return send_from_directory(FRONTEND_DIR, path)

@app.errorhandler(404)
def handle_not_found(e):
    req_path = request.path.lstrip("/")
    if req_path.startswith("frontend/"):
        req_path = req_path[len("frontend/"):]
    target_file = os.path.join(FRONTEND_DIR, req_path)
    if os.path.isfile(target_file):
        return send_from_directory(FRONTEND_DIR, req_path)
    if not req_path.endswith(".html") and req_path:
        html_file = os.path.join(FRONTEND_DIR, f"{req_path}.html")
        if os.path.isfile(html_file):
            return send_from_directory(FRONTEND_DIR, f"{req_path}.html")
    if request.path.startswith("/api/"):
        return jsonify({"success": False, "error": f"API endpoint {request.path} not found"}), 404
    return send_from_directory(FRONTEND_DIR, "index.html")

# Safe database initialization on module load (Vercel Serverless / local)
try:
    init_db()
except Exception as e:
    print(f"⚠️ [Startup DB Init Notice]: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001)) if hasattr(os, "environ") else 5001
    print(f"🚀 Starting EDUPREDICT Server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
