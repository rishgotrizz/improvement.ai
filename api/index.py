import os
import sys
import types

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

candidate_roots = [
    PROJECT_ROOT,
    os.path.dirname(PROJECT_ROOT),
    BASE_DIR,
    os.getcwd(),
    "/var/task"
]

for cand in candidate_roots:
    if os.path.exists(os.path.join(cand, "backend")):
        PROJECT_ROOT = cand
        break

for p in [PROJECT_ROOT, os.path.join(PROJECT_ROOT, "backend"), BASE_DIR, "/var/task"]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

if "backend" not in sys.modules:
    try:
        import backend
    except ImportError:
        target_dir = None
        for cand in [os.path.join(PROJECT_ROOT, "backend"), PROJECT_ROOT, "/var/task"]:
            if os.path.exists(os.path.join(cand, "services")):
                target_dir = cand
                break
        if target_dir:
            backend_mod = types.ModuleType("backend")
            backend_mod.__path__ = [target_dir]
            backend_mod.__file__ = os.path.join(target_dir, "__init__.py")
            sys.modules["backend"] = backend_mod

from backend.app import app

# Export app for Vercel Python Serverless Function entrypoint
app = app
