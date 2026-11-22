import os
import sys
import types

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

candidate_roots = [
    BASE_DIR,
    os.path.abspath(os.path.join(BASE_DIR, "..")),
    os.getcwd(),
    "/var/task"
]

PROJECT_ROOT = None
for cand in candidate_roots:
    if os.path.exists(os.path.join(cand, "backend")) or os.path.exists(os.path.join(cand, "frontend")):
        PROJECT_ROOT = cand
        break

if not PROJECT_ROOT:
    PROJECT_ROOT = BASE_DIR

for p in [PROJECT_ROOT, BASE_DIR, os.path.join(PROJECT_ROOT, "backend"), "/var/task"]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

if "backend" not in sys.modules:
    try:
        import backend
    except ImportError:
        target_dir = None
        for cand in [os.path.join(PROJECT_ROOT, "backend"), BASE_DIR, "/var/task"]:
            if os.path.exists(os.path.join(cand, "services")):
                target_dir = cand
                break
        if target_dir:
            backend_mod = types.ModuleType("backend")
            backend_mod.__path__ = [target_dir]
            backend_mod.__file__ = os.path.join(target_dir, "__init__.py")
            sys.modules["backend"] = backend_mod

from backend.app import app

# Export Flask WSGI application instance for Vercel Python runtime
app = app
