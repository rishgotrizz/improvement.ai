import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensure root repository directory is at position 0 of sys.path
if BASE_DIR in sys.path:
    sys.path.remove(BASE_DIR)
sys.path.insert(0, BASE_DIR)

from backend.app import app

# Export Flask WSGI application instance for Vercel Python runtime
app = app
