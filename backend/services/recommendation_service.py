"""
EDUPREDICT - Recommendation Service Layer
Wraps deterministic recommendation engine for API endpoints and student academic evaluations.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from recommendations.engine import generate_academic_recommendations

def fetch_recommendations_for_input(input_data, student_subjects=None):
    """
    Generate prioritized academic recommendations from student inputs.
    """
    if not input_data or not isinstance(input_data, dict):
        return []

    return generate_academic_recommendations(input_data, student_subjects=student_subjects)
