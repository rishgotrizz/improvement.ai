"""
EDUPREDICT Recommendation Engine Package
"""

from .rules import DEMO_THRESHOLDS, evaluate_factor_status, get_rule_recommendation
from .engine import generate_academic_recommendations, compute_severity_score

__all__ = [
    "DEMO_THRESHOLDS",
    "evaluate_factor_status",
    "get_rule_recommendation",
    "generate_academic_recommendations",
    "compute_severity_score"
]
