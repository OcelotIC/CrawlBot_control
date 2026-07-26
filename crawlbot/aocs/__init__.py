"""AOCS: momentum disturbance estimator and wheel command generation."""

from .force_estimator import (
    MomentumDisturbanceEstimator,
    EstimatorConfig,
    compute_aocs_command,
)

# Re-exported package API. Declared so the names are not read as
# unused imports — they are the interface, not leftovers.
__all__ = [
    'MomentumDisturbanceEstimator',
    'EstimatorConfig',
    'compute_aocs_command',
]
