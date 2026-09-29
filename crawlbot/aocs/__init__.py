"""AOCS: the reaction-wheel torque law (legacy_pid_numerical)."""

from .force_estimator import compute_aocs_command_legacy_pid_numerical

# Re-exported package API. Declared so the name is not read as an unused
# import — it is the interface, not a leftover.
__all__ = [
    'compute_aocs_command_legacy_pid_numerical',
]
