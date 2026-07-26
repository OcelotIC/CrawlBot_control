"""Core interfaces: robot model, state conversions, inverse kinematics."""

from .robot_interface import RobotInterface, RobotState
from .state_conversions import mujoco_to_pinocchio, pinocchio_to_mujoco
from .state_conversions import quat_wxyz_to_euler_deg
from .ik import dock_configuration, solve_ik

# Re-exported package API. Declared so the names are not read as
# unused imports — they are the interface, not leftovers.
__all__ = [
    'RobotInterface',
    'RobotState',
    'mujoco_to_pinocchio',
    'pinocchio_to_mujoco',
    'quat_wxyz_to_euler_deg',
    'dock_configuration',
    'solve_ik',
]
