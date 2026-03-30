"""Core interfaces: robot model, state conversions, inverse kinematics."""

from .robot_interface import RobotInterface, RobotState
from .state_conversions import mujoco_to_pinocchio, pinocchio_to_mujoco
from .state_conversions import quat_wxyz_to_euler_deg
from .ik import dock_configuration, solve_ik
