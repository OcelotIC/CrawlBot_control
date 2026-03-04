"""SPART – Spacecraft-Manipulator Robot Toolkit (Python re-implementation).

Pure NumPy implementation of the SPART kinematics and dynamics algorithms.
"""

from .attitude import (
    quat_dcm, euler_dcm, dcm_quat, dcm_euler,
    angles321_dcm, dcm_angles321, angles123_dcm,
    quat_angles321,
)
from .robot_model import urdf2robot, dh_serial2robot, connectivity_map
from .kinematics import kinematics
from .diff_kinematics import diff_kinematics
from .velocities import velocities
from .accelerations import accelerations
from .inertia import inertia_projection, mass_composite_body
from .gim import generalized_inertia_matrix
from .cim import convective_inertia_matrix
from .jacobian import geometric_jacobian
from .dynamics import forward_dynamics, inverse_dynamics
