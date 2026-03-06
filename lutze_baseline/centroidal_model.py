"""Centroidal rigid-body reduction of VISPA robot via Pinocchio.

Matches Lutze et al. (2023) abstraction: the robot is represented by its
centroidal quantities (CoM position/velocity, angular momentum, locked
inertia tensor) rather than the full multi-body state.

This module wraps RobotState properties that already exist in
robot_interface.py and adds the locked centroidal inertia computation.
"""

import numpy as np
import pinocchio as pin
from dataclasses import dataclass


@dataclass
class CentroidalState:
    """Reduced centroidal representation of the robot."""
    r_com: np.ndarray       # (3,) CoM position in world frame
    v_com: np.ndarray       # (3,) CoM velocity in world frame
    L_com: np.ndarray       # (3,) centroidal angular momentum
    I_locked: np.ndarray    # (3,3) locked rotational inertia about CoM
    mass: float             # total robot mass [kg]


def compute_centroidal_state(robot, rs) -> CentroidalState:
    """Extract centroidal state from a Pinocchio RobotState.

    Parameters
    ----------
    robot : RobotInterface
        Robot interface instance (needed for model/data access).
    rs : RobotState
        Most recent state from robot.update(q, v).

    Returns
    -------
    CentroidalState
        Rigid-body reduction of the robot.
    """
    # CoM quantities are already in RobotState
    r_com = rs.r_com.copy()
    v_com = rs.v_com.copy()
    L_com = rs.L_com.copy()

    # Locked centroidal inertia: upper-left 3x3 of centroidal momentum
    # matrix Ag evaluated at current q with v=0.
    # Ag (6x nv) maps generalized velocities to centroidal momentum:
    #   h = Ag @ v,  h = [linear(3); angular(3)]
    # The upper-left 3x3 of Ag relates base angular velocity to angular
    # momentum when all joints are locked — this is the locked inertia.
    Ag = pin.computeCentroidalMap(robot.model, robot.data, rs.q)
    I_locked = Ag[:3, :3].copy()  # (3,3) maps omega_base → L

    return CentroidalState(
        r_com=r_com,
        v_com=v_com,
        L_com=L_com,
        I_locked=I_locked,
        mass=rs.total_mass,
    )
