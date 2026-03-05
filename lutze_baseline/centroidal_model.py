"""Centroidal rigid-body reduction of VISPA robot via Pinocchio.

Matches Lutze et al. (2023) abstraction: the robot is represented by its
centroidal quantities (CoM position/velocity, angular momentum,
locked inertia, total mass).
"""

from dataclasses import dataclass
import numpy as np
import pinocchio as pin


@dataclass
class CentroidalState:
    """Reduced centroidal representation of the robot."""
    r_com: np.ndarray       # (3,) CoM position in world frame
    v_com: np.ndarray       # (3,) CoM velocity in world frame
    L_com: np.ndarray       # (3,) centroidal angular momentum
    I_locked: np.ndarray    # (3,3) locked inertia tensor
    mass: float             # total robot mass


def compute_centroidal_state(model, data, q, v):
    """Extract centroidal state from Pinocchio model.

    Parameters
    ----------
    model : pin.Model
        Pinocchio model.
    data : pin.Data
        Pinocchio data (will be modified in-place).
    q : (nq,) joint configuration.
    v : (nv,) joint velocities.

    Returns
    -------
    CentroidalState
    """
    # Forward kinematics + centroidal computations
    pin.forwardKinematics(model, data, q, v)
    pin.computeCentroidalMap(model, data, q)
    pin.centerOfMass(model, data, q, v)

    r_com = data.com[0].copy()
    v_com = data.vcom[0].copy()

    # Centroidal momentum matrix Ag (6 x nv)
    # Top 3 rows: angular momentum, bottom 3: linear momentum
    Ag = data.Ag.copy()
    h = Ag @ v  # centroidal momentum (6,)
    L_com = h[:3].copy()  # angular part

    # Locked inertia: upper-left 3x3 of Ag @ M^{-1} @ Ag^T
    # Simpler: I_locked = Ag[:3, :3] when base is identity
    # More robust: compute from centroidal momentum matrix
    # I_locked approximation: Ag_angular @ pinv(Ag_linear) * mass
    # Actually: the locked inertia is just the 3x3 upper-left block
    # of the centroidal inertia matrix, which for a fixed-base robot
    # can be computed from the composite rigid body algorithm
    pin.ccrba(model, data, q, v)
    I_locked = data.Ig.inertia.copy()  # 3x3 rotational inertia at CoM

    mass = pin.computeTotalMass(model)

    return CentroidalState(
        r_com=r_com,
        v_com=v_com,
        L_com=L_com,
        I_locked=I_locked,
        mass=mass,
    )
