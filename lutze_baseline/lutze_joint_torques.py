"""Convert optimal contact wrenches to joint torques.

Uses Pinocchio contact Jacobians to map spatial contact wrenches to
the generalized force vector, then extracts the actuated joint torques.

    tau_full = J_tool_a^T @ Fc_a + J_tool_b^T @ Fc_b
    tau_joints = tau_full[6:]   (strip unactuated free-flyer DOFs)
"""

import numpy as np


def compute_joint_torques(
    Fc_a: np.ndarray,
    Fc_b: np.ndarray,
    J_tool_a: np.ndarray,
    J_tool_b: np.ndarray,
    active_a: bool,
    active_b: bool,
    tau_max: float = 10.0,
) -> np.ndarray:
    """Map contact wrenches to joint torques.

    Parameters
    ----------
    Fc_a : (6,) – Contact wrench at tool_a [torque(3); force(3)].
    Fc_b : (6,) – Contact wrench at tool_b.
    J_tool_a : (6, 18) – Tool-A Jacobian (LOCAL_WORLD_ALIGNED).
    J_tool_b : (6, 18) – Tool-B Jacobian.
    active_a : bool – Whether contact A is active.
    active_b : bool – Whether contact B is active.
    tau_max : float – Per-joint torque limit [Nm].

    Returns
    -------
    tau_joints : (12,) – Joint torques (clipped to ±tau_max).
    """
    nv = J_tool_a.shape[1]  # 18
    tau_full = np.zeros(nv)

    if active_a:
        tau_full += J_tool_a.T @ Fc_a
    if active_b:
        tau_full += J_tool_b.T @ Fc_b

    # Extract actuated joints (skip 6 free-flyer DOFs)
    tau_joints = tau_full[6:]

    # Clip to actuator limits
    tau_joints = np.clip(tau_joints, -tau_max, tau_max)

    return tau_joints
