"""Convert optimal contact wrenches to joint torques.

Uses Pinocchio contact Jacobians to map spatial contact wrenches to
the generalized force vector, then extracts the actuated joint torques.

    tau = J_a^T @ Fc_a + J_b^T @ Fc_b
"""

import numpy as np
import pinocchio as pin


def compute_joint_torques(Fc_a, Fc_b, J_tool_a, J_tool_b,
                          active_a=True, active_b=True, tau_max=50.0):
    """Map contact wrenches to joint torques.

    Parameters
    ----------
    Fc_a : (6,) - Contact wrench at tool_a [torque(3); force(3)].
    Fc_b : (6,) - Contact wrench at tool_b.
    J_tool_a : (6, nv) - Full Jacobian at tool_a frame.
    J_tool_b : (6, nv) - Full Jacobian at tool_b frame.
    active_a : bool - Whether contact A is active.
    active_b : bool - Whether contact B is active.
    tau_max : float - Joint torque limit (Nm).

    Returns
    -------
    tau : (nv,) generalized forces (only actuated joints are nonzero).
    """
    nv = J_tool_a.shape[1]
    tau = np.zeros(nv)

    if active_a:
        tau += J_tool_a.T @ Fc_a

    if active_b:
        tau += J_tool_b.T @ Fc_b

    # Saturate
    tau = np.clip(tau, -tau_max, tau_max)

    return tau


def get_contact_jacobians(model, data, q):
    """Compute Jacobians at tool_a and tool_b in world frame.

    Parameters
    ----------
    model : pin.Model
    data : pin.Data
    q : (nq,) joint configuration.

    Returns
    -------
    J_tool_a : (6, nv)
    J_tool_b : (6, nv)
    """
    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)

    fid_a = model.getFrameId('tool_a')
    fid_b = model.getFrameId('tool_b')

    J_tool_a = pin.getFrameJacobian(model, data, fid_a, pin.WORLD)
    J_tool_b = pin.getFrameJacobian(model, data, fid_b, pin.WORLD)

    return J_tool_a, J_tool_b
