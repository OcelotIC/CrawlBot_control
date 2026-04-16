"""CoM-to-torso reference mapping (M1, v1: with delta_dot).

Converts centroidal references (r_com, v_com, a_com) into torso references
(r_b, v_b, a_b) using the mass-weighted identity:

    r_b_ref = (m_total/m_b) * r_com_ref - (1/m_b) * delta(q)
    v_b_ref = (m_total/m_b) * v_com_ref - (1/m_b) * delta_dot(q, dq)
    a_b_ff  = (m_total/m_b) * a_com_ff                   [drop delta_ddot]

where  delta(q)        = sum_{i != torso} m_i * r_i(q)
       delta_dot(q,dq) = sum_{i != torso} m_i * J_i(q) @ dq

is the mass-weighted sum of non-torso body CoM positions/velocities
(world frame). Per-body translational Jacobians J_i are computed in
LOCAL_WORLD_ALIGNED. delta_ddot is dropped at v1 (PD handles it); see
docs/architecture/STATUS.md §7 (cascade bisection 2026-04-16).

This exists so the whole-body QP can track the torso position (rigid-body
task) instead of the CoM position (centroidal task). The two are exactly
equivalent at the Jacobian level (test T3):

    J_b_pos = (m_total/m_b) * J_com - (1/m_b) * sum_{i != torso} m_i * J_i

See docs/architecture/brainstorming_reworked_architecture.md §4.4-4.5.
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin


# Pinocchio joint index of the torso body (root joint attached to Link_0).
# In a free-flyer URDF, joint 1 is root_joint and its body is the torso.
TORSO_JOINT_IDX = 1


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2],  v[1]],
                     [v[2],  0.0, -v[0]],
                     [-v[1], v[0],  0.0]])


class CoMToTorsoMapping:
    """Mass-weighted mapping from NMPC centroidal outputs to torso references.

    Parameters
    ----------
    robot_interface : RobotInterface
        Wrapper exposing the Pinocchio model, total mass, and helper
        methods. Used only for FK and mass access — no simulation state.

    Attributes
    ----------
    m_total : float
        Total mass of the robot.
    m_b : float
        Mass of the torso body.
    m_others : np.ndarray, shape (n_bodies-2,)
        Masses of the non-torso bodies (in Pinocchio joint order, skipping
        the universe joint and the root/torso joint).
    non_torso_joints : list of int
        Pinocchio joint indices of the non-torso bodies.
    """

    def __init__(self, robot_interface):
        self.ri = robot_interface
        self.model = robot_interface.model
        self.m_total = float(robot_interface._total_mass)
        self.m_b = float(self.model.inertias[TORSO_JOINT_IDX].mass)
        if self.m_b <= 0:
            raise ValueError(
                f"Torso mass must be positive (got {self.m_b}). "
                "Check URDF or pass torso_mass= to RobotInterface.")

        self.non_torso_joints = [
            i for i in range(1, self.model.njoints)
            if i != TORSO_JOINT_IDX and self.model.inertias[i].mass > 0
        ]
        self.m_others = np.array(
            [self.model.inertias[i].mass for i in self.non_torso_joints])

        # Sanity: m_b + sum(m_others) == m_total (within roundoff)
        m_check = self.m_b + float(np.sum(self.m_others))
        if not np.isclose(m_check, self.m_total, atol=1e-6):
            raise ValueError(
                f"Mass inconsistency: m_b + sum(m_others) = {m_check} "
                f"!= m_total = {self.m_total}")

        # Precompute scalar ratios used in every call
        self.ratio = self.m_total / self.m_b  # = m_total/m_b

    # ------------------------------------------------------------------ #
    # delta(q): mass-weighted sum of non-torso body CoM positions          #
    # ------------------------------------------------------------------ #

    def compute_delta(self, q: np.ndarray) -> np.ndarray:
        """Compute delta(q) = sum_{i != torso} m_i * r_i(q) in world frame.

        Requires a configuration-space vector q (Pinocchio layout).
        """
        data = self.model.createData()
        pin.forwardKinematics(self.model, data, q)
        delta = np.zeros(3)
        for i, m_i in zip(self.non_torso_joints, self.m_others):
            r_i = data.oMi[i].act(self.model.inertias[i].lever)
            delta += m_i * r_i
        return delta

    def compute_delta_dot(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """Compute delta_dot(q, dq) = sum_{i != torso} m_i * J_i(q) @ dq.

        Mathematically d/dt of delta(q): consistent with the time
        derivative of the position-level identity
            r_b * m_b + delta = m_total * r_com.

        J_i is the translational (3, nv) Jacobian of body i's CoM in the
        LOCAL_WORLD_ALIGNED frame (see body_com_jacobian).
        """
        data = self.model.createData()
        pin.forwardKinematics(self.model, data, q)
        pin.computeJointJacobians(self.model, data, q)
        delta_dot = np.zeros(3)
        for i, m_i in zip(self.non_torso_joints, self.m_others):
            J_i = self.body_com_jacobian(data, i)
            delta_dot += m_i * (J_i @ dq)
        return delta_dot

    # ------------------------------------------------------------------ #
    # Forward mapping: (r_com, v_com, a_com, q) -> torso refs              #
    # ------------------------------------------------------------------ #

    def compute(self, r_com_ref: np.ndarray, v_com_ref: np.ndarray,
                a_com_ff: np.ndarray, q_current: np.ndarray,
                dq_current: np.ndarray = None):
        """Forward mapping.

        Parameters
        ----------
        r_com_ref : (3,)   CoM position reference (world frame)
        v_com_ref : (3,)   CoM velocity reference (world frame)
        a_com_ff  : (3,)   CoM acceleration feedforward (world frame)
        q_current : (nq,)  current configuration (Pinocchio layout)
        dq_current : (nv,) current velocity. When provided, the
                           delta_dot/m_b correction is added to v_b_ref
                           (v1 fix). When None, behaviour matches v0
                           (no delta_dot term).

        Returns
        -------
        r_b_ref : (3,)   torso position reference
        v_b_ref : (3,)   torso linear velocity reference
        a_b_ff  : (3,)   torso linear acceleration feedforward
        delta   : (3,)   Sum_{i != torso} m_i * r_i (for monitoring)
        """
        r_com_ref = np.asarray(r_com_ref, dtype=float).reshape(3)
        v_com_ref = np.asarray(v_com_ref, dtype=float).reshape(3)
        a_com_ff = np.asarray(a_com_ff, dtype=float).reshape(3)
        q_current = np.asarray(q_current, dtype=float).ravel()

        delta = self.compute_delta(q_current)
        r_b_ref = self.ratio * r_com_ref - delta / self.m_b
        v_b_ref = self.ratio * v_com_ref
        if dq_current is not None:
            dq_current = np.asarray(dq_current, dtype=float).ravel()
            delta_dot = self.compute_delta_dot(q_current, dq_current)
            v_b_ref = v_b_ref - delta_dot / self.m_b
        # delta_ddot dropped at v1; PD handles the residual accel.
        a_b_ff = self.ratio * a_com_ff
        return r_b_ref, v_b_ref, a_b_ff, delta

    # ------------------------------------------------------------------ #
    # Per-body Jacobian (for T3 equivalence test)                          #
    # ------------------------------------------------------------------ #

    def body_com_jacobian(self, data, joint_idx: int) -> np.ndarray:
        """Translational Jacobian (3, nv) of body `joint_idx`'s CoM position.

        Uses the LOCAL_WORLD_ALIGNED frame:
            d(oMi @ lever)/dt = J_lin - skew(R_i @ lever) @ J_ang

        Assumes `pin.computeJointJacobians` has already been called on
        `data` at the desired configuration.
        """
        J_i = pin.getJointJacobian(
            self.model, data, joint_idx, pin.LOCAL_WORLD_ALIGNED)
        R_i = data.oMi[joint_idx].rotation
        lever_world = R_i @ self.model.inertias[joint_idx].lever
        return J_i[:3] - _skew(lever_world) @ J_i[3:]

    def torso_pos_jacobian_from_com(self, q: np.ndarray) -> np.ndarray:
        """Return J_b_pos constructed via the CoM equivalence identity.

            J_b_pos = (m_total/m_b) * J_com - (1/m_b) * sum_{i != torso} m_i * J_i

        Used by tests/T3 to verify the identity matches the direct
        Pinocchio torso Jacobian.
        """
        data = self.model.createData()
        pin.forwardKinematics(self.model, data, q)
        pin.computeJointJacobians(self.model, data, q)
        pin.centerOfMass(self.model, data, q)
        J_com = pin.jacobianCenterOfMass(self.model, data, q)

        J_sum = np.zeros_like(J_com)
        for i, m_i in zip(self.non_torso_joints, self.m_others):
            J_sum += m_i * self.body_com_jacobian(data, i)
        return self.ratio * J_com - J_sum / self.m_b
