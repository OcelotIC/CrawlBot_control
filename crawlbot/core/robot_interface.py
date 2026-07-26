"""
RobotInterface — Pinocchio wrapper for the VISPA crawling controller.

Computes at each timestep all quantities needed by the two-stage controller:
    Stage 1 (CentroidalNMPC): r_com, v_com, L_com
    Stage 2 (WholeBodyQP):    H_robot, C_robot, J_com, J̇_com·q̇,
                               J_contacts, J̇_contacts·q̇, joint limits

DOF-generic: all dimensions and frame/joint IDs are derived from the
Pinocchio model at construction time. Supports 6-DOF and 7-DOF arms.

Conventions (Pinocchio free-flyer):
    q  = [pos(3), quat_xyzw(4), joints(N)]   → nq = 7 + N
    v  = [twist(6), qdot(N)]                  → nv = 6 + N
    τ  = [base_wrench(6)=0, torques(N)]       → nv-dim

Frame IDs are looked up by name from the URDF:
    FRAME_TOOL_A → model.getFrameId("tool_a")
    FRAME_TOOL_B → model.getFrameId("tool_b")
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import pinocchio as pin


def _detect_arm_slices(model: pin.Model):
    """Detect arm joint indices from the model by name convention.

    Joints ending in '_a' belong to arm A, '_b' to arm B.
    Returns velocity-space slices and position-space slices.
    """
    arm_a_joints = []
    arm_b_joints = []
    for i in range(2, model.njoints):  # skip universe(0) and root_joint(1)
        name = model.names[i]
        if name.endswith('_a'):
            arm_a_joints.append(i)
        elif name.endswith('_b'):
            arm_b_joints.append(i)

    if not arm_a_joints or not arm_b_joints:
        raise RuntimeError("Could not detect arm joints from URDF names "
                           "(expected Joint_*_a and Joint_*_b)")

    # Velocity-space slices
    va0 = model.idx_vs[arm_a_joints[0]]
    va1 = model.idx_vs[arm_a_joints[-1]] + model.nvs[arm_a_joints[-1]]
    vb0 = model.idx_vs[arm_b_joints[0]]
    vb1 = model.idx_vs[arm_b_joints[-1]] + model.nvs[arm_b_joints[-1]]

    # Position-space slices
    qa0 = model.idx_qs[arm_a_joints[0]]
    qa1 = model.idx_qs[arm_a_joints[-1]] + model.nqs[arm_a_joints[-1]]
    qb0 = model.idx_qs[arm_b_joints[0]]
    qb1 = model.idx_qs[arm_b_joints[-1]] + model.nqs[arm_b_joints[-1]]

    return {
        'arm_a_v': slice(va0, va1),
        'arm_b_v': slice(vb0, vb1),
        'arm_a_q': slice(qa0, qa1),
        'arm_b_q': slice(qb0, qb1),
        'n_per_arm_a': va1 - va0,
        'n_per_arm_b': vb1 - vb0,
        'n_joints': (va1 - va0) + (vb1 - vb0),
        'joints_q': slice(qa0, qb1),  # all joints combined (position)
        'joints_v': slice(va0, vb1),  # all joints combined (velocity)
    }


# ── Frame / joint IDs — set at module load for backward compatibility,
#    but should be accessed via RobotInterface properties for new code.
# These are OVERWRITTEN by RobotInterface.__init__() with model-derived values.
FRAME_TORSO = 4
FRAME_TOOL_A = 18
FRAME_TOOL_B = 32
JOINT_6A_ID = 7
JOINT_6B_ID = 13

# Legacy constants — DEPRECATED, use model-derived values
N_JOINTS = 12
NQ = 19
NV = 18


@dataclass
class RobotState:
    """All Pinocchio-computed quantities at one timestep.

    Populated by RobotInterface.update(q, v).
    """
    # ── State ────────────────────────────────────────────────
    q: np.ndarray           # (19,) Pinocchio generalized positions
    v: np.ndarray           # (18,) Pinocchio generalized velocities
    q_joints: np.ndarray    # (12,) joint positions only
    dq_joints: np.ndarray   # (12,) joint velocities only
    q_torso: np.ndarray     # (7,)  torso pose [pos(3), quat_xyzw(4)]
    dq_torso: np.ndarray    # (6,)  torso twist

    # ── Dynamics ─────────────────────────────────────────────
    H: np.ndarray           # (18,18) Mass matrix (CRBA)
    C: np.ndarray           # (18,) Bias term (Coriolis + gravity via RNEA)
    C_matrix: np.ndarray    # (18,18) Coriolis matrix C(q,v) for GMO

    # ── Center of Mass ───────────────────────────────────────
    r_com: np.ndarray       # (3,) CoM position
    v_com: np.ndarray       # (3,) CoM velocity
    J_com: np.ndarray       # (3,18) CoM Jacobian
    Jdot_dq_com: np.ndarray # (3,) J̇_com · q̇

    # ── Centroidal momentum ──────────────────────────────────
    h_centroidal: np.ndarray  # (6,) [linear(3); angular(3)]
    L_com: np.ndarray         # (3,) angular part only

    # ── Contact frames ───────────────────────────────────────
    oMf_tool_a: Any         # SE3 placement of tool_a in world
    oMf_tool_b: Any         # SE3 placement of tool_b in world
    J_tool_a: np.ndarray    # (6,18) Jacobian of tool_a (LOCAL_WORLD_ALIGNED)
    J_tool_b: np.ndarray    # (6,18) Jacobian of tool_b
    Jdot_dq_tool_a: np.ndarray  # (6,) J̇_a · q̇
    Jdot_dq_tool_b: np.ndarray  # (6,) J̇_b · q̇

    # ── Torso frame ──────────────────────────────────────────
    oMf_torso: Any          # SE3 placement of torso in world
    J_torso: np.ndarray     # (6,18) Jacobian of torso frame (LOCAL_WORLD_ALIGNED)
    Jdot_dq_torso: np.ndarray  # (6,) J̇_torso · q̇

    # ── Joint limits ─────────────────────────────────────────
    q_min: np.ndarray       # (12,) lower joint limits
    q_max: np.ndarray       # (12,) upper joint limits
    tau_max: np.ndarray     # (12,) torque limits (absolute)

    # ── Total mass ───────────────────────────────────────────
    total_mass: float


class RobotInterface:
    """Pinocchio computation wrapper for the VISPA controller.

    Call update(q, v) at each timestep to compute all required quantities.
    Then access them via the `state` property.

    Parameters
    ----------
    urdf_path : str
        Path to the VISPA URDF (free-flyer base).
    tau_max : float or ndarray (12,), optional
        Joint torque limits. Default: 10 Nm per joint.
    gravity : str, optional
        'zero'/'micro' (default) for orbital operations, 'earth' for ground tests.
    """

    def __init__(self, urdf_path: str, tau_max=10.0, gravity: str = 'zero',
                 torso_mass: float = None):
        global FRAME_TORSO, FRAME_TOOL_A, FRAME_TOOL_B
        global JOINT_6A_ID, JOINT_6B_ID, N_JOINTS, NQ, NV

        # Build model with free-flyer
        self.model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self.data = self.model.createData()

        # Override torso mass if specified (URDF value may be incorrect)
        if torso_mass is not None:
            old_I = self.model.inertias[1]  # idx 1 = root_joint body
            ratio = torso_mass / max(old_I.mass, 1e-6)
            new_inertia = old_I.inertia * ratio  # scale tensor
            self.model.inertias[1] = pin.Inertia(
                torso_mass, old_I.lever, new_inertia)
            self.data = self.model.createData()

        # Gravity setting
        if gravity == 'zero' or gravity == 'micro':
            self.model.gravity = pin.Motion.Zero()
        elif gravity == 'earth':
            pass  # keep default 9.81

        # ── DOF-generic detection from model ────────────────────
        slices = _detect_arm_slices(self.model)
        self.arm_a_v_slice = slices['arm_a_v']      # velocity-space slice for arm A
        self.arm_b_v_slice = slices['arm_b_v']      # velocity-space slice for arm B
        self.arm_a_q_slice = slices['arm_a_q']      # position-space slice for arm A
        self.arm_b_q_slice = slices['arm_b_q']      # position-space slice for arm B
        self.joints_q_slice = slices['joints_q']    # all joints (position)
        self.joints_v_slice = slices['joints_v']    # all joints (velocity)
        self.n_arm_a = slices['n_per_arm_a']        # DOFs per arm A
        self.n_arm_b = slices['n_per_arm_b']        # DOFs per arm B
        self.n_joints = slices['n_joints']           # total actuated joints

        # ── Armature consistency with MJCF (M7 armature decomposition) ──
        # The MJCF's robot_joint default assigns armature=0.05 to each of
        # the 14 arm joints; Pinocchio's URDF loader ignores any such
        # attribute and leaves model.armature at zero. Install the same
        # rotor-inertia diagonal here so pin.crba produces
        # H = H_link + diag(armature) matching MuJoCo's integrator mass.
        # The free-flyer base (v[0:6]) keeps armature=0.
        _arm = np.zeros(self.model.nv)
        _arm[slices['joints_v']] = 0.05
        self.model.armature = _arm
        self.data = self.model.createData()

        # Frame IDs by name lookup
        self.frame_tool_a = self.model.getFrameId("tool_a")
        self.frame_tool_b = self.model.getFrameId("tool_b")
        self.frame_torso = self.model.getFrameId("Link_0")

        # Last joint of each arm (parent of tool frame)
        self.joint_tip_a = self.model.frames[self.frame_tool_a].parentJoint
        self.joint_tip_b = self.model.frames[self.frame_tool_b].parentJoint

        # Update module-level constants for backward compatibility
        FRAME_TOOL_A = self.frame_tool_a
        FRAME_TOOL_B = self.frame_tool_b
        FRAME_TORSO = self.frame_torso
        JOINT_6A_ID = self.joint_tip_a
        JOINT_6B_ID = self.joint_tip_b
        N_JOINTS = self.n_joints
        NQ = self.model.nq
        NV = self.model.nv

        # Joint limits (from URDF) — all actuated joints
        self._q_min = self.model.lowerPositionLimit[
            self.joints_q_slice].copy()
        self._q_max = self.model.upperPositionLimit[
            self.joints_q_slice].copy()

        # Torque limits
        if np.isscalar(tau_max):
            self._tau_max = np.full(self.n_joints, tau_max)
        else:
            self._tau_max = np.array(tau_max).ravel()

        # Total mass (computed once)
        data_tmp = self.model.createData()
        pin.computeTotalMass(self.model, data_tmp)
        self._total_mass = data_tmp.mass[0]

        # State placeholder
        self._state: Optional[RobotState] = None

    def update(self, q: np.ndarray, v: np.ndarray,
               omega_struct: Optional[np.ndarray] = None) -> 'RobotState':
        """Compute all controller-required quantities from (q, v).

        Parameters
        ----------
        q : ndarray (19,)
            Pinocchio generalized positions.
        v : ndarray (18,)
            Pinocchio generalized velocities.
        omega_struct : ndarray (3,), optional
            Structure angular velocity in the structure body frame [rad/s].
            Reserved for future non-inertial frame corrections.
            Currently stored but not used in dynamics computation —
            see Misc/reports/test_findings_report.md for analysis of why Layers 1
            (apparent gravity) and 2 (J̇ correction) destabilize the
            existing controller gains.

        Returns
        -------
        state : RobotState
            All computed quantities.
        """
        model = self.model
        data = self.data

        # Store structure angular velocity for diagnostics / future use.
        # Non-inertial corrections (centrifugal gravity, J̇ += ad(ω_s)·J)
        # are NOT applied: they are mathematically correct but destabilize
        # the existing QP gains. The controller was designed assuming a
        # quasi-static (inertial) structure frame. Adding non-inertial
        # terms requires re-tuning all QP weights and AOCS gains.
        self._omega_struct = omega_struct.copy() if omega_struct is not None else None

        # ── Kinematics + dynamics in one pass ────────────────────
        # computeAllTerms computes: CRBA, RNEA(bias), CoM, Jacobians, etc.
        pin.computeAllTerms(model, data, q, v)
        # Frame placements need explicit update
        pin.updateFramePlacements(model, data)

        # ── Mass matrix H = M(q) ────────────────────────────────
        # CRBA is already called by computeAllTerms
        H = data.M.copy()
        # Make symmetric (Pinocchio fills upper triangle)
        H = (H + H.T) / 2.0

        # ── Bias vector C(q, v) = h(q, v) ───────────────────────
        # RNEA bias = h(q, v, 0) = C(q,v)q̇ + g(q)
        # computeAllTerms already computed nle (non-linear effects)
        C = data.nle.copy()

        # ── Coriolis matrix C(q,v) for GMO ───────────────────────
        pin.computeCoriolisMatrix(model, data, q, v)
        C_matrix = data.C.copy()

        # ── Center of Mass ───────────────────────────────────────
        r_com = data.com[0].copy()

        # CoM velocity: v_com = J_com @ v
        J_com = data.Jcom.copy()  # (3, nv)
        v_com = J_com @ v

        # J̇_com · q̇: computeAllTerms does NOT populate data.acom.
        # Must call centerOfMass(q, v, a=0) explicitly:
        #   acom = J_com @ q̈ + J̇_com @ q̇  with q̈=0 → acom = J̇_com @ q̇
        a_zero = np.zeros(model.nv)
        pin.centerOfMass(model, data, q, v, a_zero)
        Jdot_dq_com = data.acom[0].copy()

        # ── Centroidal momentum ──────────────────────────────────
        # Already computed by computeAllTerms (via computeCentroidalMomentum)
        h_centroidal = data.hg.vector.copy()  # (6,)
        L_com = h_centroidal[3:6]  # angular part

        # ── Contact frame Jacobians ──────────────────────────────
        # Full frame Jacobians in LOCAL_WORLD_ALIGNED
        fid_a = self.frame_tool_a
        fid_b = self.frame_tool_b
        fid_t = self.frame_torso

        J_tool_a = pin.getFrameJacobian(
            model, data, fid_a, pin.LOCAL_WORLD_ALIGNED).copy()
        J_tool_b = pin.getFrameJacobian(
            model, data, fid_b, pin.LOCAL_WORLD_ALIGNED).copy()

        # J̇·q̇ for contact frames
        # We need computeJointJacobiansTimeVariation which is NOT part
        # of computeAllTerms → call it explicitly
        pin.computeJointJacobiansTimeVariation(model, data, q, v)
        dJ_a = pin.getFrameJacobianTimeVariation(
            model, data, fid_a, pin.LOCAL_WORLD_ALIGNED)
        dJ_b = pin.getFrameJacobianTimeVariation(
            model, data, fid_b, pin.LOCAL_WORLD_ALIGNED)

        # ── Non-inertial frame correction (Layer 2) — DISABLED ──
        # In a rotating frame, J̇_true = J̇_pinocchio + ad(ξ_s)·J
        # where ad(0,ω_s) = diag([ω_s]×, [ω_s]×) for LOCAL_WORLD_ALIGNED.
        # This is mathematically correct but destabilizes the QP because
        # the controller gains were tuned assuming an inertial frame.
        # Enabling requires re-tuning all QP weights and feedback gains.
        # See Misc/reports/test_findings_report.md for experimental evidence.

        Jdot_dq_tool_a = (dJ_a @ v).copy()
        Jdot_dq_tool_b = (dJ_b @ v).copy()

        # ── Frame placements ─────────────────────────────────────
        oMf_tool_a = data.oMf[fid_a].copy()
        oMf_tool_b = data.oMf[fid_b].copy()
        oMf_torso = data.oMf[fid_t].copy()

        # ── Torso frame Jacobian ─────────────────────────────────
        J_torso = pin.getFrameJacobian(
            model, data, fid_t, pin.LOCAL_WORLD_ALIGNED).copy()
        dJ_torso = pin.getFrameJacobianTimeVariation(
            model, data, fid_t, pin.LOCAL_WORLD_ALIGNED)
        Jdot_dq_torso = (dJ_torso @ v).copy()

        # ── Assemble state ───────────────────────────────────────
        self._state = RobotState(
            q=q.copy(),
            v=v.copy(),
            q_joints=q[self.joints_q_slice].copy(),
            dq_joints=v[self.joints_v_slice].copy(),
            q_torso=q[0:7].copy(),
            dq_torso=v[0:6].copy(),
            H=H,
            C=C,
            C_matrix=C_matrix,
            r_com=r_com,
            v_com=v_com,
            J_com=J_com,
            Jdot_dq_com=Jdot_dq_com,
            h_centroidal=h_centroidal,
            L_com=L_com,
            oMf_tool_a=oMf_tool_a,
            oMf_tool_b=oMf_tool_b,
            J_tool_a=J_tool_a,
            J_tool_b=J_tool_b,
            Jdot_dq_tool_a=Jdot_dq_tool_a,
            Jdot_dq_tool_b=Jdot_dq_tool_b,
            oMf_torso=oMf_torso,
            J_torso=J_torso,
            Jdot_dq_torso=Jdot_dq_torso,
            q_min=self._q_min,
            q_max=self._q_max,
            tau_max=self._tau_max,
            total_mass=self._total_mass,
        )
        return self._state

    @property
    def state(self) -> RobotState:
        """Most recently computed state."""
        if self._state is None:
            raise RuntimeError("Call update(q, v) first.")
        return self._state

    # ── Convenience methods ──────────────────────────────────────

    def compute_gjm(self, swing_arm: str) -> np.ndarray:
        """Generalized Jacobian Matrix for the swing arm.

        Accounts for base recoil: J_GJM = J_arm - J_base @ inv(H_bb) @ H_ba.
        When the swing arm accelerates, the base recoils. The GJM gives the
        net EE motion in the inertial (structure) frame.

        Parameters
        ----------
        swing_arm : 'a' or 'b'

        Returns
        -------
        J_GJM : ndarray (6, n_arm)
            Generalized Jacobian for the swing arm's tool frame.
        """
        s = self.state
        if swing_arm == 'a':
            J_full = s.J_tool_a
            arm_slice = self.arm_a_v_slice
        else:
            J_full = s.J_tool_b
            arm_slice = self.arm_b_v_slice

        J_base = J_full[:, :6]           # (6, 6)
        J_arm = J_full[:, arm_slice]     # (6, n_arm)
        H_bb = s.H[:6, :6]              # (6, 6) base inertia
        H_ba = s.H[:6, arm_slice]       # (6, n_arm) coupling

        # J_GJM = J_arm - J_base @ H_bb^{-1} @ H_ba
        return J_arm - J_base @ np.linalg.solve(H_bb, H_ba)

    def get_contact_jacobians(
        self, active_A: bool, active_B: bool
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return stacked contact Jacobian and J̇·q̇ for active contacts.

        Returns
        -------
        J_contacts : ndarray (6*nc, 18) or None
            Stacked Jacobians, A then B.
        Jdot_dq_contacts : ndarray (6*nc,) or None
            Stacked J̇·q̇ vectors.
        """
        s = self.state
        Js, Jds = [], []
        if active_A:
            Js.append(s.J_tool_a)
            Jds.append(s.Jdot_dq_tool_a)
        if active_B:
            Js.append(s.J_tool_b)
            Jds.append(s.Jdot_dq_tool_b)
        if not Js:
            return None, None
        return np.vstack(Js), np.concatenate(Jds)

    def neutral_configuration(self) -> np.ndarray:
        """Pinocchio neutral configuration (identity base, zero joints)."""
        return pin.neutral(self.model)
