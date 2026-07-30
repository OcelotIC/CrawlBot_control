"""
WholeBodyQP - Whole-body Quadratic Program for high-rate tracking.

Wraps HierarchicalQP with the whole-body dynamics of the crawling
space robot. Tracks centroidal references from Stage 1 (CentroidalNMPC)
while enforcing full multibody dynamics, actuator limits, contact
constraints, and momentum safety bounds.

Architecture:
    Stage 2 of the two-stage controller (see Chelikh et al., IEEE Access 2024).
    Instantaneous (single time-step) optimization, run at 1/dt_qp — 100 Hz on
    the frozen canonical config.

Decision variables:
    z = [q̈_t (6), q̈ (nq), λ (6·nc_max), τ_q (nq)]
    - q̈_t:  Torso (floating-base) acceleration
    - q̈:    Joint accelerations
    - λ:     Contact wrenches [f1, τ1, f2, τ2]
    - τ_q:   Joint torques (direct output to actuators)

Equality constraints:
    1. Full robot dynamics (Eq. VI-F.7):
       H_robot q̈_robot + C_robot = B_u τ_q + J_robot^T λ
    2. Contact acceleration constraint (bilateral, q̈_s ≈ 0):
       J_contact q̈_robot = -J̇_contact q̇_robot

Inequality constraints:
    1. Momentum safety (the "box"):
       h_min ≤ hw - dt·M_λ·λ ≤ h_max
    2. Joint torque limits:    τ_min ≤ τ_q ≤ τ_max
    3. Joint acceleration limits (from barrier functions)

Tasks (fully WEIGHTED — no null-space projection anywhere; at weight_ratio=1
the α magnitudes ARE the hierarchy and `priority=` is a nominal label):

    Single support (the two-task stack, Phase-2.1 — the canonical controller)
        T-MOM linear      α = ss_alpha_mom       (400)
        torso-pose 6-D    α = alpha_torso_pose   (2000)
        swing-EE 6-D      α = alpha_ee           (1000)
        posture           α = alpha_posture      (20)

    Double support
        joint-space settle, or — when ds_centroidal_mode — CoM 3-D +
        torso-angular 3-D + posture, with energy dissipation handled by the
        passivity *inequality* rather than a cost.
        internal-stress regularization on the welded-loop λ (alpha_lambda_int)

    All phases
        contact-wrench tracking (alpha_wrench), joint-torque minimization
        (alpha_torque), acceleration regularization (alpha_reg, the cost
        floor), h_w slack penalty (w_hw_slack)

    Canonical α ordering: torso-pose 2000 > EE 1000 > T-MOM 400 > posture 20
    > torque 5 > accel-reg 1 ≈ wrench 1. Keep torque-min ≳ 5× the accel-reg
    floor (see SimConfig / the CANONICAL-2p5 freeze).

Reference:
    Eq. (VI-F.1)-(VI-F.11) of the paper.
"""

import numpy as np
import pinocchio as pin
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict

from .hierarchical_qp import HierarchicalQP, QPSolveInfo
from .contact_phase import ContactConfig, skew, compute_momentum_map


def as_gain_matrix(g, n: int, name: str = 'gain') -> np.ndarray:
    """Normalize a task PD gain to an ``(n, n)`` matrix.

    Accepts the three forms the call sites actually use and rejects
    everything else:

    ==============  ======================================================
    input shape     result
    ==============  ======================================================
    scalar          ``g · I_n``
    ``(n,)``        ``diag(g)``  — per-axis gains given as a vector
    ``(n, n)``      ``g`` unchanged — per-axis gains given as ``np.diag(v)``
    ==============  ======================================================

    Why this exists (COM-GAIN-AUDIT). ``Kp_com`` is declared as an ``(n,)``
    default but every caller in the repository passes ``np.diag(...)`` — a
    matrix (``sim_loop._build_qp``, and the ``Misc/`` baselines, two of them
    anisotropic). The consumer used to be a bare ``np.diag(cfg.Kp_com)``,
    which is *shape-polymorphic in the worst way*: on a vector it BUILDS a
    matrix, but on a matrix it EXTRACTS the diagonal. The extracted diagonal
    then contracted with the error as ``diag_vec @ e``, producing a SCALAR
    that NumPy broadcast back over all n axes:

        applied  =  (Σᵢ kᵢ·eᵢ) · [1, …, 1]          rank one
        intended =  diag(k) · e                     rank n

    Both are ``(n,)`` and finite, so nothing raised and no test caught it —
    the canonical run executed the rank-one law on 8458/8458 solves. Axes are
    coupled, per-axis anisotropy is destroyed, and any error component
    orthogonal to ``[1, …, 1]`` is structurally invisible (81 % of the
    canonical SS CoM error, median). Normalizing here makes the diagonal
    contract explicit and the failure mode unrepresentable.

    Raises
    ------
    ValueError
        If the shape is not scalar, ``(n,)`` or ``(n, n)``, or if an
        ``(n, n)`` input carries non-negligible off-diagonal terms (a full
        matrix is not a supported gain form — flag it rather than silently
        using or dropping the coupling).
    """
    a = np.asarray(g, dtype=float)
    if a.ndim == 0:
        return float(a) * np.eye(n)
    if a.shape == (n,):
        return np.diag(a)
    if a.shape == (n, n):
        off = a - np.diag(np.diag(a))
        scale = max(float(np.abs(np.diag(a)).max()), 1.0)
        if float(np.abs(off).max()) > 1e-12 * scale:
            raise ValueError(
                f"{name}: off-diagonal gain terms are not supported "
                f"(max |off-diag| = {np.abs(off).max():.3e}); pass a scalar, "
                f"an ({n},) vector, or a diagonal ({n},{n}) matrix")
        return a
    raise ValueError(
        f"{name}: expected scalar, ({n},) or ({n},{n}); got shape {a.shape}")


@dataclass
class WholeBodyQPConfig:
    """Configuration for WholeBodyQP.

    Parameters
    ----------
    nq : int
        Number of active robot joints (excluding 6-DoF torso).
    nc_max : int
        Maximum number of contacts (typically 2 for dual-arm).
    """
    nq: int = 14                  # Joint DoFs (2 × 7-DoF arms)
    nc_max: int = 2               # Max contacts

    # QP method
    method: str = 'weighted'      # 'weighted' or 'strict'
    solver: str = 'qpoases'
    # HierarchicalQP weight ratio between priority levels.
    # The task stack is fully WEIGHTED — there is no null-space projection
    # anywhere in this file. At weight_ratio = 1.0 each task enters the cost
    # at its face-value α, so the α magnitudes ARE the hierarchy and the
    # `priority=` arguments are nominal labels only. Do not raise this: at
    # weight_ratio > 1 the priority integers start scaling the weights too
    # and the tuned α ratios stop meaning what they say.
    weight_ratio: float = 1.0

    # Task weights (Eq. VI-F.6)
    alpha_ee: float = 5e2         # End-effector tracking (swing arm)
    alpha_posture: float = 1e2    # Posture regulation
    alpha_wrench: float = 1e1     # Wrench tracking (from NMPC)
    alpha_torque: float = 1e0     # Joint torque minimization
    alpha_reg: float = 1e-2       # Acceleration regularization (lowest)
    alpha_lambda_int: float = 0.0  # Internal-stress regularization (DS only).
    # ‖(I − G⁺G)·λ‖² where G (6×12) maps λ → net wrench on the structure.
    # When >0 and both contacts active, drives the QP to pick the
    # minimum-net-wrench λ — equivalent to zero internal preload on
    # the welded loop. No effect in SS (only one contact ⇒ rank-6 λ,
    # null space is empty). Default 0 ⇒ bit-identical legacy.

    # DS centroidal-control mode: when True and settle_mode is True,
    # replaces the joint-vel-damping cost task (P1, weight 1000) with
    # CoM 3D + torso angular 3D tracking tasks at P1, plus posture at
    # P3. Energy dissipation is handled by the passivity *inequality*
    # (not a cost), which the sim_loop activates concurrently. Default
    # False ⇒ legacy behavior (joint-vel damping cost).
    ds_centroidal_mode: bool = False
    ds_alpha_com: float = 1e2
    ds_alpha_torso_ori: float = 2e2
    ds_alpha_posture: float = 5e1

    # ── SS two-task fully-weighted stack (Phase-2.1 reformulation) ──
    # THE canonical SS controller. T-MOM linear (ss_alpha_mom) + 6-D
    # torso-pose on J_torso (alpha_torso_pose, fed the raw TorsoPlanner
    # quintic+SLERP reference — NO δ-mapping) + swing-EE (alpha_ee) +
    # posture (alpha_posture). All WEIGHTED, with NO null-space projection,
    # so at weight_ratio=1 the α magnitudes ARE the hierarchy.
    #
    # CLEANUP-6/7 removed the stacks this superseded (legacy CoM / torso-6D
    # P1, the cooperative split, the Option D tube, the projected EE task,
    # T-MOM v1, the soft-CoM residual) and their config fields.
    # NOTE: SimConfig.use_m2_stack survives — it gates torso-reference
    # routing and DS passivity in sim_loop. Only the QP-side copy was removed.
    ss_two_task_mode: bool = False
    ss_alpha_mom: float = 5e2
    alpha_torso_pose: float = 1e3

    # ── Passivity constraint (DS only) ──
    alpha_passivity: float = 1.0  # Energy decay rate α [1/s] (α < 50 at 100 Hz)
    # Dock-floor audit: a CONSTANT positive work budget added to the
    # passivity RHS:  dqⱼᵀτ_q + 2α·T_kin ≤ W_budget  (vs strict ≤0). This is
    # a provisional relaxation knob to probe whether positive joint work
    # changes the achievable dock distance — NOT the envelope-coupled
    # Piste A. Default 0.0 ⇒ strict (unchanged).
    passivity_W_budget: float = 0.0
    # Piste A LOT B (FLAG 2): use the EXACT origin-referenced Ḣ_s in the
    # momentum-rate envelope box (|M_exact·λ| ≤ τ_w_max) instead of the
    # |M_λ·λ| proxy. Default False ⇒ proxy box (byte-identical).
    qp_envelope_exact: bool = False

    # ── M5: soft slack on momentum safety backup constraint ──
    # Replaces the hard inequality h_w(k+1) ∈ [h_min, h_max] with
    #   h_w(k+1) ≤ h_max + s_upper, s_upper ≥ 0
    #   h_w(k+1) ≥ h_min - s_lower, s_lower ≥ 0
    # plus a heavy penalty w_slack * (||s_upper||² + ||s_lower||²)
    # in the cost. When h_w is within the box, both slacks go to 0
    # (constraint is effectively hard). When h_w is physically over
    # the box, the slack allows the QP to remain feasible while the
    # penalty drives the solution toward the maximum corrective
    # wrench available.
    w_hw_slack: float = 8e2       # Quadratic penalty on hw slack (CANONICAL-2p5 / Add-5 freeze; was 1e4)

    # PD gains for CoM tracking (Eq. VI-F.4).
    # Contract (COM-GAIN-AUDIT): scalar, (3,) vector, or diagonal (3,3) matrix
    # — all three are normalized by `as_gain_matrix` at the point of use, so
    # `3.0`, `np.ones(3)*3` and `np.diag([3.,3.,3.])` are equivalent. Callers
    # in this repository pass `np.diag(...)`; a full (non-diagonal) matrix is
    # rejected rather than silently truncated.
    Kp_com: np.ndarray = field(default_factory=lambda: 100.0 * np.ones(3))
    Kd_com: np.ndarray = field(default_factory=lambda: 20.0 * np.ones(3))

    # PD gains for Torso 6D tracking [linear(3), angular(3)]
    Kp_torso: np.ndarray = field(default_factory=lambda: np.array([8., 8., 8., 5., 5., 5.]))
    Kd_torso: np.ndarray = field(default_factory=lambda: np.array([6., 6., 6., 4., 4., 4.]))

    # PD gains for end-effector tracking (swing arm, 6D: position + orientation)
    Kp_ee: np.ndarray = field(default_factory=lambda: 80.0 * np.ones(3))
    Kd_ee: np.ndarray = field(default_factory=lambda: 15.0 * np.ones(3))
    Kp_ee_ang: np.ndarray = field(default_factory=lambda: 5.0 * np.ones(3))
    Kd_ee_ang: np.ndarray = field(default_factory=lambda: 3.0 * np.ones(3))

    # PD gains for posture regulation
    Kp_posture: float = 25.0
    Kd_posture: float = 10.0

    # DS settling: joint velocity damping (torso/CoM tasks skipped)
    Kd_settle: float = 10.0           # Joint velocity damping [1/s]
    alpha_settle: float = 1e3         # Weight (high priority)

    # Actuator limits
    tau_max: np.ndarray = field(default_factory=lambda: 50.0 * np.ones(14))  # [Nm]

    # Joint acceleration limits
    qdd_max: float = 50.0         # [rad/s²]

    # Momentum safety
    dt_qp: float = 0.008          # QP time step [s] (125 Hz)

    # Contact wrench limits (HOTDOCK)
    f_max: float = 3000.0         # [N]
    tau_contact_max: float = 300.0  # [Nm]

    # Robot angular momentum constraints
    L_max: float = np.inf          # |L_robot| ≤ L_max [Nms]
    tau_w_max: float = np.inf      # |L̇_robot| ≤ τ_w_max [Nm]


class WholeBodyQP:
    """Whole-body QP for high-rate tracking of centroidal references.

    Parameters
    ----------
    config : WholeBodyQPConfig
        Problem configuration.
    """

    def __init__(self, config: Optional[WholeBodyQPConfig] = None):
        if config is None:
            config = WholeBodyQPConfig()
        self.config = config

        nq = config.nq
        nc_max = config.nc_max

        # Decision variable dimensions
        self._dim_qdd_t = 6                   # Torso acceleration
        self._dim_qdd = nq                    # Joint accelerations
        self._dim_lambda = 6 * nc_max         # Contact wrenches
        self._dim_tau = nq                     # Joint torques
        # M5: slack variables for the momentum safety backup constraint.
        # 3 for the upper bound (hw ≤ h_max + s_up) and 3 for the lower
        # bound (hw ≥ h_min - s_lo). Always allocated; they're zero-cost
        # (and zero-bound) whenever the hw constraint is not active.
        self._dim_slack_hw = 6

        self._n_vars = (self._dim_qdd_t + self._dim_qdd +
                        self._dim_lambda + self._dim_tau +
                        self._dim_slack_hw)

        # Variable index ranges in z
        self._idx = self._compute_indices()

        # Nominal posture (set by user, default: zero)
        self._q_nominal = np.zeros(nq)

        # hw-slack telemetry (always populated). After each QP solve we
        # record the norms of the upper/lower slack variables; non-zero
        # values mean the momentum-box safety constraint was active and
        # the w_hw_slack=1e4 cost was consuming QP budget. Each entry:
        # dict(label, slack_up_max, slack_lo_max, slack_norm). Set
        # diag_label externally per QP solve to tag entries with
        # sim context (e.g. "step02/SS").
        self.hw_slack_log: list = []
        self.diag_label: str = ''

    def set_nominal_posture(self, q_nom: np.ndarray) -> None:
        """Set the nominal joint posture for regularization.

        Parameters
        ----------
        q_nom : ndarray (nq,)
            Preferred joint configuration (e.g., arms mid-range).
        """
        self._q_nominal = np.asarray(q_nom).ravel()

    def solve(
        self,
        # Robot state
        dq_t: np.ndarray,
        q: np.ndarray,
        dq: np.ndarray,
        # References from Stage 1 (CentroidalNMPC)
        r_com_ref: np.ndarray,
        v_com_ref: np.ndarray,
        lambda_ref: np.ndarray,
        a_com_ff: np.ndarray,
        # Dynamics data (from Pinocchio)
        H_robot: np.ndarray,
        C_robot: np.ndarray,
        J_com: np.ndarray,
        Jdot_dq_com: np.ndarray,
        # Contact data
        contact_config: ContactConfig,
        J_contacts: Optional[np.ndarray] = None,
        Jdot_dq_contacts: Optional[np.ndarray] = None,
        # Momentum data
        hw_current: Optional[np.ndarray] = None,
        hw_min: Optional[np.ndarray] = None,
        hw_max: Optional[np.ndarray] = None,
        # Current CoM (for momentum map computation)
        r_com: Optional[np.ndarray] = None,
        # Current robot angular momentum
        L_com_current: Optional[np.ndarray] = None,
        # End-effector tracking (swing arm, optional, 6D)
        J_ee: Optional[np.ndarray] = None,         # (6, 6+nq) tool Jacobian
        Jdot_dq_ee: Optional[np.ndarray] = None,   # (6,) J̇_ee · q̇
        p_ee_ref: Optional[np.ndarray] = None,      # (3,) desired EE position
        R_ee_ref: Optional[np.ndarray] = None,      # (3,3) desired EE rotation
        v_ee_ref: Optional[np.ndarray] = None,      # (6,) desired EE twist [lin(3), ang(3)]
        a_ee_ff: Optional[np.ndarray] = None,       # (6,) feedforward EE accel [lin(3), ang(3)]
        p_ee: Optional[np.ndarray] = None,           # (3,) current EE position
        R_ee: Optional[np.ndarray] = None,           # (3,3) current EE rotation
        # Torso 6D tracking (optional, replaces CoM task when active)
        J_torso: Optional[np.ndarray] = None,       # (6, 6+nq) torso Jacobian
        Jdot_dq_torso: Optional[np.ndarray] = None, # (6,) J̇_torso · q̇
        p_torso: Optional[np.ndarray] = None,        # (3,) current torso position
        R_torso: Optional[np.ndarray] = None,        # (3,3) current torso rotation
        p_torso_ref: Optional[np.ndarray] = None,    # (3,) desired torso position
        R_torso_ref: Optional[np.ndarray] = None,    # (3,3) desired torso rotation
        v_torso_ref: Optional[np.ndarray] = None,    # (6,) desired torso twist [lin(3), ang(3)]
        a_torso_ff: Optional[np.ndarray] = None,     # (6,) feedforward torso accel [lin(3), ang(3)]
        # DS settling mode (skip torso/CoM, damp velocities)
        settle_mode: bool = False,
        # M2 passivity constraint: enforce dq_j^T*tau_q + 2*alpha*T <= 0
        # Intended to be ON during DS phase only (see §3.6, §5.7).
        passivity_active: bool = False,
        ds_centroidal_active: bool = False,
        # Piste A LOT A: per-tick passivity RHS budget (envelope-coupled,
        # computed by sim_loop). None ⇒ fall back to cfg.passivity_W_budget.
        passivity_W_budget: Optional[float] = None,
        # Chatter fix (J2): in settle_mode, override the wrench-tracking
        # weight α_wrench with a larger (strictly-convex) value. cfg.alpha_wrench
        # (≈0.01) is below the active-set solver's degeneracy tolerance, so when
        # the exact envelope box binds the solver alternates between the two
        # equal-norm saturating vertices (A≈−B) → period-2 chatter. A larger
        # weight makes the λ-cost strictly convex ⇒ unique min-norm wrench (the
        # midpoint), killing the limit cycle. None ⇒ cfg.alpha_wrench
        # (byte-identical). Passed ONLY from the inter-step settle loop, so SS
        # and the _step DWELL are untouched.
        settle_alpha_wrench: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, QPSolveInfo]:
        """Solve the whole-body QP.

        Parameters
        ----------
        dq_t : ndarray (6,)
            Torso twist [angular(3), linear(3)].
        q : ndarray (nq,)
            Joint positions.
        dq : ndarray (nq,)
            Joint velocities.
        r_com_ref : ndarray (3,)
            Reference CoM position from Stage 1.
        v_com_ref : ndarray (3,)
            Reference CoM velocity from Stage 1.
        lambda_ref : ndarray (12,)
            Reference contact wrenches from Stage 1.
        a_com_ff : ndarray (3,)
            Feedforward CoM acceleration = (1/m)Σf_j_ref.
        H_robot : ndarray (6+nq, 6+nq)
            Robot mass/inertia matrix (from Pinocchio CRBA).
        C_robot : ndarray (6+nq,)
            Coriolis/centrifugal + gravity terms (from Pinocchio RNEA).
        J_com : ndarray (3, 6+nq)
            Robot CoM Jacobian (from Pinocchio).
        Jdot_dq_com : ndarray (3,)
            J̇_com · q̇_robot (Pinocchio: computeAllTerms provides this).
        contact_config : ContactConfig
            Contact phase and positions.
        J_contacts : ndarray (6·nc, 6+nq), optional
            Stacked contact Jacobians (6 rows per active contact).
            Required if nc > 0.
        Jdot_dq_contacts : ndarray (6·nc,), optional
            J̇_contact · q̇_robot for active contacts.
            Required if nc > 0.
        hw_current : ndarray (3,), optional
            Current wheel momentum. Required for momentum constraint.
        hw_min, hw_max : ndarray (3,), optional
            Wheel momentum bounds. Required for momentum constraint.
        r_com : ndarray (3,), optional
            Current CoM position. Required for momentum constraint.

        Returns
        -------
        qdd_t : ndarray (6,)
            Optimal torso acceleration.
        qdd : ndarray (nq,)
            Optimal joint accelerations.
        lambda_opt : ndarray (12,)
            Optimal contact wrenches.
        tau_q : ndarray (nq,)
            Optimal joint torques (direct actuator commands).
        info : QPSolveInfo
            Solver information.
        """
        cfg = self.config
        idx = self._idx
        n = self._n_vars
        nq = cfg.nq

        # --- Build QP ---
        # weight_ratio comes from the config and is 1 on the canonical: the
        # stack is fully weighted, so each task enters at its face-value α
        # and the `priority=` labels below do not scale anything.
        # See WholeBodyQPConfig.weight_ratio.
        qp = HierarchicalQP(
            n_vars=n, method=cfg.method, solver=cfg.solver,
            weight_ratio=cfg.weight_ratio,
        )

        self._add_equality_constraints(
            qp, H_robot, C_robot, J_contacts, Jdot_dq_contacts, contact_config)

        hw_constraint_active = self._add_inequality_constraints(
            qp, H_robot, dq, r_com, hw_current, hw_min, hw_max,
            L_com_current, contact_config, passivity_active,
            passivity_W_budget)

        self._set_variable_bounds(qp, contact_config, hw_constraint_active)

        # ============================================================ #
        #  TASKS                                                        #
        # ============================================================ #

        dq_robot = np.concatenate([dq_t, dq])
        A_com, b_com = self._com_task_rows(
            J_com, Jdot_dq_com, dq_robot, r_com, r_com_ref, v_com_ref,
            a_com_ff)

        # ── SS: the two-task stack — THE canonical single-support controller ──
        # T-MOM linear (ss_alpha_mom) + 6-D torso-pose on J_torso
        # (alpha_torso_pose; TorsoPlanner quintic+SLERP reference, NO δ) +
        # swing-EE (alpha_ee) + posture (alpha_posture). All WEIGHTED with NO
        # null-space projection (strict-P1 abandoned, lecture B). Supersedes
        # the cooperative split; the legacy torso/EE/CoM/posture blocks below
        # are gated off via `_two_task`. weight_ratio=1 ⇒ the α's set the
        # hierarchy directly (momentum+EE high, torso-pose just below, posture
        # low). Constraints (dynamics, contact, momentum box) added earlier.
        _two_task = cfg.ss_two_task_mode and not settle_mode
        if _two_task:
            # (1) Momentum task — linear CMM rows (CoM-Jacobian form).
            qp.add_task(A_com, b_com, cfg.ss_alpha_mom, priority=2)
            # (2) 6-D torso-pose task on J_torso (position + orientation).
            if J_torso is not None and p_torso_ref is not None:
                Kp_t = as_gain_matrix(cfg.Kp_torso, 6, 'Kp_torso')
                Kd_t = as_gain_matrix(cfg.Kd_torso, 6, 'Kd_torso')
                v_t_act = J_torso @ dq_robot
                p_t = p_torso if p_torso is not None else np.zeros(3)
                R_t = R_torso if R_torso is not None else np.eye(3)
                R_rt = R_torso_ref if R_torso_ref is not None else np.eye(3)
                e6 = np.concatenate([p_torso_ref - p_t, pin.log3(R_t.T @ R_rt)])
                v_rt = v_torso_ref if v_torso_ref is not None else np.zeros(6)
                a_ft = a_torso_ff if a_torso_ff is not None else np.zeros(6)
                a_t_des = a_ft + Kp_t @ e6 + Kd_t @ (v_rt - v_t_act)
                jdq_t = (Jdot_dq_torso if Jdot_dq_torso is not None
                         else np.zeros(6))
                A_tp = np.zeros((6, n))
                A_tp[:, idx['qdd_t'][0]: idx['qdd_t'][1]] = J_torso[:, :6]
                A_tp[:, idx['qdd'][0]: idx['qdd'][1]] = J_torso[:, 6:]
                qp.add_task(A_tp, a_t_des - jdq_t,
                            cfg.alpha_torso_pose, priority=2)
            # (3) Swing-EE 6-D task (direct, no projection).
            if J_ee is not None and p_ee_ref is not None:
                jdq_ee = Jdot_dq_ee if Jdot_dq_ee is not None else np.zeros(6)
                v_ee_act = J_ee @ dq_robot
                p_ea = p_ee if p_ee is not None else np.zeros(3)
                R_ea = R_ee if R_ee is not None else np.eye(3)
                R_er = R_ee_ref if R_ee_ref is not None else np.eye(3)
                e6e = np.concatenate([p_ee_ref - p_ea, pin.log3(R_ea.T @ R_er)])
                Kp_e = np.diag(np.concatenate([cfg.Kp_ee, cfg.Kp_ee_ang]))
                Kd_e = np.diag(np.concatenate([cfg.Kd_ee, cfg.Kd_ee_ang]))
                v_re = v_ee_ref if v_ee_ref is not None else np.zeros(6)
                a_fe = a_ee_ff if a_ee_ff is not None else np.zeros(6)
                a_e_des = a_fe + Kp_e @ e6e + Kd_e @ (v_re - v_ee_act)
                A_ee_task = np.zeros((6, n))
                A_ee_task[:, idx['qdd_t'][0]: idx['qdd_t'][1]] = J_ee[:, :6]
                A_ee_task[:, idx['qdd'][0]: idx['qdd'][1]] = J_ee[:, 6:]
                qp.add_task(A_ee_task, a_e_des - jdq_ee, cfg.alpha_ee, priority=2)
            # (4) Posture — redundancy resolution (low weight, unprojected).
            A_post = np.zeros((nq, n))
            A_post[:, idx['qdd'][0]: idx['qdd'][1]] = np.eye(nq)
            qp.add_task(A_post,
                        cfg.Kp_posture * (self._q_nominal - q) - cfg.Kd_posture * dq,
                        cfg.alpha_posture, priority=3)

        # ── Posture regulation — SS (when the two-task stack is off) and DS ──
        # q̈_posture = Kp_post (q_nom - q) + Kd_post (0 - dq)
        # Skipped in settle_mode (legacy joint-vel-damping path) because
        # the settle task already dampens velocities and posture would
        # interfere (T10 regression). Re-enabled when ds_centroidal_mode
        # since the joint-vel cost is gone — posture is needed to
        # constrain the 2 arm-null-space DOFs.
        _posture_in_ds = (settle_mode and cfg.ds_centroidal_mode
                          and ds_centroidal_active)
        if ((not settle_mode) or _posture_in_ds) and not _two_task:
            qdd_posture = (cfg.Kp_posture * (self._q_nominal - q) -
                           cfg.Kd_posture * dq)

            A_posture = np.zeros((nq, n))
            A_posture[:, idx['qdd'][0]: idx['qdd'][1]] = np.eye(nq)
            b_posture = qdd_posture

            # Added unprojected — the stack is weighted, not projected.
            _post_w = (cfg.ds_alpha_posture if _posture_in_ds
                       else cfg.alpha_posture)
            qp.add_task(A_posture, b_posture,
                        _post_w, priority=3)

        # ── DS: joint-space settle (damp all velocities to zero) ──
        # In settle mode, torso/CoM tasks are skipped (they conflict with
        # the constrained equilibrium). This task drives all joint velocities
        # to zero via pure damping. No position term — with 8 DOF remaining
        # (6 base + 2 redundant from 7-DOF arms) and welds constraining the
        # EEs, the system can only stop at the current configuration.
        #
        # When cfg.ds_centroidal_mode is True, this cost task is REPLACED
        # by CoM + torso-ori tracking at P1 (below), with energy
        # dissipation handled by the passivity inequality (sim_loop
        # activates passivity_active=True concurrently).
        if settle_mode and not (cfg.ds_centroidal_mode and ds_centroidal_active):
            A_settle = np.zeros((nq, n))
            A_settle[:, idx['qdd'][0]: idx['qdd'][1]] = np.eye(nq)
            b_settle = -cfg.Kd_settle * dq
            qp.add_task(A_settle, b_settle, cfg.alpha_settle, priority=1)

        # ── DS: centroidal tasks — CoM 3-D + torso-angular 3-D ──
        # 6-D centroidal tracking during DS: CoM 3D + torso angular 3D.
        # The captured Stage 3 reference (r_torso_ref, R_torso_ref, r_com_ref)
        # becomes load-bearing here. Energy dissipation is enforced by
        # the passivity inequality (added below if passivity_active).
        # The NMPC's planned a_com_ff is already in a_com_des.
        if settle_mode and cfg.ds_centroidal_mode and ds_centroidal_active:
            # CoM 3D task — reuses a_com_des / A_com computed earlier
            # (always-on at the top of the task block).
            qp.add_task(A_com, b_com, cfg.ds_alpha_com, priority=1)

            # Torso angular 3D task — reuse the angular rows of the
            # 6D torso target. Note torso_task_active is gated False
            # by settle_mode, so a_torso_des / A_torso_full aren't
            # computed; recompute the angular part here.
            if (J_torso is not None and R_torso is not None
                    and R_torso_ref is not None):
                e_ori = pin.log3(R_torso.T @ R_torso_ref)
                v_torso_actual = J_torso @ dq_robot  # (6,)
                v_ref_t = (v_torso_ref if v_torso_ref is not None
                           else np.zeros(6))
                a_torso_ang_des = (
                    as_gain_matrix(cfg.Kp_torso, 6, 'Kp_torso')[3:, 3:] @ e_ori
                    + as_gain_matrix(cfg.Kd_torso, 6, 'Kd_torso')[3:, 3:] @ (
                        v_ref_t[3:] - v_torso_actual[3:]))
                jdq = (Jdot_dq_torso[3:] if Jdot_dq_torso is not None
                       else np.zeros(3))
                A_torso_ang = np.zeros((3, n))
                A_torso_ang[:, idx['qdd_t'][0]: idx['qdd_t'][1]] = J_torso[3:, :6]
                A_torso_ang[:, idx['qdd'][0]: idx['qdd'][1]] = J_torso[3:, 6:]
                b_torso_ang = a_torso_ang_des - jdq
                qp.add_task(A_torso_ang, b_torso_ang,
                            cfg.ds_alpha_torso_ori, priority=1)

        # ── Contact-wrench tracking (all phases) ──
        A_wrench = np.zeros((self._dim_lambda, n))
        A_wrench[:, idx['lambda'][0]: idx['lambda'][1]] = np.eye(self._dim_lambda)
        b_wrench = lambda_ref.copy()

        # Chatter fix: settle-only α_wrench boost (strictly convex ⇒ unique
        # min-norm λ; breaks the period-2 active-set degeneracy). Localized to
        # settle_mode and only when the caller passes an override.
        _aw = (settle_alpha_wrench
               if (settle_mode and settle_alpha_wrench is not None)
               else cfg.alpha_wrench)
        qp.add_task(A_wrench, b_wrench, _aw, priority=4)

        # ── DS: internal-stress regularization on the welded-loop λ ──
        # In DS both grippers are welded → contact-wrench space is 12-D
        # while only 6-D acts on the robot CoM dynamics. The remaining
        # 6-D subspace is internal stress: combinations of (f_A, τ_A,
        # f_B, τ_B) producing zero net wrench on the robot but a
        # non-zero couple on the structure body. The QP has no other
        # cost on this subspace (α_wrench=0.01 with λ_ref weak), so
        # it picks an arbitrary internal-stress component; that
        # arbitrariness is the suspected driver of the welded-redundancy
        # drift observed in trailing-DS settle.
        #
        # G (6×12) maps λ → net wrench on the structure body (about
        # struct CoM ≈ world origin at small struct rotation, which is
        # our regime). (I − G⁺G) projects onto the 6-D internal-stress
        # null space. We minimize the projection at low priority.
        #
        # Gated on cfg.alpha_lambda_int > 0 AND exactly two active
        # contacts. In SS there's no internal-stress subspace (rank
        # of G_struct is 6 = dim(λ)).
        if (cfg.alpha_lambda_int > 0.0
                and contact_config.nc == 2
                and contact_config.active_contacts[0]
                and contact_config.active_contacts[1]):
            r_CA = contact_config.r_contact_A
            r_CB = contact_config.r_contact_B
            G = np.zeros((6, 12))
            G[0:3, 0:3] = np.eye(3)     # f_A → net force
            G[0:3, 6:9] = np.eye(3)     # f_B → net force
            G[3:6, 0:3] = skew(r_CA)    # r_CA × f_A → torque about origin
            G[3:6, 3:6] = np.eye(3)     # τ_A → torque
            G[3:6, 6:9] = skew(r_CB)    # r_CB × f_B → torque about origin
            G[3:6, 9:12] = np.eye(3)    # τ_B → torque
            G_pinv = np.linalg.pinv(G, rcond=1e-8)
            P_int = np.eye(12) - G_pinv @ G  # projector onto internal-stress

            A_lint = np.zeros((12, n))
            A_lint[:, idx['lambda'][0]: idx['lambda'][1]] = P_int
            b_lint = np.zeros(12)
            qp.add_task(A_lint, b_lint,
                        cfg.alpha_lambda_int, priority=4)

        # ── Joint-torque minimization (all phases) ──
        A_torque = np.zeros((nq, n))
        A_torque[:, idx['tau'][0]: idx['tau'][1]] = np.eye(nq)
        b_torque = np.zeros(nq)

        qp.add_task(A_torque, b_torque, cfg.alpha_torque, priority=5)

        # ── Acceleration regularization — the cost floor (all phases) ──
        A_reg = np.zeros((6 + nq, n))
        A_reg[:6, idx['qdd_t'][0]: idx['qdd_t'][1]] = np.eye(6)
        A_reg[6:, idx['qdd'][0]: idx['qdd'][1]] = np.eye(nq)
        b_reg = np.zeros(6 + nq)

        qp.add_task(A_reg, b_reg, cfg.alpha_reg, priority=6)

        # ── h_w slack penalty (momentum-box softening) ──
        # Drive the slack variables to zero as fast as the actuator
        # limits allow. NOTE: at weight_ratio=1 the priority integer is
        # inert — the α magnitudes alone set the hierarchy, so this task
        # ranks by w_hw_slack (800), below torso/EE. The slacks are only
        # active when the hw safety box itself is violated.
        if cfg.w_hw_slack > 0:
            A_slack = np.zeros((6, n))
            A_slack[0: 3, idx['slack_hw_up'][0]: idx['slack_hw_up'][1]] = np.eye(3)
            A_slack[3: 6, idx['slack_hw_lo'][0]: idx['slack_hw_lo'][1]] = np.eye(3)
            b_slack = np.zeros(6)
            qp.add_task(A_slack, b_slack, cfg.w_hw_slack, priority=1)

        # ============================================================ #
        #  SOLVE                                                        #
        # ============================================================ #

        z_opt, info = qp.solve()

        # --- Extract solution ---
        qdd_t_opt = z_opt[idx['qdd_t'][0]: idx['qdd_t'][1]]
        qdd_opt = z_opt[idx['qdd'][0]: idx['qdd'][1]]
        lambda_opt = z_opt[idx['lambda'][0]: idx['lambda'][1]]
        tau_q_opt = z_opt[idx['tau'][0]: idx['tau'][1]]

        # --- hw_slack telemetry ---
        # Capture the 3-vector slacks on the upper/lower momentum-box
        # constraint. Non-zero values indicate the soft-slack cost
        # (w_hw_slack, default 1e4) was active and consuming QP budget.
        try:
            s_up = z_opt[idx['slack_hw_up'][0]: idx['slack_hw_up'][1]]
            s_lo = z_opt[idx['slack_hw_lo'][0]: idx['slack_hw_lo'][1]]
            self.hw_slack_log.append({
                'label': self.diag_label,
                's_up_max': float(np.max(np.abs(s_up))),
                's_lo_max': float(np.max(np.abs(s_lo))),
                's_norm': float(np.linalg.norm(np.concatenate([s_up, s_lo]))),
            })
        except (KeyError, IndexError, ValueError):
            pass

        # Torso-task debug capture removed with the legacy SS stack: every
        # quantity it reported (a_torso_des, e_pos/e_ori, v_ref_t, a_ff_t)
        # existed only on the pre-two-task path. Consumers read it via
        # getattr(..., None), so leaving it None is compatible.
        self.last_torso_debug = None

        return qdd_t_opt, qdd_opt, lambda_opt, tau_q_opt, info

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    #  QP assembly helpers                                                 #
    #                                                                      #
    #  Extracted verbatim from solve() (CLEANUP-11). They only APPEND to   #
    #  the `qp` accumulator — no shared state is mutated — so the call     #
    #  order in solve() is the assembly order.                             #
    # ------------------------------------------------------------------ #

    def _add_equality_constraints(
        self, qp, H_robot, C_robot, J_contacts, Jdot_dq_contacts,
        contact_config,
    ) -> None:
        """Full robot dynamics + the bilateral contact-acceleration constraint."""
        cfg = self.config
        idx = self._idx
        n = self._n_vars
        nq = cfg.nq
        n_robot = 6 + nq  # dimension of q̈_robot = [q̈_t; q̈]


        # 1. Full robot dynamics: H q̈_robot + C = B_u τ_q + J_robot^T λ
        #    → [H, -J_robot^T, -B_u] [q̈_robot; λ; τ_q] = -C
        #
        # Decision vector z = [q̈_t, q̈, λ, τ_q]
        # q̈_robot = [q̈_t; q̈] are the first (6+nq) components

        A_dyn = np.zeros((n_robot, n))

        # H_robot @ q̈_robot
        A_dyn[:, idx['qdd_t'][0]: idx['qdd_t'][1]] = H_robot[:, :6]
        A_dyn[:, idx['qdd'][0]: idx['qdd'][1]] = H_robot[:, 6:]

        # -J_robot^T @ λ  (only active contacts contribute)
        n_lambda = self._dim_lambda
        J_robot_T = np.zeros((n_robot, n_lambda))
        if J_contacts is not None and J_contacts.size > 0:
            # J_contacts is (6·nc_active, n_robot) for active contacts
            # We need to place it in the right columns of the full λ vector
            contact_idx = 0
            for j in range(cfg.nc_max):
                if contact_config.active_contacts[j]:
                    rows = slice(contact_idx * 6, (contact_idx + 1) * 6)
                    J_robot_T[:, j * 6: (j + 1) * 6] = J_contacts[rows, :].T
                    contact_idx += 1

        A_dyn[:, idx['lambda'][0]: idx['lambda'][1]] = -J_robot_T

        # -B_u @ τ_q  (B_u = [0_{6×nq}; I_nq])
        A_dyn[6:, idx['tau'][0]: idx['tau'][1]] = -np.eye(nq)

        b_dyn = -C_robot

        qp.add_equality_constraint(A_dyn, b_dyn)

        # 2. Contact acceleration constraint: J_contact q̈_robot = -J̇_contact q̇_robot
        if J_contacts is not None and J_contacts.size > 0:
            nc_active_rows = J_contacts.shape[0]
            A_contact = np.zeros((nc_active_rows, n))
            A_contact[:, idx['qdd_t'][0]: idx['qdd_t'][1]] = J_contacts[:, :6]
            A_contact[:, idx['qdd'][0]: idx['qdd'][1]] = J_contacts[:, 6:]

            b_contact = np.zeros(nc_active_rows)
            if Jdot_dq_contacts is not None:
                b_contact = -Jdot_dq_contacts

            qp.add_equality_constraint(A_contact, b_contact)

    def _add_inequality_constraints(
        self, qp, H_robot, dq, r_com, hw_current, hw_min, hw_max,
        L_com_current, contact_config, passivity_active, passivity_W_budget,
    ) -> bool:
        """Momentum box (+ h_w slack), L_com box, Ḣ_s rate box, passivity.

        Returns ``hw_constraint_active`` — the bounds block needs it to decide
        whether the slack variables are free or pinned to zero.
        """
        cfg = self.config
        idx = self._idx
        n = self._n_vars


        # 1. Momentum safety (M5 soft slack): h_min ≤ hw - dt·M_λ·λ ≤ h_max
        #    With slack s_up, s_lo ≥ 0:
        #    Upper: -dt·M_λ·λ - s_up ≤ h_max - hw
        #    Lower:  dt·M_λ·λ - s_lo ≤ hw - h_min
        #    Bound: s_up, s_lo ≥ 0  (set later in the bounds block)
        #    Cost: w_hw_slack * (||s_up||² + ||s_lo||²)  (added as a task)
        #
        # When h_w is within the box, both slacks go to 0 and the
        # constraint is effectively hard. When physical h_w is beyond
        # the box (e.g., during actuator saturation), the slack lets
        # the QP stay feasible while the heavy penalty drives the
        # slack toward zero — giving the maximum corrective wrench.
        hw_constraint_active = (
            hw_current is not None and hw_min is not None
            and r_com is not None)
        if hw_constraint_active:
            M_lambda = compute_momentum_map(r_com, contact_config)

            A_mom_upper = np.zeros((3, n))
            A_mom_upper[:, idx['lambda'][0]: idx['lambda'][1]] = -cfg.dt_qp * M_lambda
            A_mom_upper[:, idx['slack_hw_up'][0]: idx['slack_hw_up'][1]] = -np.eye(3)
            b_mom_upper = hw_max - hw_current

            A_mom_lower = np.zeros((3, n))
            A_mom_lower[:, idx['lambda'][0]: idx['lambda'][1]] = cfg.dt_qp * M_lambda
            A_mom_lower[:, idx['slack_hw_lo'][0]: idx['slack_hw_lo'][1]] = -np.eye(3)
            b_mom_lower = hw_current - hw_min

            qp.add_inequality_constraint(
                np.vstack([A_mom_upper, A_mom_lower]),
                np.concatenate([b_mom_upper, b_mom_lower])
            )

            # 2. Robot angular momentum box: |L_com + dt·M_λ·λ| ≤ L_max
            if np.isfinite(cfg.L_max) and L_com_current is not None:
                A_L_upper = np.zeros((3, n))
                A_L_upper[:, idx['lambda'][0]: idx['lambda'][1]] = cfg.dt_qp * M_lambda
                b_L_upper = cfg.L_max * np.ones(3) - L_com_current

                A_L_lower = np.zeros((3, n))
                A_L_lower[:, idx['lambda'][0]: idx['lambda'][1]] = -cfg.dt_qp * M_lambda
                b_L_lower = cfg.L_max * np.ones(3) + L_com_current

                qp.add_inequality_constraint(
                    np.vstack([A_L_upper, A_L_lower]),
                    np.concatenate([b_L_upper, b_L_lower])
                )

            # 3. Momentum rate box: |Ḣ_s| ≤ τ_w_max.
            # Proxy (default): Ḣ_s ≈ M_λ·λ (lever from robot CoM — omits the
            # orbital term r_com×Σf). Piste A LOT B (qp_envelope_exact): use
            # the EXACT origin-referenced Ḣ_s = M_exact·λ, M_exact the
            # momentum map with levers from O_s (= compute_momentum_map at
            # r_com=0). Stays LINEAR in λ (r_com is a per-tick parameter).
            if np.isfinite(cfg.tau_w_max):
                if cfg.qp_envelope_exact:
                    M_env = compute_momentum_map(np.zeros(3), contact_config)
                else:
                    M_env = M_lambda
                A_Ld_upper = np.zeros((3, n))
                A_Ld_upper[:, idx['lambda'][0]: idx['lambda'][1]] = M_env
                b_Ld_upper = cfg.tau_w_max * np.ones(3)

                A_Ld_lower = np.zeros((3, n))
                A_Ld_lower[:, idx['lambda'][0]: idx['lambda'][1]] = -M_env
                b_Ld_lower = cfg.tau_w_max * np.ones(3)

                qp.add_inequality_constraint(
                    np.vstack([A_Ld_upper, A_Ld_lower]),
                    np.concatenate([b_Ld_upper, b_Ld_lower])
                )

        # 4. Passivity constraint (M2, DS only): dq_j^T * τ_q + 2α·T ≤ 0
        #
        # Enforces exponential kinetic-energy decay T(t) ≤ T(t0)·exp(-2α·t)
        # during double-support. T = 0.5 * dq_j^T * H_jj * dq_j uses the
        # joint block of the mass matrix (the full v=[dq_t; dq] is
        # constrained by welds at both EEs so only the joint kinetic
        # energy matters here). Linear in τ_q only:
        #
        #     [0, 0, 0, dq]^T · z ≤ -2α·T
        if passivity_active and cfg.alpha_passivity > 0:
            H_jj = H_robot[6:, 6:]
            T_kin = 0.5 * float(dq @ H_jj @ dq)
            A_pass = np.zeros((1, n))
            A_pass[0, idx['tau'][0]: idx['tau'][1]] = dq
            # + W_budget relaxes the strict ≤0 RHS to allow bounded positive
            # joint work. Piste A LOT A passes an envelope-coupled per-tick
            # W_budget (kwarg); else the constant cfg.passivity_W_budget
            # (dock-floor audit; default 0 ⇒ strict, byte-identical).
            W_budget = (passivity_W_budget if passivity_W_budget is not None
                        else cfg.passivity_W_budget)
            b_pass = np.array([-2.0 * cfg.alpha_passivity * T_kin + W_budget])
            qp.add_inequality_constraint(A_pass, b_pass)

        return hw_constraint_active

    def _set_variable_bounds(self, qp, contact_config,
                             hw_constraint_active: bool) -> None:
        """Box bounds on q̈, τ_q, the contact wrenches and the h_w slacks."""
        cfg = self.config
        idx = self._idx
        n = self._n_vars


        lb = np.full(n, -np.inf)
        ub = np.full(n, np.inf)

        # Joint acceleration bounds
        lb[idx['qdd'][0]: idx['qdd'][1]] = -cfg.qdd_max
        ub[idx['qdd'][0]: idx['qdd'][1]] = cfg.qdd_max

        # Joint torque bounds
        lb[idx['tau'][0]: idx['tau'][1]] = -cfg.tau_max
        ub[idx['tau'][0]: idx['tau'][1]] = cfg.tau_max

        # Contact wrench bounds (zero for inactive contacts)
        for j in range(cfg.nc_max):
            s = idx['lambda'][0] + j * 6
            if contact_config.active_contacts[j]:
                lb[s: s + 3] = -cfg.f_max
                ub[s: s + 3] = cfg.f_max
                lb[s + 3: s + 6] = -cfg.tau_contact_max
                ub[s + 3: s + 6] = cfg.tau_contact_max
            else:
                lb[s: s + 6] = 0.0
                ub[s: s + 6] = 0.0

        # M5 hw slack bounds: non-negativity when constraint is active,
        # pinned to 0 otherwise.
        if hw_constraint_active:
            lb[idx['slack_hw_up'][0]: idx['slack_hw_up'][1]] = 0.0
            ub[idx['slack_hw_up'][0]: idx['slack_hw_up'][1]] = np.inf
            lb[idx['slack_hw_lo'][0]: idx['slack_hw_lo'][1]] = 0.0
            ub[idx['slack_hw_lo'][0]: idx['slack_hw_lo'][1]] = np.inf
        else:
            lb[idx['slack_hw_up'][0]: idx['slack_hw_up'][1]] = 0.0
            ub[idx['slack_hw_up'][0]: idx['slack_hw_up'][1]] = 0.0
            lb[idx['slack_hw_lo'][0]: idx['slack_hw_lo'][1]] = 0.0
            ub[idx['slack_hw_lo'][0]: idx['slack_hw_lo'][1]] = 0.0

        qp.set_bounds(lb, ub)

    def _com_task_rows(self, J_com, Jdot_dq_com, dq_robot, r_com, r_com_ref,
                       v_com_ref, a_com_ff):
        """Build the CoM cost row ``J_com @ qdd = a_com_des - J̇dq``.

        This is the linear centroidal-momentum (T-MOM) task in CoM-Jacobian
        form. Returned as ``(A_com, b_com)`` and consumed by both the SS
        two-task stack and the DS centroidal task.
        """
        cfg = self.config
        idx = self._idx
        n = self._n_vars

        r_com_actual = r_com if r_com is not None else np.zeros(3)
        v_com_actual = J_com @ dq_robot
        # COM-GAIN-AUDIT: `as_gain_matrix`, not `np.diag`. Every caller hands
        # these over as `np.diag(...)` (a matrix), on which a bare `np.diag`
        # EXTRACTS the diagonal instead of building it — collapsing the PD law
        # to rank one. See `as_gain_matrix` for the full derivation.
        Kp_com_mat = as_gain_matrix(cfg.Kp_com, 3, 'Kp_com')
        Kd_com_mat = as_gain_matrix(cfg.Kd_com, 3, 'Kd_com')
        a_com_des = (a_com_ff
                     + Kp_com_mat @ (r_com_ref - r_com_actual)
                     + Kd_com_mat @ (v_com_ref - v_com_actual))
        A_com = np.zeros((3, n))
        A_com[:, idx['qdd_t'][0]: idx['qdd_t'][1]] = J_com[:, :6]
        A_com[:, idx['qdd'][0]: idx['qdd'][1]] = J_com[:, 6:]
        b_com = a_com_des - Jdot_dq_com

        return A_com, b_com

    def _compute_indices(self) -> Dict[str, Tuple[int, int]]:
        """Compute start/end indices for each variable block in z.

        Layout: [qdd_t(6), qdd(nq), lambda(6*nc_max), tau(nq),
                 slack_hw_upper(3), slack_hw_lower(3)]
        """
        nq = self.config.nq
        n_lambda = 6 * self.config.nc_max

        s = 0
        idx = {}

        idx['qdd_t'] = (s, s + 6);           s += 6
        idx['qdd'] = (s, s + nq);            s += nq
        idx['lambda'] = (s, s + n_lambda);    s += n_lambda
        idx['tau'] = (s, s + nq);             s += nq
        idx['slack_hw_up'] = (s, s + 3);     s += 3
        idx['slack_hw_lo'] = (s, s + 3);     s += 3

        assert s == self._n_vars
        return idx

    @property
    def n_vars(self) -> int:
        return self._n_vars

    @property
    def variable_indices(self) -> Dict[str, Tuple[int, int]]:
        """Index ranges for each variable block."""
        return self._idx

    def __repr__(self) -> str:
        return (
            f"WholeBodyQP(nq={self.config.nq}, nc_max={self.config.nc_max}, "
            f"n_vars={self._n_vars}, method='{self.config.method}')"
        )
