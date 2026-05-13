"""
WholeBodyQP - Whole-body Quadratic Program for high-rate tracking.

Wraps HierarchicalQP with the whole-body dynamics of the crawling
space robot. Tracks centroidal references from Stage 1 (CentroidalNMPC)
while enforcing full multibody dynamics, actuator limits, contact
constraints, and momentum safety bounds.

Architecture:
    Stage 2 of the two-stage controller (see Chelikh et al., IEEE Access 2024).
    Runs at 125 Hz with instantaneous (single time-step) optimization.

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

Tasks (weighted hierarchy, Eq. VI-F.6):
    Priority 1 (α=10³): CoM tracking
    Priority 2 (α=10²): Posture regulation
    Priority 3 (α=10¹): Contact wrench tracking (from NMPC)
    Priority 4 (α=10⁰): Joint torque minimization
    Priority 5 (α=10⁻²): Acceleration regularization

Reference:
    Eq. (VI-F.1)-(VI-F.11) of the paper.
"""

import numpy as np
import pinocchio as pin
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any

from .hierarchical_qp import HierarchicalQP, QPSolveInfo
from .contact_phase import ContactConfig, ContactPhase, skew, compute_momentum_map


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
    # With the M2 null-space-projected task stack, task isolation is
    # provided geometrically by the projections N_torso, N_torso·N_ee,
    # etc. — NOT by weight scaling. weight_ratio = 1.0 therefore lets
    # every task use its face-value α. Legacy (non-M2) users that
    # rely on priority-by-scaling should override this to ~1000.
    weight_ratio: float = 1.0

    # Task weights (Eq. VI-F.6)
    alpha_com: float = 1e3        # Legacy CoM tracking as primary task (weighted hierarchy)
    alpha_torso: float = 0.0      # Torso 6D tracking (P1 in M2 stack)
    alpha_ee: float = 5e2         # End-effector tracking (swing arm)
    alpha_posture: float = 1e2    # Posture regulation
    alpha_wrench: float = 1e1     # Wrench tracking (from NMPC)
    alpha_torque: float = 1e0     # Joint torque minimization
    alpha_reaction: float = 0.0   # Reaction null-space (minimize base disturbance from swing arm)
    alpha_reg: float = 1e-2       # Acceleration regularization (lowest)

    # ── M2 stack: torso P1 + EE null-space P2 + posture P3 + soft CoM ──
    use_m2_stack: bool = False    # Enable M2 reworked task stack (§5.6-5.7)
    alpha_com_soft: float = 5.0   # Soft CoM residual weight (1-10 per spec)
    ee_null_space: bool = False   # Null-space project EE task against torso task

    # ── Option D: torso linear soft tube ────────────────────────
    # See SimConfig docstring for the full rationale. r_tube > 0 enables
    # the split. Angular torso stays at α_torso; linear torso is gated by
    # ||p_torso - p_ref|| > r_tube with cost weight
    # w_tube_lin * (|e|^2 - r_tube^2). Inside the tube the EE / posture
    # null-space projections only need to avoid the 3D angular Jacobian.
    r_tube: float = 0.0           # [m] tube radius; 0 = legacy 6D equality
    w_tube_lin: float = 50.0      # cost scale on (|e|² - r²) when violated

    # ── Cooperative-arms task stack (deviation from spec §4.3) ──
    # See SimConfig.cooperative_arms_mode for the full rationale.
    # When True, torso angular is P1 (alpha_torso_ang); torso linear +
    # EE 6D are co-equal P2 weighted-LS (alpha_torso_lin, alpha_ee).
    # Posture P3 is projected against the combined P1+P2 stack with
    # rcond=1e-4 on the combined 12×n pinv (looser than the 1e-8 used
    # on individual task pinvs) to handle the wider conditioning of
    # the stacked Jacobian. r_tube is ignored when cooperative_arms_mode
    # is True (split is structural, not violation-gated).
    cooperative_arms_mode: bool = False
    alpha_torso_ang: float = 5e2
    alpha_torso_lin: float = 5e2

    # ── Passivity constraint (DS only) ──
    alpha_passivity: float = 1.0  # Energy decay rate α [1/s] (α < 50 at 100 Hz)

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
    w_hw_slack: float = 1e4       # Quadratic penalty on hw slack

    # PD gains for CoM tracking (Eq. VI-F.4)
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

        # Option D tube telemetry (populated by solve() when r_tube > 0).
        # _tube_total_calls    : total QP solves with torso task active
        # _tube_violations     : solves where ||e_lin|| > r_tube (linear active)
        # _tube_max_e_lin_m    : max ||e_lin|| observed [m]
        # _tube_last_violated  : last solve outcome (True if violated)
        self._tube_total_calls: int = 0
        self._tube_violations: int = 0
        self._tube_max_e_lin_m: float = 0.0
        self._tube_last_violated: bool = False

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
        q_t: np.ndarray,
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
        # Reaction null-space (minimize base disturbance from swing arm)
        H_base_swing: Optional[np.ndarray] = None,  # (6, n_swing) coupling block
        swing_v_slice: Optional[slice] = None,       # velocity-space slice of swing arm joints
        # DS settling mode (skip torso/CoM, damp velocities)
        settle_mode: bool = False,
        # M2 passivity constraint: enforce dq_j^T*tau_q + 2*alpha*T <= 0
        # Intended to be ON during DS phase only (see §3.6, §5.7).
        passivity_active: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, QPSolveInfo]:
        """Solve the whole-body QP.

        Parameters
        ----------
        q_t : ndarray (7,)
            Torso pose [quaternion(4), position(3)]. Used for state only.
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
        n_robot = 6 + nq  # dimension of q̈_robot = [q̈_t; q̈]

        # --- Build QP ---
        # weight_ratio is taken from the config. In the M2 task stack
        # task isolation comes from null-space projection (N_torso,
        # N_torso·N_ee, ...), NOT from weight scaling, so the default
        # weight_ratio = 1 lets every task use its face-value α. See
        # WholeBodyQPConfig.weight_ratio and the P1→P6 stack below.
        qp = HierarchicalQP(
            n_vars=n, method=cfg.method, solver=cfg.solver,
            weight_ratio=cfg.weight_ratio,
        )

        # ============================================================ #
        #  EQUALITY CONSTRAINTS                                         #
        # ============================================================ #

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
            nc_active = J_contacts.shape[0] // 6
            col_offset = 0
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

        # ============================================================ #
        #  INEQUALITY CONSTRAINTS                                       #
        # ============================================================ #

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

            # 3. Momentum rate box: |L̇_robot| = |M_λ·λ| ≤ τ_w_max
            if np.isfinite(cfg.tau_w_max):
                A_Ld_upper = np.zeros((3, n))
                A_Ld_upper[:, idx['lambda'][0]: idx['lambda'][1]] = M_lambda
                b_Ld_upper = cfg.tau_w_max * np.ones(3)

                A_Ld_lower = np.zeros((3, n))
                A_Ld_lower[:, idx['lambda'][0]: idx['lambda'][1]] = -M_lambda
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
            b_pass = np.array([-2.0 * cfg.alpha_passivity * T_kin])
            qp.add_inequality_constraint(A_pass, b_pass)

        # ============================================================ #
        #  BOUNDS                                                       #
        # ============================================================ #

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

        # ============================================================ #
        #  TASKS                                                        #
        # ============================================================ #

        dq_robot = np.concatenate([dq_t, dq])

        # ──────────────────────────────────────────────────────────── #
        # Helper: build the CoM cost row `J_com @ qdd = a_com_des - Jdq`
        # (used for legacy Task 1 and M2 soft CoM)
        # ──────────────────────────────────────────────────────────── #
        r_com_actual = r_com if r_com is not None else np.zeros(3)
        v_com_actual = J_com @ dq_robot
        Kp_com_mat = np.diag(cfg.Kp_com)
        Kd_com_mat = np.diag(cfg.Kd_com)
        a_com_des = (a_com_ff
                     + Kp_com_mat @ (r_com_ref - r_com_actual)
                     + Kd_com_mat @ (v_com_ref - v_com_actual))
        A_com = np.zeros((3, n))
        A_com[:, idx['qdd_t'][0]: idx['qdd_t'][1]] = J_com[:, :6]
        A_com[:, idx['qdd'][0]: idx['qdd'][1]] = J_com[:, 6:]
        b_com = a_com_des - Jdot_dq_com

        # --- Task 1: Legacy CoM tracking (skipped in M2 stack) ---
        if cfg.alpha_com > 0 and not settle_mode and not cfg.use_m2_stack:
            qp.add_task(A_com, b_com, cfg.alpha_com, priority=1)

        # --- Task 1b: Torso 6D tracking (primary P1 task in M2 stack) ---
        # Controls both position and orientation of the torso body frame.
        # a_torso_des = [a_lin_ff + Kp_lin(p_ref - p) + Kd_lin(v_ref - v);
        #                a_ang_ff + Kp_ang(log3(R^T R_ref)) + Kd_ang(ω_ref - ω)]
        torso_task_active = (
            (cfg.alpha_torso > 0 or cfg.use_m2_stack)
            and J_torso is not None
            and p_torso_ref is not None
            and not settle_mode
        )
        A_torso = None
        b_torso = None
        if torso_task_active:
            Kp_t = np.diag(cfg.Kp_torso)   # (6,6)
            Kd_t = np.diag(cfg.Kd_torso)   # (6,6)

            v_torso_actual = J_torso @ dq_robot  # (6,): [lin(3), ang(3)]
            p_t = p_torso if p_torso is not None else np.zeros(3)
            e_pos = p_torso_ref - p_t
            R_t = R_torso if R_torso is not None else np.eye(3)
            R_ref_t = R_torso_ref if R_torso_ref is not None else np.eye(3)
            e_ori = pin.log3(R_t.T @ R_ref_t)
            e_6d = np.concatenate([e_pos, e_ori])

            v_ref_t = v_torso_ref if v_torso_ref is not None else np.zeros(6)
            a_ff_t = a_torso_ff if a_torso_ff is not None else np.zeros(6)
            a_torso_des = a_ff_t + Kp_t @ e_6d + Kd_t @ (v_ref_t - v_torso_actual)

            jdq = Jdot_dq_torso if Jdot_dq_torso is not None else np.zeros(6)
            A_torso_full = np.zeros((6, n))
            A_torso_full[:, idx['qdd_t'][0]: idx['qdd_t'][1]] = J_torso[:, :6]
            A_torso_full[:, idx['qdd'][0]: idx['qdd'][1]] = J_torso[:, 6:]
            b_torso_full = a_torso_des - jdq

            # Effective weight in M2 stack (sim_loop sets alpha_torso via config;
            # in pure M2 testing, fall back to 1e3 if user left alpha_torso=0).
            w_torso = cfg.alpha_torso if cfg.alpha_torso > 0 else 1e3

            # ── Cooperative-arms mode: split angular/linear ───────────
            # Angular goes to P1 at alpha_torso_ang (strict equality
            # strength). Linear is stashed and added later as a P2
            # weighted-LS co-contributor with EE (same null-space
            # projection N_torso_ang). The "effective" A_torso /
            # b_torso / J_torso_eff used downstream collapse to the
            # angular 3D subset — matching the inside-tube branch.
            # Stance-arm thrust still flows passively via the dynamics
            # equality (no explicit cost on stance thrust).
            _coop_A_lin = None
            _coop_b_lin = None
            if cfg.cooperative_arms_mode:
                A_torso_ang = A_torso_full[3:, :]
                b_torso_ang = b_torso_full[3:]
                _coop_A_lin = A_torso_full[:3, :]
                _coop_b_lin = b_torso_full[:3]
                # P1 — angular only, strict-equality strength
                qp.add_task(A_torso_ang, b_torso_ang,
                            cfg.alpha_torso_ang, priority=1)
                # Effective torso (used for N_torso, EE FF, residuals)
                A_torso = A_torso_ang
                b_torso = b_torso_ang
                J_torso_eff = J_torso[3:, :]
                a_torso_des_eff = a_torso_des[3:]
                # Skip Option D and legacy paths.
                continue_torso_block = False
            else:
                continue_torso_block = True

            # ── Option D: torso linear soft tube ──────────────────────
            # r_tube > 0 splits the 6D task: angular stays at full weight,
            # linear is gated by ||p_torso - p_ref|| > r_tube with the
            # violation-magnitude weight. Inside the tube the linear task
            # is skipped entirely, and the effective A_torso (used for
            # null-space projections and EE feedforward downstream)
            # collapses to the 3 angular rows — freeing the linear-base
            # DOFs for EE tracking.
            r_tube = float(getattr(cfg, 'r_tube', 0.0))
            if continue_torso_block and r_tube > 0.0:
                e_lin_norm = float(np.linalg.norm(e_pos))
                linear_active = (e_lin_norm > r_tube)

                A_torso_ang = A_torso_full[3:, :]   # (3, n)
                b_torso_ang = b_torso_full[3:]      # (3,)
                A_torso_lin = A_torso_full[:3, :]   # (3, n)
                b_torso_lin = b_torso_full[:3]      # (3,)

                # Angular task is always enforced at the full weight.
                qp.add_task(A_torso_ang, b_torso_ang, w_torso, priority=1)

                if linear_active:
                    # Outside the tube: add linear task with weight that
                    # scales linearly with the squared-violation; this is
                    # the QP-friendly surrogate for the user-specified
                    # quartic cost w_tube * (|e|² - r²)² in position space.
                    violation_sq = max(0.0, e_lin_norm ** 2 - r_tube ** 2)
                    w_lin = float(cfg.w_tube_lin) * violation_sq
                    if w_lin > 0.0:
                        qp.add_task(
                            A_torso_lin, b_torso_lin, w_lin, priority=1)
                    # Effective task (for null-space / FF) stays 6D.
                    A_torso = A_torso_full
                    b_torso = b_torso_full
                    J_torso_eff = J_torso
                    a_torso_des_eff = a_torso_des
                else:
                    # Inside the tube: angular-only for projections / FF.
                    A_torso = A_torso_ang
                    b_torso = b_torso_ang
                    J_torso_eff = J_torso[3:, :]
                    a_torso_des_eff = a_torso_des[3:]

                self._tube_total_calls += 1
                if linear_active:
                    self._tube_violations += 1
                self._tube_max_e_lin_m = max(
                    self._tube_max_e_lin_m, e_lin_norm)
                self._tube_last_violated = bool(linear_active)
            elif continue_torso_block:
                # Legacy: full 6D equality task at P1.
                qp.add_task(A_torso_full, b_torso_full, w_torso, priority=1)
                A_torso = A_torso_full
                b_torso = b_torso_full
                J_torso_eff = J_torso
                a_torso_des_eff = a_torso_des

        # ──────────────────────────────────────────────────────────── #
        # Null-space projectors used by the rest of the M2 stack
        # N_torso   = I − A_torso^+ A_torso  (projects x into null(A_torso))
        # N_torso_ee = N_torso · (I − (A_ee·N_torso)^+ (A_ee·N_torso))
        #           (further projects out EE subspace so lower-priority
        #           tasks don't interfere with either torso or EE)
        # These are shared between the EE task (priority 2) and the
        # posture task (priority 3). Under the M2 stack this gives
        # geometric isolation regardless of the task weights, so the
        # HierarchicalQP weight_ratio can stay at 1.
        # ──────────────────────────────────────────────────────────── #
        N_torso = None
        A_torso_pinv = None
        if cfg.use_m2_stack and A_torso is not None:
            A_torso_pinv = np.linalg.pinv(A_torso, rcond=1e-8)
            N_torso = np.eye(n) - A_torso_pinv @ A_torso

        # --- Task 2: End-effector 6D tracking (swing arm, optional) ---
        # In the M2 stack the EE task is projected into the null space of
        # the torso task via N_torso = I - J_torso^+ @ J_torso. This ensures
        # EE tracking does not interfere with torso tracking.
        A_ee = None
        b_ee = None
        A_ee_proj = None
        b_ee_res = None
        if J_ee is not None and p_ee_ref is not None:
            jdq_ee = Jdot_dq_ee if Jdot_dq_ee is not None else np.zeros(6)

            v_ee_actual = J_ee @ dq_robot  # (6,): [lin(3), ang(3)]

            p_ee_actual = p_ee if p_ee is not None else np.zeros(3)
            e_pos_ee = p_ee_ref - p_ee_actual

            R_ee_actual = R_ee if R_ee is not None else np.eye(3)
            R_ee_ref_actual = R_ee_ref if R_ee_ref is not None else np.eye(3)
            e_ori_ee = pin.log3(R_ee_actual.T @ R_ee_ref_actual)

            e_6d_ee = np.concatenate([e_pos_ee, e_ori_ee])
            Kp_ee_full = np.diag(np.concatenate([cfg.Kp_ee, cfg.Kp_ee_ang]))
            Kd_ee_full = np.diag(np.concatenate([cfg.Kd_ee, cfg.Kd_ee_ang]))

            v_ref_ee = v_ee_ref if v_ee_ref is not None else np.zeros(6)
            a_ff_ee = a_ee_ff if a_ee_ff is not None else np.zeros(6)

            # M7 v17: task-consistent feedforward. When the torso accelerates
            # at a_torso_des, the EE sees J_ee · J_torso^+ · a_torso_des of
            # induced motion through the shared floating base. Pre-add this
            # to the EE feedforward so the PD doesn't have to chase the
            # coupling reactively. J_torso^+ is the damped pseudo-inverse
            # (rcond=1e-8), matching the one used for the null-space
            # projector above.
            if torso_task_active and J_torso is not None:
                # Use the effective torso Jacobian / desired-acceleration
                # (3D angular only when the Option D tube is satisfied;
                # full 6D otherwise — matches the null-space projection).
                J_torso_pinv = np.linalg.pinv(J_torso_eff, rcond=1e-8)
                a_ff_ee = a_ff_ee + J_ee @ J_torso_pinv @ a_torso_des_eff

            a_ee_des = a_ff_ee + Kp_ee_full @ e_6d_ee + Kd_ee_full @ (v_ref_ee - v_ee_actual)

            A_ee = np.zeros((6, n))
            A_ee[:, idx['qdd_t'][0]: idx['qdd_t'][1]] = J_ee[:, :6]
            A_ee[:, idx['qdd'][0]: idx['qdd'][1]] = J_ee[:, 6:]
            b_ee = a_ee_des - jdq_ee

            # Null-space projection against the torso task: replace (A_ee,b_ee)
            # with (N_torso @ A_ee, b_ee residual). Because we only scale
            # the rows of the cost, the QP solution is equivalent to
            # strict prioritization even when α_ee > α_torso.
            apply_null_space = (
                (cfg.use_m2_stack or cfg.ee_null_space)
                and N_torso is not None
            )
            if apply_null_space:
                A_ee_proj = A_ee @ N_torso
                b_ee_res = b_ee - A_ee @ A_torso_pinv @ b_torso
                qp.add_task(A_ee_proj, b_ee_res, cfg.alpha_ee, priority=2)
            else:
                A_ee_proj = A_ee
                b_ee_res = b_ee
                qp.add_task(A_ee, b_ee, cfg.alpha_ee, priority=2)
        else:
            A_ee_proj = None
            b_ee_res = None

        # --- Cooperative-arms P2: torso LINEAR (co-equal with EE) ---
        # Projected through the SAME N_torso (= N_torso_ang) used for EE
        # so the two P2 tasks share a null-space frame. The QP arbitrates
        # between them via alpha_torso_lin vs alpha_ee — at the defaults
        # (500 vs 3000) EE dominates 6:1 on body-linear motion, letting
        # the swing arm pull the body forward when reach margin demands.
        A_coop_lin_proj = None
        b_coop_lin_res = None
        if (cfg.cooperative_arms_mode and _coop_A_lin is not None
                and N_torso is not None and A_torso_pinv is not None):
            A_coop_lin_proj = _coop_A_lin @ N_torso
            b_coop_lin_res = _coop_b_lin - _coop_A_lin @ A_torso_pinv @ b_torso
            qp.add_task(A_coop_lin_proj, b_coop_lin_res,
                        cfg.alpha_torso_lin, priority=2)

        # --- Task 2b: Soft CoM residual (M2 stack only) ---
        # Quadratic cost α_com_soft * ||J_com @ qdd - a_com_des||² with
        # desired acceleration from NMPC ff + PD (NOT from the mapping).
        # Weight is deliberately small (1-10) so it biases the residual
        # null-space DOFs toward momentum-consistent trajectories
        # without overriding the torso or EE tasks.
        #
        # With weight_ratio=1 the soft CoM is directly visible, and
        # since J_com rows correlate with J_torso rows through the
        # mass ratio, we null-space project it into N(A_torso) ∩ N(A_ee)
        # — same as the posture task — so it never degrades the torso
        # or EE targets.
        if cfg.use_m2_stack and cfg.alpha_com_soft > 0 and not settle_mode:
            if A_torso is not None:
                if A_ee is not None:
                    A_combo_com = np.vstack([A_torso, A_ee])
                    b_combo_com = np.concatenate([b_torso, b_ee])
                else:
                    A_combo_com = A_torso
                    b_combo_com = b_torso
                A_combo_com_pinv = np.linalg.pinv(A_combo_com, rcond=1e-8)
                N_com = np.eye(n) - A_combo_com_pinv @ A_combo_com
                A_com_proj = A_com @ N_com
                b_com_res = b_com - A_com @ A_combo_com_pinv @ b_combo_com
                qp.add_task(A_com_proj, b_com_res,
                            cfg.alpha_com_soft, priority=4)
            else:
                qp.add_task(A_com, b_com, cfg.alpha_com_soft, priority=4)

        # --- Task 3: Posture regulation ---
        # q̈_posture = Kp_post (q_nom - q) + Kd_post (0 - dq)
        # Skipped in settle_mode: the settle task (P1) already dampens
        # all joint velocities, and at weight_ratio=1 the posture PD
        # would interfere with that dissipation (T10 regression).
        if not settle_mode:
            qdd_posture = (cfg.Kp_posture * (self._q_nominal - q) -
                           cfg.Kd_posture * dq)

            A_posture = np.zeros((nq, n))
            A_posture[:, idx['qdd'][0]: idx['qdd'][1]] = np.eye(nq)
            b_posture = qdd_posture

            # M2: project posture into null(A_torso) ∩ null(A_ee) so
            # that its contribution to q̈ perturbs NEITHER the torso
            # task nor the EE task. This is what makes weight_ratio=1
            # safe — task isolation is geometric, not via weight
            # scaling.
            #
            #   A_combo = [A_torso; A_ee]
            #   N_combo = I − A_combo^+ A_combo
            #
            # We also subtract the "particular solution" contribution
            # so the posture residual is consistent with the torso / EE
            # targets:
            #   b_posture_res = b_posture − A_posture · A_combo^+ · b_combo
            if cfg.use_m2_stack and A_torso is not None:
                # Cooperative-arms: include A_torso_lin (the P2 linear task)
                # in the combined null-space basis so posture doesn't fight
                # either component of the split torso task. rcond=1e-4 on
                # the combined 12×n pinv handles the wider conditioning of
                # the stacked Jacobian (individual task pinvs above stay at
                # the tighter rcond=1e-8).
                rows = [A_torso]
                rhs = [b_torso]
                if (cfg.cooperative_arms_mode and _coop_A_lin is not None):
                    rows.append(_coop_A_lin)
                    rhs.append(_coop_b_lin)
                if A_ee is not None:
                    rows.append(A_ee)
                    rhs.append(b_ee)
                A_combo = np.vstack(rows)
                b_combo = np.concatenate(rhs)
                _rcond = 1e-4 if cfg.cooperative_arms_mode else 1e-8
                A_combo_pinv = np.linalg.pinv(A_combo, rcond=_rcond)
                N_combo = np.eye(n) - A_combo_pinv @ A_combo
                A_posture_proj = A_posture @ N_combo
                b_posture_res = b_posture - A_posture @ A_combo_pinv @ b_combo
                qp.add_task(A_posture_proj, b_posture_res,
                            cfg.alpha_posture, priority=3)
            else:
                qp.add_task(A_posture, b_posture,
                            cfg.alpha_posture, priority=3)

        # --- Task 3b: DS joint-space settle (damp all velocities to zero) ---
        # In settle mode, torso/CoM tasks are skipped (they conflict with
        # the constrained equilibrium). This task drives all joint velocities
        # to zero via pure damping. No position term — with 8 DOF remaining
        # (6 base + 2 redundant from 7-DOF arms) and welds constraining the
        # EEs, the system can only stop at the current configuration.
        if settle_mode:
            A_settle = np.zeros((nq, n))
            A_settle[:, idx['qdd'][0]: idx['qdd'][1]] = np.eye(nq)
            b_settle = -cfg.Kd_settle * dq
            qp.add_task(A_settle, b_settle, cfg.alpha_settle, priority=1)

        # --- Task 3: Contact wrench tracking ---
        A_wrench = np.zeros((self._dim_lambda, n))
        A_wrench[:, idx['lambda'][0]: idx['lambda'][1]] = np.eye(self._dim_lambda)
        b_wrench = lambda_ref

        qp.add_task(A_wrench, b_wrench, cfg.alpha_wrench, priority=4)

        # --- Task 3b: Reaction null-space (minimize base disturbance from swing arm) ---
        # Penalizes ||H_bt @ qdd_swing||² where H_bt is the base-swing coupling.
        # This makes the QP prefer swing arm motions that don't torque the base.
        if (cfg.alpha_reaction > 0 and H_base_swing is not None
                and swing_v_slice is not None):
            n_sw = swing_v_slice.stop - swing_v_slice.start
            # Map swing arm qdd to base reaction: tau_base = H_bt @ qdd_swing
            # In the QP, qdd_swing is part of the qdd block.
            # qdd indices in z: idx['qdd'][0] + (swing_v_slice.start - 6)
            # because qdd block starts after qdd_t(6), and swing_v_slice
            # is in velocity space (starting from 6 for first arm joint).
            sw_offset = swing_v_slice.start - 6  # offset within qdd block
            A_react = np.zeros((6, n))
            A_react[:, idx['qdd'][0] + sw_offset:
                       idx['qdd'][0] + sw_offset + n_sw] = H_base_swing
            b_react = np.zeros(6)
            qp.add_task(A_react, b_react, cfg.alpha_reaction, priority=4)

        # --- Task 4: Joint torque minimization ---
        A_torque = np.zeros((nq, n))
        A_torque[:, idx['tau'][0]: idx['tau'][1]] = np.eye(nq)
        b_torque = np.zeros(nq)

        qp.add_task(A_torque, b_torque, cfg.alpha_torque, priority=5)

        # --- Task 5: Acceleration regularization ---
        A_reg = np.zeros((6 + nq, n))
        A_reg[:6, idx['qdd_t'][0]: idx['qdd_t'][1]] = np.eye(6)
        A_reg[6:, idx['qdd'][0]: idx['qdd'][1]] = np.eye(nq)
        b_reg = np.zeros(6 + nq)

        qp.add_task(A_reg, b_reg, cfg.alpha_reg, priority=6)

        # --- M5 Task: hw slack penalty (heavy, high priority) ---
        # Drive the slack variables to zero as fast as the actuator
        # limits allow. Uses priority 1 so it can't be traded off
        # against posture / torque minimisation. Weight = w_hw_slack
        # (default 1e4 >> alpha_torso ≈ 1e3).
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

        # --- Debug capture: torso task pre- vs post-solve (M7 diagnosis) ---
        if torso_task_active and A_torso is not None:
            qdd_full = np.concatenate([qdd_t_opt, qdd_opt])
            x_dd_torso_post = J_torso @ qdd_full
            self.last_torso_debug = {
                'a_torso_des_pre': a_torso_des.copy(),   # PD+ff target (6,)
                'x_dd_torso_post': x_dd_torso_post.copy(),  # J_torso @ qdd_opt (6,)
                'e_pos': e_pos.copy(), 'e_ori': e_ori.copy(),
                'v_torso_actual': v_torso_actual.copy(),
                'v_ref_t': v_ref_t.copy(),
                'a_ff_t': a_ff_t.copy(),
                'Kp_t_diag': cfg.Kp_torso.copy(),
                'Kd_t_diag': cfg.Kd_torso.copy(),
                'Jdot_dq_torso': jdq.copy(),
            }
        else:
            self.last_torso_debug = None

        return qdd_t_opt, qdd_opt, lambda_opt, tau_q_opt, info

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

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
