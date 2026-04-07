"""
CentroidalNMPC - Centroidal NMPC for momentum-feasible trajectory generation.

Wraps NMPCSolver with the robot-centric centroidal dynamics model for
crawling space robots. Generates momentum-feasible CoM reference
trajectories that respect the spacecraft reaction wheel envelope.

Architecture:
    Stage 1 of the two-stage controller (see Chelikh et al., IEEE Access 2024).
    Runs at 20 Hz with a 1-second prediction horizon (N=20, dt=0.05s).

State vector (nx=9):
    x = [r_com (3), v_com (3), L_com (3)]
    - r_com: Robot CoM position in spacecraft frame R_s
    - v_com: Robot CoM velocity in R_s
    - L_com: Robot centroidal angular momentum about CoM, in R_s

    Note: h_w (wheel momentum) removed — AOCS manages wheels independently.
    Linear momentum ||m·v_com|| is bounded to limit orbital disturbance.

Control vector (nu=12, for nc_max=2):
    u = [f_1 (3), τ_1 (3), f_2 (3), τ_2 (3)]
    - f_j:   Contact force at contact j
    - τ_j:   Contact moment at contact j
    Inactive contacts are zeroed via bounds.

Parameters (np=12):
    p = [r_ref (3), v_ref (3), r_C1 (3), r_C2 (3)]
    - r_ref, v_ref: CoM reference position/velocity
    - r_C1, r_C2:   Contact point positions in R_s

Constraints:
    - Dynamics:  RK4 integration of centroidal equations
    - Angular:   |L_com,i| <= L_max     (angular momentum bounds)
    - Linear:    ||m·v_com||² <= p_max² (linear momentum → orbital bound)
    - SOC:       ||f_j||² <= f_max²     (force norm limits)
    - SOC:       ||τ_j||² <= τ_max²     (torque norm limits)

Reference:
    Eq. (VI-E.12), (VI-E.17), (VI-E.22)-(VI-E.26) of the paper.
"""

import numpy as np
import casadi as ca
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple

from .nmpc_solver import NMPCSolver, NMPCSolveInfo
from .contact_phase import ContactPhase, ContactConfig


@dataclass
class CentroidalNMPCConfig:
    """Configuration for CentroidalNMPC.

    All physical parameters needed to instantiate the problem.
    """
    # Robot properties
    robot_mass: float = 90.0                 # Total robot mass [kg]

    # Horizon
    N: int = 20                              # Prediction horizon steps
    dt: float = 0.05                         # Time step [s] → 1s horizon

    # Cost weights (Eq. VI-E.17)
    Wr: np.ndarray = field(default_factory=lambda: 100.0 * np.ones(3))  # Position tracking
    Wv: np.ndarray = field(default_factory=lambda: 10.0 * np.ones(3))   # Velocity tracking
    Wu_f: float = 0.01                       # Force regularization
    Wu_tau: float = 0.001                    # Torque regularization
    Qf_r: np.ndarray = field(default_factory=lambda: 1000.0 * np.ones(3))  # Terminal position
    Qf_v: np.ndarray = field(default_factory=lambda: 100.0 * np.ones(3))   # Terminal velocity

    # Contact wrench limits (HOTDOCK specs)
    f_max: float = 3000.0                    # Max contact force norm [N]
    tau_max: float = 300.0                   # Max contact torque norm [Nm]

    # 6D centroidal momentum constraints
    L_max: float = np.inf                    # |L_com,i| ≤ L_max [Nms]  (angular)
    tau_w_max: float = np.inf                # |L̇_com,i| ≤ τ_w_max [Nm] (angular rate)
    p_max: float = np.inf                    # ||m·v_com|| ≤ p_max [kg·m/s] (linear momentum)
                                             # Bounds orbital disturbance: τ_orbital ≤ |r_com|·p_max
    tau_struct_max: float = np.inf           # |Ḣ_s,i| ≤ tau_struct_max [Nm] (structure disturbance)

    # Solver
    solver_name: str = 'ipopt'
    solver_opts: Dict[str, Any] = field(default_factory=dict)


class CentroidalNMPC:
    """Centroidal NMPC for momentum-feasible trajectory generation.

    Parameters
    ----------
    config : CentroidalNMPCConfig
        Problem configuration.
    """

    # Dimensions (fixed)
    NX = 9     # [r_com(3), v_com(3), L_com(3)]  — hw removed, AOCS independent
    NU = 12    # [f1(3), τ1(3), f2(3), τ2(3)]
    NP = 12    # [r_ref(3), v_ref(3), r_C1(3), r_C2(3)]

    def __init__(self, config: Optional[CentroidalNMPCConfig] = None):
        if config is None:
            config = CentroidalNMPCConfig()
        self.config = config
        self._nmpc: Optional[NMPCSolver] = None
        self._built = False

    def build(self, solver_opts: Optional[Dict[str, Any]] = None) -> None:
        """Build the NMPC solver.

        Must be called once before solve(). Can be rebuilt with different
        solver options if needed.
        """
        cfg = self.config
        m = cfg.robot_mass

        # --- Create generic NMPC solver ---
        nmpc = NMPCSolver(
            nx=self.NX, nu=self.NU, N=cfg.N, dt=cfg.dt,
            solver_name=cfg.solver_name,
        )

        # --- Parameters ---
        nmpc.set_parameters(self.NP)

        # --- Continuous dynamics (RK4 integration) ---
        # State: x = [r_com(3), v_com(3), L_com(3)]  — NX=9
        # hw removed: AOCS manages wheels independently.
        def centroidal_ode(x, u, p):
            r_com = x[0:3]
            v_com = x[3:6]
            # L_com = x[6:9]

            f1 = u[0:3];  tau1 = u[3:6]
            f2 = u[6:9];  tau2 = u[9:12]

            r_C1 = p[6:9]
            r_C2 = p[9:12]

            # Linear momentum: m v̇ = Σf_j  (no gravity in orbit)
            v_dot = (f1 + f2) / m

            # Centroidal angular momentum rate (about robot CoM):
            # L̇_com = Σ [(r_Cj - r_com) × f_j + τ_j]
            L_dot = (ca.cross(r_C1 - r_com, f1) + tau1 +
                     ca.cross(r_C2 - r_com, f2) + tau2)

            return ca.vertcat(v_com, v_dot, L_dot)

        nmpc.set_continuous_dynamics(centroidal_ode)

        # --- Stage cost ---
        Wr = np.diag(cfg.Wr)
        Wv = np.diag(cfg.Wv)
        Wu = np.diag(np.concatenate([
            cfg.Wu_f * np.ones(3), cfg.Wu_tau * np.ones(3),   # contact 1
            cfg.Wu_f * np.ones(3), cfg.Wu_tau * np.ones(3),   # contact 2
        ]))

        def stage_cost(x, u, p):
            r_com = x[0:3]
            v_com = x[3:6]
            r_ref = p[0:3]
            v_ref = p[3:6]

            e_r = r_com - r_ref
            e_v = v_com - v_ref

            return e_r.T @ Wr @ e_r + e_v.T @ Wv @ e_v + u.T @ Wu @ u

        nmpc.set_stage_cost(stage_cost)

        # --- Terminal cost ---
        Qf_r = np.diag(cfg.Qf_r)
        Qf_v = np.diag(cfg.Qf_v)

        def terminal_cost(x, p):
            r_com = x[0:3]
            v_com = x[3:6]
            r_ref = p[0:3]
            v_ref = p[3:6]

            e_r = r_com - r_ref
            e_v = v_com - v_ref

            return e_r.T @ Qf_r @ e_r + e_v.T @ Qf_v @ e_v

        nmpc.set_terminal_cost(terminal_cost)

        # --- Path constraints: SOC on contact wrenches ---
        # g(x, u, p) <= 0:
        #   ||f1||² - f_max²  <= 0
        #   ||τ1||² - τ_max²  <= 0
        #   ||f2||² - f_max²  <= 0
        #   ||τ2||² - τ_max²  <= 0
        f_max_sq = cfg.f_max ** 2
        tau_max_sq = cfg.tau_max ** 2

        p_max_sq = (cfg.p_max ** 2) if np.isfinite(cfg.p_max) else np.inf
        has_Hdot_s = np.isfinite(cfg.tau_struct_max)

        def path_constraints(x, u, p):
            f1 = u[0:3];  tau1 = u[3:6]
            f2 = u[6:9];  tau2 = u[9:12]

            r_com = x[0:3]
            v_com = x[3:6]
            r_C1 = p[6:9]
            r_C2 = p[9:12]

            # SOC on contact wrenches
            soc = ca.vertcat(
                ca.dot(f1, f1) - f_max_sq,
                ca.dot(tau1, tau1) - tau_max_sq,
                ca.dot(f2, f2) - f_max_sq,
                ca.dot(tau2, tau2) - tau_max_sq,
            )

            # L̇_com rate constraint: |L̇_com,i| ≤ τ_w_max
            L_dot = (ca.cross(r_C1 - r_com, f1) + tau1 +
                     ca.cross(r_C2 - r_com, f2) + tau2)
            tw = cfg.tau_w_max
            Ldot_ineq = ca.vertcat(L_dot - tw, -L_dot - tw)

            parts = [soc, Ldot_ineq]

            # Structure disturbance constraint: Ḣ_s = Σ [r_Cⱼ × fⱼ + τⱼ]
            # r_Cⱼ are lever arms from structure CoM (= origin in R_s).
            # Unlike L̇_com which uses (r_Cⱼ - r_com), this uses r_Cⱼ directly
            # to bound the torque the structure AOCS must absorb.
            if has_Hdot_s:
                H_dot_s = (ca.cross(r_C1, f1) + tau1 +
                           ca.cross(r_C2, f2) + tau2)
                ts = cfg.tau_struct_max
                Hdot_s_ineq = ca.vertcat(H_dot_s - ts, -H_dot_s - ts)
                parts.append(Hdot_s_ineq)

            # Linear momentum constraint: ||m·v_com||² ≤ p_max²
            p_lin = m * v_com
            p_ineq = ca.dot(p_lin, p_lin) - p_max_sq
            parts.append(p_ineq)

            return ca.vertcat(*parts)

        ng_path = 4 + 6 + 1  # 4 SOC + 6 L̇ bilateral + 1 linear momentum
        if has_Hdot_s:
            ng_path += 6        # + 6 Ḣ_s bilateral
        nmpc.set_path_constraints(path_constraints, ng=ng_path)

        # --- State bounds ---
        L_max_safe = cfg.L_max
        x_min = np.concatenate([
            np.full(3, -np.inf),         # r_com: unbounded
            np.full(3, -np.inf),         # v_com: unbounded (norm bounded via path constraint)
            np.full(3, -L_max_safe),     # L_com: bounded by wheel capacity
        ])
        x_max = np.concatenate([
            np.full(3, np.inf),
            np.full(3, np.inf),
            np.full(3, L_max_safe),      # L_com: bounded by wheel capacity
        ])
        nmpc.set_state_bounds(x_min, x_max)

        # --- Control bounds (default: all contacts active, box around SOC) ---
        # These are overridden per-solve based on contact phase
        u_min_default = np.full(self.NU, -cfg.f_max)
        u_max_default = np.full(self.NU, cfg.f_max)
        # Torque components have different limits
        for j in range(2):
            u_min_default[6 * j + 3: 6 * j + 6] = -cfg.tau_max
            u_max_default[6 * j + 3: 6 * j + 6] = cfg.tau_max
        nmpc.set_control_bounds(u_min_default, u_max_default)

        # --- Build ---
        opts = cfg.solver_opts.copy()
        if solver_opts:
            opts.update(solver_opts)
        nmpc.build(opts)

        self._nmpc = nmpc
        self._built = True

    def solve(
        self,
        r_com: np.ndarray,
        v_com: np.ndarray,
        L_com: np.ndarray,
        r_com_ref: np.ndarray,
        v_com_ref: np.ndarray,
        contact_config: ContactConfig,
        warm_start: bool = True,
        hw_current: Optional[np.ndarray] = None,  # deprecated, ignored
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, 'NMPCSolveInfo']:
        """Solve the centroidal NMPC.

        Parameters
        ----------
        r_com : ndarray (3,)
            Current robot CoM position in R_s.
        v_com : ndarray (3,)
            Current robot CoM velocity in R_s.
        L_com : ndarray (3,)
            Current robot centroidal angular momentum.
        hw_current : ndarray (3,)
            Current spacecraft wheel momentum (from AOCS telemetry).
        r_com_ref : ndarray (3,)
            Desired CoM position reference.
        v_com_ref : ndarray (3,)
            Desired CoM velocity reference.
        contact_config : ContactConfig
            Current contact phase and positions.
        warm_start : bool
            Use previous solution as initial guess.

        Returns
        -------
        r_com_plan : ndarray (3,)
            Planned CoM position at t+dt (for Stage 2 reference).
        v_com_plan : ndarray (3,)
            Planned CoM velocity at t+dt.
        L_com_plan : ndarray (3,)
            Planned angular momentum at t+dt.
        lambda_plan : ndarray (12,)
            Planned contact wrenches at t=0 (for Stage 2 reference).
        info : NMPCSolveInfo
            Solver information.

        Notes
        -----
        The returned references correspond to the first time step of the
        NMPC solution, intended for use by the Stage 2 whole-body QP.
        """
        if not self._built:
            raise RuntimeError("Call build() before solve().")

        cfg = self.config

        # --- Apply contact phase to control bounds ---
        self._apply_contact_bounds(contact_config)

        # --- Assemble initial state (NX=9, no hw) ---
        x0 = np.concatenate([r_com, v_com, L_com])

        # --- Assemble parameters ---
        params = np.concatenate([
            r_com_ref, v_com_ref,
            contact_config.r_contact_A,
            contact_config.r_contact_B,
        ])

        # --- Solve ---
        x_opt, u_opt, info = self._nmpc.solve(
            x0, params=params, warm_start=warm_start
        )

        # --- Shift warm-start for next call ---
        if info.success:
            self._nmpc.shift_warm_start()

        # --- Extract first-step references for Stage 2 ---
        r_com_plan = x_opt[0:3, 1]
        v_com_plan = x_opt[3:6, 1]
        L_com_plan = x_opt[6:9, 1]
        lambda_plan = u_opt[:, 0]

        return r_com_plan, v_com_plan, L_com_plan, lambda_plan, info

    def get_full_trajectory(
        self,
        r_com: np.ndarray,
        v_com: np.ndarray,
        L_com: np.ndarray,
        r_com_ref: np.ndarray,
        v_com_ref: np.ndarray,
        contact_config: ContactConfig,
        hw_current: Optional[np.ndarray] = None,  # deprecated, ignored
    ) -> Tuple[np.ndarray, np.ndarray, NMPCSolveInfo]:
        """Solve and return the full predicted trajectory over the horizon.

        Returns
        -------
        x_opt : ndarray (9, N+1)
            Full state trajectory [r_com, v_com, L_com].
        u_opt : ndarray (12, N)
            Full control trajectory.
        info : NMPCSolveInfo
        """
        if not self._built:
            raise RuntimeError("Call build() before solve().")

        self._apply_contact_bounds(contact_config)

        x0 = np.concatenate([r_com, v_com, L_com])
        params = np.concatenate([
            r_com_ref, v_com_ref,
            contact_config.r_contact_A,
            contact_config.r_contact_B,
        ])

        x_opt, u_opt, info = self._nmpc.solve(x0, params=params)
        return x_opt, u_opt, info

    def compute_feedforward_acceleration(
        self, lambda_ref: np.ndarray
    ) -> np.ndarray:
        """Compute feedforward CoM acceleration from planned wrenches.

        r̈_com_ff = (1/m) Σ f_j_ref

        Used by Stage 2 PD law (Eq. VI-F.4).

        Parameters
        ----------
        lambda_ref : ndarray (12,)
            Planned contact wrenches [f1, τ1, f2, τ2].

        Returns
        -------
        a_com_ff : ndarray (3,)
            Feedforward CoM acceleration.
        """
        f1 = lambda_ref[0:3]
        f2 = lambda_ref[6:9]
        return (f1 + f2) / self.config.robot_mass

    # ------------------------------------------------------------------ #
    #  Private                                                             #
    # ------------------------------------------------------------------ #

    def _apply_contact_bounds(self, contact_config: ContactConfig) -> None:
        """Update control bounds based on active contacts.

        Inactive contacts are zeroed: u_min = u_max = 0.
        """
        cfg = self.config

        u_min = np.zeros(self.NU)
        u_max = np.zeros(self.NU)

        # Contact A: indices 0:6
        if contact_config.active_contacts[0]:
            u_min[0:3] = -cfg.f_max
            u_max[0:3] = cfg.f_max
            u_min[3:6] = -cfg.tau_max
            u_max[3:6] = cfg.tau_max

        # Contact B: indices 6:12
        if contact_config.active_contacts[1]:
            u_min[6:9] = -cfg.f_max
            u_max[6:9] = cfg.f_max
            u_min[9:12] = -cfg.tau_max
            u_max[9:12] = cfg.tau_max

        self._nmpc.u_min = u_min
        self._nmpc.u_max = u_max

        # Update the stored bounds in the built solver
        # Rebuild lbw/ubw for control components
        if self._nmpc._lbw is not None:
            nx = self.NX
            nu = self.NU
            for k in range(cfg.N):
                # Layout: [X0, U0, X1, U1, ..., X_N]
                # Offset to U_k: nx + k*(nx+nu) + ... 
                # More precisely: first nx (X0), then for each k: nu (Uk) + nx (Xk+1)
                u_start = nx + k * (nx + nu)
                self._nmpc._lbw[u_start: u_start + nu] = u_min
                self._nmpc._ubw[u_start: u_start + nu] = u_max

    def __repr__(self) -> str:
        status = "built" if self._built else "not built"
        return (
            f"CentroidalNMPC(m={self.config.robot_mass}kg, "
            f"N={self.config.N}, dt={self.config.dt}s, "
            f"hw=[{self.config.hw_min[0]:.0f}, {self.config.hw_max[0]:.0f}] Nms, "
            f"{status})"
        )
