"""
CoarsePrePlanner — momentum-feasible CoM trajectory optimization (M6).

Runs **once per step** before the step starts. Solves a centroidal NLP
over the full step horizon to produce a CoM reference trajectory that
satisfies the position-dependent momentum box at every collocation
point.

Mathematical formulation (spec §6.2):
------------------------------------
Decision variables at M+1 collocation points:
    r_com^s(k), v_com^s(k), L_com^s(k)   for k = 0..M
    f_stance(k),  tau_stance(k)          for k = 0..M-1
(Only one active contact during SS → f, τ are 3-vectors each.)

Dynamics (centroidal ODE, one active contact):
    ṙ_com^s = v_com^s
    v̇_com^s = f_stance / m
    L̇_com^s = (r_C - r_com^s) × f_stance + τ_stance

Constraints:
 1. Momentum box at every k:
    c − L_com^s(k) − r_com^s(k) × m v_com^s(k) ∈ [−h_max', +h_max']
 2. Rate bound: |L̇_com^s|_∞ ≤ τ_{w,max}
 3. Force/torque box: |f|_∞ ≤ f_max, |τ|_∞ ≤ τ_max
 4. Boundary: state(0) = x0, r_com(M) = r_goal, v_com(M) ≈ 0, L_com(M) ≈ 0
 5. Terminal margin: same box tightened by κ < 1 at k = M

Cost:
    J = Σ_k [ w_L ||L_com^s(k)||² + w_u ( ||f(k)||² + ||τ(k)||² ) ]

The NLP is built **once** via CasADi's Opti interface with parameters
bound to the initial state, goal, contact point, conservation constant
and the step duration. Subsequent calls to `solve()` only update the
parameter values and re-run IPOPT.

Output: CoarsePlanResult with a dense interpolant for r_com(t) and
v_com(t) plus the raw collocation trajectory. The result is consumed
by `sim_loop.py` as the NMPC reference in place of the TorsoPlanner's
geometric path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import casadi as ca


# ---------------------------------------------------------------------------
# Config + result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CoarsePrePlannerConfig:
    """Configuration for the coarse pre-planner.

    Most defaults mirror `SimConfig` / `CentroidalNMPCConfig` so a
    pre-planner result is always consistent with what the downstream
    NMPC will accept.
    """
    # Problem size
    M: int = 15                      # number of collocation intervals
    robot_mass: float = 71.0         # [kg]

    # Momentum box (same units as hw: Nms). Component-wise.
    h_max: np.ndarray = field(default_factory=lambda: np.full(3, 5.0))
    kappa_terminal: float = 0.7      # terminal margin multiplier (< 1)

    # Contact wrench bounds (componentwise, fast for IPOPT)
    f_max: float = 25.0              # [N] per active contact
    tau_max: float = 8.0             # [Nm] per active contact

    # Rate bound on L̇ (wheel torque equivalent)
    tau_w_max: float = 5.0           # [Nm] componentwise

    # Cost weights
    w_L: float = 1.0                 # ||L_com||² weight
    w_u: float = 1e-2                # ||[f; τ]||² weight
    w_v_terminal: float = 1e2        # soft penalty on ||v_com(M)||²
    w_L_terminal: float = 1e2        # soft penalty on ||L_com(M)||²

    # Terminal velocity/momentum tolerance for the equality "≈ 0"
    eps_v_terminal: float = 5e-3     # [m/s]
    eps_L_terminal: float = 5e-2     # [Nms]

    # Step duration handling
    T_step_default: float = 6.0      # [s] default, can be overridden per solve

    # Solver
    ipopt_print_level: int = 0
    ipopt_max_iter: int = 300
    ipopt_tol: float = 1e-6


@dataclass
class CoarsePlanResult:
    """Result of a single pre-planner solve.

    Provides both the raw collocation trajectory and smooth linear
    interpolants for evaluation at arbitrary times inside [0, T_step].
    """
    T_step: float
    t_grid: np.ndarray              # (M+1,)
    r_com: np.ndarray               # (M+1, 3)
    v_com: np.ndarray               # (M+1, 3)
    L_com: np.ndarray               # (M+1, 3)
    f_stance: np.ndarray            # (M, 3)
    tau_stance: np.ndarray          # (M, 3)

    success: bool
    solve_time_ms: float
    cost: float
    status: str
    iter_count: int = 0

    def r_com_at(self, t: float) -> np.ndarray:
        """Linear interpolation of r_com at time t ∈ [0, T_step]."""
        return self._interp(t, self.r_com)

    def v_com_at(self, t: float) -> np.ndarray:
        """Linear interpolation of v_com at time t ∈ [0, T_step]."""
        return self._interp(t, self.v_com)

    def L_com_at(self, t: float) -> np.ndarray:
        """Linear interpolation of L_com at time t ∈ [0, T_step]."""
        return self._interp(t, self.L_com)

    def _interp(self, t: float, traj: np.ndarray) -> np.ndarray:
        if t <= self.t_grid[0]:
            return traj[0].copy()
        if t >= self.t_grid[-1]:
            return traj[-1].copy()
        idx = int(np.searchsorted(self.t_grid, t) - 1)
        idx = max(0, min(idx, len(self.t_grid) - 2))
        t0 = self.t_grid[idx]
        t1 = self.t_grid[idx + 1]
        alpha = (t - t0) / (t1 - t0)
        return (1.0 - alpha) * traj[idx] + alpha * traj[idx + 1]

    def hw_at_knots(self, c_const: np.ndarray) -> np.ndarray:
        """Reconstruct h_w^s(k) = c − L_com(k) − r_com(k) × m v_com(k).

        Used in tests to verify the momentum box is satisfied. The mass
        must be passed in through the trajectory — this method assumes
        r_com, v_com are already expressed in the structure frame.
        """
        hw = np.zeros_like(self.r_com)
        for k in range(self.r_com.shape[0]):
            hw[k] = (
                c_const
                - self.L_com[k]
                - np.cross(self.r_com[k], self._mass * self.v_com[k])
            )
        return hw

    # ── Test-only constructor ─────────────────────────────────────────
    @classmethod
    def from_heuristic(
        cls,
        r_com_0: np.ndarray,
        r_com_goal: np.ndarray,
        h_max: np.ndarray,
        robot_mass: float,
        M: int = 15,
        lever_arm: float = 1.0,
    ) -> 'CoarsePlanResult':
        """Analytic CoarsePlanResult from the momentum feasibility envelope.

        Provides a valid `T_step` + straight-line `r_com` trajectory
        without invoking IPOPT. **Test fixtures only** — production paths
        always call `CoarsePrePlanner.solve()`, and `sim_loop.py` skips
        the step if the solve fails rather than falling back here.

        Reasoning
        ---------
        Momentum feasibility requires ‖h_w‖_∞ ≤ h_max. Ignoring L_com
        (conservative), h_w ≈ r_com × m v_com, so
            v_max_transverse ≈ min(h_max) / (m · lever_arm)
        and T_step ≈ ‖Δr_com‖ / v_max_transverse. A floor of 0.5 s
        guards against degenerate goals.

        Parameters
        ----------
        r_com_0, r_com_goal : (3,)
            CoM endpoints in the structure frame.
        h_max : (3,)
            Per-axis momentum box.
        robot_mass : float
            Total robot mass [kg].
        M : int
            Number of collocation intervals (must match downstream
            consumers that index r_com[k]).
        lever_arm : float
            Nominal |r_com| used in the envelope formula. 1.0 m is a
            conservative default for ASTROHUB-scale geometries.

        Returns
        -------
        CoarsePlanResult with ``status='heuristic'``, ``success=True``,
        ``L_com=0``, and a quintic straight-line ``r_com``.
        """
        r0 = np.asarray(r_com_0, dtype=float).reshape(3)
        r1 = np.asarray(r_com_goal, dtype=float).reshape(3)
        h_arr = np.asarray(h_max, dtype=float).reshape(3)

        distance = float(np.linalg.norm(r1 - r0))
        v_max = float(np.min(np.abs(h_arr))) / max(float(robot_mass), 1e-6) / max(float(lever_arm), 1e-6)
        if v_max <= 0.0:
            T_step = 1.0
        else:
            T_step = max(0.5, distance / v_max)

        t_grid = np.linspace(0.0, T_step, M + 1)
        # Quintic s(τ) = 10τ³ − 15τ⁴ + 6τ⁵ interpolation
        tau = t_grid / T_step
        s = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
        s_dot = (30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4) / T_step
        r_com = r0[None, :] + (r1 - r0)[None, :] * s[:, None]
        v_com = (r1 - r0)[None, :] * s_dot[:, None]
        L_com = np.zeros((M + 1, 3))
        f_stance = np.zeros((M, 3))
        tau_stance = np.zeros((M, 3))

        result = cls(
            T_step=T_step,
            t_grid=t_grid,
            r_com=r_com,
            v_com=v_com,
            L_com=L_com,
            f_stance=f_stance,
            tau_stance=tau_stance,
            success=True,
            solve_time_ms=0.0,
            cost=0.0,
            status='heuristic',
            iter_count=0,
        )
        result._mass = float(robot_mass)
        return result


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


class CoarsePrePlanner:
    """Coarse pre-planner using CasADi/IPOPT.

    Usage:
        planner = CoarsePrePlanner(CoarsePrePlannerConfig(robot_mass=71.0))
        planner.build()
        result = planner.solve(
            r_com_0, v_com_0, L_com_0,
            r_com_goal, r_C_stance, c_const,
            T_step=6.0)
    """

    def __init__(self, config: Optional[CoarsePrePlannerConfig] = None):
        self.config = config if config is not None else CoarsePrePlannerConfig()
        self._built = False
        # Opti handles
        self._opti: Optional[ca.Opti] = None
        self._X: Optional[ca.MX] = None
        self._U: Optional[ca.MX] = None
        self._p_x0: Optional[ca.MX] = None
        self._p_r_goal: Optional[ca.MX] = None
        self._p_r_C: Optional[ca.MX] = None
        self._p_c_const: Optional[ca.MX] = None
        self._p_h_max: Optional[ca.MX] = None
        self._p_dt: Optional[ca.MX] = None

    # ── Build ──────────────────────────────────────────────────────────

    def build(self) -> None:
        """Build the CasADi NLP once. Subsequent solves reuse it."""
        cfg = self.config
        M = cfg.M
        m = float(cfg.robot_mass)

        opti = ca.Opti()

        # --- Decision variables ---
        X = opti.variable(9, M + 1)       # [r(3); v(3); L(3)]
        U = opti.variable(6, M)           # [f(3); τ(3)]

        # --- Parameters ---
        p_x0 = opti.parameter(9)          # initial state
        p_r_goal = opti.parameter(3)      # goal r_com
        p_r_C = opti.parameter(3)         # stance contact point in R_s
        p_c_const = opti.parameter(3)     # conservation constant c
        p_h_max = opti.parameter(3)       # momentum box (Nms)
        p_dt = opti.parameter()           # collocation step [s]

        # --- Centroidal ODE (one active contact) ---
        def f_ode(x: ca.MX, u: ca.MX) -> ca.MX:
            r = x[0:3]
            v = x[3:6]
            # L = x[6:9]  # not used directly in ODE
            f = u[0:3]
            tau = u[3:6]
            r_dot = v
            v_dot = f / m
            L_dot = ca.cross(p_r_C - r, f) + tau
            return ca.vertcat(r_dot, v_dot, L_dot)

        # --- Multiple shooting with RK4 ---
        for k in range(M):
            xk = X[:, k]
            uk = U[:, k]
            xkp1 = X[:, k + 1]

            k1 = f_ode(xk,                uk)
            k2 = f_ode(xk + 0.5 * p_dt * k1, uk)
            k3 = f_ode(xk + 0.5 * p_dt * k2, uk)
            k4 = f_ode(xk +       p_dt * k3, uk)
            xk_next = xk + (p_dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            opti.subject_to(xkp1 == xk_next)

            # Wrench box
            fk = uk[0:3]
            tauk = uk[3:6]
            opti.subject_to(opti.bounded(-cfg.f_max, fk, cfg.f_max))
            opti.subject_to(opti.bounded(-cfg.tau_max, tauk, cfg.tau_max))

            # Rate bound on L̇ evaluated at the start of the interval.
            # |L̇|_∞ ≤ τ_{w,max} is the same bound the NMPC enforces.
            rk = xk[0:3]
            L_dot = ca.cross(p_r_C - rk, fk) + tauk
            opti.subject_to(opti.bounded(-cfg.tau_w_max, L_dot, cfg.tau_w_max))

        # --- Boundary conditions ---
        opti.subject_to(X[:, 0] == p_x0)
        opti.subject_to(X[0:3, M] == p_r_goal)

        # Terminal velocity and momentum: soft box + penalty in cost.
        eps_v = cfg.eps_v_terminal
        eps_L = cfg.eps_L_terminal
        opti.subject_to(opti.bounded(-eps_v, X[3:6, M], eps_v))
        opti.subject_to(opti.bounded(-eps_L, X[6:9, M], eps_L))

        # --- Momentum box at every collocation point ---
        # h_w^s(k) = c − L_com(k) − r_com(k) × m v_com(k)
        kappa = float(cfg.kappa_terminal)
        for k in range(M + 1):
            rk = X[0:3, k]
            vk = X[3:6, k]
            Lk = X[6:9, k]
            hw_k = p_c_const - Lk - ca.cross(rk, m * vk)
            if k == M:
                bound = kappa * p_h_max
            else:
                bound = p_h_max
            opti.subject_to(opti.bounded(-bound, hw_k, bound))

        # --- Cost ---
        cost = 0
        for k in range(M):
            Lk = X[6:9, k]
            fk = U[0:3, k]
            tauk = U[3:6, k]
            cost += cfg.w_L * ca.dot(Lk, Lk)
            cost += cfg.w_u * (ca.dot(fk, fk) + ca.dot(tauk, tauk))
        # Terminal penalty on residual v, L (on top of the box constraint)
        cost += cfg.w_v_terminal * ca.dot(X[3:6, M], X[3:6, M])
        cost += cfg.w_L_terminal * ca.dot(X[6:9, M], X[6:9, M])
        opti.minimize(cost)

        # --- Solver ---
        p_opts = {"expand": True, "print_time": 0}
        s_opts = {
            "print_level": cfg.ipopt_print_level,
            "max_iter": cfg.ipopt_max_iter,
            "tol": cfg.ipopt_tol,
            "sb": "yes",  # suppress banner
        }
        opti.solver("ipopt", p_opts, s_opts)

        # Retain handles
        self._opti = opti
        self._X = X
        self._U = U
        self._p_x0 = p_x0
        self._p_r_goal = p_r_goal
        self._p_r_C = p_r_C
        self._p_c_const = p_c_const
        self._p_h_max = p_h_max
        self._p_dt = p_dt
        self._built = True

    # ── Solve ──────────────────────────────────────────────────────────

    def solve(
        self,
        r_com_0: np.ndarray,
        v_com_0: np.ndarray,
        L_com_0: np.ndarray,
        r_com_goal: np.ndarray,
        r_C_stance: np.ndarray,
        c_const: np.ndarray,
        T_step: Optional[float] = None,
        h_max: Optional[np.ndarray] = None,
    ) -> CoarsePlanResult:
        """Solve the coarse pre-planner NLP for one crawl step.

        Parameters
        ----------
        r_com_0, v_com_0, L_com_0 : (3,)
            Initial centroidal state in the structure frame R_s.
        r_com_goal : (3,)
            Target CoM position at the end of the step (from IK).
        r_C_stance : (3,)
            Stance contact point in R_s (constant during SS).
        c_const : (3,)
            Conservation constant
            c = h_w^{s,0} + L_com^{s,0} + r_com^{s,0} × m v_com^{s,0}.
        T_step : float, optional
            Step duration [s]. Defaults to `config.T_step_default`.
        h_max : (3,), optional
            Per-component momentum box. Defaults to `config.h_max`.

        Returns
        -------
        CoarsePlanResult
        """
        if not self._built:
            self.build()

        cfg = self.config
        if T_step is None:
            T_step = cfg.T_step_default
        if h_max is None:
            h_max = cfg.h_max

        M = cfg.M
        dt = T_step / M

        x0 = np.concatenate([
            np.asarray(r_com_0, dtype=float).reshape(3),
            np.asarray(v_com_0, dtype=float).reshape(3),
            np.asarray(L_com_0, dtype=float).reshape(3),
        ])
        r_goal_arr = np.asarray(r_com_goal, dtype=float).reshape(3)
        r_C_arr = np.asarray(r_C_stance, dtype=float).reshape(3)
        c_arr = np.asarray(c_const, dtype=float).reshape(3)
        h_max_arr = np.asarray(h_max, dtype=float).reshape(3)

        opti = self._opti
        opti.set_value(self._p_x0, x0)
        opti.set_value(self._p_r_goal, r_goal_arr)
        opti.set_value(self._p_r_C, r_C_arr)
        opti.set_value(self._p_c_const, c_arr)
        opti.set_value(self._p_h_max, h_max_arr)
        opti.set_value(self._p_dt, dt)

        # Initial guess: linear interpolation for r_com, zero for v, L, u.
        X_init = np.zeros((9, M + 1))
        for k in range(M + 1):
            alpha = k / M
            X_init[0:3, k] = (1.0 - alpha) * x0[0:3] + alpha * r_goal_arr
            X_init[3:6, k] = (r_goal_arr - x0[0:3]) / T_step
        # Zero v at the terminal knot
        X_init[3:6, M] = 0.0
        U_init = np.zeros((6, M))
        opti.set_initial(self._X, X_init)
        opti.set_initial(self._U, U_init)

        t0 = time.perf_counter()
        success = True
        status = "Solve_Succeeded"
        sol = None
        try:
            sol = opti.solve()
        except RuntimeError as e:
            success = False
            status = f"{type(e).__name__}: {e}".splitlines()[0][:200]
        solve_ms = (time.perf_counter() - t0) * 1000.0

        # Extract trajectory (use .debug values on failure — still useful)
        source = sol if sol is not None else opti.debug
        try:
            X_opt = np.asarray(source.value(self._X), dtype=float).reshape(9, M + 1)
            U_opt = np.asarray(source.value(self._U), dtype=float).reshape(6, M)
            cost_val = float(source.value(opti.f))
        except Exception:
            X_opt = X_init.copy()
            U_opt = np.zeros((6, M))
            cost_val = float("inf")

        try:
            stats = opti.stats() if sol is not None else {}
            iter_count = int(stats.get("iter_count", 0)) if stats else 0
        except Exception:
            iter_count = 0

        t_grid = np.linspace(0.0, T_step, M + 1)
        result = CoarsePlanResult(
            T_step=float(T_step),
            t_grid=t_grid,
            r_com=X_opt[0:3, :].T.copy(),
            v_com=X_opt[3:6, :].T.copy(),
            L_com=X_opt[6:9, :].T.copy(),
            f_stance=U_opt[0:3, :].T.copy(),
            tau_stance=U_opt[3:6, :].T.copy(),
            success=success,
            solve_time_ms=solve_ms,
            cost=cost_val,
            status=status,
            iter_count=iter_count,
        )
        # Attach mass so hw_at_knots can reconstruct the box residual.
        result._mass = m = float(self.config.robot_mass)  # noqa: E501
        return result
