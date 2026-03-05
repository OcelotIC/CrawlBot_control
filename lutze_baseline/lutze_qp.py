"""Lutze et al. (2023) single-step QP wrench optimizer for dual contacts.

Extends the paper's 6-dim QP (single contact) to 12-dim (dual arm).
Minimizes satellite/structure disturbance while tracking desired robot motion.

Wrench convention: [f(3), tau(3)] per contact (force first, matching MPC).
"""

from dataclasses import dataclass, field
import numpy as np
import qpsolvers


@dataclass
class LutzeQPConfig:
    """Configuration for the Lutze wrench-optimization QP."""
    Qr: np.ndarray = field(default_factory=lambda: np.eye(6) * 1.0)
    Qb: np.ndarray = field(default_factory=lambda: np.eye(3) * 10.0)
    Qc: float = 0.01
    F_max: float = 3000.0    # N per contact
    tau_max: float = 300.0    # Nm per contact
    h_dot_max: float = None   # Optional angular momentum rate limit (Nms/s)

    def __post_init__(self):
        for attr in ['Qr', 'Qb']:
            val = getattr(self, attr)
            if not isinstance(val, np.ndarray):
                setattr(self, attr, np.asarray(val, dtype=float))


class LutzeQP:
    """Single-step QP wrench optimizer (Lutze et al. 2023, dual-arm variant)."""

    def __init__(self, config=None):
        self.config = config or LutzeQPConfig()
        self._build_bounds()

    def _build_bounds(self):
        """Build per-contact wrench bounds [force(3); torque(3)]."""
        cfg = self.config
        single_lb = np.array([-cfg.F_max]*3 + [-cfg.tau_max]*3)
        single_ub = np.array([cfg.F_max]*3 + [cfg.tau_max]*3)
        # Dual contact: stack [Fc_a; Fc_b]
        self.lb_dual = np.tile(single_lb, 2)
        self.ub_dual = np.tile(single_ub, 2)
        self.lb_single = single_lb
        self.ub_single = single_ub

    def solve(self, Ad_a, Ad_b, M_lambda, F_d_r, F_d_b):
        """Solve the wrench optimization QP.

        Parameters
        ----------
        Ad_a : (6, 6) or None - Adjoint wrench map for contact A.
        Ad_b : (6, 6) or None - Adjoint wrench map for contact B.
        M_lambda : (3, n) - Momentum map (n=12 dual, 6 single, 0 none).
        F_d_r : (6,) - Desired robot tracking wrench.
        F_d_b : (6,) - Desired structure stabilization wrench (unused directly;
                        structure objective uses M_lambda instead).

        Returns
        -------
        Fc_a : (6,) contact wrench at A (zeros if inactive).
        Fc_b : (6,) contact wrench at B (zeros if inactive).
        info : dict with 'status', 'L_dot'.
        """
        cfg = self.config
        active_a = Ad_a is not None
        active_b = Ad_b is not None

        if not active_a and not active_b:
            return np.zeros(6), np.zeros(6), {'status': 'no_contacts'}

        # Build A_wrench: maps stacked Fc → net wrench in world frame
        Ad_blocks = []
        if active_a:
            Ad_blocks.append(Ad_a)
        if active_b:
            Ad_blocks.append(Ad_b)
        A_wrench = np.hstack(Ad_blocks)  # (6, n)
        n = A_wrench.shape[1]

        # --- Hessian ---
        # Term 1: tracking  ||A_wrench @ Fc - F_d_r||^2_Qr
        P1 = A_wrench.T @ cfg.Qr @ A_wrench
        q1 = -2.0 * A_wrench.T @ cfg.Qr @ F_d_r

        # Term 2: minimize angular momentum rate  ||M_lambda @ Fc||^2_Qb
        M = M_lambda
        if M.shape[1] == n:
            P2 = M.T @ cfg.Qb @ M
            q2 = np.zeros(n)
        else:
            P2 = np.zeros((n, n))
            q2 = np.zeros(n)

        # Term 3: regularization  ||Fc||^2_Qc
        P3 = np.eye(n) * cfg.Qc
        q3 = np.zeros(n)

        P = P1 + P2 + P3
        q = q1 + q2 + q3

        # Symmetrize and regularize
        P = 0.5 * (P + P.T)
        P += np.eye(n) * 1e-8

        # --- Bounds ---
        if active_a and active_b:
            lb, ub = self.lb_dual, self.ub_dual
        else:
            lb, ub = self.lb_single, self.ub_single

        # --- Optional angular momentum rate constraint ---
        G, h_ineq = None, None
        if cfg.h_dot_max is not None and M.shape[1] == n:
            # |M @ Fc| <= h_dot_max  →  M @ Fc <= h_dot_max, -M @ Fc <= h_dot_max
            G = np.vstack([M, -M])
            h_ineq = np.ones(2 * M.shape[0]) * cfg.h_dot_max

        # --- Solve ---
        try:
            from scipy.sparse import csc_matrix
            P_sparse = csc_matrix(P)
            Fc_opt = qpsolvers.solve_qp(
                P=P_sparse, q=q,
                G=G, h=h_ineq,
                lb=lb[:n], ub=ub[:n],
                solver='osqp',
                eps_abs=1e-6, eps_rel=1e-6,
                max_iter=200,
            )
        except Exception:
            Fc_opt = None

        if Fc_opt is None:
            # Fallback: pseudoinverse projection
            Fc_opt = np.linalg.pinv(A_wrench) @ F_d_r
            Fc_opt = np.clip(Fc_opt, lb[:n], ub[:n])
            status = 'fallback'
        else:
            status = 'optimal'

        # Unpack
        Fc_a = np.zeros(6)
        Fc_b = np.zeros(6)
        idx = 0
        if active_a:
            Fc_a = Fc_opt[idx:idx+6]
            idx += 6
        if active_b:
            Fc_b = Fc_opt[idx:idx+6]

        L_dot = M @ Fc_opt if M.shape[1] == n else np.zeros(3)

        return Fc_a, Fc_b, {'status': status, 'L_dot': L_dot}
