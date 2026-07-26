"""Lutze et al. (2023) single-step QP wrench optimizer for dual contacts.

Extends the paper's 6-dim QP (single contact) to 12-dim (dual arm).
Minimizes satellite/structure disturbance while tracking desired robot motion.

    min  ||A_wrench @ Fc - F_d_r||²_Qr      (robot tracking)
       + ||M_lambda @ Fc||²_Qb               (minimize angular momentum rate)
       + ||Fc||²_Qc                           (wrench regularization)

    s.t.  Fc_min <= Fc <= Fc_max              (SI / actuator limits)
          (optional) |M_lambda @ Fc| <= tau_w_max  (momentum rate constraint)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple

try:
    from qpsolvers import solve_qp
    HAS_QPSOLVERS = True
except ImportError:
    HAS_QPSOLVERS = False


@dataclass
class LutzeQPConfig:
    """Configuration for the Lutze wrench-optimization QP."""
    # QP weights
    Qr: np.ndarray = None   # (6,6) robot tracking weight
    Qb: np.ndarray = None   # (3,3) momentum rate minimization weight
    Qc: np.ndarray = None   # wrench regularization weight (sized at runtime)

    # Contact wrench bounds (per contact)
    f_max: float = 25.0     # [N] force magnitude per axis
    tau_max: float = 8.0    # [Nm] torque magnitude per axis

    # Momentum rate constraint (optional)
    tau_w_max: float = np.inf  # [Nm] max |L_dot| per axis (inf = disabled)

    # Solver
    solver: str = 'osqp'
    reg: float = 1e-6       # Hessian regularization

    def __post_init__(self):
        if self.Qr is None:
            self.Qr = np.eye(6) * 1.0
        if self.Qb is None:
            self.Qb = np.eye(3) * 10.0
        if self.Qc is None:
            self.Qc = None  # will be set to eye(n_fc) * 0.01 at solve time


class LutzeQP:
    """Single-step QP wrench optimizer (Lutze et al. 2023, dual-arm variant)."""

    def __init__(self, config: LutzeQPConfig = None):
        self.cfg = config or LutzeQPConfig()

    def solve(
        self,
        Ad_a: Optional[np.ndarray],
        Ad_b: Optional[np.ndarray],
        M_lambda: np.ndarray,
        F_d_r: np.ndarray,
        F_d_b: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Solve the wrench optimization QP.

        Parameters
        ----------
        Ad_a : (6, 6) or None – Adjoint wrench map for contact A.
        Ad_b : (6, 6) or None – Adjoint wrench map for contact B.
        M_lambda : (3, n_fc) – Momentum map (contact wrenches → L_dot).
        F_d_r : (6,) – Desired robot tracking wrench (world frame).
        F_d_b : (6,) – Desired structure stabilization wrench (unused when Kb=Db=0).
        Returns
        -------
        Fc_a : (6,) – Optimal wrench for contact A (zeros if inactive).
        Fc_b : (6,) – Optimal wrench for contact B (zeros if inactive).
        info : dict – Solver info {'status', 'cost', 'Fc_full'}.
        """
        cfg = self.cfg

        # Determine active contacts and build stacked adjoint
        contacts = []
        if Ad_a is not None:
            contacts.append(('a', Ad_a))
        if Ad_b is not None:
            contacts.append(('b', Ad_b))

        n_contacts = len(contacts)
        if n_contacts == 0:
            return np.zeros(6), np.zeros(6), {'status': 'no_contacts'}

        n_fc = 6 * n_contacts

        # Stacked adjoint: A_wrench (6 x n_fc) maps Fc → world wrench
        A_wrench = np.zeros((6, n_fc))
        for i, (name, Ad) in enumerate(contacts):
            A_wrench[:, 6*i:6*(i+1)] = Ad

        # Weight matrices
        Qr = cfg.Qr
        Qb = cfg.Qb
        Qc = cfg.Qc if cfg.Qc is not None else np.eye(n_fc) * 0.01

        # Ensure Qc is right size
        if Qc.shape[0] != n_fc:
            Qc = np.eye(n_fc) * 0.01

        # --- Build QP Hessian and gradient ---
        # Term 1: ||A_wrench @ Fc - F_d_r||²_Qr
        H1 = A_wrench.T @ Qr @ A_wrench
        g1 = -A_wrench.T @ Qr @ F_d_r

        # Term 2: ||M_lambda @ Fc||²_Qb (minimize angular momentum rate)
        H2 = M_lambda.T @ Qb @ M_lambda
        g2 = np.zeros(n_fc)

        # Term 3: ||Fc||²_Qc
        H3 = Qc
        g3 = np.zeros(n_fc)

        H = H1 + H2 + H3
        g = g1 + g2 + g3

        # Regularize
        H = H + cfg.reg * np.eye(n_fc)
        # Ensure symmetry
        H = 0.5 * (H + H.T)

        # --- Bounds ---
        lb = np.zeros(n_fc)
        ub = np.zeros(n_fc)
        for i in range(n_contacts):
            lb[6*i:6*i+3] = -cfg.tau_max
            lb[6*i+3:6*i+6] = -cfg.f_max
            ub[6*i:6*i+3] = cfg.tau_max
            ub[6*i+3:6*i+6] = cfg.f_max

        # --- Inequality constraints for momentum rate (optional) ---
        G = None
        h_ineq = None
        if cfg.tau_w_max < np.inf and M_lambda.shape[1] == n_fc:
            # |M_lambda @ Fc| <= tau_w_max  →  ±M_lambda @ Fc <= tau_w_max
            G = np.vstack([M_lambda, -M_lambda])
            h_ineq = np.ones(6) * cfg.tau_w_max

        # --- Solve ---
        info = {'status': 'unknown', 'cost': np.inf, 'Fc_full': None}

        Fc_opt = None
        if HAS_QPSOLVERS:
            try:
                Fc_opt = solve_qp(
                    P=H, q=g, G=G, h=h_ineq,
                    lb=lb, ub=ub,
                    solver=cfg.solver,
                    max_iter=200,
                )
            except Exception:
                Fc_opt = None

        if Fc_opt is None:
            # Fallback: unconstrained least-squares
            try:
                Fc_opt = np.linalg.solve(H, -g)
                Fc_opt = np.clip(Fc_opt, lb, ub)
            except np.linalg.LinAlgError:
                Fc_opt = np.zeros(n_fc)
            info['status'] = 'fallback'
        else:
            info['status'] = 'optimal'

        info['Fc_full'] = Fc_opt.copy()
        info['cost'] = 0.5 * Fc_opt @ H @ Fc_opt + g @ Fc_opt

        # Split back to per-contact wrenches
        Fc_a = np.zeros(6)
        Fc_b = np.zeros(6)
        idx = 0
        for name, _ in contacts:
            if name == 'a':
                Fc_a = Fc_opt[idx:idx+6]
            else:
                Fc_b = Fc_opt[idx:idx+6]
            idx += 6

        return Fc_a, Fc_b, info
