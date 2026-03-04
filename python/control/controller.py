"""Lutze QP Controller.

Translated from LutzeQPController.m.
Implements the QP controller from Lutze et al. (2023), equations (12-22).
"""

import numpy as np
import warnings
from ..utils.spatial import adjoint_se3
from .feedforward import compute_feedforward_wrench

try:
    from qpsolvers import solve_qp as _solve_qp
    HAS_QPSOLVERS = True
except ImportError:
    HAS_QPSOLVERS = False


class LutzeQPController:
    """QP-based controller for minimizing satellite disturbances.

    The optimization trades off:
    - Robot trajectory tracking (weight Qr)
    - Satellite stabilization (weight Qb)
    - Contact wrench magnitude (weight Qc)
    """

    def __init__(self, env):
        self.env = env
        self.Qr = env['control']['Qr']
        self.Qb = env['control']['Qb']
        self.Qc = env['control']['Qc']
        self.Kr = env['control']['Kr']
        self.Dr = env['control']['Dr']
        self.Kb = env['control']['Kb']
        self.Db = env['control']['Db']

        # Wrench limits (eq. 22)
        self.F_min = np.concatenate([
            -env['SI']['F_max'] * np.ones(3),
            -env['SI']['tau_max'] * np.ones(3),
        ])
        self.F_max = np.concatenate([
            env['SI']['F_max'] * np.ones(3),
            env['SI']['tau_max'] * np.ones(3),
        ])

    def solve(self, sys, traj_des, t_idx, use_optimization):
        """Solve for optimal contact wrench and joint torques.

        Parameters
        ----------
        sys : dict – System state from compute_system_state.
        traj_des : dict – Desired trajectory.
        t_idx : int – Time index (0-based).
        use_optimization : bool – If True, use QP; else direct control.

        Returns
        -------
        Fc_opt : (6,) – Optimal contact wrench.
        tau_joints : (n_q,) – Joint torques.
        """
        time = traj_des['t'][t_idx] if 'time' not in sys else sys['time']

        gains = {
            'Kr': self.Kr, 'Dr': self.Dr,
            'Kb': self.Kb, 'Db': self.Db,
        }

        # Desired wrenches (eq. 12-16)
        F_d_r, F_d_b = compute_feedforward_wrench(sys, traj_des, t_idx, gains)

        if use_optimization:
            Fc_opt = self._solve_qp(F_d_r, F_d_b, sys, time)
        else:
            # Non-optimized: robot tracking only
            g_cb = self._build_transform(sys)
            Ad_gcb = adjoint_se3(g_cb)
            Fc_opt = np.linalg.pinv(Ad_gcb.T) @ F_d_r
            Fc_opt = np.clip(Fc_opt, self.F_min, self.F_max)

        # Map Fc -> tau_joints
        tau_joints = sys['Jc_joints'].T @ Fc_opt
        return Fc_opt, tau_joints

    def _solve_qp(self, F_d_r, F_d_b, sys, time):
        """QP formulation (eq. 18-22) with fallback."""
        # Validate inputs
        if np.any(np.isnan(F_d_r)) or np.any(np.isinf(F_d_r)):
            warnings.warn(f"[QP] F_d_r contains NaN/Inf at t={time:.2f}")
            return np.zeros(6)
        if np.any(np.isnan(F_d_b)) or np.any(np.isinf(F_d_b)):
            warnings.warn(f"[QP] F_d_b contains NaN/Inf at t={time:.2f}")
            return np.zeros(6)

        # Contact -> base transform
        try:
            g_cb = self._build_transform(sys)
        except Exception as e:
            warnings.warn(f"[QP] Error building transform: {e}")
            return np.zeros(6)

        Ad_gcb = adjoint_se3(g_cb)

        if np.any(np.isnan(Ad_gcb)) or np.any(np.isinf(Ad_gcb)):
            warnings.warn(f"[QP] Adjoint invalid at t={time:.2f}")
            return np.zeros(6)

        # Residuals (eq. 19-21)
        A_r = Ad_gcb.T
        b_r = F_d_r
        A_b = Ad_gcb.T
        b_b = -F_d_b
        A_c = np.eye(6)
        b_c = np.zeros(6)

        # Hessian and gradient
        H = A_r.T @ self.Qr @ A_r + A_b.T @ self.Qb @ A_b + A_c.T @ self.Qc @ A_c
        H = (H + H.T) / 2  # Symmetrize

        # Regularization if ill-conditioned
        cond_H = np.linalg.cond(H)
        if cond_H > 1e12 or np.any(np.isnan(H)) or np.any(np.isinf(H)):
            warnings.warn(f"[QP] H ill-conditioned (cond={cond_H:.2e}) at t={time:.2f}")
            H = H + 1e-6 * np.eye(6)

        f = -2 * (b_r @ self.Qr @ A_r + b_b @ self.Qb @ A_b + b_c @ self.Qc @ A_c)

        if np.any(np.isnan(f)) or np.any(np.isinf(f)):
            warnings.warn(f"[QP] Gradient invalid at t={time:.2f}")
            return np.zeros(6)

        # Solve QP
        Fc_opt = self._qp_solve(H, f, self.F_min, self.F_max)

        if Fc_opt is None:
            # Fallback: pseudoinverse projection
            warnings.warn(f"[QP] Infeasible at t={time:.2f}, using fallback")
            Fc_opt = np.linalg.pinv(Ad_gcb.T, rcond=1e-6) @ F_d_r
            Fc_opt = np.clip(Fc_opt, self.F_min, self.F_max)
            if np.any(np.isnan(Fc_opt)) or np.any(np.isinf(Fc_opt)):
                warnings.warn("[QP] Fallback failed, returning zero")
                Fc_opt = np.zeros(6)

        return Fc_opt

    @staticmethod
    def _qp_solve(H, f, lb, ub):
        """Solve the box-constrained QP: min 0.5 x'Hx + f'x, lb <= x <= ub.

        Uses qpsolvers with OSQP if available, else falls back to scipy.
        """
        if HAS_QPSOLVERS:
            try:
                result = _solve_qp(
                    P=H, q=f,
                    lb=lb, ub=ub,
                    solver='osqp',
                    eps_abs=1e-6, eps_rel=1e-6,
                    max_iter=200,
                    verbose=False,
                )
                if result is not None:
                    return result
            except Exception:
                pass

        # Fallback to scipy
        try:
            from scipy.optimize import minimize

            def objective(x):
                return 0.5 * x @ H @ x + f @ x

            def gradient(x):
                return H @ x + f

            x0 = np.zeros(len(f))
            bounds = list(zip(lb, ub))
            res = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': 200})
            if res.success:
                return res.x
        except Exception:
            pass

        return None

    @staticmethod
    def _build_transform(sys):
        """Build contact-to-base homogeneous transform.

        g_cb = [R_end, r_contact; 0 0 0 1]
        """
        # RL is a list of (3,3) arrays
        R_end = sys['RL'][-1]
        r_contact = sys['r_contact']

        if R_end.shape != (3, 3):
            raise ValueError(f"R_end must be [3x3], got {R_end.shape}")
        if len(r_contact) != 3:
            raise ValueError(f"r_contact must be [3x1], got {r_contact.shape}")

        g_cb = np.eye(4)
        g_cb[:3, :3] = R_end
        g_cb[:3, 3] = r_contact
        return g_cb
