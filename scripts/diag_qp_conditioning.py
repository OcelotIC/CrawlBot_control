#!/usr/bin/env python3
"""Phase QP-cond — instrument the canonical C (h_max=5) run to measure QP conditioning
and NMPC prediction. DIAGNOSTICS ONLY, ZERO crawlbot/ change (monkeypatch only).

Captures (canonical config unchanged; C's raw sim_log is ephemeral so we rerun it — the run
reproduces committed C bit-for-bit: docks [4.94,4.41,4.88,4.42,4.76,4.92]):
  TASK A — every 10th whole-body-QP solve: cond(H) of the assembled cost Hessian
    (HierarchicalQP._solve_weighted: H = Σ Aᵢᵀ Wᵢ Aᵢ + reg·I, hierarchical_qp.py:271),
    cond(H) with the max-weight (torso, α=24000) block removed, the dominant task by
    weight and by block spectral norm, qpOASES return_status (solver.stats), success/cost,
    and the phase tag (SS/DS) from WholeBodyQP.diag_label (set by sim_loop:3032).
  TASK B feed — every CentroidalNMPC.solve: inputs (r/v/L_com measured) and outputs
    (r/v/L_com_plan at t+dt, lambda_plan) so we can form planned Ḣ_s and the k+1 residual
    (NMPC-predicted next state vs the measured state fed in at the next NMPC tick).

Dumps: results/j2_ablation_envelope/qp_cond_raw.json (qp[], nmpc[]).
tau_q, hw, tau_w, phase per tick come from the run's sim_log (results/figC_qpcond/).

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_qp_conditioning.py
"""
import json
import os
import runpy
import sys

import numpy as np

import scripts.run_m7_single_step as r_single
import crawlbot.solvers.hierarchical_qp as hq
import crawlbot.solvers.wholebody_qp as wq
import crawlbot.solvers.centroidal_nmpc as cn

_orig_make = r_single._make_m7_config


def _mk(*a, **k):
    c = _orig_make(*a, **k)
    c.h_max_tight = np.full(3, 5.0)      # canonical
    return c


r_single._make_m7_config = _mk

WEIGHT_NAME = {24000.0: 'torso_pose', 10000.0: 'hw_slack', 5000.0: 'ss_alpha_mom'}


def _name(w):
    for wv, nm in WEIGHT_NAME.items():
        if abs(w - wv) / max(wv, 1) < 0.01:
            return nm
    return f'w={w:g}'


# ---- context from the whole-body QP (phase tag) ----
_ctx = {'label': '', 'settle': False}
_orig_wq = wq.WholeBodyQP.solve


def _wq_solve(self, *a, **k):
    _ctx['label'] = str(getattr(self, 'diag_label', ''))
    _ctx['settle'] = bool(k.get('settle_mode', False))
    return _orig_wq(self, *a, **k)


wq.WholeBodyQP.solve = _wq_solve

# ---- TASK A: cond(H) per (every 10th) QP solve ----
_qp = []
_n = [0]
_orig_raw = hq.HierarchicalQP._solve_qp_raw


def _cond(M):
    try:
        c = float(np.linalg.cond(M))
        return c if np.isfinite(c) else 1e300
    except Exception:
        return float('nan')


def _raw(self, H, g, C_eq, d_eq, C_ineq, d_ineq, lb, ub, x0=None):
    i = _n[0]; _n[0] += 1
    sample = (i % 10 == 0)
    if sample:
        n = H.shape[0]
        cond_H = _cond(H)
        tasks = list(self._tasks)
        weights = [float(np.max(np.diag(t.W))) if t.W.size else 0.0 for t in tasks]
        blk = [float(np.linalg.norm(t.A.T @ t.W @ t.A, 2)) if t.A.size else 0.0 for t in tasks]
        i_w = int(np.argmax(weights)) if weights else -1
        i_b = int(np.argmax(blk)) if blk else -1
        # cond(H) with the max-WEIGHT (torso) block removed
        cond_no_torso = float('nan')
        if tasks and i_w >= 0:
            Hnd = np.zeros((n, n))
            for j, t in enumerate(tasks):
                if j == i_w:
                    continue
                wp = self.weight_ratio ** (t.priority - 1)
                AtW = t.A.T @ (t.W / wp)
                Hnd += AtW @ t.A
            Hnd += self.regularization * np.eye(n)
            Hnd = 0.5 * (Hnd + Hnd.T)
            cond_no_torso = _cond(Hnd)
    x, info = _orig_raw(self, H, g, C_eq, d_eq, C_ineq, d_ineq, lb, ub, x0)
    if sample:
        m = 0
        if C_eq is not None and np.size(C_eq):
            m += np.atleast_2d(C_eq).shape[0]
        if C_ineq is not None and np.size(C_ineq):
            m += np.atleast_2d(C_ineq).shape[0]
        st = ''
        try:
            solver = self._solver_cache.get((H.shape[0], m))
            if solver is not None:
                st = str(solver.stats().get('return_status', ''))
        except Exception:
            pass
        lab = _ctx['label']
        phase = 'DS' if '/DS/' in lab or _ctx['settle'] else ('SS' if '/SS/' in lab else '?')
        _qp.append(dict(i=i, label=lab, phase=phase, settle=_ctx['settle'], n=int(H.shape[0]),
                        cond_H=cond_H, cond_no_torso=cond_no_torso,
                        dom_weight=weights[i_w] if i_w >= 0 else 0.0,
                        dom_task_byweight=_name(weights[i_w]) if i_w >= 0 else '',
                        dom_task_byblock=_name(weights[i_b]) if i_b >= 0 else '',
                        success=bool(info['success']), cost=float(info.get('cost', np.inf)),
                        status=st))
    return x, info


hq.HierarchicalQP._solve_qp_raw = _raw

# ---- TASK B feed: NMPC in/out ----
_nmpc = []
_orig_nmpc = cn.CentroidalNMPC.solve


def _nmpc_solve(self, r_com, v_com, L_com, *a, **k):
    out = _orig_nmpc(self, r_com, v_com, L_com, *a, **k)
    try:
        r_p, v_p, L_p, lam_p, info = out
        _nmpc.append(dict(
            r_in=[float(x) for x in np.ravel(r_com)], v_in=[float(x) for x in np.ravel(v_com)],
            L_in=[float(x) for x in np.ravel(L_com)],
            r_plan=[float(x) for x in np.ravel(r_p)], v_plan=[float(x) for x in np.ravel(v_p)],
            L_plan=[float(x) for x in np.ravel(L_p)],
            lam_plan=[float(x) for x in np.ravel(lam_p)]))
    except Exception:
        pass
    return out


cn.CentroidalNMPC.solve = _nmpc_solve

sys.argv = [
    'diag_cooperative_arms.py',
    '--ss-two-task', '--ss-alpha-mom', '5000', '--alpha-torso-pose', '24000',
    '--ss-kp-torso', '3', '--ss-kd-torso', '2.5',
    '--aocs_mode', 'legacy_pid_numerical', '--K_omega', '50', '--qp-envelope-exact',
    '--interstep-settle-alpha-wrench', '3', '--interstep-settle-epsilon-v', '5e-3',
    '--n-steps', '6', '--envelope-constraint', 'on', '--out-dir', 'figC_qpcond',
]
print('[qp-cond] instrumenting canonical C (h_max=5); sampling every 10th QP solve for cond(H)')
try:
    runpy.run_module('scripts.diag_cooperative_arms', run_name='__main__')
except SystemExit:
    pass

os.makedirs('results/j2_ablation_envelope', exist_ok=True)
json.dump({'qp': _qp, 'nmpc': _nmpc, 'dt_nmpc': 0.1, 'dt_qp': 0.01},
          open('results/j2_ablation_envelope/qp_cond_raw.json', 'w'))
print(f'\n[qp-cond] captured {len(_qp)} QP-cond samples (every 10th of {_n[0]} solves), '
      f'{len(_nmpc)} NMPC solves -> results/j2_ablation_envelope/qp_cond_raw.json')
