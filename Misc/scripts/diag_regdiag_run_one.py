#!/usr/bin/env python3
"""Phase REG-DIAG — run canonical C at one regularization epsilon (measurement).

Monkeypatches HierarchicalQP._solve_weighted to (1) force self.regularization = EPS
(the WBC never passes it, so the shipped default 1e-6 is otherwise used), and (2)
capture lambda_min/lambda_max of the task-only Hessian H_LS = Sum W A^T A (NO reg) at
strided ticks. Since eig(H_LS + eps I) = eig(H_LS) + eps, one run's (lam_min,lam_max)
gives kappa(H) for ANY eps analytically; capturing them here (they are eps-independent)
also cross-checks the well-posedness floor min lam_min(H_LS) across the trajectory.

Runs the EE=3000 canonical C baseline (NOT ee6000 — epsilon isolated), full 6-step.
Nothing committed to crawlbot/.

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 Misc/scripts/diag_regdiag_run_one.py <eps> <tag>
"""
import os
import sys
import json
import numpy as np
import crawlbot.solvers.hierarchical_qp as hq

EPS = float(sys.argv[1])
TAG = sys.argv[2]
STRIDE = 20
CALLS = [0]
CAP = []   # (idx, n_eq, lam_min_LS, lam_max_LS)

_orig_w = hq.HierarchicalQP._solve_weighted


def _assemble_LS(tasks, n):
    H = np.zeros((n, n))
    for t in tasks:
        AtW = t.A.T @ t.W
        H += AtW @ t.A
    return 0.5 * (H + H.T)


def _patched_w(self, sorted_tasks, x0):
    self.regularization = EPS               # force the swept epsilon
    idx = CALLS[0]; CALLS[0] += 1
    if idx % STRIDE == 0:
        n = self.n_vars
        H_LS = _assemble_LS(sorted_tasks, n)
        try:
            w = np.linalg.eigvalsh(H_LS)
            lam_min, lam_max = float(w[0]), float(w[-1])
        except np.linalg.LinAlgError:
            lam_min, lam_max = float('nan'), float('nan')
        n_eq = 0 if self._C_eq is None else int(np.atleast_2d(self._C_eq).shape[0])
        CAP.append((idx, n_eq, lam_min, lam_max))
    return _orig_w(self, sorted_tasks, x0)


hq.HierarchicalQP._solve_weighted = _patched_w

import scripts.diag_cooperative_arms as dca
from scripts.diag_cooperative_arms import _mutate_mjcf, _mjcf_md5, MJCF

out_dir = f'figC_reg_{TAG}'
with open(MJCF) as f:
    _original = f.read()
_pre = _mjcf_md5(MJCF)
try:
    _mutate_mjcf(damping=0.0, armature=0.05, anchor_dx=0.8, mass_ratio=0.01)
    try:
        dca.main(
            legacy=False, alpha_torso_lin=0.0, anchor_dx=0.8, mass_ratio=0.01,
            aocs_mode='legacy_pid_numerical', settle_seconds=20.0,
            K_theta=1.0, K_omega=50.0, tau_w_max=5.0,
            n_steps=6, ss_two_task=True, ss_alpha_mom=5000.0,
            alpha_torso_pose=24000.0, ss_alpha_ee=3000.0, ss_alpha_posture=2e1,
            ss_alpha_wrench=1e-2, ss_kp_torso=3.0, ss_kd_torso=2.5,
            qp_envelope_exact=True,
            interstep_settle_alpha_wrench=3.0, interstep_settle_epsilon_v=5e-3,
            out_dir_override=out_dir)
    except Exception as e:
        print(f'[note] main() raised after sim: {type(e).__name__}: {str(e)[:160]}')
finally:
    with open(MJCF, 'w') as f:
        f.write(_original)
    assert _mjcf_md5(MJCF) == _pre, 'MJCF restore failed'

cap = np.array(CAP, float) if CAP else np.zeros((0, 4))
os.makedirs(f'results/{out_dir}', exist_ok=True)
json.dump({'eps': EPS, 'tag': TAG, 'n_solves': CALLS[0],
           'cap': CAP}, open(f'results/{out_dir}/regcap.json', 'w'))
if len(cap):
    ss = cap[:, 1] == 26; ds = cap[:, 1] == 32
    print(f'eps={EPS:g} tag={TAG}: {CALLS[0]} solves; '
          f'min lam_min(H_LS)={cap[:,2].min():.4e} (over run); '
          f'SS lam_min med={np.median(cap[ss,2]) if ss.any() else float("nan"):.4e}; '
          f'DS lam_min med={np.median(cap[ds,2]) if ds.any() else float("nan"):.4e}')
print(f'wrote results/{out_dir}/regcap.json')
