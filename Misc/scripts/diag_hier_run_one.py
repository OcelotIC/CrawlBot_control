#!/usr/bin/env python3
"""Phase HIER-STANDARD-SWEEP — one canonical-stack run at (torso, EE), momentum FIXED 5000.

Canonical stack (posture 20, torque 1, wrench 0.01, accel-reg 0.01, eps 1e-6) — ONLY
torso-pose and swing-EE set via CLI; ss_alpha_mom held at 5000. Captures lambda_min/max
of H_LS (task-only) at strided ticks for kappa. Full 6-step canonical C.

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 Misc/scripts/diag_hier_run_one.py <torso> <ee> <tag>
"""
import sys
import json
import numpy as np
import crawlbot.solvers.hierarchical_qp as hq

TORSO = float(sys.argv[1]); EE = float(sys.argv[2]); TAG = sys.argv[3]
MOM = 5000.0
CAP = []; CALLS = [0]
_orig = hq.HierarchicalQP._solve_weighted


def _asm_LS(tasks, n):
    H = np.zeros((n, n))
    for t in tasks:
        H += (t.A.T @ t.W) @ t.A
    return 0.5 * (H + H.T)


def _pw(self, tasks, x0):
    idx = CALLS[0]; CALLS[0] += 1
    if idx % 20 == 0:
        n = self.n_vars
        try:
            w = np.linalg.eigvalsh(_asm_LS(tasks, n))
            ne = 0 if self._C_eq is None else int(np.atleast_2d(self._C_eq).shape[0])
            CAP.append((idx, ne, float(w[0]), float(w[-1])))
        except np.linalg.LinAlgError:
            pass
    return _orig(self, tasks, x0)


hq.HierarchicalQP._solve_weighted = _pw

import scripts.diag_cooperative_arms as dca
from scripts.diag_cooperative_arms import _mutate_mjcf, _mjcf_md5, MJCF

out = f'figC_hier_{TAG}'
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
            n_steps=6, ss_two_task=True, ss_alpha_mom=MOM,
            alpha_torso_pose=TORSO, ss_alpha_ee=EE, ss_alpha_posture=2e1,
            ss_alpha_wrench=1e-2, ss_kp_torso=3.0, ss_kd_torso=2.5,
            qp_envelope_exact=True,
            interstep_settle_alpha_wrench=3.0, interstep_settle_epsilon_v=5e-3,
            out_dir_override=out)
    except Exception as e:
        print(f'[note] main() raised after sim: {type(e).__name__}: {str(e)[:160]}')
finally:
    with open(MJCF, 'w') as f:
        f.write(_original)
    assert _mjcf_md5(MJCF) == _pre, 'MJCF restore failed'

import os
os.makedirs(f'results/{out}', exist_ok=True)
json.dump({'torso': TORSO, 'ee': EE, 'mom': MOM, 'cap': CAP},
          open(f'results/{out}/regcap.json', 'w'))
print(f'wrote results/{out}/regcap.json ({CALLS[0]} solves, {len(CAP)} kappa caps)')
