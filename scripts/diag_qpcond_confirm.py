#!/usr/bin/env python3
"""Phase QP-COND — closed-loop confirmation: does a common-factor weight rescale
change the canonical traversal?

Monkeypatches HierarchicalQP._solve_weighted to divide EVERY task weight W by a
common divisor D before assembly (ratios strictly preserved). Two modes:
  reg_scaled=False : leave HierarchicalQP.regularization = 1e-6 FIXED — this is
                     EXACTLY what editing the config task weights (÷D) would do,
                     since the 1e-6 reg is a separate absolute constructor arg.
  reg_scaled=True  : also divide reg by D -> pure scalar H->H/D (exact invariance).

Runs the full canonical C traversal (CANON, n_steps from arg) and prints the 6
docks + T_steps. Compare to the committed baseline (figC_sw_s5_x1).

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_qpcond_confirm.py <D> <reg_scaled 0|1> <n_steps>
"""
import os
import sys
import json
import numpy as np
import crawlbot.solvers.hierarchical_qp as hq

D = float(sys.argv[1]) if len(sys.argv) > 1 else 24.0
REG_SCALED = bool(int(sys.argv[2])) if len(sys.argv) > 2 else False
N_STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 6

_orig_w = hq.HierarchicalQP._solve_weighted


def _patched_w(self, sorted_tasks, x0):
    for t in sorted_tasks:
        t.W = t.W / D                      # scale EVERY task weight by 1/D
    if REG_SCALED:
        _saved = self.regularization
        self.regularization = self.regularization / D
        try:
            return _orig_w(self, sorted_tasks, x0)
        finally:
            self.regularization = _saved
    return _orig_w(self, sorted_tasks, x0)


hq.HierarchicalQP._solve_weighted = _patched_w

import scripts.diag_cooperative_arms as dca
from scripts.diag_cooperative_arms import _mutate_mjcf, _mjcf_md5, MJCF

SCRATCH = ('/tmp/claude-0/-home-user-CrawlBot-control/'
           '51ef2e0d-fbab-59f4-a937-ec828e0d7eda/scratchpad')
tag = f'qpcond_D{D:g}_reg{int(REG_SCALED)}_n{N_STEPS}'
out_dir = f'{SCRATCH}/{tag}'
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
            n_steps=N_STEPS, ss_two_task=True, ss_alpha_mom=5000.0,
            alpha_torso_pose=24000.0, ss_alpha_ee=3e3, ss_alpha_posture=2e1,
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

sl = json.load(open(f'{out_dir}/sim_log.json'))
from scripts.export_figure_data import phase_per_tick
si = np.asarray(sl['step_idx']); dg = np.asarray(sl['d_grip_swing'], float)
ph = np.array(phase_per_tick(sl))
ts = np.asarray(sl.get('preplanner_T_steps', []), float)
# SS-phase closest approach (the C5 dock metric, matches TSTEP/dockcause)
docks = {}
for k in sorted(set(int(x) for x in si)):
    m = (si == k) & (ph == 'SS') & np.isfinite(dg)
    if m.any():
        docks[k] = float(1000 * dg[m].min())
print(f'\n=== {tag}: D={D:g} reg_scaled={REG_SCALED} ===')
print('  docks[mm]: ' + '  '.join(f's{k}:{v:.4f}' for k, v in docks.items()))
print('  T_steps:   ' + '  '.join(f'{t:.3f}' for t in ts[:N_STEPS]))
json.dump({'D': D, 'reg_scaled': REG_SCALED, 'n_steps': N_STEPS,
           'docks_mm': docks, 'T_steps': ts[:N_STEPS].tolist()},
          open(f'{SCRATCH}/{tag}_docks.json', 'w'), indent=2)
print(f'wrote {SCRATCH}/{tag}_docks.json')
