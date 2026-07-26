#!/usr/bin/env python3
"""Canonical (C, managed) replay for the CLEANUP gate — isolated subprocess.

Re-runs the frozen *managed* scenario into ``results/gate_run_scratch/`` using
the EXACT ``dca.main`` kwargs AND the regularization setting that generated the
committed canonical artifacts (``Misc/scripts/diag_canonical2p5_run.py`` C run).
Writes ``sim_log.json`` there and nothing else — no committed artifact is
touched (scratch out_dir; the MJCF mutate is a no-op on the canonical and is
restored under an md5 assert).

Run standalone:
    MUJOCO_GL=disabled PYTHONPATH=. python3 gate/replay_canonical.py
"""
import os
import sys
os.environ.setdefault('MUJOCO_GL', 'disabled')

import crawlbot.solvers.hierarchical_qp as hq

EPS = 1e-6                       # == HierarchicalQP default; set explicitly to
_ow = hq.HierarchicalQP._solve_weighted   # match the artifact-generation run.


def _pw(self, tasks, x0):
    self.regularization = EPS
    return _ow(self, tasks, x0)


hq.HierarchicalQP._solve_weighted = _pw

import scripts.diag_cooperative_arms as dca
from scripts.diag_cooperative_arms import _mutate_mjcf, _mjcf_md5, MJCF

OUT = 'gate_run_scratch'         # -> results/gate_run_scratch/sim_log.json

# C-run kwargs — verbatim from diag_canonical2p5_run.py:_run('C', 2.5, ...).
C_KWARGS = dict(
    legacy=False, alpha_torso_lin=0.0, anchor_dx=0.8, mass_ratio=0.01,
    aocs_mode='legacy_pid_numerical', settle_seconds=20.0,
    K_theta=1.0, K_omega=50.0, tau_w_max=2.5,
    n_steps=6, ss_two_task=True, ss_alpha_mom=400.0,
    alpha_torso_pose=2000.0, ss_alpha_ee=1000.0, ss_alpha_posture=2e1,
    ss_alpha_wrench=1.0, ss_kp_torso=3.0, ss_kd_torso=2.5,
    qp_envelope_exact=True,
    interstep_settle_alpha_wrench=3.0, interstep_settle_epsilon_v=5e-3,
    out_dir_override=OUT,
)

# Delete last run's log FIRST. Without this a replay that dies mid-sim leaves
# the previous artifact in place, the export re-exports it, and the identity
# check passes against stale data — a false PASS. That is not hypothetical:
# CLEANUP-32 shipped a broken TickState and the gate reported PASS on a
# 33-minute-old log.
_LOG = f'results/{OUT}/sim_log.json'
if os.path.exists(_LOG):
    os.remove(_LOG)

with open(MJCF) as f:
    _orig = f.read()
_pre = _mjcf_md5(MJCF)
try:
    _mutate_mjcf(damping=0.0, armature=0.05, anchor_dx=0.8, mass_ratio=0.01)
    try:
        dca.main(**C_KWARGS)
    except Exception as e:
        # dca.main can raise AFTER writing the log (post-sim reporting). That is
        # tolerable. Raising BEFORE the log exists is a failed replay and must
        # not exit 0 — the gate would otherwise validate a stale artifact.
        print(f'[replay] main() raised: '
              f'{type(e).__name__}: {str(e)[:200]}', flush=True)
        if not os.path.exists(_LOG):
            print('[replay] FAILED — no sim_log.json produced; the exception '
                  'came before the log was written.', flush=True)
            with open(MJCF, 'w') as f:
                f.write(_orig)
            sys.exit(1)
finally:
    with open(MJCF, 'w') as f:
        f.write(_orig)
    assert _mjcf_md5(MJCF) == _pre, 'MJCF restore failed'
print('[replay] done -> results/gate_run_scratch/sim_log.json', flush=True)
