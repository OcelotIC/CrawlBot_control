#!/usr/bin/env python3
"""Phase-1c side-ask — max|s_hw| (QP soft-box slack) over the full NOMINAL traversal.

Confirms the QP momentum soft-slack s_hw is inert (~0) in the nominal QP, so the
Task-1 numbers are not distorted by it. C's raw sim_log is ephemeral/gitignored, so
this re-runs the genuine canonical config (h_max_tight=5, enforce=True — NO loosening)
and reads qp.hw_slack_log after the run. ZERO crawlbot/ change: only registers each
WholeBodyQP instance to read its per-solve slack telemetry.

Canonical CANON flags == runfix/C, --n-steps 6 (6-step orientation-fix scenario).

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_nominal_shw.py
"""
import json
import os
import runpy
import sys

from crawlbot.solvers.wholebody_qp import WholeBodyQP

_qps = []
_orig_qp_init = WholeBodyQP.__init__


def _qp_init(self, *a, **k):
    _orig_qp_init(self, *a, **k)
    _qps.append(self)


WholeBodyQP.__init__ = _qp_init

sys.argv = [
    'diag_cooperative_arms.py',
    '--ss-two-task', '--ss-alpha-mom', '5000', '--alpha-torso-pose', '24000',
    '--ss-kp-torso', '3', '--ss-kd-torso', '2.5',
    '--aocs_mode', 'legacy_pid_numerical', '--K_omega', '50', '--qp-envelope-exact',
    '--interstep-settle-alpha-wrench', '3', '--interstep-settle-epsilon-v', '5e-3',
    '--n-steps', '6', '--envelope-constraint', 'on', '--out-dir', 'figC_nominal',
]
print('[nominal-s_hw] canonical h_max_tight=5, enforce=True (NO loosening); '
      'capturing qp.hw_slack_log across all QP instances')
try:
    runpy.run_module('scripts.diag_cooperative_arms', run_name='__main__')
except SystemExit:
    pass

allslack = []
for qp in _qps:
    allslack += list(getattr(qp, 'hw_slack_log', []) or [])


def _mx(key):
    return max((float(e.get(key, 0.0)) for e in allslack), default=0.0)


summary = dict(
    config='canonical C (h_max_tight=5, enforce=True, hw_max soft box=5)',
    n_qp_instances=len(_qps), n_slack_entries=len(allslack),
    max_slack_up=_mx('slack_up_max'), max_slack_lo=_mx('slack_lo_max'),
    max_slack_norm=_mx('slack_norm'), hw_max_soft_box=5.0)
os.makedirs('results/j2_ablation_envelope', exist_ok=True)
json.dump(summary, open('results/j2_ablation_envelope/nominal_hw_slack.json', 'w'), indent=2)
print('\n' + '=' * 72)
print('PHASE-1C  max|s_hw| over the full NOMINAL traversal (QP soft box hw_max=5):')
print(f'  QP instances={summary["n_qp_instances"]}  slack log entries={summary["n_slack_entries"]}')
print(f'  max|s_hw| upper={summary["max_slack_up"]:.4e}  lower={summary["max_slack_lo"]:.4e}  '
      f'norm={summary["max_slack_norm"]:.4e}  Nms')
print('  -> wrote results/j2_ablation_envelope/nominal_hw_slack.json')
print('=' * 72)
