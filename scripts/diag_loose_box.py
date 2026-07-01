#!/usr/bin/env python3
"""Phase-1c diagnostic — storage box ACTIVE but LOOSE (h_max_tight=10, enforce=True).

Falsifiable test of whether Task-2's storage-off blocker was the box's BOUND VALUE
or its ABSENCE. Task 2 removed the box (enforce_hw_conservation=False + h_max=1e6);
IPOPT then hit a local-infeasibility restoration failure with the last iterate's
h_w only 0.774 Nms — nowhere near any bound. Hypothesis: keeping the box ACTIVE but
LOOSE (at 2x the real 5 bound) converges fine, because the box then acts as a
regularizer that never binds (C's solution peaks at 4.886 < 5 < 10). If so, the
Task-2 blocker was the ABSENCE of the box, not its value.

ZERO crawlbot/ change. Two monkeypatches:
  (1) r_single._make_m7_config -> post-set cfg.h_max_tight = [10,10,10] and assert
      enforce_hw_conservation stays True. ONLY the storage bound changes; the rate
      cap (tau_w_max=5), the QP soft box (hw_max=5) and the AOCS clamp
      (aocs_tau_w_max=5) are untouched => the whole-body QP is bit-identical to C.
      Plumbing (verified): cfg.h_max_tight feeds the NMPC box (sim_loop:388-396) and
      the pre-planner box (sim_loop:406-409 build + :1886-1900 solve); enforce gates
      the NMPC box (:395). The QP uses tau_w_max/hw_max, NOT h_max_tight.
  (2) WholeBodyQP.__init__ -> register each instance so we can read qp.hw_slack_log
      after the run (per-solve slack telemetry: slack_up_max/slack_lo_max/slack_norm),
      to report max|s_hw| for the nominal QP. C's raw sim_log is ephemeral/gitignored,
      but its QP soft box (hw_max=5) is identical to this run's and h_w<5 in both, so
      this max|s_hw| IS the nominal (C) QP soft-slack value.

Canonical CANON flags == runfix/C (doc INTERNAL_canonical_revalidation.md:21-23),
--n-steps 6 (the 6-step orientation-fix scenario; runfix has 6 docks / steps 0-5).

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_loose_box.py
"""
import json
import os
import runpy
import sys

import numpy as np

import scripts.run_m7_single_step as r_single
from crawlbot.solvers.wholebody_qp import WholeBodyQP

H_MAX_LOOSE = 10.0

# (1) storage box ACTIVE but LOOSE ------------------------------------------------
_orig_make = r_single._make_m7_config


def _loose_make(*a, **k):
    cfg = _orig_make(*a, **k)
    assert bool(cfg.enforce_hw_conservation), \
        "canonical must keep enforce_hw_conservation=True (box active)"
    cfg.h_max_tight = np.full(3, H_MAX_LOOSE)   # 2x the real 5 bound; box stays active
    return cfg


r_single._make_m7_config = _loose_make

# (2) capture QP slack telemetry --------------------------------------------------
_qps = []
_orig_qp_init = WholeBodyQP.__init__


def _qp_init(self, *a, **k):
    _orig_qp_init(self, *a, **k)
    _qps.append(self)


WholeBodyQP.__init__ = _qp_init

# run the canonical scenario, box active-loose ------------------------------------
sys.argv = [
    'diag_cooperative_arms.py',
    '--ss-two-task', '--ss-alpha-mom', '5000', '--alpha-torso-pose', '24000',
    '--ss-kp-torso', '3', '--ss-kd-torso', '2.5',
    '--aocs_mode', 'legacy_pid_numerical', '--K_omega', '50',
    '--n-steps', '6', '--qp-envelope-exact',
    '--interstep-settle-alpha-wrench', '3', '--interstep-settle-epsilon-v', '5e-3',
    '--envelope-constraint', 'on', '--out-dir', 'figU_loose10',
]
print(f'[loose-box] h_max_tight={H_MAX_LOOSE} (box ACTIVE), enforce_hw_conservation=True, '
      f'tau_w_max/hw_max/aocs untouched; argv={" ".join(sys.argv[1:])}')
try:
    runpy.run_module('scripts.diag_cooperative_arms', run_name='__main__')
except SystemExit:
    pass

# aggregate hw_slack telemetry across all QP instances ----------------------------
allslack = []
for qp in _qps:
    allslack += list(getattr(qp, 'hw_slack_log', []) or [])


def _mx(key):
    return max((float(e.get(key, 0.0)) for e in allslack), default=0.0)


summary = dict(
    h_max_tight=H_MAX_LOOSE, enforce_hw_conservation=True,
    n_qp_instances=len(_qps), n_slack_entries=len(allslack),
    max_slack_up=_mx('slack_up_max'), max_slack_lo=_mx('slack_lo_max'),
    max_slack_norm=_mx('slack_norm'), hw_max_soft_box=5.0,
    note=('QP soft box hw_max=5 identical to C (only NMPC/pre-planner storage bound '
          'differs, 10 vs 5, and it does not bind since h_w<5). So max|s_hw| here is '
          'the nominal C QP soft-slack value; C raw sim_log is ephemeral/gitignored.'))
os.makedirs('results/j2_ablation_envelope', exist_ok=True)
json.dump(summary, open('results/j2_ablation_envelope/loose10_hw_slack.json', 'w'), indent=2)
print('\n' + '=' * 72)
print('PHASE-1C  hw-slack telemetry (nominal QP; hw_max soft box = 5 Nms):')
print(f'  QP instances={summary["n_qp_instances"]}  slack log entries={summary["n_slack_entries"]}')
print(f'  max|s_hw| upper={summary["max_slack_up"]:.4e}  lower={summary["max_slack_lo"]:.4e}  '
      f'norm={summary["max_slack_norm"]:.4e}  Nms')
print('  -> wrote results/j2_ablation_envelope/loose10_hw_slack.json')
print('=' * 72)
