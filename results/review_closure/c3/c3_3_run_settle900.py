#!/usr/bin/env python3
"""C3.3 follow-up — re-run the 900 s settle WITH the C2.x instrumentation.

    MUJOCO_GL=disabled PYTHONPATH=. python3 \
        results/review_closure/c3/c3_3_run_settle900.py

Why: C3.3 §3 found that mean commanded tau_w exceeds the realized dh_w/dt by
45-81x, sign reversed, on ALL THREE axes. Closing that needs the AOCS
decomposition (C2.3) and the wheel state logged together over a long settle.
The committed T4b artifacts predate the instrumentation and carry only the
five aggregate channels, so the question cannot be answered from them.

Config: the canonical kwargs verbatim (gate/replay_canonical.py, itself verbatim
Misc/scripts/diag_canonical2p5_run.py) with settle_seconds 20 -> 900, exactly as
Misc/scripts/diag_t4_settle450.py does for its own sweep. Nothing else changes,
so the traversal portion should reproduce the canonical and the settle is the
committed T4b run with more channels.

~19 min: ~4.5 min traversal + 9000 settle ticks at 10 Hz.
"""
import os
import sys

os.environ.setdefault('MUJOCO_GL', 'disabled')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import crawlbot.solvers.hierarchical_qp as hq                    # noqa: E402

EPS = 1e-6                       # == the artifact-generation regularization
_ow = hq.HierarchicalQP._solve_weighted


def _pw(self, tasks, x0):
    self.regularization = EPS
    return _ow(self, tasks, x0)


hq.HierarchicalQP._solve_weighted = _pw

import scripts.diag_cooperative_arms as dca                      # noqa: E402
from scripts.diag_cooperative_arms import (                      # noqa: E402
    _mutate_mjcf, _mjcf_md5, MJCF)

OUT = 'review_closure/c3/settle900'
SETTLE = 900.0


def main():
    log_path = os.path.join(ROOT, 'results', OUT, 'sim_log.json')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if os.path.exists(log_path):
        os.remove(log_path)          # never let a dead run re-export stale data

    with open(MJCF) as f:
        orig = f.read()
    pre = _mjcf_md5(MJCF)
    try:
        _mutate_mjcf(damping=0.0, armature=0.05, anchor_dx=0.8, mass_ratio=0.01)
        try:
            dca.main(
                legacy=False, alpha_torso_lin=0.0, anchor_dx=0.8,
                mass_ratio=0.01, aocs_mode='legacy_pid_numerical',
                settle_seconds=SETTLE,
                K_theta=1.0, K_omega=50.0, tau_w_max=2.5,
                n_steps=6, ss_two_task=True, ss_alpha_mom=400.0,
                alpha_torso_pose=2000.0, ss_alpha_ee=1000.0,
                ss_alpha_posture=2e1, ss_alpha_wrench=1.0,
                ss_kp_torso=3.0, ss_kd_torso=2.5,
                qp_envelope_exact=True,
                interstep_settle_alpha_wrench=3.0,
                interstep_settle_epsilon_v=5e-3,
                out_dir_override=OUT)
        except Exception as e:                                  # noqa: BLE001
            print(f'[settle900] main() raised: '
                  f'{type(e).__name__}: {str(e)[:200]}', flush=True)
            if not os.path.exists(log_path):
                print('[settle900] FAILED — no sim_log.json produced.')
                return 1
    finally:
        with open(MJCF, 'w') as f:
            f.write(orig)
        assert _mjcf_md5(MJCF) == pre, 'MJCF restore failed'
    print(f'[settle900] done -> results/{OUT}/sim_log.json', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
