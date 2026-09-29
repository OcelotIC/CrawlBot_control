#!/usr/bin/env python3
"""Reproduce the paper's Table 2 arms — one arm per invocation.

    MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/run_table2.py {full|rate|none|u25}
    MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/run_table2.py station_keeping

Adapted from results/review_closure/c4_ablation/c4_run_ablation.py (e37b9ca,
branch claude/review-closure-bloc-2-uwu1x7), which needed SimConfig
nmpc_tau_w_max — ported to this branch in a44a3b6.

    arm      NMPC rate cap           NMPC storage box        QP box + AOCS clip
    full     2.5                     on   (= the canonical)  2.5
    rate     2.5                     off                     2.5
    none     off (nmpc_tau_w_max=inf) off                    2.5  (held)
    u25      lifted (tau_w_max=1e6)  on                      lifted (published unmanaged run)
    station_keeping  = full with settle_seconds=900 (T4b)

The kwargs are otherwise verbatim the canonical set of gate/replay_canonical.py
(itself verbatim Misc/scripts/diag_canonical2p5_run.py), including its
regularization setting. Output: results/table2/<arm>/ (sim_log.json + the
driver's files) and the 66-column fulldiag export next to it.
"""
import os
import subprocess
import sys

os.environ.setdefault('MUJOCO_GL', 'disabled')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import crawlbot.solvers.hierarchical_qp as hq                    # noqa: E402

_ow = hq.HierarchicalQP._solve_weighted


def _pw(self, tasks, x0):
    self.regularization = 1e-6          # == the artifact-generation run
    return _ow(self, tasks, x0)


hq.HierarchicalQP._solve_weighted = _pw

import scripts.diag_cooperative_arms as dca                      # noqa: E402
from scripts.diag_cooperative_arms import (                      # noqa: E402
    _mutate_mjcf, _mjcf_md5, MJCF)

CANONICAL = dict(
    legacy=False, alpha_torso_lin=0.0, anchor_dx=0.8, mass_ratio=0.01,
    aocs_mode='legacy_pid_numerical', settle_seconds=20.0,
    K_theta=1.0, K_omega=50.0, tau_w_max=2.5,
    n_steps=6, ss_two_task=True, ss_alpha_mom=400.0,
    alpha_torso_pose=2000.0, ss_alpha_ee=1000.0, ss_alpha_posture=2e1,
    ss_alpha_wrench=1.0, ss_kp_torso=3.0, ss_kd_torso=2.5,
    qp_envelope_exact=True,
    interstep_settle_alpha_wrench=3.0, interstep_settle_epsilon_v=5e-3,
)
ARMS = {
    'full': {},
    'rate': dict(enforce_hw_conservation=False),
    'none': dict(nmpc_tau_w_max=float('inf'), enforce_hw_conservation=False),
    'u25': dict(tau_w_max=1e6),
    'station_keeping': dict(settle_seconds=900.0),
}


def main():
    arm = sys.argv[1] if len(sys.argv) > 1 else ''
    if arm not in ARMS:
        print(f'usage: {sys.argv[0]} {{{"|".join(ARMS)}}}')
        return 2
    out = f'table2/{arm}'
    log_path = os.path.join('results', out, 'sim_log.json')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if os.path.exists(log_path):       # never re-export a stale log
        os.remove(log_path)
    kwargs = dict(CANONICAL, **ARMS[arm], out_dir_override=out)
    print(f'=== Table 2 arm {arm!r}: {ARMS[arm] or "canonical"} ===', flush=True)
    with open(MJCF) as f:
        orig = f.read()
    pre = _mjcf_md5(MJCF)
    try:
        _mutate_mjcf(damping=0.0, armature=0.05, anchor_dx=0.8, mass_ratio=0.01)
        try:
            dca.main(**kwargs)
        except Exception as e:                                   # noqa: BLE001
            print(f'[{arm}] main() raised: {type(e).__name__}: {e}', flush=True)
    finally:
        with open(MJCF, 'w') as f:
            f.write(orig)
        assert _mjcf_md5(MJCF) == pre, 'MJCF restore failed'
    if not os.path.exists(log_path):
        print(f'[{arm}] FAILED — no sim_log.json')
        return 1
    subprocess.run([sys.executable, 'scripts/diag_full_diag_export.py',
                    '--run-dir', f'results/{out}',
                    '--out-prefix', f'results/{out}/{arm}'], check=False)
    print(f'[{arm}] done -> results/{out}/', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
