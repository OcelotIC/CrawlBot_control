#!/usr/bin/env python3
"""C4 — clean three-way envelope ablation. Runs one arm per invocation.

    MUJOCO_GL=disabled PYTHONPATH=. python3 \
        results/review_closure/c4_ablation/c4_run_ablation.py {none|rate|full}

Design: the three arms differ ONLY in the NMPC constraint set.

    arm     NMPC rate bound        NMPC storage bound
    none    off  (nmpc_tau_w_max=inf)   off (enforce_hw_conservation=False)
    rate    on   (2.5)                  off
    full    on   (2.5)                  on   (= canonical)

Held FIXED in all three arms, deliberately:
  * the AOCS output clip, ±2.5 N·m — the actuator's physical cap does not
    disappear when the planner stops respecting it, so holding it is what makes
    the comparison a statement about the PLANNER;
  * the whole-body QP envelope box, |Ḣ_s| ≤ 2.5 — not part of the NMPC
    constraint set;
  * every weight, gain, gait timing and threshold.

`cfg.tau_w_max` is read by all three of NMPC / QP / AOCS, which is why the arms
move `cfg.nmpc_tau_w_max` (an NMPC-only override, default None) instead. The
published u25 run moves `tau_w_max` itself and so shifts all three at once, and
keeps the storage box on — see `c4_config_diff.py`. It is not the `none` arm.

The kwargs below are otherwise verbatim the canonical set from
`gate/replay_canonical.py`, which is verbatim
`Misc/scripts/diag_canonical2p5_run.py`. No tuning.
"""
import os
import sys

os.environ.setdefault('MUJOCO_GL', 'disabled')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import crawlbot.solvers.hierarchical_qp as hq                    # noqa: E402

# Match the artifact-generation run's regularization exactly (as the gate does).
EPS = 1e-6
_ow = hq.HierarchicalQP._solve_weighted


def _pw(self, tasks, x0):
    self.regularization = EPS
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
    # nmpc_tau_w_max=None ⇒ NMPC uses cfg.tau_w_max (2.5). float('inf') ⇒ the
    # NMPC rate constraint is skipped entirely (`np.isfinite` guard).
    'none': dict(nmpc_tau_w_max=float('inf'), enforce_hw_conservation=False),
    'rate': dict(nmpc_tau_w_max=None,         enforce_hw_conservation=False),
    'full': dict(nmpc_tau_w_max=None,         enforce_hw_conservation=True),
}


def main():
    arm = sys.argv[1] if len(sys.argv) > 1 else ''
    if arm not in ARMS:
        print(f'usage: {sys.argv[0]} {{none|rate|full}}')
        return 2
    out = f'review_closure/c4_ablation/{arm}'
    log_path = os.path.join(ROOT, 'results', out, 'sim_log.json')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    # Delete first: a run that dies mid-sim would otherwise leave the previous
    # log in place and the export would silently re-export stale data (the
    # CLEANUP-32 false-PASS failure mode).
    if os.path.exists(log_path):
        os.remove(log_path)

    kwargs = dict(CANONICAL, **ARMS[arm], out_dir_override=out)
    print(f'=== C4 arm {arm!r} ===')
    for k in ('nmpc_tau_w_max', 'enforce_hw_conservation'):
        print(f'    {k} = {kwargs[k]}')
    print(f'    tau_w_max (QP box + AOCS clip, HELD) = {kwargs["tau_w_max"]}')

    with open(MJCF) as f:
        orig = f.read()
    pre = _mjcf_md5(MJCF)
    try:
        _mutate_mjcf(damping=0.0, armature=0.05, anchor_dx=0.8,
                     mass_ratio=0.01)
        try:
            dca.main(**kwargs)
        except Exception as e:                                  # noqa: BLE001
            # dca.main can raise AFTER writing the log (post-sim reporting).
            print(f'[arm {arm}] main() raised: '
                  f'{type(e).__name__}: {str(e)[:200]}', flush=True)
            if not os.path.exists(log_path):
                print(f'[arm {arm}] FAILED — no sim_log.json produced.')
                return 1
    finally:
        with open(MJCF, 'w') as f:
            f.write(orig)
        assert _mjcf_md5(MJCF) == pre, 'MJCF restore failed'
    print(f'[arm {arm}] done -> results/{out}/sim_log.json', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
