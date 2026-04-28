"""T15 — M7 v22 three-step traversal with FK-on-smoothed-q references.

Validates that the task-space-smoothed constrained-geodesic reference
architecture (plan §2.2) resolves the step-2 dock failure observed
in every prior T15 run. The smoother eliminates the structural
failure mode (kinematically-uncoupled task-space refs producing
infeasible interior triples) by deriving torso + swing references
via FK on a single q-sequence that satisfies the stance constraint
by construction.

Phase-0 measurement (commits b924ded, eb38c72) confirms:
  - Stance compliance ≤40 μm (gate 50 mm).
  - Step-2 swing-EE world-frame inflation +0.2 % vs raw chord.
  - 21/21 3-task IK convergence with zero fallbacks.

This run is identical to the IK-fix runner
(scripts/run_m7_v22_1pct_3step_t15_ik_fix.py) except for:
  cfg.reference_source       = 'joint_space_fk'   (was default 'task_space')
  cfg.geodesic_n_tau         = 21                 (defaults)
  cfg.geodesic_n_iter        = 120                (defaults)
  cfg.geodesic_tol           = 1e-5               (defaults)

Pass criteria (plan §4.3, test E.7):
  - All 3 steps DOCKED at d ≤ 5 mm and ori ≤ 5°.
  - min(w_actual) per step ≥ 1e-2.
  - AOCS tau_w peak ≤ 5 Nm; hw peak ≤ 5 Nms.
  - NMPC Infeasible_Problem_Detected count ≤ baseline.

Run:
  MUJOCO_GL=osmesa PYTHONPATH=. python3 \\
      scripts/run_m7_v22_1pct_3step_t15_fk.py
(MUJOCO_GL=disabled when OSMesa is unavailable — affects rendering
only, not physics.)

Output:
    results/M7_1pct_3step_v22_t15_fk/
        sim_log.json, metrics.csv, physics_trace.pkl, ik_trace.json,
        diagnostic plots.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')

import scripts.run_m7_single_step as r_single


MJCF = os.path.join(_root, 'models', 'VISPA_crawling_rwa3.xml')
OUT = os.path.join(_root, 'results', 'M7_1pct_3step_v22_t15_fk_margin5')

ROBOT_JOINT_RE = re.compile(
    r'(<default class="robot_joint">\s*\n\s*<joint damping=")[^"]+'
    r'(" armature=")[^"]+(")')


def _mjcf_md5(path: str) -> str:
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def _mutate_mjcf(damping: float, armature: float):
    with open(MJCF, 'r') as f:
        text = f.read()
    new, n = ROBOT_JOINT_RE.subn(
        rf'\g<1>{damping}\g<2>{armature}\g<3>', text, count=1)
    if n != 1:
        raise RuntimeError('robot_joint default not found')
    with open(MJCF, 'w') as f:
        f.write(new)


def _dump_ik_trace(sim, out_dir: str):
    trace = getattr(sim, '_debug_ik_trace', []) or []
    safe = []
    for e in trace:
        safe.append({
            'step_ab_end': list(e.get('step_ab_end', ())),
            'mode': e.get('mode'),
            'theta_deg': float(e.get('theta_deg', float('nan'))),
            'dp_mm': float(e.get('dp_mm', float('nan'))),
            'w_fixed': float(e.get('w_fixed', float('nan'))),
            'w_sigma_min_fixed': float(e.get('w_sigma_min_fixed', float('nan'))),
            'traj_drift': float(e.get('traj_drift', float('nan'))),
            'traj_w_worst': float(e.get('traj_w_worst', float('nan'))),
            'traj_w_end': float(e.get('traj_w_end', float('nan'))),
            'traj_ik_elapsed_s': float(e.get('traj_ik_elapsed_s', float('nan'))),
        })
    with open(os.path.join(out_dir, 'ik_trace.json'), 'w') as f:
        json.dump(safe, f, indent=2)
    return safe


def main():
    with open(MJCF, 'r') as f:
        original = f.read()
    pre_hash = _mjcf_md5(MJCF)
    print(f'[T15-FK-MARGIN5] MJCF md5 (pre-run, pre-mutation):  {pre_hash}')
    try:
        _mutate_mjcf(damping=0.0, armature=0.05)
        mid_hash = _mjcf_md5(MJCF)
        print(f'[T15-FK-MARGIN5] MJCF md5 (during run, mutated): {mid_hash}')

        cfg = r_single._make_m7_config()
        # Baseline T15 settings (byte-identical to run_m7_v22_1pct_3step_t15.py
        # and IK-fix runner).
        cfg.preplanner_a_cruise_max = 0.01
        cfg.preplanner_cruise_ramp_frac = 0.2
        cfg.mapping_bypass_in_ss = True
        cfg.swing_early_finish_fraction = 0.80
        cfg.aocs_off_in_ds = True
        # IK-fix branch carries forward — trajectory-aware IK ON.
        cfg.use_trajectory_aware_ik = True
        cfg.trajectory_ik_qstart_tolerance = 0.05  # unused in on-demand path
        cfg.trajectory_ik_n_samples = 5
        cfg.trajectory_ik_w_min_threshold = 1e-3  # §9.3 safety check
        # FK-on-smoothed-q reference architecture (the only delta vs IK-fix).
        cfg.reference_source = 'joint_space_fk'
        cfg.geodesic_n_tau = 21
        cfg.geodesic_n_iter = 120
        cfg.geodesic_tol = 1e-5
        # Bump SS margin from 1.0s to 5.0s. Per the QP isolation analysis,
        # the (3,4) anchor traversal has a long asymptotic tail in
        # d_swing convergence; default 1s margin is insufficient.
        cfg.t_ss_margin = 5.0

        print(f'[T15-FK-MARGIN5] reference_source                = '
              f'{cfg.reference_source!r}')
        print(f'[T15-FK-MARGIN5] geodesic_n_tau / n_iter / tol   = '
              f'{cfg.geodesic_n_tau} / {cfg.geodesic_n_iter} / '
              f'{cfg.geodesic_tol:.1e}')
        print(f'[T15-FK-MARGIN5] use_trajectory_aware_ik         = '
              f'{cfg.use_trajectory_aware_ik}')

        sim, log, metrics = r_single.run_case(
            '1pct 3-step v22 T15-FK-MARGIN5 '
            '(reference_source=joint_space_fk; n_tau=21; n_iter=120)',
            OUT, n_steps=3, config=cfg)

        ik_trace = _dump_ik_trace(sim, OUT)
        print('\n' + '=' * 70)
        print('  Per-step IK trace (FK reference architecture)')
        print('=' * 70)
        for i, e in enumerate(ik_trace):
            print(
                f"  step {i}: pair={tuple(e['step_ab_end'])}  "
                f"mode={e['mode']}  theta={e['theta_deg']:.2f}°  "
                f"dp={e['dp_mm']:.1f} mm  "
                f"t_ik={e['traj_ik_elapsed_s']:.2f} s  "
                f"w_worst={e['traj_w_worst']:.2e}  "
                f"w_end={e['traj_w_end']:.2e}"
            )
    finally:
        with open(MJCF, 'w') as f:
            f.write(original)
        with open(MJCF, 'r') as f:
            assert f.read() == original, 'MJCF restoration failed'
        post_hash = _mjcf_md5(MJCF)
        print(f'[T15-FK-MARGIN5] MJCF md5 (post-run, restored):   {post_hash}')
        print(f'[mjcf restored byte-exactly] {post_hash == pre_hash}')


if __name__ == '__main__':
    main()
