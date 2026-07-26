"""T15 — M7 v22 three-step traversal, trajectory-aware IK with the
IK_FORMULATION §9 fixes applied (manipulability-ik-fix branch).

Validates that the four post-Phase-4 fixes resolve the step-2 dock
failure observed in Phase 4:
  §9.1  deterministic inner-solve seed from q_start
  §9.2  broadened 7-seed multi-start
  §9.3  post-convergence safety check (w_min_threshold=1e-3)
  §10   metric-mismatch fix (sigma-min reported on both IKs)

All other controller configuration is byte-identical to the T15
baseline (scripts/run_m7_v22_1pct_3step_t15.py) and to the Phase 4
on-demand run (scripts/run_m7_v22_1pct_3step_t15_trajIK_ondemand.py).
The only differences are the four IK code fixes applied on this
branch.

Run:
  MUJOCO_GL=osmesa PYTHONPATH=. python3 \\
      scripts/run_m7_v22_1pct_3step_t15_trajIK_ondemand.py
(MUJOCO_GL=disabled when OSMesa is unavailable — affects rendering
only, not physics.)

Output:
    Misc/runs/M7_1pct_3step_v22_t15_ik_fix/
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
OUT = os.path.join(_root, 'Misc', 'runs', 'M7_1pct_3step_v22_t15_ik_fix')

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
    print(f'[T15-ik-fix] MJCF md5 (pre-run, pre-mutation):  {pre_hash}')
    try:
        _mutate_mjcf(damping=0.0, armature=0.05)
        mid_hash = _mjcf_md5(MJCF)
        print(f'[T15-ik-fix] MJCF md5 (during run, mutated): {mid_hash}')

        cfg = r_single._make_m7_config()
        # Baseline T15 settings (byte-identical to run_m7_v22_1pct_3step_t15.py).
        cfg.preplanner_a_cruise_max = 0.01
        cfg.preplanner_cruise_ramp_frac = 0.2
        cfg.mapping_bypass_in_ss = True
        cfg.swing_early_finish_fraction = 0.80
        cfg.aocs_off_in_ds = True
        # IK-fix branch: flag ON → on-demand path with §9 fixes.
        cfg.use_trajectory_aware_ik = True
        cfg.trajectory_ik_qstart_tolerance = 0.05  # unused in on-demand path
        cfg.trajectory_ik_n_samples = 5
        cfg.trajectory_ik_w_min_threshold = 1e-3  # §9.3 safety check

        print(f'[T15-ik-fix] use_trajectory_aware_ik           = '
              f'{cfg.use_trajectory_aware_ik}')
        print(f'[T15-ik-fix] trajectory_ik_n_samples           = '
              f'{cfg.trajectory_ik_n_samples}')
        print(f'[T15-ik-fix] trajectory_ik_w_min_threshold     = '
              f'{cfg.trajectory_ik_w_min_threshold:.1e}')

        sim, log, metrics = r_single.run_case(
            '1pct 3-step v22 T15-ik-fix '
            '(IK_FORMULATION §9 fixes; K=5; w_min=1e-3)',
            OUT, n_steps=3, config=cfg)

        ik_trace = _dump_ik_trace(sim, OUT)
        print('\n' + '=' * 70)
        print('  Per-step IK trace (Phase 4 — on-demand)')
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
        print(f'[T15-ik-fix] MJCF md5 (post-run, restored):   {post_hash}')
        print(f'[mjcf restored byte-exactly] {post_hash == pre_hash}')


if __name__ == '__main__':
    main()
