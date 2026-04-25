"""T15 — M7 v22 three-step traversal with mid-waypoint reshape (Option B).

Phase 7 validation of T15_step2_path_geometry.md §7.3 Option B:
the IK fix from manipulability-ik-fix branch (IK_FORMULATION §9.1–9.3
+ §10) is unchanged; on top of it, this run enables the mid-waypoint
reshape that splits the SS reference path into two quintic
sub-segments through a manipulability-aware mid-waypoint when the
single-quintic reference is flagged as singular.

Three new SimConfig flags (all default False so behavior is
byte-identical to the IK-fix run with flags off):

  use_path_feasibility_check  = True  (run + log the gate)
  use_mid_waypoint_reshape    = True  (insert q_mid when gate fires)
  mid_waypoint_force_on       = True  (bypass gate, always insert)

The force_on flag is set because the gate's simplified planner-style
references underreport vs the actual mapped sim_log references
(check_path_feasibility caveat in T15_step2_path_geometry.md §7).

Run:
  MUJOCO_GL=disabled PYTHONPATH=. python3 \\
      scripts/run_m7_v22_1pct_3step_t15_midwaypoint.py

Output:
    results/M7_1pct_3step_v22_t15_midwaypoint/
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
OUT = os.path.join(_root, 'results', 'M7_1pct_3step_v22_t15_midwaypoint')

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
            # Phase 4 / Option B fields:
            'mid_wp_used': bool(e.get('mid_wp_used', False)),
            'mid_w_worst': float(e.get('mid_w_worst', float('nan'))),
            'mid_wp_p_t': e.get('mid_wp_p_t', None),
            'path_feasibility_w_min': e.get('path_feasibility_w_min', None),
            'path_feasibility_all_ok': e.get('path_feasibility_all_ok', None),
        })
    with open(os.path.join(out_dir, 'ik_trace.json'), 'w') as f:
        json.dump(safe, f, indent=2)
    return safe


def main():
    with open(MJCF, 'r') as f:
        original = f.read()
    pre_hash = _mjcf_md5(MJCF)
    print(f'[T15-midwp] MJCF md5 (pre-run, pre-mutation):  {pre_hash}')
    try:
        _mutate_mjcf(damping=0.0, armature=0.05)
        mid_hash = _mjcf_md5(MJCF)
        print(f'[T15-midwp] MJCF md5 (during run, mutated): {mid_hash}')

        cfg = r_single._make_m7_config()
        # Baseline T15 settings (byte-identical to run_m7_v22_1pct_3step_t15.py).
        cfg.preplanner_a_cruise_max = 0.01
        cfg.preplanner_cruise_ramp_frac = 0.2
        cfg.mapping_bypass_in_ss = True
        cfg.swing_early_finish_fraction = 0.80
        cfg.aocs_off_in_ds = True

        # IK-fix branch flags.
        cfg.use_trajectory_aware_ik = True
        cfg.trajectory_ik_n_samples = 5
        cfg.trajectory_ik_w_min_threshold = 1e-3

        # Phase 4 / Option B: enable mid-waypoint reshape.
        cfg.use_path_feasibility_check = True
        cfg.use_mid_waypoint_reshape = True
        cfg.mid_waypoint_force_on = True
        cfg.path_feasibility_w_threshold = 1e-3

        print(f'[T15-midwp] use_trajectory_aware_ik           = '
              f'{cfg.use_trajectory_aware_ik}')
        print(f'[T15-midwp] use_path_feasibility_check        = '
              f'{cfg.use_path_feasibility_check}')
        print(f'[T15-midwp] use_mid_waypoint_reshape          = '
              f'{cfg.use_mid_waypoint_reshape}')
        print(f'[T15-midwp] mid_waypoint_force_on             = '
              f'{cfg.mid_waypoint_force_on}')
        print(f'[T15-midwp] path_feasibility_w_threshold      = '
              f'{cfg.path_feasibility_w_threshold:.1e}')

        sim, log, metrics = r_single.run_case(
            '1pct 3-step v22 T15-midwaypoint '
            '(Option B: piecewise-quintic SS reference)',
            OUT, n_steps=3, config=cfg)

        ik_trace = _dump_ik_trace(sim, OUT)
        print('\n' + '=' * 70)
        print('  Per-step IK trace (Phase 7 — mid-waypoint reshape)')
        print('=' * 70)
        for i, e in enumerate(ik_trace):
            mid_str = (f"  mid_used=Y  w_worst_mid={e['mid_w_worst']:.2e}"
                       if e['mid_wp_used']
                       else "  mid_used=N (single-quintic)")
            print(
                f"  step {i}: pair={tuple(e['step_ab_end'])}  "
                f"theta={e['theta_deg']:.2f}°  "
                f"dp={e['dp_mm']:.1f} mm  "
                f"w_end={e['traj_w_end']:.2e}{mid_str}"
            )
    finally:
        with open(MJCF, 'w') as f:
            f.write(original)
        with open(MJCF, 'r') as f:
            assert f.read() == original, 'MJCF restoration failed'
        post_hash = _mjcf_md5(MJCF)
        print(f'[T15-midwp] MJCF md5 (post-run, restored):   {post_hash}')
        print(f'[mjcf restored byte-exactly] {post_hash == pre_hash}')


if __name__ == '__main__':
    main()
