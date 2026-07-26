"""NMPC warm-start diagnosis — 5-step traversal.

Instruments NMPCSolver.solve to record per-cycle (iter_count, return_status,
solve_time_ms, warm_x, warm_duals) tagged by locomotion step + phase, then
runs a 5-step traversal at the most recent stable T15-FK configuration.

Output:
    Misc/runs/diag_nmpc_warm_start_5step/
        sim_log.json            (standard sim log)
        nmpc_step_log.json      (every NMPC.solve() call)
        nmpc_per_step.txt       (per-step aggregate table)

Interpretation:
    Signature 1 (warm start is the cause):
        iter_count grows monotonically across steps; status flips to
        Maximum_Iterations_Exceeded by step 3.
    Signature 2 (formulation problem):
        IPOPT converges fast but flips to Infeasible_Problem_Detected.
    Signature 3 (WBC / null space):
        NMPC stays clean across all steps; failure is downstream.

Run:
    MUJOCO_GL=disabled PYTHONPATH=. python3 \\
        Misc/scripts/diag_nmpc_warm_start_5step.py
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import Counter

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')

import scripts.run_m7_single_step as r_single  # noqa: E402


MJCF = os.path.join(_root, 'models', 'VISPA_crawling_rwa3.xml')
OUT = os.path.join(_root, 'Misc', 'runs', 'diag_nmpc_warm_start_5step')

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


def _parse_label(label: str):
    """Parse 'stepNN/PHASE/t=...' into (step:int, phase:str)."""
    if not label:
        return -1, '?'
    parts = label.split('/')
    try:
        step = int(parts[0].replace('step', ''))
    except Exception:
        step = -1
    phase = parts[1] if len(parts) >= 2 else '?'
    return step, phase


def _print_per_step_table(step_log: list, out_path: str):
    """Aggregate per (step, phase) and write a clean table."""
    groups: dict = {}
    for e in step_log:
        step, phase = _parse_label(e.get('label', ''))
        key = (step, phase)
        groups.setdefault(key, []).append(e)

    lines = []
    header = (f"{'Step':>4s} {'Phase':>5s} {'N':>4s} "
              f"{'iter_min':>8s} {'iter_max':>8s} {'iter_mean':>9s} "
              f"{'t_max_ms':>8s} {'t_mean_ms':>9s} "
              f"{'warm_x':>6s} {'warm_du':>7s} "
              f"status_distribution")
    lines.append(header)
    lines.append('-' * len(header))
    for key in sorted(groups.keys()):
        step, phase = key
        entries = groups[key]
        iters = [e['iter'] for e in entries]
        times = [e['time_ms'] for e in entries]
        wx = sum(1 for e in entries if e['warm_x'])
        wd = sum(1 for e in entries if e['warm_duals'])
        statuses = Counter(e['status'] for e in entries)
        status_str = ', '.join(f"{s}:{c}" for s, c in statuses.most_common())
        lines.append(
            f"{step:>4d} {phase:>5s} {len(entries):>4d} "
            f"{min(iters):>8d} {max(iters):>8d} "
            f"{sum(iters)/len(iters):>9.1f} "
            f"{max(times):>8.1f} {sum(times)/len(times):>9.1f} "
            f"{wx:>6d} {wd:>7d} "
            f"{status_str}")
    text = '\n'.join(lines) + '\n'
    with open(out_path, 'w') as f:
        f.write(text)
    print('\n' + '=' * 70)
    print('  NMPC per-(step, phase) aggregate')
    print('=' * 70)
    print(text)


def main():
    with open(MJCF, 'r') as f:
        original = f.read()
    pre_hash = _mjcf_md5(MJCF)
    print(f'[NMPC-DIAG-5STEP] MJCF md5 (pre-run, pre-mutation):  {pre_hash}')
    try:
        _mutate_mjcf(damping=0.0, armature=0.05)
        mid_hash = _mjcf_md5(MJCF)
        print(f'[NMPC-DIAG-5STEP] MJCF md5 (during run, mutated): {mid_hash}')

        cfg = r_single._make_m7_config()
        # Use the most recent stable T15-FK configuration (matches
        # run_m7_v22_1pct_3step_t15_fk_margin5.py).
        cfg.preplanner_a_cruise_max = 0.01
        cfg.preplanner_cruise_ramp_frac = 0.2
        cfg.mapping_bypass_in_ss = True
        cfg.swing_early_finish_fraction = 0.80
        cfg.aocs_off_in_ds = True
        cfg.use_trajectory_aware_ik = True
        cfg.trajectory_ik_qstart_tolerance = 0.05
        cfg.trajectory_ik_n_samples = 5
        cfg.trajectory_ik_w_min_threshold = 1e-3
        cfg.reference_source = 'joint_space_fk'
        cfg.geodesic_n_tau = 21
        cfg.geodesic_n_iter = 120
        cfg.geodesic_tol = 1e-5
        cfg.t_ss_margin = 5.0

        # Build the sim manually (instead of run_case) so we can attach the
        # NMPC diag flag *before* setup, and dump step_log after the run.
        from crawlbot.simulation.sim_loop import SimulationLoop
        URDF = os.path.join(_root, 'models', 'VISPA_crawling_fixed.urdf')

        n_steps = 5
        start_a, start_b = 2, 2
        print('\n' + '=' * 70)
        print(f"  NMPC warm-start diagnosis: 5-step traversal "
              f"start=({start_a},{start_b})")
        print('=' * 70)
        sim = SimulationLoop(mjcf_path=MJCF, urdf_path=URDF, config=cfg)
        sim.setup(n_steps=n_steps, start_a=start_a, start_b=start_b)
        # Diagnostic capture from run_case (physics trace, L_com_ref).
        sim._debug_l_com_ref_trace_limit = 5
        sim._debug_physics_trace_limit = 400
        sim._debug_physics_sample_every = 2

        # Enable NMPC per-cycle logging. Keep verbose printing OFF
        # (would emit thousands of lines); rely on the JSON dump and
        # per-step aggregate.
        sim.nmpc._nmpc.diag_verbose = False
        sim.nmpc._nmpc.step_log.clear()

        log = sim.run(verbose=True)

        os.makedirs(OUT, exist_ok=True)
        log.save(os.path.join(OUT, 'sim_log.json'))

        step_log = list(sim.nmpc._nmpc.step_log)
        with open(os.path.join(OUT, 'nmpc_step_log.json'), 'w') as f:
            json.dump(step_log, f, indent=2)

        _print_per_step_table(step_log,
                              os.path.join(OUT, 'nmpc_per_step.txt'))

        # Summary
        print('\n' + '=' * 70)
        print('  Dock / abort summary')
        print('=' * 70)
        print(f"  Dock events:  {len(log.dock_events)}")
        for ev in log.dock_events:
            print(f"    t={ev['t']:.2f}s arm={ev['arm']} anchor={ev['anchor']}"
                  f" d={ev['d_mm']:.2f}mm ori={ev['ori_deg']:.2f}deg")
        print(f"  Aborted steps: {len(log.aborted_steps)}")
        for ab in log.aborted_steps:
            print(f"    step {ab['step_idx']}: reason={ab['reason']}"
                  f" t={ab.get('t', float('nan')):.2f}s")
        print(f"\n  Outputs: {OUT}")
    finally:
        with open(MJCF, 'w') as f:
            f.write(original)
        post_hash = _mjcf_md5(MJCF)
        print(f'[NMPC-DIAG-5STEP] MJCF md5 (post-run, restored):  {post_hash}')
        assert post_hash == pre_hash, 'MJCF restoration failed'


if __name__ == '__main__':
    main()
