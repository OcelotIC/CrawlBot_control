"""t_ss_margin parametric sweep — 5-step T15-FK traversal.

Single-parameter variation on `cfg.t_ss_margin`. Same T15-FK configuration
as the existing 5-step baseline (run A, margin=5s, already on disk in
Misc/runs/diag_nmpc_warm_start_5step/). This runner produces runs B (10s)
and C (15s).

Outputs per run:
    Misc/runs/diag_nmpc_margin_sweep/margin_{seconds}s/
        sim_log.json
        nmpc_step_log.json
        nmpc_per_step.txt
        step_metrics.txt           (dock / torso-recoil / EE-error summary)

Run:
    MUJOCO_GL=disabled PYTHONPATH=. python3 \\
        scripts/diag_nmpc_margin_sweep.py --margin 10
    MUJOCO_GL=disabled PYTHONPATH=. python3 \\
        scripts/diag_nmpc_margin_sweep.py --margin 15
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')

import numpy as np  # noqa: E402

import scripts.run_m7_single_step as r_single  # noqa: E402


MJCF = os.path.join(_root, 'models', 'VISPA_crawling_rwa3.xml')

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
    if not label:
        return -1, '?'
    parts = label.split('/')
    try:
        step = int(parts[0].replace('step', ''))
    except Exception:
        step = -1
    phase = parts[1] if len(parts) >= 2 else '?'
    return step, phase


def _write_per_step_table(step_log: list, out_path: str):
    groups: dict = {}
    for e in step_log:
        key = _parse_label(e.get('label', ''))
        groups.setdefault(key, []).append(e)
    lines = []
    header = (f"{'Step':>4s} {'Phase':>5s} {'N':>4s} "
              f"{'iter_min':>8s} {'iter_max':>8s} {'iter_p95':>8s} "
              f"{'t_max':>7s} {'t_mean':>7s} {'fail':>4s}  status_dist")
    lines.append(header)
    lines.append('-' * len(header))
    for key in sorted(groups.keys()):
        step, phase = key
        entries = groups[key]
        iters = np.array([e['iter'] for e in entries])
        times = np.array([e['time_ms'] for e in entries])
        failed = sum(1 for e in entries
                     if 'Succeeded' not in e['status']
                     and 'Acceptable' not in e['status'])
        statuses = Counter(e['status'] for e in entries)
        status_str = ', '.join(f"{s}:{c}" for s, c in statuses.most_common())
        lines.append(
            f"{step:>4d} {phase:>5s} {len(entries):>4d} "
            f"{iters.min():>8d} {iters.max():>8d} "
            f"{int(np.percentile(iters, 95)):>8d} "
            f"{times.max():>7.1f} {times.mean():>7.1f} {failed:>4d}  "
            f"{status_str}")
    text = '\n'.join(lines) + '\n'
    with open(out_path, 'w') as f:
        f.write(text)
    return text


def _step_metrics(log, n_steps: int):
    """Per-step dock/recoil/EE-error summary."""
    t = np.array(log.t)
    phase = np.array(log.phase, dtype=object)
    step_idx_arr = np.array(log.step_idx)
    p_torso = np.array(log.p_torso) if log.p_torso else None
    e_ee_pos = np.array(log.e_ee_pos) if log.e_ee_pos else None

    dock_by_step = {ev.get('step', None): ev for ev in log.dock_events}
    abort_by_step = {ab['step_idx']: ab for ab in log.aborted_steps}

    rows = []
    for s in range(n_steps):
        # SS samples for this step
        mask_ss = (step_idx_arr == s) & (phase == 'SS')
        if not mask_ss.any():
            rows.append({
                'step': s, 'outcome': 'NO_DATA',
                'd_mm': None, 'ori_deg': None,
                'peak_torso_recoil_mm': None, 'peak_ee_err_mm': None,
            })
            continue
        # Peak torso recoil — displacement from SS-entry position
        if p_torso is not None:
            p0 = p_torso[mask_ss][0]
            disp = np.linalg.norm(p_torso[mask_ss] - p0, axis=1)
            recoil_mm = float(disp.max() * 1000.0)
        else:
            recoil_mm = None
        # Peak EE tracking error during SS swing
        if e_ee_pos is not None:
            ee_err = e_ee_pos[mask_ss]
            # e_ee_pos entries may be scalars (norm) or 3-vectors — handle both
            if ee_err.ndim == 1:
                peak_ee_mm = float(np.abs(ee_err).max() * 1000.0)
            else:
                peak_ee_mm = float(np.linalg.norm(ee_err, axis=1).max() * 1000.0)
        else:
            peak_ee_mm = None
        # Outcome
        if s in abort_by_step:
            ab = abort_by_step[s]
            outcome = ab['reason'].upper()
            d_mm = ab.get('d_mm', None)
            ori_deg = ab.get('ori_deg', None)
        elif s in dock_by_step:
            ev = dock_by_step[s]
            outcome = 'DOCK'
            d_mm = ev.get('d_mm', None)
            ori_deg = ev.get('ori_deg', None)
        else:
            outcome = 'UNKNOWN'
            d_mm = ori_deg = None
        rows.append({
            'step': s, 'outcome': outcome,
            'd_mm': d_mm, 'ori_deg': ori_deg,
            'peak_torso_recoil_mm': recoil_mm,
            'peak_ee_err_mm': peak_ee_mm,
        })

    # Terminal kinetic energy: last sim_log entry
    T_kin_final = None
    if hasattr(log, 'qvel_joints_a') and log.qvel_joints_a:
        # We don't have T directly — approximate from torso + arm velocities
        # via 0.5 sum of velocity² (not mass-weighted; just a proxy).
        # Better: pull from inter_step_settles if available, else NaN.
        pass

    return rows, T_kin_final


def _write_metrics_table(rows, out_path: str):
    lines = []
    header = (f"{'Step':>4s} {'Outcome':>22s} {'d [mm]':>8s} {'ori [°]':>8s} "
              f"{'recoil [mm]':>12s} {'EE err [mm]':>12s}")
    lines.append(header)
    lines.append('-' * len(header))
    for r in rows:
        d = f"{r['d_mm']:.2f}" if r['d_mm'] is not None else '--'
        o = f"{r['ori_deg']:.2f}" if r['ori_deg'] is not None else '--'
        recoil = (f"{r['peak_torso_recoil_mm']:.1f}"
                  if r['peak_torso_recoil_mm'] is not None else '--')
        ee = (f"{r['peak_ee_err_mm']:.1f}"
              if r['peak_ee_err_mm'] is not None else '--')
        lines.append(
            f"{r['step']:>4d} {r['outcome']:>22s} {d:>8s} {o:>8s} "
            f"{recoil:>12s} {ee:>12s}")
    text = '\n'.join(lines) + '\n'
    with open(out_path, 'w') as f:
        f.write(text)
    return text


def run_one(margin_s: float, out_dir: str):
    cfg = r_single._make_m7_config()
    # T15-FK config (identical to baseline runner).
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
    # Single-parameter sweep variable.
    cfg.t_ss_margin = float(margin_s)

    from crawlbot.simulation.sim_loop import SimulationLoop
    URDF = os.path.join(_root, 'models', 'VISPA_crawling_fixed.urdf')

    n_steps = 5
    start_a, start_b = 2, 2
    print('\n' + '=' * 70)
    print(f"  Margin sweep: t_ss_margin = {margin_s:.1f} s, "
          f"start=({start_a},{start_b}), n_steps={n_steps}")
    print('=' * 70)
    sim = SimulationLoop(mjcf_path=MJCF, urdf_path=URDF, config=cfg)
    sim.setup(n_steps=n_steps, start_a=start_a, start_b=start_b)
    sim._debug_l_com_ref_trace_limit = 5
    sim._debug_physics_trace_limit = 400
    sim._debug_physics_sample_every = 2
    sim.nmpc._nmpc.diag_verbose = False
    sim.nmpc._nmpc.step_log.clear()

    log = sim.run(verbose=True)

    os.makedirs(out_dir, exist_ok=True)
    log.save(os.path.join(out_dir, 'sim_log.json'))

    step_log = list(sim.nmpc._nmpc.step_log)
    with open(os.path.join(out_dir, 'nmpc_step_log.json'), 'w') as f:
        json.dump(step_log, f, indent=2)

    text_nmpc = _write_per_step_table(
        step_log, os.path.join(out_dir, 'nmpc_per_step.txt'))
    print('\n' + '=' * 70)
    print(f'  NMPC per-(step, phase) aggregate (margin={margin_s:.1f}s)')
    print('=' * 70)
    print(text_nmpc)

    rows, _ = _step_metrics(log, n_steps)
    text_metrics = _write_metrics_table(
        rows, os.path.join(out_dir, 'step_metrics.txt'))
    print('=' * 70)
    print(f'  Per-step metrics (margin={margin_s:.1f}s)')
    print('=' * 70)
    print(text_metrics)
    print(f'  Outputs: {out_dir}')
    return log, step_log, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--margin', type=float, required=True,
                        help='t_ss_margin value in seconds (e.g. 10, 15)')
    args = parser.parse_args()

    out_dir = os.path.join(
        _root, 'Misc', 'runs', 'diag_nmpc_margin_sweep',
        f'margin_{int(args.margin)}s')

    with open(MJCF, 'r') as f:
        original = f.read()
    pre_hash = _mjcf_md5(MJCF)
    print(f'[MARGIN-SWEEP m={args.margin}s] MJCF md5 pre:  {pre_hash}')
    try:
        _mutate_mjcf(damping=0.0, armature=0.05)
        run_one(args.margin, out_dir)
    finally:
        with open(MJCF, 'w') as f:
            f.write(original)
        post_hash = _mjcf_md5(MJCF)
        print(f'[MARGIN-SWEEP m={args.margin}s] MJCF md5 post: {post_hash}')
        assert post_hash == pre_hash, 'MJCF restoration failed'


if __name__ == '__main__':
    main()
