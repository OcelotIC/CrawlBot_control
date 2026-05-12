"""5-step traversal with q_current mapping fix applied.

Single change from previous bypass=OFF run: sim_loop.py now uses
rs.q, rs.v in the mapping call during SS (instead of the planned-arm
q_planned). Everything else identical to baseline.

Reports per step:
  Outcome, dock accuracy, peak torso recoil, peak EE error, plus
  r_b_ref oscillation amplitude at 100 Hz (max ||r_b_ref[i] -
  r_b_ref[i-1]||) — stability indicator.

Outputs:
    results/diag_qcurrent_fix/
        sim_log.json, step_metrics.txt, rb_ref_stability.txt,
        nmpc_per_step.txt, step_log.json
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')

import scripts.run_m7_single_step as r_single  # noqa: E402


MJCF = os.path.join(_root, 'models', 'VISPA_crawling_rwa3.xml')
OUT = os.path.join(_root, 'results', 'diag_frate_fsat')

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


def _step_metrics_table(log, diag_log, n_steps=5, out_path=None):
    t = np.array(log.t)
    phase = np.array(log.phase, dtype=object)
    sidx = np.array(log.step_idx)
    p_torso = np.array(log.p_torso) if log.p_torso else None
    e_ee_raw = log.e_ee_pos
    e_ee_pos = np.array(e_ee_raw) if e_ee_raw else None
    dock_by_step = {ev.get('step', None): ev for ev in log.dock_events}
    abort_by_step = {ab['step_idx']: ab for ab in log.aborted_steps}

    # Group the diag log by step_idx for r_b_ref oscillation calc
    by_step = defaultdict(list)
    for e in diag_log:
        by_step[e.get('step_idx', -1)].append(e)

    lines = []
    header = (f"{'Step':>4s} {'Outcome':>22s} {'d [mm]':>8s} {'ori [°]':>8s} "
              f"{'recoil [mm]':>12s} {'EE err [mm]':>12s} "
              f"{'rb_osc_max [mm]':>16s} {'rb_osc_p95 [mm]':>16s}")
    lines += [header, '-' * len(header)]
    for s in range(n_steps):
        m = (sidx == s) & (phase == 'SS')
        if not m.any():
            lines.append(f"{s:>4d} {'NO_DATA':>22s}")
            continue
        # Recoil + EE error
        recoil_mm = None
        if p_torso is not None and p_torso.size:
            p0 = p_torso[m][0]
            recoil_mm = float(np.linalg.norm(
                p_torso[m] - p0, axis=1).max() * 1000.0)
        peak_ee_mm = None
        if e_ee_pos is not None and e_ee_pos.size:
            ee = e_ee_pos[m]
            peak_ee_mm = float(
                (np.linalg.norm(ee, axis=1) if ee.ndim == 2
                 else np.abs(ee)).max() * 1000.0)
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
        # r_b_ref oscillation: max ||r_b_ref[i] - r_b_ref[i-1]|| at 100Hz
        entries = by_step.get(s, [])
        if len(entries) >= 2:
            r_b = np.array([e['r_b_ref'] for e in entries])
            step_diffs = np.linalg.norm(np.diff(r_b, axis=0), axis=1)
            osc_max_mm = float(step_diffs.max() * 1000.0)
            osc_p95_mm = float(np.percentile(step_diffs, 95) * 1000.0)
        else:
            osc_max_mm = osc_p95_mm = None
        d = f"{d_mm:.2f}" if d_mm is not None else '--'
        o = f"{ori_deg:.2f}" if ori_deg is not None else '--'
        r = f"{recoil_mm:.1f}" if recoil_mm is not None else '--'
        e = f"{peak_ee_mm:.1f}" if peak_ee_mm is not None else '--'
        om = f"{osc_max_mm:.3f}" if osc_max_mm is not None else '--'
        op = f"{osc_p95_mm:.3f}" if osc_p95_mm is not None else '--'
        lines.append(f"{s:>4d} {outcome:>22s} {d:>8s} {o:>8s} "
                     f"{r:>12s} {e:>12s} {om:>16s} {op:>16s}")
    text = '\n'.join(lines) + '\n'
    if out_path:
        with open(out_path, 'w') as f:
            f.write(text)
    return text


def _nmpc_table(step_log, out_path):
    groups = defaultdict(list)
    for e in step_log:
        lbl = e.get('label', '')
        parts = lbl.split('/')
        try:
            si = int(parts[0].replace('step', ''))
        except Exception:
            si = -1
        ph = parts[1] if len(parts) >= 2 else '?'
        groups[(si, ph)].append(e)
    lines = []
    header = (f"{'Step':>4s} {'Phase':>5s} {'N':>4s} "
              f"{'iter_min':>8s} {'iter_max':>8s} {'iter_p95':>8s} "
              f"{'t_max':>7s} {'t_mean':>7s} {'fail':>4s}  status_dist")
    lines += [header, '-' * len(header)]
    for key in sorted(groups.keys()):
        s, ph = key
        entries = groups[key]
        iters = np.array([e['iter'] for e in entries])
        times = np.array([e['time_ms'] for e in entries])
        failed = sum(1 for e in entries
                     if 'Succeeded' not in e['status']
                     and 'Acceptable' not in e['status'])
        statuses = Counter(e['status'] for e in entries)
        sd = ', '.join(f"{x}:{c}" for x, c in statuses.most_common())
        lines.append(
            f"{s:>4d} {ph:>5s} {len(entries):>4d} "
            f"{iters.min():>8d} {iters.max():>8d} "
            f"{int(np.percentile(iters, 95)):>8d} "
            f"{times.max():>7.1f} {times.mean():>7.1f} {failed:>4d}  {sd}")
    text = '\n'.join(lines) + '\n'
    with open(out_path, 'w') as f:
        f.write(text)
    return text


def main():
    cfg = r_single._make_m7_config()
    cfg.preplanner_a_cruise_max = 0.01
    cfg.preplanner_cruise_ramp_frac = 0.2
    cfg.mapping_bypass_in_ss = False   # bypass stays OFF
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
    cfg.ss_alpha_torso = 500.0
    cfg.r_tube = 0.0

    from crawlbot.simulation.sim_loop import SimulationLoop
    URDF = os.path.join(_root, 'models', 'VISPA_crawling_fixed.urdf')

    sim = SimulationLoop(mjcf_path=MJCF, urdf_path=URDF, config=cfg)
    sim.setup(n_steps=5, start_a=2, start_b=2)
    sim._debug_l_com_ref_trace_limit = 5
    sim._debug_physics_trace_limit = 400
    sim._debug_physics_sample_every = 2
    sim.nmpc._nmpc.diag_verbose = False
    sim.nmpc._nmpc.step_log.clear()
    sim.qp_ss.hw_slack_log.clear()
    sim._step2_diag_enabled = True   # captures every SS step now
    sim._step2_diag_log.clear()

    log = sim.run(verbose=True)

    os.makedirs(OUT, exist_ok=True)
    log.save(os.path.join(OUT, 'sim_log.json'))
    with open(os.path.join(OUT, 'step_log.json'), 'w') as f:
        json.dump(sim._step2_diag_log, f, indent=2)

    text_metrics = _step_metrics_table(
        log, sim._step2_diag_log, 5,
        os.path.join(OUT, 'step_metrics.txt'))
    print('\n=== Per-step outcomes (q_current fix) ===\n' + text_metrics)

    step_log = list(sim.nmpc._nmpc.step_log)
    with open(os.path.join(OUT, 'nmpc_step_log.json'), 'w') as f:
        json.dump(step_log, f, indent=2)
    _nmpc_table(step_log, os.path.join(OUT, 'nmpc_per_step.txt'))

    # F-SAT telemetry
    tot = int(sim._sat_total_calls)
    clip = int(sim._sat_clipped_calls)
    ratio = (clip / tot * 100.0) if tot > 0 else 0.0
    sat_text = (
        f"F-SAT telemetry\n"
        f"  total saturator calls: {tot}\n"
        f"  cycles clipped:        {clip}  ({ratio:.2f}%)\n"
        f"  max clip magnitude:    {sim._sat_max_clip_mm:.3f} mm "
        f"(per-cycle excess over the threshold)\n"
    )
    print('\n' + sat_text)
    with open(os.path.join(OUT, 'sat_stats.txt'), 'w') as f:
        f.write(sat_text)
    print(f'  Outputs: {OUT}')


if __name__ == '__main__':
    with open(MJCF, 'r') as f:
        original = f.read()
    pre_hash = _mjcf_md5(MJCF)
    print(f'[FRATE-FSAT] MJCF md5 pre:  {pre_hash}')
    try:
        _mutate_mjcf(damping=0.0, armature=0.05)
        main()
        # F-SAT telemetry summary
        try:
            from crawlbot.simulation.sim_loop import SimulationLoop  # noqa
        except Exception:
            pass
    finally:
        with open(MJCF, 'w') as f:
            f.write(original)
        post_hash = _mjcf_md5(MJCF)
        print(f'[FRATE-FSAT] MJCF md5 post: {post_hash}')
        assert post_hash == pre_hash, 'MJCF restoration failed'
