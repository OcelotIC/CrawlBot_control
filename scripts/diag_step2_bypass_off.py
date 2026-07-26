"""Step 2 with mapping_bypass_in_ss DISABLED + diag (a)+(b).

Same baseline config as Run A (T15-FK, α_torso=500, t_ss_margin=5s),
ONLY changing mapping_bypass_in_ss = False. The live mapping layer
(M1 CoMToTorsoMapping) computes r_b_ref = (m/m_b)·c_ref − δ(q)/m_b
at every WBC cycle.

Outputs:
    Misc/runs/diag_step2_bypass_off/
        sim_log.json, step2_log.json
        step2_report.txt, mapping_check.txt
        nmpc_per_step.txt, step_metrics.txt
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
OUT = os.path.join(_root, 'Misc', 'runs', 'diag_step2_bypass_off')

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


def _write_step_metrics(log, n_steps=5, out_path=None):
    t = np.array(log.t)
    phase = np.array(log.phase, dtype=object)
    sidx = np.array(log.step_idx)
    p_torso = np.array(log.p_torso) if log.p_torso else None
    e_ee_raw = log.e_ee_pos
    e_ee_pos = np.array(e_ee_raw) if e_ee_raw else None
    dock_by_step = {ev.get('step', None): ev for ev in log.dock_events}
    abort_by_step = {ab['step_idx']: ab for ab in log.aborted_steps}
    lines = []
    header = (f"{'Step':>4s} {'Outcome':>22s} {'d [mm]':>8s} {'ori [°]':>8s} "
              f"{'recoil [mm]':>12s} {'EE err [mm]':>12s}")
    lines += [header, '-' * len(header)]
    for s in range(n_steps):
        m = (sidx == s) & (phase == 'SS')
        if not m.any():
            lines.append(f"{s:>4d} {'NO_DATA':>22s}")
            continue
        p0 = p_torso[m][0] if (p_torso is not None and p_torso.size) else None
        recoil_mm = (float(np.linalg.norm(
            p_torso[m] - p0, axis=1).max() * 1000.0) if p0 is not None else None)
        ee = e_ee_pos[m] if e_ee_pos is not None and e_ee_pos.size else None
        if ee is not None:
            peak_ee_mm = float(
                (np.linalg.norm(ee, axis=1) if ee.ndim == 2
                 else np.abs(ee)).max() * 1000.0)
        else:
            peak_ee_mm = None
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
        d = f"{d_mm:.2f}" if d_mm is not None else '--'
        o = f"{ori_deg:.2f}" if ori_deg is not None else '--'
        r = f"{recoil_mm:.1f}" if recoil_mm is not None else '--'
        e = f"{peak_ee_mm:.1f}" if peak_ee_mm is not None else '--'
        lines.append(f"{s:>4d} {outcome:>22s} {d:>8s} {o:>8s} {r:>12s} {e:>12s}")
    text = '\n'.join(lines) + '\n'
    if out_path:
        with open(out_path, 'w') as f:
            f.write(text)
    return text


def _write_nmpc(step_log, out_path):
    groups = defaultdict(list)
    for e in step_log:
        groups[_parse_label(e.get('label', ''))].append(e)
    lines = []
    header = (f"{'Step':>4s} {'Phase':>5s} {'N':>4s} "
              f"{'iter_min':>8s} {'iter_max':>8s} {'iter_p95':>8s} "
              f"{'t_max':>7s} {'t_mean':>7s} {'fail':>4s}  status_dist")
    lines += [header, '-' * len(header)]
    for key in sorted(groups.keys()):
        step, ph = key
        entries = groups[key]
        iters = np.array([e['iter'] for e in entries])
        times = np.array([e['time_ms'] for e in entries])
        failed = sum(1 for e in entries
                     if 'Succeeded' not in e['status']
                     and 'Acceptable' not in e['status'])
        statuses = Counter(e['status'] for e in entries)
        sd = ', '.join(f"{s}:{c}" for s, c in statuses.most_common())
        lines.append(
            f"{step:>4d} {ph:>5s} {len(entries):>4d} "
            f"{iters.min():>8d} {iters.max():>8d} "
            f"{int(np.percentile(iters, 95)):>8d} "
            f"{times.max():>7.1f} {times.mean():>7.1f} {failed:>4d}  {sd}")
    text = '\n'.join(lines) + '\n'
    with open(out_path, 'w') as f:
        f.write(text)
    return text


def analyse(diag_log, out_dir):
    """Diag B (QP realization) + Diag C (mapping layer) per the user spec."""
    os.makedirs(out_dir, exist_ok=True)
    if not diag_log:
        print('Empty diag log')
        return
    t = np.array([e['t'] for e in diag_log])
    c_ref = np.array([e['c_ref'] for e in diag_log])
    r_b_ref = np.array([e['r_b_ref'] for e in diag_log])
    p_torso = np.array([e['p_torso'] for e in diag_log])
    a_des = np.array([e['a_torso_des'] if e['a_torso_des'] is not None
                      else [np.nan] * 6 for e in diag_log])
    a_qp = np.array([e['a_torso_qp'] if e['a_torso_qp'] is not None
                     else [np.nan] * 6 for e in diag_log])
    # δ(q): None on cycles where the bypass was active, else 3-vec
    delta_q_raw = [e.get('delta_q') for e in diag_log]
    delta_q = np.array([d if d is not None else [np.nan] * 3
                        for d in delta_q_raw])
    mapping_active = np.array([d is not None for d in delta_q_raw])

    # B
    delta_a = a_des - a_qp
    delta_a_lin = np.linalg.norm(delta_a[:, :3], axis=1)
    delta_a_ang = np.linalg.norm(delta_a[:, 3:], axis=1)

    # C
    delta_map = r_b_ref - c_ref
    delta_track = p_torso - r_b_ref
    delta_total = p_torso - c_ref
    n_map_n = np.linalg.norm(delta_map, axis=1)
    n_track_n = np.linalg.norm(delta_track, axis=1)
    n_total_n = np.linalg.norm(delta_total, axis=1)

    # Mapping verification: how many cycles where δ(q) is finite, how
    # many are stale (identical to the previous cycle), δ statistics,
    # and the update-rate sanity check (consecutive entries differ).
    stale_count = 0
    n_finite = int(mapping_active.sum())
    if n_finite >= 2:
        finite_idx = np.where(mapping_active)[0]
        for i in range(1, len(finite_idx)):
            a, b = finite_idx[i - 1], finite_idx[i]
            if np.allclose(delta_q[a], delta_q[b], atol=1e-12):
                stale_count += 1
        delta_q_disp = float(np.linalg.norm(
            delta_q[finite_idx[-1]] - delta_q[finite_idx[0]]))
        delta_q_max_norm = float(np.linalg.norm(
            delta_q[mapping_active], axis=1).max())
        delta_q_min_norm = float(np.linalg.norm(
            delta_q[mapping_active], axis=1).min())
    else:
        delta_q_disp = float('nan')
        delta_q_max_norm = float('nan')
        delta_q_min_norm = float('nan')

    n_cycles = len(t)

    lines = []
    lines.append('=' * 84)
    lines.append('  Step 2 SS diagnostics — α=500, mapping_bypass_in_ss=False')
    lines.append('=' * 84)
    lines.append(f"  n_cycles: {n_cycles}  (t={t[0]:.2f}..{t[-1]:.2f} s, "
                 f"{t[-1]-t[0]:.2f}s)")
    lines.append(f"  mapping invoked on {n_finite}/{n_cycles} cycles "
                 f"({100*n_finite/n_cycles:.1f}%)")
    lines.append('')
    lines.append('--- B: QP realization (a_torso_des vs a_torso_qp) ---')
    lines.append(f"  ||δa|| linear  mean={np.nanmean(delta_a_lin):.4e} "
                 f"max={np.nanmax(delta_a_lin):.4e}  [m/s²]")
    lines.append(f"  ||δa|| angular mean={np.nanmean(delta_a_ang):.4e} "
                 f"max={np.nanmax(delta_a_ang):.4e}  [rad/s²]")
    lines.append('')
    lines.append('--- C: mapping layer (r_b_ref vs c_ref) ---')
    lines.append(f"  c_ref     displacement: "
                 f"{np.linalg.norm(c_ref[-1] - c_ref[0])*1000:.1f} mm")
    lines.append(f"  r_b_ref   displacement: "
                 f"{np.linalg.norm(r_b_ref[-1] - r_b_ref[0])*1000:.1f} mm")
    lines.append(f"  p_torso   displacement: "
                 f"{np.linalg.norm(p_torso[-1] - p_torso[0])*1000:.1f} mm")
    lines.append(f"  r_b_ref ≡ const? {np.allclose(r_b_ref, r_b_ref[0], atol=1e-6)}")
    lines.append(f"  ||δ_map||   = r_b_ref - c_ref   mean={n_map_n.mean()*1000:.1f}mm "
                 f"max={n_map_n.max()*1000:.1f}mm "
                 f"std={n_map_n.std()*1000:.1f}mm")
    lines.append(f"  ||δ_track|| = p_torso - r_b_ref mean={n_track_n.mean()*1000:.1f}mm "
                 f"max={n_track_n.max()*1000:.1f}mm")
    lines.append(f"  ||δ_total|| = p_torso - c_ref   mean={n_total_n.mean()*1000:.1f}mm "
                 f"max={n_total_n.max()*1000:.1f}mm")
    lines.append('')
    lines.append('--- Mapping δ(q) verification ---')
    lines.append(f"  δ(q) finite on {n_finite}/{n_cycles} cycles")
    if n_finite >= 2:
        lines.append(f"  ||δ(q)|| range: "
                     f"min={delta_q_min_norm:.4f} max={delta_q_max_norm:.4f} (m)")
        lines.append(f"  ||δ(q)_end - δ(q)_start|| = {delta_q_disp:.4f} m "
                     f"(zero ⇒ δ(q) was constant; non-zero ⇒ updated)")
        lines.append(f"  stale cycles (δ unchanged from prev): "
                     f"{stale_count}/{max(0, n_finite-1)}")
        lines.append(f"  Update rate: WBC tick rate (100 Hz) — δ(q) "
                     f"recomputed every QP cycle via mapping.compute() "
                     f"called inside the qs loop.")
    lines.append('')
    # Quintile breakdown
    edges = np.linspace(t.min(), t.max(), 6)
    lines.append('--- Quintile time profile ---')
    lines.append(f"  {'q':>2s} {'t_lo':>7s} {'t_hi':>7s} {'n':>5s} "
                 f"{'δa_lin_max':>10s} {'δa_ang_max':>10s} "
                 f"{'δ_map_max':>10s} {'δ_track_max':>11s} "
                 f"{'δ_total_max':>11s}")
    for q in range(5):
        m = ((t >= edges[q]) & (t <= edges[q + 1])) if q == 4 \
            else ((t >= edges[q]) & (t < edges[q + 1]))
        if not m.any():
            continue
        lines.append(
            f"  {q+1:>2d} {edges[q]:>7.2f} {edges[q+1]:>7.2f} {int(m.sum()):>5d} "
            f"{np.nanmax(delta_a_lin[m]):>10.4f} "
            f"{np.nanmax(delta_a_ang[m]):>10.4f} "
            f"{(n_map_n[m].max()*1000):>10.1f} "
            f"{(n_track_n[m].max()*1000):>11.1f} "
            f"{(n_total_n[m].max()*1000):>11.1f}")

    text = '\n'.join(lines) + '\n'
    with open(os.path.join(out_dir, 'step2_report.txt'), 'w') as f:
        f.write(text)
    print(text)


def main():
    cfg = r_single._make_m7_config()
    cfg.preplanner_a_cruise_max = 0.01
    cfg.preplanner_cruise_ramp_frac = 0.2
    # ── THE single change vs baseline ─────────────────────────
    cfg.mapping_bypass_in_ss = False
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
    sim._step2_diag_enabled = True
    sim._step2_diag_log.clear()

    log = sim.run(verbose=True)

    os.makedirs(OUT, exist_ok=True)
    log.save(os.path.join(OUT, 'sim_log.json'))
    with open(os.path.join(OUT, 'step2_log.json'), 'w') as f:
        json.dump(sim._step2_diag_log, f, indent=2)

    text_metrics = _write_step_metrics(
        log, 5, os.path.join(OUT, 'step_metrics.txt'))
    print('\n=== (a) Per-step outcomes ===\n' + text_metrics)

    step_log = list(sim.nmpc._nmpc.step_log)
    with open(os.path.join(OUT, 'nmpc_step_log.json'), 'w') as f:
        json.dump(step_log, f, indent=2)
    _write_nmpc(step_log, os.path.join(OUT, 'nmpc_per_step.txt'))

    print('\n=== (b) Diag B+C ===')
    analyse(sim._step2_diag_log, OUT)
    print(f'  Outputs: {OUT}')


if __name__ == '__main__':
    with open(MJCF, 'r') as f:
        original = f.read()
    pre_hash = _mjcf_md5(MJCF)
    print(f'[STEP2-BYPASS-OFF] MJCF md5 pre:  {pre_hash}')
    try:
        _mutate_mjcf(damping=0.0, armature=0.05)
        main()
    finally:
        with open(MJCF, 'w') as f:
            f.write(original)
        post_hash = _mjcf_md5(MJCF)
        print(f'[STEP2-BYPASS-OFF] MJCF md5 post: {post_hash}')
        assert post_hash == pre_hash, 'MJCF restoration failed'
