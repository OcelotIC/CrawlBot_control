"""Step 2 diagnostics B + C: QP realization + mapping layer.

Runs the α=500 baseline (T15-FK, t_ss_margin=5s) with sim._step2_diag_enabled
to capture per-WBC-cycle (c_ref, r_b_ref, p_torso, a_torso_des, a_torso_qp)
across all step 2 SS cycles. Then analyses + dumps:

  - delta_a_torso = a_torso_des - a_torso_qp   (QP realization error, 6D)
  - delta_map     = r_b_ref - c_ref            (mapping offset, 3D, position)
  - delta_track   = p_torso - r_b_ref          (tracking vs mapped ref, 3D)
  - delta_total   = p_torso - c_ref            (total error vs planner ref, 3D)

Outputs:
    Misc/runs/diag_step2_bc/
        step2_log.json
        step2_report.txt
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys

import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')

import scripts.run_m7_single_step as r_single  # noqa: E402


MJCF = os.path.join(_root, 'models', 'VISPA_crawling_rwa3.xml')
OUT = os.path.join(_root, 'Misc', 'runs', 'diag_step2_bc')

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


def analyse(log: list, out_dir: str):
    """Compute deltas + summary statistics."""
    os.makedirs(out_dir, exist_ok=True)
    if not log:
        print('Empty diag log — step 2 SS was never entered, '
              'or _step2_diag_enabled not set')
        return
    n = len(log)
    t = np.array([e['t'] for e in log])
    c_ref = np.array([e['c_ref'] for e in log])
    r_b_ref = np.array([e['r_b_ref'] for e in log])
    p_torso = np.array([e['p_torso'] for e in log])

    # Filter for entries that have a_torso_des (torso task was active)
    have_torso = [e['a_torso_des'] is not None for e in log]
    a_des = np.array([e['a_torso_des'] if e['a_torso_des'] is not None
                      else [np.nan] * 6 for e in log])  # (N, 6)
    a_qp = np.array([e['a_torso_qp'] if e['a_torso_qp'] is not None
                     else [np.nan] * 6 for e in log])    # (N, 6)

    delta_a = a_des - a_qp                  # (N, 6) [lin(3), ang(3)]
    delta_a_lin_n = np.linalg.norm(delta_a[:, :3], axis=1)
    delta_a_ang_n = np.linalg.norm(delta_a[:, 3:], axis=1)
    a_des_lin_n = np.linalg.norm(a_des[:, :3], axis=1)
    a_des_ang_n = np.linalg.norm(a_des[:, 3:], axis=1)
    a_qp_lin_n = np.linalg.norm(a_qp[:, :3], axis=1)
    a_qp_ang_n = np.linalg.norm(a_qp[:, 3:], axis=1)

    delta_map = r_b_ref - c_ref             # (N, 3)
    delta_track = p_torso - r_b_ref         # (N, 3)
    delta_total = p_torso - c_ref           # (N, 3)
    delta_map_n = np.linalg.norm(delta_map, axis=1)
    delta_track_n = np.linalg.norm(delta_track, axis=1)
    delta_total_n = np.linalg.norm(delta_total, axis=1)

    # Time profile: split step 2 SS into 5 quintiles
    quintile_edges = np.linspace(t.min(), t.max(), 6)
    bins = []
    for q in range(5):
        m = (t >= quintile_edges[q]) & (t < quintile_edges[q + 1])
        if q == 4:
            m = (t >= quintile_edges[q]) & (t <= quintile_edges[q + 1])
        if not m.any():
            continue
        bins.append({
            'q': q + 1,
            't_lo': float(quintile_edges[q]),
            't_hi': float(quintile_edges[q + 1]),
            'n': int(m.sum()),
            'delta_a_lin_max_mm_s2': float(delta_a_lin_n[m].max() * 1000) if a_des_lin_n[m].max() > 0 or True else None,
            'delta_a_lin_mean': float(delta_a_lin_n[m].mean()),
            'delta_a_ang_max': float(delta_a_ang_n[m].max()),
            'delta_a_ang_mean': float(delta_a_ang_n[m].mean()),
            'delta_map_max_mm': float(delta_map_n[m].max() * 1000),
            'delta_map_mean_mm': float(delta_map_n[m].mean() * 1000),
            'delta_track_max_mm': float(delta_track_n[m].max() * 1000),
            'delta_total_max_mm': float(delta_total_n[m].max() * 1000),
        })

    # Top-level summary
    summary = {
        'n_cycles': n,
        't_range_s': [float(t.min()), float(t.max())],
        'torso_task_active_pct': 100.0 * sum(have_torso) / n,
        # B — QP realization
        'B': {
            'delta_a_lin_max_m_s2': float(np.nanmax(delta_a_lin_n)),
            'delta_a_lin_mean_m_s2': float(np.nanmean(delta_a_lin_n)),
            'delta_a_ang_max_rad_s2': float(np.nanmax(delta_a_ang_n)),
            'delta_a_ang_mean_rad_s2': float(np.nanmean(delta_a_ang_n)),
            'a_des_lin_max_m_s2': float(np.nanmax(a_des_lin_n)),
            'a_des_ang_max_rad_s2': float(np.nanmax(a_des_ang_n)),
            'a_qp_lin_max_m_s2': float(np.nanmax(a_qp_lin_n)),
            'a_qp_ang_max_rad_s2': float(np.nanmax(a_qp_ang_n)),
        },
        # C — mapping layer
        'C': {
            'delta_map_max_mm': float(delta_map_n.max() * 1000),
            'delta_map_mean_mm': float(delta_map_n.mean() * 1000),
            'delta_track_max_mm': float(delta_track_n.max() * 1000),
            'delta_track_mean_mm': float(delta_track_n.mean() * 1000),
            'delta_total_max_mm': float(delta_total_n.max() * 1000),
            'delta_total_mean_mm': float(delta_total_n.mean() * 1000),
            'r_b_ref_constant': bool(np.allclose(
                r_b_ref, r_b_ref[0], atol=1e-6)),
            'c_ref_displacement_mm': float(np.linalg.norm(
                c_ref[-1] - c_ref[0]) * 1000),
            'r_b_ref_displacement_mm': float(np.linalg.norm(
                r_b_ref[-1] - r_b_ref[0]) * 1000),
            'p_torso_displacement_mm': float(np.linalg.norm(
                p_torso[-1] - p_torso[0]) * 1000),
        },
        'quintiles': bins,
    }
    with open(os.path.join(out_dir, 'step2_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    lines = []
    lines.append('=' * 84)
    lines.append('  Step 2 SS diagnostics (B + C) — α=500 baseline')
    lines.append('=' * 84)
    lines.append(f"  n_cycles: {n} ({summary['torso_task_active_pct']:.0f}% "
                 f"with torso task active)")
    lines.append(f"  t range: {t.min():.3f} ... {t.max():.3f} s "
                 f"({t.max()-t.min():.3f} s)")
    lines.append('')
    lines.append('--- B: QP realization (a_torso_des vs a_torso_qp) ---')
    lines.append(f"  ||δa|| linear  mean={summary['B']['delta_a_lin_mean_m_s2']:.4e} "
                 f"max={summary['B']['delta_a_lin_max_m_s2']:.4e}  [m/s²]")
    lines.append(f"  ||δa|| angular mean={summary['B']['delta_a_ang_mean_rad_s2']:.4e} "
                 f"max={summary['B']['delta_a_ang_max_rad_s2']:.4e}  [rad/s²]")
    lines.append(f"  ||a_des|| max  linear={summary['B']['a_des_lin_max_m_s2']:.4e} "
                 f"angular={summary['B']['a_des_ang_max_rad_s2']:.4e}")
    lines.append(f"  ||a_qp||  max  linear={summary['B']['a_qp_lin_max_m_s2']:.4e} "
                 f"angular={summary['B']['a_qp_ang_max_rad_s2']:.4e}")
    lines.append('')
    lines.append('--- C: mapping layer (c_ref, r_b_ref, p_torso) ---')
    lines.append(f"  c_ref displacement   (planner)        = "
                 f"{summary['C']['c_ref_displacement_mm']:.1f} mm")
    lines.append(f"  r_b_ref displacement (WBC torso ref)  = "
                 f"{summary['C']['r_b_ref_displacement_mm']:.1f} mm")
    lines.append(f"  p_torso displacement (actual)         = "
                 f"{summary['C']['p_torso_displacement_mm']:.1f} mm")
    lines.append(f"  r_b_ref ≡ const (frozen)? "
                 f"{summary['C']['r_b_ref_constant']}")
    lines.append(f"  ||δ_map||    = r_b_ref - c_ref   mean="
                 f"{summary['C']['delta_map_mean_mm']:.1f}mm "
                 f"max={summary['C']['delta_map_max_mm']:.1f}mm")
    lines.append(f"  ||δ_track||  = p_torso - r_b_ref mean="
                 f"{summary['C']['delta_track_mean_mm']:.1f}mm "
                 f"max={summary['C']['delta_track_max_mm']:.1f}mm")
    lines.append(f"  ||δ_total||  = p_torso - c_ref   mean="
                 f"{summary['C']['delta_total_mean_mm']:.1f}mm "
                 f"max={summary['C']['delta_total_max_mm']:.1f}mm")
    lines.append('')
    lines.append('--- Quintile time profile ---')
    lines.append(f"  {'q':>2s} {'t_lo':>7s} {'t_hi':>7s} {'n':>5s} "
                 f"{'δa_lin_max':>10s} {'δa_ang_max':>10s} "
                 f"{'δ_map_max':>10s} {'δ_track_max':>11s} "
                 f"{'δ_total_max':>11s}")
    for b in bins:
        lines.append(
            f"  {b['q']:>2d} {b['t_lo']:>7.2f} {b['t_hi']:>7.2f} "
            f"{b['n']:>5d} "
            f"{b['delta_a_lin_max_mm_s2']/1000:>10.4f} "
            f"{b['delta_a_ang_max']:>10.4f} "
            f"{b['delta_map_max_mm']:>10.1f} "
            f"{b['delta_track_max_mm']:>11.1f} "
            f"{b['delta_total_max_mm']:>11.1f}")
    text = '\n'.join(lines) + '\n'
    with open(os.path.join(out_dir, 'step2_report.txt'), 'w') as f:
        f.write(text)
    print(text)


def main():
    cfg = r_single._make_m7_config()
    # T15-FK baseline (identical to Run A in margin sweep).
    cfg.preplanner_a_cruise_max = 0.01
    cfg.preplanner_cruise_ramp_frac = 0.2
    cfg.mapping_bypass_in_ss = True   # <-- keep baseline value
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
    cfg.ss_alpha_torso = 500.0  # baseline
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
    analyse(sim._step2_diag_log, OUT)
    print(f'  Outputs: {OUT}')


if __name__ == '__main__':
    with open(MJCF, 'r') as f:
        original = f.read()
    pre_hash = _mjcf_md5(MJCF)
    print(f'[STEP2-BC] MJCF md5 pre:  {pre_hash}')
    try:
        _mutate_mjcf(damping=0.0, armature=0.05)
        main()
    finally:
        with open(MJCF, 'w') as f:
            f.write(original)
        post_hash = _mjcf_md5(MJCF)
        print(f'[STEP2-BC] MJCF md5 post: {post_hash}')
        assert post_hash == pre_hash, 'MJCF restoration failed'
