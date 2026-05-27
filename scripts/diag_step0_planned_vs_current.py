"""One-line diagnostic: compare δ(q_planned) vs δ(q_current).

Re-runs the bypass=OFF α=500 baseline with the per-cycle diag now
capturing both δ(q_planned) (used by the live mapping during SS) and
δ(q_current) (live arm state, computed alongside for diagnostic
purposes only). Step 0 is the relevant failure point here because
bypass=OFF makes step 0 timeout at 62 mm and step 2 is never reached.

Reports for step 0 SS:
  - max ||r_b_ref_current - r_b_ref_planned||
        (candidate-1 magnitude: is the planned-vs-current arm-config
        mismatch the dominant source of r_b_ref drift?)
  - max ||r_b_ref_current|| displacement over step 0
        (candidate-2 magnitude: is the m_total/m_b=1.78 scaling on its
        own producing > 800 mm of body motion?)
  - max ||δ(q_planned) - δ(q_current)|| [m·kg, before /m_b]
  - max per-component differences

Outputs:
    results/diag_step0_planned_vs_current/
        sim_log.json, step0_log.json, planned_vs_current_report.txt
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys

import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')

import scripts.run_m7_single_step as r_single  # noqa: E402


MJCF = os.path.join(_root, 'models', 'VISPA_crawling_rwa3.xml')
OUT = os.path.join(_root, 'results', 'diag_step0_planned_vs_current')

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


def analyse(diag_log, sim, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    if not diag_log:
        print('Empty diag log')
        return
    m_total = float(sim.mapping.m_total)
    m_b = float(sim.mapping.m_b)
    ratio = m_total / m_b

    # Filter to step 0 SS only
    s0 = [e for e in diag_log if e.get('step_idx') == 0]
    if not s0:
        print('No step 0 SS samples')
        return

    c_ref = np.array([e['c_ref'] for e in s0])
    delta_planned = np.array([e['delta_q'] for e in s0 if e['delta_q'] is not None])
    delta_current = np.array([e['delta_q_current'] for e in s0 if e['delta_q_current'] is not None])

    # Both deltas should be present on every cycle where mapping was
    # invoked (i.e. bypass=OFF). Align lengths if any are None.
    have_pair = [(e['delta_q'] is not None and e['delta_q_current'] is not None)
                 for e in s0]
    n_pair = sum(have_pair)
    c_ref_p = np.array([e['c_ref'] for e, ok in zip(s0, have_pair) if ok])
    delta_p = np.array([e['delta_q'] for e, ok in zip(s0, have_pair) if ok])
    delta_c = np.array([e['delta_q_current'] for e, ok in zip(s0, have_pair) if ok])

    # r_b_ref under each choice of q in δ
    r_b_ref_planned = ratio * c_ref_p - delta_p / m_b
    r_b_ref_current = ratio * c_ref_p - delta_c / m_b

    # Magnitudes
    diff_norm = np.linalg.norm(r_b_ref_current - r_b_ref_planned, axis=1)
    rb_current_disp = np.linalg.norm(
        r_b_ref_current - r_b_ref_current[0], axis=1)
    rb_planned_disp = np.linalg.norm(
        r_b_ref_planned - r_b_ref_planned[0], axis=1)
    delta_norm_diff = np.linalg.norm(delta_p - delta_c, axis=1)

    n_cycles = len(s0)
    lines = []
    lines.append('=' * 84)
    lines.append('  Step 0 SS — δ(q_planned) vs δ(q_current) diagnostic')
    lines.append(f'  (bypass=OFF, α=500, T15-FK baseline; m_total/m_b = {ratio:.3f})')
    lines.append('=' * 84)
    lines.append(f"  Step 0 SS cycles logged: {n_cycles}")
    lines.append(f"  cycles with both δs present: {n_pair}")
    lines.append('')
    lines.append('--- Candidate 1: planned-vs-current arm-config mismatch ---')
    lines.append(f"  max ||δ(q_planned) - δ(q_current)|| "
                 f"= {delta_norm_diff.max():.4f} m·kg")
    lines.append(f"  mean ||δ(q_planned) - δ(q_current)|| "
                 f"= {delta_norm_diff.mean():.4f} m·kg")
    lines.append(f"  max ||r_b_ref_current - r_b_ref_planned|| "
                 f"= {diff_norm.max()*1000:.1f} mm")
    lines.append(f"  mean ||r_b_ref_current - r_b_ref_planned|| "
                 f"= {diff_norm.mean()*1000:.1f} mm")
    verdict1 = ('CANDIDATE 1 IS DOMINANT — fix to use q_current'
                if diff_norm.max() * 1000 > 50
                else ('CANDIDATE 1 IS NEGLIGIBLE — do not change q'
                      if diff_norm.max() * 1000 < 10
                      else 'CANDIDATE 1 IS MARGINAL — between 10 and 50 mm'))
    lines.append(f"  VERDICT(1): {verdict1}")
    lines.append('')
    lines.append('--- Candidate 2: mass-ratio scaling magnitude ---')
    lines.append(f"  max ||r_b_ref_current|| displacement over step 0 = "
                 f"{rb_current_disp.max()*1000:.1f} mm")
    lines.append(f"  max ||r_b_ref_planned|| displacement over step 0 = "
                 f"{rb_planned_disp.max()*1000:.1f} mm")
    verdict2 = ('CANDIDATE 2 IS LOAD-BEARING — mass-ratio scaling alone '
                'produces > 800 mm of body motion regardless of (1)'
                if rb_current_disp.max() * 1000 > 800
                else 'CANDIDATE 2 IS NOT LOAD-BEARING (< 800 mm)')
    lines.append(f"  VERDICT(2): {verdict2}")
    lines.append('')
    lines.append('--- Component-wise max differences ---')
    diff = r_b_ref_current - r_b_ref_planned
    for ax, name in enumerate(['x', 'y', 'z']):
        lines.append(f"  max |Δ_{name}|: planned-vs-current = "
                     f"{np.abs(diff[:, ax]).max()*1000:.2f} mm; "
                     f"r_b_ref_current range = "
                     f"[{r_b_ref_current[:, ax].min()*1000:.1f}, "
                     f"{r_b_ref_current[:, ax].max()*1000:.1f}] mm")
    text = '\n'.join(lines) + '\n'
    print(text)
    with open(os.path.join(out_dir, 'planned_vs_current_report.txt'), 'w') as f:
        f.write(text)


def main():
    cfg = r_single._make_m7_config()
    cfg.preplanner_a_cruise_max = 0.01
    cfg.preplanner_cruise_ramp_frac = 0.2
    cfg.mapping_bypass_in_ss = False   # keep OFF (user's hard constraint)
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
    with open(os.path.join(OUT, 'step0_log.json'), 'w') as f:
        json.dump(sim._step2_diag_log, f, indent=2)

    print(f'  Total diag cycles: {len(sim._step2_diag_log)}')
    analyse(sim._step2_diag_log, sim, OUT)
    print(f'  Outputs: {OUT}')


if __name__ == '__main__':
    with open(MJCF, 'r') as f:
        original = f.read()
    pre_hash = _mjcf_md5(MJCF)
    print(f'[STEP0-DIAG] MJCF md5 pre:  {pre_hash}')
    try:
        _mutate_mjcf(damping=0.0, armature=0.05)
        main()
    finally:
        with open(MJCF, 'w') as f:
            f.write(original)
        post_hash = _mjcf_md5(MJCF)
        print(f'[STEP0-DIAG] MJCF md5 post: {post_hash}')
        assert post_hash == pre_hash, 'MJCF restoration failed'
