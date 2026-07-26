#!/usr/bin/env python3
"""Attribution experiment 3 — loop-free mapping ON.

Re-runs the 1% canonical cooperative-arms 5-step traversal with
`use_local_delta_mapping=True` (the loop-free base-relative δ
reformulation), everything else identical to the canonical run
(scripts/diag_cooperative_arms.py, cooperative_arms_mode=True,
alpha_torso_lin=500). This removes the world-frame δ(q_current) feedback
and the F-SAT rate-clamp band-aid that depends on it.

Hypothesis under test (§7c, candidate c): the 10.8× QP-vs-NMPC contact
LINEAR-force excess at the step-2 peak is driven by the δ(q_current)+F-SAT
mapping injecting jerky base-acceleration demands. Falsifier:
  - if the step-2 |f1| peak / a_com excess COLLAPSES toward the NMPC plan
    with loop-free mapping → mapping/F-SAT debt is the driver.
  - if it PERSISTS (~10×) → mapping is NOT the driver; the excess comes
    from the task structure (torso-lin vs EE co-equal) or the swing
    reaching dynamics → CMM-feedforward / hierarchy territory.

Reads canonical baseline:  Misc/runs/diag_cooperative_arms/sim_log.json
Writes: Misc/runs/diag_attribution/loopfree_mapping/{sim_log.json, compare.json}
"""
import os
import sys
import json

_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('FRAMES_PER_STEP', '0')  # no render capture

import numpy as np

import scripts.run_m7_single_step as r_single
from crawlbot.simulation.sim_loop import SimulationLoop

MJCF = os.path.join(_root, 'models', 'VISPA_crawling_rwa3.xml')
URDF = os.path.join(_root, 'models', 'VISPA_crawling_fixed.urdf')
OUT = os.path.join(_root, 'Misc', 'runs', 'diag_attribution', 'loopfree_mapping')
BASELINE = os.path.join(_root, 'Misc', 'runs', 'diag_cooperative_arms',
                        'sim_log.json')
M_ROBOT = 71.0


def _make_canonical_cfg():
    """Exact canonical cfg from diag_cooperative_arms.main (non-legacy,
    alpha_torso_lin=500, anchor_dx=0.8)."""
    cfg = r_single._make_m7_config()
    cfg.gait_anchor_dx = 0.8
    cfg.mapping_bypass_in_ss = False
    cfg.t_ss_margin = 5.0
    cfg.r_tube = 0.0
    cfg.preplanner_f_max = 25.0
    cfg.use_com_z_standoff = True
    cfg.com_z_standoff = -0.35
    cfg.dock_vel_max = 0.01
    cfg.cooperative_arms_mode = True
    cfg.ss_alpha_torso_lin = 500.0
    cfg.frames_per_step = 0
    cfg.ik_level_axis = np.array([0.0, 0.0, 1.0])
    cfg.ik_q_nominal = np.array([
        0.297781, 1.526727, 0.842257, 1.147432,
        0.751390, 0.251893, 1.332349,
        -0.297781, -1.526727, -0.842257, -1.147432,
        -0.751390, -0.251893, 1.332349])
    cfg.ik_w_posture = 0.2
    return cfg


def _step2_peak_metrics(log_dict):
    t = np.array(log_dict['t'])
    ph = np.array(log_dict['phase'])
    step = np.array(log_dict['step_idx'])
    lam = np.array(log_dict['lambda_qp'])
    lref = np.array(log_dict['lambda_ref'])
    vcom = np.array(log_dict['v_com'])
    dt = np.gradient(t)
    acom = np.gradient(vcom, axis=0) / dt[:, None]
    ss2 = (step == 2) & (ph == 'SS')
    idx = np.where(ss2)[0]
    f1n = np.linalg.norm(lam[:, 0:3], axis=1)
    if len(idx) == 0:
        return {'step2_SS_ticks': 0, 'note': 'no step-2 SS (timed out earlier?)'}
    j = int(idx[int(np.argmax(f1n[idx]))])
    return {
        'step2_SS_ticks': int(len(idx)),
        'peak_t': float(t[j]),
        'f1_peak_N': float(f1n[j]),
        'tau1_peak_Nm': float(np.linalg.norm(lam[j, 3:6])),
        'nmpc_f_N': float(np.linalg.norm(lref[j, 0:3])),
        'qp_over_nmpc_force_ratio': float(
            f1n[j] / max(np.linalg.norm(lref[j, 0:3]), 1e-9)),
        'm_a_com_peak_N': float(M_ROBOT * np.linalg.norm(acom[j])),
        'f1_rms_over_SS_N': float(np.sqrt(np.mean(f1n[idx] ** 2))),
        'f2_max_step2_N': float(np.linalg.norm(lam[idx, 6:9], axis=1).max()),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = _make_canonical_cfg()
    cfg.use_local_delta_mapping = True   # <<< the only change vs canonical

    sim = SimulationLoop(mjcf_path=MJCF, urdf_path=URDF, config=cfg)
    sim.setup(n_steps=5, start_a=2, start_b=2)
    print('=' * 70)
    print('  Exp 3 — loop-free mapping ON (use_local_delta_mapping=True)')
    print(f'  Output: {os.path.relpath(OUT, _root)}')
    print('=' * 70)
    log = sim.run(verbose=True)
    log.save(os.path.join(OUT, 'sim_log.json'))

    lf = json.load(open(os.path.join(OUT, 'sim_log.json')))
    base = json.load(open(BASELINE))

    cmp = {
        'baseline_canonical': {
            'n_docks': len(base['dock_events']),
            'step2': _step2_peak_metrics(base),
        },
        'loopfree_mapping': {
            'n_docks': len(lf['dock_events']),
            'step2': _step2_peak_metrics(lf),
        },
    }
    with open(os.path.join(OUT, 'compare.json'), 'w') as fh:
        json.dump(cmp, fh, indent=2)

    print('\n' + '=' * 70)
    print('  COMPARISON  (baseline canonical  vs  loop-free mapping)')
    print('=' * 70)
    b, l = cmp['baseline_canonical'], cmp['loopfree_mapping']
    print(f"  docks (of 5)            : {b['n_docks']}   vs   {l['n_docks']}")
    bs, ls = b['step2'], l['step2']
    for k in ('f1_peak_N', 'tau1_peak_Nm', 'qp_over_nmpc_force_ratio',
              'm_a_com_peak_N', 'f1_rms_over_SS_N'):
        bv = bs.get(k, float('nan'))
        lv = ls.get(k, float('nan'))
        print(f"  step2 {k:<24}: {bv:8.2f}   vs   {lv:8.2f}")
    print(f"\n  saved: {os.path.relpath(OUT, _root)}/"
          "{sim_log.json, compare.json}")


if __name__ == '__main__':
    main()
