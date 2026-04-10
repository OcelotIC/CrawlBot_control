#!/usr/bin/env python3
"""Sweep alpha_com_soft to find the M5 cascade's stability boundary.

Runs a 1-step closed-loop sim at 1% mass ratio for each value and reports:
  - nmpc_infeasibility_rate
  - hw_saturation_ratio_peak
  - platform_rotation_total_deg
  - com_tracking_err_rms_mm
  - torso_pos_err_peak_mm
  - Dock events

Output: stdout table + results/M5_alpha_sweep/metrics.csv
"""
import os
import sys

_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')

import numpy as np
import csv

from crawlbot.simulation.sim_loop import SimulationLoop
from crawlbot.simulation.config import SimConfig
from crawlbot.diagnostics.metrics import compute_metrics

URDF = os.path.join(_root, 'models', 'VISPA_crawling_fixed.urdf')
MJCF = os.path.join(_root, 'models', 'VISPA_crawling_rwa3.xml')
OUTPUT_DIR = 'results/M5_alpha_sweep'

SWEEP_VALUES = [1.0, 5.0, 10.0, 25.0, 50.0]


def run_one(alpha_com_soft: float):
    cfg = SimConfig(
        use_m2_stack=True,
        alpha_com_soft=alpha_com_soft,
        alpha_passivity=1.0,
        enforce_hw_conservation=True,
        h_max_tight=np.full(3, 5.0),
        w_L_nmpc=1.0,
        kappa_terminal=1.0,
        aocs_mode='legacy_corrected',
        aocs_use_legacy_corrected=True,
    )
    # Silence setup prints
    from io import StringIO
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        sim = SimulationLoop(mjcf_path=MJCF, urdf_path=URDF, config=cfg)
        sim.setup(n_steps=1, start_a=2, start_b=2)
        log = sim.run(verbose=False)
    finally:
        sys.stdout = old_stdout

    m = compute_metrics(log, cfg=cfg)

    def v(name):
        entry = m.get(name, (None, None, 'SKIP'))
        return entry[0]

    return {
        'alpha_com_soft': alpha_com_soft,
        'nmpc_infeasibility': v('nmpc_infeasibility_rate'),
        'hw_sat_peak': v('hw_saturation_ratio_peak'),
        'hw_sat_rms': v('hw_saturation_ratio_rms'),
        'platform_rotation_deg': v('platform_rotation_total_deg'),
        'com_err_rms_mm': v('com_tracking_err_rms_mm'),
        'torso_err_peak_mm': v('torso_pos_err_peak_mm'),
        'n_dock': len(log.dock_events),
    }


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("\n" + "=" * 110)
    print("  M5 alpha_com_soft sweep — 1% single-step, M1+M2+M3+M4+M5 all active")
    print("=" * 110)
    header = (f"{'α_com_soft':>11} {'nmpc infeas':>14} "
              f"{'hw_sat_peak':>12} {'hw_sat_rms':>12} "
              f"{'rot [deg]':>11} {'torso pk [mm]':>14} "
              f"{'CoM RMS [mm]':>14} {'docks':>6}")
    print(header)
    print("-" * 110)

    results = []
    for alpha in SWEEP_VALUES:
        try:
            r = run_one(alpha)
        except Exception as e:
            print(f"{alpha:>11.1f}  FAILED: {type(e).__name__}: {e}")
            continue
        results.append(r)
        print(f"{r['alpha_com_soft']:>11.1f} "
              f"{r['nmpc_infeasibility']*100:>13.1f}% "
              f"{r['hw_sat_peak']:>12.3f} "
              f"{r['hw_sat_rms']:>12.3f} "
              f"{r['platform_rotation_deg']:>11.2f} "
              f"{r['torso_err_peak_mm']:>14.1f} "
              f"{r['com_err_rms_mm']:>14.1f} "
              f"{r['n_dock']:>6d}")

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, 'metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"\nSaved: {csv_path}")
