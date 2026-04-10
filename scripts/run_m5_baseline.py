#!/usr/bin/env python3
"""M5 baseline: single-step closed-loop sim with full M1+M2+M3+M4+M5.

All flags enabled (mapping wired, L_com_ref passed to NMPC, SwingPlanner
6D with delayed cosine). Output: results/M5_baseline_1pct/.
"""
import os
import sys

_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')

import numpy as np

from crawlbot.simulation.sim_loop import SimulationLoop
from crawlbot.simulation.config import SimConfig
from crawlbot.diagnostics import run_diagnostics

URDF = os.path.join(_root, 'models', 'VISPA_crawling_fixed.urdf')
MJCF = os.path.join(_root, 'models', 'VISPA_crawling_rwa3.xml')


def _make_m5_config():
    return SimConfig(
        # M2: reworked QP stack
        use_m2_stack=True,
        alpha_com_soft=5.0,
        alpha_passivity=1.0,
        # M3: NMPC conservation-law box
        enforce_hw_conservation=True,
        h_max_tight=np.full(3, 5.0),
        w_L_nmpc=1.0,
        kappa_terminal=1.0,
        # M4: corrected legacy AOCS
        aocs_mode='legacy_corrected',
        aocs_use_legacy_corrected=True,
        aocs_use_H_estimator=False,
    )


def run_case(tag, output_dir, n_steps=1):
    print("\n" + "=" * 70)
    print(f"  M5 baseline: {tag}, n_steps={n_steps}")
    print("=" * 70)

    cfg = _make_m5_config()
    sim = SimulationLoop(mjcf_path=MJCF, urdf_path=URDF, config=cfg)
    sim.setup(n_steps=n_steps, start_a=2, start_b=2)
    log = sim.run(verbose=True)

    os.makedirs(output_dir, exist_ok=True)
    log.save(os.path.join(output_dir, 'sim_log.json'))

    metrics = run_diagnostics(log, output_dir, cfg=cfg)
    print(f"\n  Outputs: {output_dir}")
    return log, metrics


if __name__ == "__main__":
    run_case("1pct", "results/M5_baseline_1pct", n_steps=1)
