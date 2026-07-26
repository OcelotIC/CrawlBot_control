#!/usr/bin/env python3
"""M6 baseline: single-step closed-loop sim with the coarse pre-planner active.

Same config as M5 baseline + `use_coarse_preplanner=True`. The
pre-planner is solved once per step before SS starts and its
momentum-feasible CoM trajectory replaces the TorsoPlanner's
geometric path as the NMPC reference.

Output: Misc/runs/M6_baseline_1pct/.
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


def _make_m6_config():
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
        # M6: coarse pre-planner
        use_coarse_preplanner=True,
        preplanner_M=15,
        preplanner_kappa=0.7,
        preplanner_f_max=25.0,
        preplanner_tau_max=8.0,
        preplanner_w_L=1.0,
        preplanner_w_u=1e-2,
    )


def run_case(tag, output_dir, n_steps=1):
    print("\n" + "=" * 70)
    print(f"  M6 baseline: {tag}, n_steps={n_steps}")
    print("=" * 70)

    cfg = _make_m6_config()
    sim = SimulationLoop(mjcf_path=MJCF, urdf_path=URDF, config=cfg)
    sim.setup(n_steps=n_steps, start_a=2, start_b=2)
    log = sim.run(verbose=True)

    os.makedirs(output_dir, exist_ok=True)
    log.save(os.path.join(output_dir, 'sim_log.json'))

    metrics = run_diagnostics(log, output_dir, cfg=cfg)

    # Extra line: pre-planner stats
    stats = sim._preplanner_stats
    n_ok = sum(1 for s in stats if s['success'])
    if stats:
        avg_ms = float(np.mean([s['solve_ms'] for s in stats]))
        max_ms = float(max(s['solve_ms'] for s in stats))
        avg_iter = float(np.mean([s['iter_count'] for s in stats]))
    else:
        avg_ms = max_ms = avg_iter = 0.0
    print(f"\n  [CoarsePrePlanner] {n_ok}/{len(stats)} solves ok, "
          f"avg {avg_ms:.1f} ms, max {max_ms:.1f} ms, "
          f"avg {avg_iter:.1f} iters")
    print(f"  Outputs: {output_dir}")
    return log, metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1,
                        help='Number of locomotion steps to simulate')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: auto per n_steps)')
    args = parser.parse_args()

    n_steps = args.steps
    if args.output is not None:
        out_dir = args.output
    elif n_steps == 1:
        out_dir = "Misc/runs/M6_baseline_1pct"
    else:
        out_dir = f"results/M6_baseline_1pct_{n_steps}step"
    run_case(f"1pct ({n_steps}-step)", out_dir, n_steps=n_steps)
