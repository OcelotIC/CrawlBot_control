#!/usr/bin/env python3
"""Instrument the NMPC call inside sim_loop to see the actual inputs
during the first few NMPC calls of the single-step M4 run.

Produces a table of (step_idx, phase, r_com_0, hw_0, c_simple, status).
Does NOT run the full sim — bails out after 20 NMPC calls.
"""
import os
import sys

_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')

import numpy as np

from crawlbot.simulation.sim_loop import SimulationLoop
from crawlbot.simulation.config import SimConfig
from crawlbot.solvers.centroidal_nmpc import CentroidalNMPC

URDF = os.path.join(_root, 'models', 'VISPA_crawling_fixed.urdf')
MJCF = os.path.join(_root, 'models', 'VISPA_crawling_rwa3.xml')


def _make_m4_config():
    return SimConfig(
        use_m2_stack=True,
        alpha_com_soft=5.0,
        alpha_passivity=1.0,
        enforce_hw_conservation=True,
        h_max_tight=np.full(3, 5.0),
        w_L_nmpc=1.0,
        kappa_terminal=0.9,
        aocs_mode='legacy_corrected',
        aocs_use_legacy_corrected=True,
    )


def main():
    # Monkey-patch CentroidalNMPC.solve to log inputs + outputs
    original_solve = CentroidalNMPC.solve
    call_log = []

    def traced_solve(self, r_com, v_com, L_com, r_com_ref, v_com_ref,
                     contact_config, warm_start=True, hw_current=None,
                     L_com_ref=None):
        c = self.compute_c_simple(r_com, v_com, L_com, hw_current)
        res = original_solve(
            self, r_com=r_com, v_com=v_com, L_com=L_com,
            r_com_ref=r_com_ref, v_com_ref=v_com_ref,
            contact_config=contact_config, warm_start=warm_start,
            hw_current=hw_current, L_com_ref=L_com_ref)
        r_plan, v_plan, L_plan, lam, info = res
        call_log.append({
            'idx': len(call_log),
            'r_com_0': r_com.copy(),
            'v_com_0': v_com.copy(),
            'L_com_0': L_com.copy(),
            'r_com_ref': r_com_ref.copy(),
            'v_com_ref': v_com_ref.copy(),
            'hw_0': (hw_current.copy() if hw_current is not None
                     else np.zeros(3)),
            'c_simple': c.copy(),
            'status': info.status,
            'success': info.success,
            'cost': float(info.cost) if np.isfinite(info.cost) else -1.0,
        })
        return res

    CentroidalNMPC.solve = traced_solve

    cfg = _make_m4_config()
    sim = SimulationLoop(mjcf_path=MJCF, urdf_path=URDF, config=cfg)
    sim.setup(n_steps=1, start_a=2, start_b=2)

    # Full run — no early bail
    sim.run(verbose=False)

    CentroidalNMPC.solve = original_solve

    # Summary: count statuses and print first / first failure / last calls
    np.set_printoptions(precision=3, suppress=True, linewidth=120)
    n_total = len(call_log)
    n_fail = sum(1 for e in call_log if not e['success'])
    print("\n" + "=" * 78)
    print(f"Total NMPC calls: {n_total}, failures: {n_fail} "
          f"({100*n_fail/max(n_total,1):.1f}%)")
    print("=" * 78)

    # Find first failure
    first_fail = None
    for i, e in enumerate(call_log):
        if not e['success']:
            first_fail = i
            break
    print(f"First failure at call idx = {first_fail}")
    print()

    # Print a compact row per call, marking failures
    print(f"{'idx':>4} {'ok':>2} {'cost':>9}  {'|r_com_ref|':>11}  "
          f"{'|r_com_0|':>9}  {'|L_com_0|':>9}  {'|hw_0|':>7}  "
          f"{'|c_simple|':>10}  status")
    print("-" * 120)
    def rnorm(v): return float(np.linalg.norm(v))
    # Show calls around the first failure + some early + some late
    target_idxs = set()
    target_idxs.update(range(0, 20))                         # early
    if first_fail is not None:
        target_idxs.update(range(max(0, first_fail - 5),
                                  min(n_total, first_fail + 10)))
    target_idxs.update(range(n_total - 5, n_total))          # end
    for i, e in enumerate(call_log):
        if i not in target_idxs:
            continue
        mark = 'OK' if e['success'] else 'X'
        cost_s = f"{e['cost']:9.2f}" if e['cost'] >= 0 else "   -inf  "
        print(f"{e['idx']:>4d} {mark:>2s} {cost_s}  "
              f"{rnorm(e['r_com_ref']):>11.4f}  "
              f"{rnorm(e['r_com_0']):>9.4f}  "
              f"{rnorm(e['L_com_0']):>9.4f}  "
              f"{rnorm(e['hw_0']):>7.3f}  "
              f"{rnorm(e['c_simple']):>10.4f}  {e['status']}")


if __name__ == "__main__":
    main()
