#!/usr/bin/env python3
"""M7 single-step closed-loop sim at 1% mass ratio with full diagnostics.

Validates the M7 two-phase state machine (DS/SS) and synchronized
trajectories. Prints:
  - T_step from the pre-planner
  - SwingPlanner phase duration (must equal T_step)
  - TorsoPlanner phase duration (must equal T_step)
  - Unique phase strings in the log (must be {'DS', 'SS'} — no 'EXT')
  - Dock success under d<5mm AND ori<5° gate

Output: results/M7_1pct_1step/.
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


def _make_m7_config():
    """M7 config: all M1–M6 features + mandatory coarse pre-planner.

    No t_swing / t_ext_max / torso_delay / use_coarse_preplanner knobs:
    the pre-planner produces T_step and both torso + swing trajectories
    are synchronized over [0, T_step].
    """
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
        # M6: coarse pre-planner (now mandatory in M7; config knobs
        # still tune the NLP)
        preplanner_M=15,
        preplanner_kappa=0.7,
        preplanner_f_max=25.0,
        preplanner_tau_max=8.0,
        preplanner_w_L=1.0,
        preplanner_w_u=1e-2,
    )


def _print_phase_sync_report(sim, log):
    """Print the M7 trajectory-sync validation block."""
    print("\n" + "=" * 70)
    print("  M7 phase / synchronization report")
    print("=" * 70)

    # (1) Phase column: only DS and SS
    phases_seen = sorted(set(log.phase))
    print(f"  Unique phase strings in log: {phases_seen}")
    forbidden = {'EXT', 'EXT_CLOSE'}.intersection(phases_seen)
    if forbidden:
        print(f"  !! FAIL: forbidden phase strings present: {forbidden}")
    else:
        print("  OK: no EXT / EXT_CLOSE phases present.")

    # (2) T_step from pre-planner + both planner durations
    stats = sim._preplanner_stats
    if not stats:
        print("  !! No pre-planner stats recorded.")
        return
    for k, s in enumerate(stats):
        print(f"  Step {k}: pre-planner "
              f"success={s['success']}, "
              f"T_step={s.get('T_step', float('nan')):.3f} s, "
              f"{s['solve_ms']:.1f} ms, {s['iter_count']} iters, "
              f"status={s['status']}")

    # SS GaitPhase.duration (set by GaitPlan.set_step_duration)
    ss_phases = [gp for gp in sim.plan.phases
                 if gp.phase.value != 'double']
    print(f"  SwingPlanner SS phase durations: "
          f"{[round(gp.duration, 3) for gp in ss_phases]}")

    # TorsoPlanner internal phases
    torso_durations = [round(p['duration'], 3)
                       for p in sim.torso_planner._phases]
    print(f"  TorsoPlanner last phase durations: {torso_durations}")

    if ss_phases and torso_durations:
        ss_d = ss_phases[0].duration
        torso_d = torso_durations[-1]
        match = abs(ss_d - torso_d) < 1e-9
        verdict = "OK" if match else "!! MISMATCH"
        print(f"  {verdict}: swing duration={ss_d:.6f} s, "
              f"torso duration={torso_d:.6f} s, "
              f"Δ={abs(ss_d - torso_d):.2e} s")

    # (3) Dock result (d<5mm AND ori<5°)
    print(f"  Dock events: {len(log.dock_events)}")
    for ev in log.dock_events:
        print(f"    t={ev['t']:.2f} s  arm={ev['arm']}  anchor={ev['anchor']}  "
              f"d={ev['d_mm']:.2f} mm  ori={ev['ori_deg']:.2f} deg  "
              f"method={ev['method']}")

    print(f"  Aborted steps: {len(log.aborted_steps)}")
    for ab in log.aborted_steps:
        print(f"    step {ab['step_idx']}: reason={ab['reason']}  "
              f"t={ab.get('t', float('nan')):.2f} s")


def run_case(tag, output_dir, n_steps=1):
    print("\n" + "=" * 70)
    print(f"  M7 closed-loop: {tag}, n_steps={n_steps}")
    print("=" * 70)

    cfg = _make_m7_config()
    sim = SimulationLoop(mjcf_path=MJCF, urdf_path=URDF, config=cfg)
    sim.setup(n_steps=n_steps, start_a=2, start_b=2)
    # M7 debug: capture L_com_ref for the first 5 SS NMPC calls so we
    # can confirm the TorsoPlanner momentum feedforward is nonzero
    # during torso reorientation (M5 wiring check).
    sim._debug_l_com_ref_trace_limit = 5
    log = sim.run(verbose=True)

    # Print the L_com_ref trace right after the run so it appears above
    # the metrics table in the output.
    trace = getattr(sim, '_debug_l_com_ref_trace', [])
    print("\n" + "=" * 70)
    print("  L_com_ref fed to NMPC (first 5 SS calls)")
    print("=" * 70)
    if not trace:
        print("  !! empty trace (no SS NMPC calls captured)")
    else:
        print(f"  {'t [s]':>8}  {'t_mid [s]':>10}  {'L_com_ref [Nms]':>36}  {'||L_com_ref||':>14}")
        for e in trace:
            L = e['L_com_ref']
            print(f"  {e['t']:8.2f}  {e['t_mid']:10.2f}  "
                  f"[{L[0]:+8.4f}, {L[1]:+8.4f}, {L[2]:+8.4f}]  "
                  f"{e['norm']:14.6f}")
        any_nonzero = any(e['norm'] > 1e-9 for e in trace)
        print(f"  L_com_ref wired: {'YES' if any_nonzero else 'NO — all zero, M5 feedforward is off'}")

    os.makedirs(output_dir, exist_ok=True)
    log.save(os.path.join(output_dir, 'sim_log.json'))

    _print_phase_sync_report(sim, log)

    print("\n" + "=" * 70)
    print("  Full diagnostic metrics")
    print("=" * 70)
    metrics = run_diagnostics(log, output_dir, cfg=cfg)

    print(f"\n  Outputs: {output_dir}")
    return sim, log, metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1,
                        help='Number of locomotion steps')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory')
    args = parser.parse_args()

    n_steps = args.steps
    if args.output is not None:
        out_dir = args.output
    elif n_steps == 1:
        out_dir = "results/M7_1pct_1step"
    else:
        out_dir = f"results/M7_1pct_{n_steps}step"
    run_case(f"1pct {n_steps}-step", out_dir, n_steps=n_steps)
