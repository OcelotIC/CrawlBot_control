#!/usr/bin/env python3
"""Swing planner velocity profile — pre-M7 vs M7 horizon comparison.

Read-only diagnostic. Builds a minimal ContactScheduler (0.8 m step,
0.03 m clearance) and samples SwingPlanner.reference_at() at 1000 points
for two step durations:
    * M7 case: T = 9.094 s (the pre-planner's T_step for this step)
    * pre-M7:  T = 6.000 s (the legacy cfg.t_swing default)

Reports peak |v_ee|, peak |a_ee|, terminal |v_ee|, mean |v_ee|, and the
analytic sanity check (15/8)·d/T vs the numerical straight-line peak.

Saves a two-panel PNG and a CSV to Misc/runs/M7_1pct_1step/.
No MuJoCo, no NMPC — just SwingPlanner math.
"""
import os
import sys

_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from crawlbot.planning.contact_scheduler import (
    ContactScheduler, make_anchor_grid)
from crawlbot.planning.swing_planner import SwingPlanner


def profile_one(T_step: float, n: int = 1000) -> dict:
    """Sample SwingPlanner for one (0.8 m, 0.03 m clearance) step.

    Builds a minimal 1-step gait plan with dt_ss=T_step so
    SwingPlanner.reference_at() returns full v_ee, a_ee vectors.
    """
    anchors_a, anchors_b = make_anchor_grid()
    # 1% sim uses start_a=start_b=2, swings arm B first to anchor 3.
    sched = ContactScheduler(
        dt_ds=0.5, dt_ss=T_step,
        anchors_a=anchors_a, anchors_b=anchors_b)
    sched.plan_traversal(start_a=2, start_b=2, n_steps=1)

    sp = SwingPlanner(sched, clearance=0.03)
    # Set release orientation to identity (same as anchor target).
    sp.set_swing_orientation(np.eye(3))

    # SS phase starts at t = dt_ds (0.5) in the scheduler's timeline.
    t_ss_start = 0.5
    ts = np.linspace(t_ss_start, t_ss_start + T_step, n + 1)
    v_mag = np.zeros(n + 1)
    a_mag = np.zeros(n + 1)
    omega_mag = np.zeros(n + 1)
    tau = np.zeros(n + 1)

    for i, t in enumerate(ts):
        # clamp slightly inside the phase so phase_at returns SS
        tq = min(t, t_ss_start + T_step - 1e-9)
        sr = sp.reference_at(tq)
        v_mag[i] = np.linalg.norm(sr.v_ee)
        a_mag[i] = np.linalg.norm(sr.a_ee)
        omega_mag[i] = np.linalg.norm(sr.omega_ee)
        tau[i] = sr.phase_progress

    # Analytic sanity: peak of straight-line quintic is 15/8 · d/T
    anchors_b_list = anchors_b
    d = float(np.linalg.norm(anchors_b_list[3] - anchors_b_list[2]))
    v_peak_analytic = (15.0 / 8.0) * d / T_step

    i_v = int(np.argmax(v_mag))
    i_a = int(np.argmax(a_mag))
    return {
        'T_step': T_step,
        'ts': ts - t_ss_start,   # time relative to SS start
        'tau': tau,
        'v_mag': v_mag,
        'a_mag': a_mag,
        'omega_mag': omega_mag,
        'd': d,
        'v_peak_numeric': float(v_mag[i_v]),
        'v_peak_tau': float(tau[i_v]),
        'v_peak_analytic': v_peak_analytic,
        'a_peak_numeric': float(a_mag[i_a]),
        'a_peak_tau': float(tau[i_a]),
        'v_terminal': float(v_mag[-1]),
        'v_mean': float(np.mean(v_mag)),
    }


def main():
    cases = [profile_one(9.094203), profile_one(6.0)]

    # ── Print report ─────────────────────────────────────────
    print("=" * 78)
    print("  Swing planner velocity profile — 0.8 m step, 0.03 m clearance")
    print("=" * 78)
    hdr = f"{'metric':<28}{'M7 (T=9.094s)':>20}{'pre-M7 (T=6.0s)':>20}"
    print(hdr)
    print("-" * 78)
    rows = [
        ('step distance d [m]', 'd'),
        ('peak |v_ee| numeric [m/s]', 'v_peak_numeric'),
        ('peak |v_ee| analytic [m/s]', 'v_peak_analytic'),
        ('peak |v_ee| at tau', 'v_peak_tau'),
        ('peak |a_ee| numeric [m/s^2]', 'a_peak_numeric'),
        ('peak |a_ee| at tau', 'a_peak_tau'),
        ('terminal |v_ee| [m/s]', 'v_terminal'),
        ('mean |v_ee| [m/s]', 'v_mean'),
    ]
    for label, key in rows:
        print(f"  {label:<26}"
              f"{cases[0][key]:>20.6f}{cases[1][key]:>20.6f}")
    print()
    print(f"  Peak |v_ee| ratio (pre-M7 / M7) = "
          f"{cases[1]['v_peak_numeric'] / cases[0]['v_peak_numeric']:.3f}")
    print(f"  Peak |a_ee| ratio (pre-M7 / M7) = "
          f"{cases[1]['a_peak_numeric'] / cases[0]['a_peak_numeric']:.3f}")

    # ── Save CSV and PNG ─────────────────────────────────────
    out_dir = os.path.join(_root, 'Misc', 'runs', 'M7_1pct_1step')
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, 'velocity_profile.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t_M7', 'tau_M7', 'v_mag_M7', 'a_mag_M7',
                    't_preM7', 'tau_preM7', 'v_mag_preM7', 'a_mag_preM7'])
        for i in range(len(cases[0]['ts'])):
            w.writerow([
                cases[0]['ts'][i], cases[0]['tau'][i],
                cases[0]['v_mag'][i], cases[0]['a_mag'][i],
                cases[1]['ts'][i], cases[1]['tau'][i],
                cases[1]['v_mag'][i], cases[1]['a_mag'][i],
            ])

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
    ax = axes[0]
    ax.plot(cases[0]['ts'], cases[0]['v_mag'] * 1000,
            label=f"M7 (T=9.094s), peak={cases[0]['v_peak_numeric']*1000:.1f} mm/s",
            color='C0', lw=2)
    ax.plot(cases[1]['ts'], cases[1]['v_mag'] * 1000,
            label=f"pre-M7 (T=6.0s), peak={cases[1]['v_peak_numeric']*1000:.1f} mm/s",
            color='C3', lw=2)
    ax.set_xlabel('t since SS start [s]')
    ax.set_ylabel('|v_ee| [mm/s]')
    ax.set_title('Swing-planner EE speed, 0.8 m step, 0.03 m clearance')
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(cases[0]['ts'], cases[0]['a_mag'] * 1000,
            label=f"M7 (T=9.094s), peak={cases[0]['a_peak_numeric']*1000:.1f} mm/s^2",
            color='C0', lw=2)
    ax.plot(cases[1]['ts'], cases[1]['a_mag'] * 1000,
            label=f"pre-M7 (T=6.0s), peak={cases[1]['a_peak_numeric']*1000:.1f} mm/s^2",
            color='C3', lw=2)
    ax.set_xlabel('t since SS start [s]')
    ax.set_ylabel('|a_ee| [mm/s^2]')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    png_path = os.path.join(out_dir, 'velocity_profile.png')
    plt.savefig(png_path, dpi=110)
    plt.close(fig)

    print()
    print(f"  CSV: {csv_path}")
    print(f"  PNG: {png_path}")


if __name__ == '__main__':
    main()
