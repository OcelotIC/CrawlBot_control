#!/usr/bin/env python3
"""Diagnostic: swing docking + hw slack + NMPC infeasibility correlation.

Three questions, answered from the latest M6 1% single-step sim_log:

 (2) Why isn't the swing arm docking?
     - Min d_swing over the sim
     - Time of min, phase at that time
     - EE task's effective acceleration command at closest approach
     - EE pos/ori tracking error at closest approach
     - Peak ||tau_q|| during EXT approach

 (3) Is the hw soft slack actually firing?
     - Peak ||hw_phys||, % of steps exceeding the ±5 Nms box
     - Correlation between hw excursions and NMPC infeasibility
     - Breakdown of NMPC failures by status code and by phase

Output:
     stdout report + results/M6_platform_diag/swing_vs_hw.png
"""
import os
import sys

_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = 'results/M6_platform_diag'
LOG_PATH = 'results/M6_baseline_1pct/sim_log.json'


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(LOG_PATH) as f:
        log = json.load(f)

    t = np.asarray(log['t'])
    phase = np.asarray(log['phase'])
    d_sw = np.asarray(log['d_grip_swing'])
    d_st = np.asarray(log['d_grip_stance'])
    e_ee_pos = np.asarray(log['e_ee_pos'])
    e_ee_ori = np.asarray(log['e_ee_ori'])
    tau = np.asarray(log['tau'])
    hw = np.asarray(log['hw_physical'])
    L_com = np.asarray(log['L_com'])
    nmpc_ok = np.asarray(log['nmpc_ok'])
    nmpc_status = np.asarray(log['nmpc_status'])

    # ──────────────────────────────────────────────────────────────── #
    # (2) SWING DOCKING
    # ──────────────────────────────────────────────────────────────── #
    print("=" * 78)
    print("  (2) SWING DOCKING")
    print("=" * 78)
    i_min = int(np.argmin(d_sw))
    print(f"  Min d_swing          : {d_sw[i_min] * 1000:.2f} mm "
          f"(dock threshold = 5 mm)")
    print(f"  Time of min          : {t[i_min]:.2f} s")
    print(f"  Phase at min         : {phase[i_min]!r}")
    print(f"  Index of min         : {i_min}")
    print(f"  Final d_swing        : {d_sw[-1] * 1000:.1f} mm")
    print(f"  Dock events          : {len(log.get('dock_events', []))}")
    print()
    print(f"  Peak EE pos err      : {e_ee_pos.max() * 1000:.1f} mm")
    print(f"  Mean EE pos err      : {e_ee_pos.mean() * 1000:.1f} mm")
    print(f"  Peak EE ori err      : {e_ee_ori.max():.1f} deg")
    print(f"  At idx={i_min}: EE pos err = {e_ee_pos[i_min]*1000:.1f} mm, "
          f"ori err = {e_ee_ori[i_min]:.1f} deg")
    print()
    # Joint torques at closest approach
    tau_pk_idx = np.max(np.abs(tau[i_min]))
    print(f"  Max |τ_j| at min d_sw: {tau_pk_idx:.2f} Nm  (limit 20)")
    ext_mask = phase == 'EXT'
    if ext_mask.any():
        tau_ext = tau[ext_mask]
        tau_pk_ext = np.max(np.abs(tau_ext), axis=1)
        print(f"  Peak |τ_j| during EXT: {tau_pk_ext.max():.2f} Nm")
        print(f"  % EXT steps saturated: "
              f"{100 * np.mean(tau_pk_ext > 19.9):.1f}%")
    # How much of d_swing's motion comes from the robot moving toward
    # the anchor vs the anchor moving away (structure rotation)?
    # We don't have per-step p_ee_ref vs r_target logs — just report the
    # trajectory of d_swing around the minimum.
    w = 30  # ±0.3 s window
    lo = max(0, i_min - w)
    hi = min(len(t), i_min + w + 1)
    print(f"  d_swing in window   (t {t[lo]:.2f} … {t[hi-1]:.2f} s):")
    for k in range(lo, hi, 3):
        marker = " ←" if k == i_min else ""
        print(f"    t={t[k]:5.2f}s  d={d_sw[k]*1000:7.2f} mm  "
              f"e_ee_pos={e_ee_pos[k]*1000:7.2f} mm  "
              f"|τ|max={np.max(np.abs(tau[k])):5.1f} Nm{marker}")

    # ──────────────────────────────────────────────────────────────── #
    # (3) hw SLACK FIRING + INFEAS CORRELATION
    # ──────────────────────────────────────────────────────────────── #
    print()
    print("=" * 78)
    print("  (3) hw SLACK FIRING + NMPC INFEASIBILITY CORRELATION")
    print("=" * 78)
    hw_mag = np.max(np.abs(hw), axis=1)
    h_box = 5.0
    in_box = hw_mag <= h_box
    over_box = hw_mag > h_box
    print(f"  |hw|_∞ stats   : peak = {hw_mag.max():.2f} Nms   "
          f"mean = {hw_mag.mean():.2f} Nms")
    print(f"  Steps inside box  : {int(in_box.sum())}/{len(hw_mag)} "
          f"({100 * in_box.mean():.1f} %)")
    print(f"  Steps OVER box    : {int(over_box.sum())}/{len(hw_mag)} "
          f"({100 * over_box.mean():.1f} %)")
    print(f"  Peak excursion    : {(hw_mag.max() - h_box):.3f} Nms "
          f"above ±{h_box}")
    print()
    # NMPC infeas summary
    n_fail = int(np.sum(~nmpc_ok))
    print(f"  NMPC failures     : {n_fail}/{len(nmpc_ok)} "
          f"({100 * (1 - nmpc_ok.mean()):.1f} %)")
    from collections import Counter
    code_counts = Counter(int(c) for c in nmpc_status)
    print(f"  NMPC status codes : {dict(code_counts)}  "
          f"(0=ok, 1=max_iter, 2=infeasible)")
    # Phase breakdown
    print()
    print("  Per-phase NMPC failure rate:")
    for ph in ['DS', 'SS', 'EXT']:
        mask = phase == ph
        if not mask.any():
            continue
        n = int(mask.sum())
        fails = int(np.sum(~nmpc_ok[mask]))
        hw_over = int(np.sum(over_box[mask]))
        print(f"    {ph:4s}  n={n:4d}  nmpc_fail={fails:4d} "
              f"({100 * fails / n:5.1f}%)  "
              f"hw_over={hw_over:4d} ({100 * hw_over / n:5.1f}%)")

    # Correlation: do NMPC failures line up with hw excursions?
    print()
    print("  Joint distribution of (hw_over × nmpc_fail):")
    n11 = int(np.sum(over_box & (~nmpc_ok)))
    n10 = int(np.sum(over_box & nmpc_ok))
    n01 = int(np.sum((~over_box) & (~nmpc_ok)))
    n00 = int(np.sum((~over_box) & nmpc_ok))
    print(f"    hw_OK,  NMPC_OK   : {n00:4d}")
    print(f"    hw_OK,  NMPC_FAIL : {n01:4d}")
    print(f"    hw_OVER,NMPC_OK   : {n10:4d}")
    print(f"    hw_OVER,NMPC_FAIL : {n11:4d}")
    p_fail_given_over = n11 / max(1, n11 + n10)
    p_fail_given_in = n01 / max(1, n01 + n00)
    print(f"  P(NMPC fail | hw over box)  = {100 * p_fail_given_over:.1f}%")
    print(f"  P(NMPC fail | hw in box)    = {100 * p_fail_given_in:.1f}%")
    if p_fail_given_over > 3 * p_fail_given_in:
        print("  → NMPC failures strongly correlated with hw excursions.")
    elif p_fail_given_over > 1.5 * p_fail_given_in:
        print("  → Weak positive correlation between hw excursions and NMPC failures.")
    else:
        print("  → No strong correlation: NMPC failures have a different "
              "root cause.")

    # Slack variables: we don't log them per-step, but we can infer from
    # hw_over that the slack is absorbing the excess (otherwise the QP
    # would hit the hard bound and fail). QP failures:
    qp_fails = int(np.sum(~np.asarray(log['qp_ok'])))
    print()
    print(f"  QP failures (soft slack doing its job?): {qp_fails}/{len(nmpc_ok)}")
    if qp_fails == 0 and over_box.any():
        print("  → soft slack is firing: hw is over the box at "
              f"{int(over_box.sum())} steps but the QP never fails.")

    # ──────────────────────────────────────────────────────────────── #
    # Plot
    # ──────────────────────────────────────────────────────────────── #
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    ax = axes[0]
    ax.plot(t, d_sw * 1000, 'r', lw=1.5, label='d_swing')
    ax.plot(t, d_st * 1000, 'g', lw=1.0, alpha=0.6, label='d_stance')
    ax.axhline(5, color='k', ls='--', lw=0.8, label='dock 5 mm')
    ax.axvline(t[i_min], color='b', ls=':', lw=0.8, label=f'min at {t[i_min]:.1f}s')
    ax.set_ylabel('gripper distance [mm]')
    ax.set_yscale('symlog', linthresh=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)
    ax.set_title('Gripper distance to dock target')

    ax = axes[1]
    ax.plot(t, e_ee_pos * 1000, 'r', lw=1.2, label='|p_ee - p_ref|')
    ax.set_ylabel('EE pos err [mm]')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[2]
    for i, lab in enumerate(['hw_x', 'hw_y', 'hw_z']):
        ax.plot(t, hw[:, i], lw=1.0, label=lab)
    ax.axhline(+5, color='k', ls='--', lw=0.6)
    ax.axhline(-5, color='k', ls='--', lw=0.6)
    ax.fill_between(t, -5, 5, color='gray', alpha=0.1)
    ax.set_ylabel('hw [Nms]')
    ax.legend(fontsize=8, loc='upper right'); ax.grid(alpha=0.3)
    ax.set_title('hw physical (shaded = ±5 Nms box)')

    ax = axes[3]
    # Red shading where NMPC failed
    fail_mask = ~nmpc_ok
    for i in range(1, len(t)):
        if fail_mask[i]:
            ax.axvspan(t[i - 1], t[i], color='red', alpha=0.25)
    # Hw-over shading
    for i in range(1, len(t)):
        if over_box[i]:
            ax.axvspan(t[i - 1], t[i], color='orange', alpha=0.1)
    ax.plot(t, hw_mag, 'b-', lw=1.0, label='|hw|_inf')
    ax.axhline(5, color='k', ls='--', lw=0.6)
    ax.set_ylabel('|hw|_inf [Nms]')
    ax.set_xlabel('time [s]')
    ax.set_title('NMPC fails (red) vs hw excursions (orange)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, 'swing_vs_hw.png')
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\n  saved: {out}")


if __name__ == "__main__":
    main()
