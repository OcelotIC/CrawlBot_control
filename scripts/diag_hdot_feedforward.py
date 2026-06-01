"""Diagnostic: AOCS Ḣ feedforward — analytical or finite-difference?

For the wheel-sizing study (PR #20), τ_w_max=2 Nm sits at the working edge with
3.9% τ_w saturation. The actuator decision (reaction wheel vs CMG) depends on
whether those τ_w peaks are:
  (a) NARROW SPIKES aligned with contact transitions  → FD/estimator artifact,
      reducible without changing actuator
  (b) BROAD demand distributed across swing phases     → physical, CMG-justified

How is Ḣ produced today?
  `legacy_corrected` (canonical AOCS, `force_estimator.py:356–366`):
      L_dot_est  = (L_com − L_com_prev) / dt
      orbital    = r_com × m·(v_com − v_com_prev)/dt
      Ḣ_FF       = L_dot_est + orbital
      τ_AOCS     = -Ḣ_FF + K_hw·hw_error + (PD/PID terms in legacy_pd/pid_*)
  `H_est`           : same finite-difference but on a LOW-PASS filtered H_rO
                      (filter_tau ≈ 16 ms).
  `nmpc_plan` mode listed in `config.py:71` but NOT implemented — no analytical
    Ḣ from the NMPC's planned contact wrenches exists in the current codebase.

So Ḣ_FF is a **one-step finite difference of measured L_com and v_com at
dt_qp = 0.01 s**. Any discontinuity in L_com or v_com (contact switches, weld
activations, mapping-jitter cycles) is amplified into a Ḣ_FF spike, which drives
τ_AOCS into saturation.

This script reconstructs Ḣ_FF post-hoc from sim_log's L_com / v_com / r_com using
the exact `legacy_corrected` formula, plots it alongside the logged τ_w, and
classifies each saturation tick as "dock-aligned" (within ±200 ms of a dock
event) or "swing-distributed".

Usage:
  PYTHONPATH=. python3 scripts/diag_hdot_feedforward.py [subdir]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_subdir = sys.argv[1] if len(sys.argv) > 1 else (
    'diag_cooperative_arms_legacy_pid_numerical_Kt36.3_Kw355.4_Tw2')
LOG = os.path.join(_root, 'results', _subdir, 'sim_log.json')
OUT = os.path.join(_root, 'results', _subdir, 'hdot_feedforward.png')

# Tolerance for "at the saturation clip" detection.
SAT_TOL = 1e-6
# Window around dock events to call a saturation "dock-aligned".
DOCK_WINDOW = 0.200  # seconds — ~20 dt_qp samples either side
DT_QP = 0.01


def main():
    sl = json.load(open(LOG))
    t = np.array(sl['t'])
    L_com = np.array(sl['L_com'])
    v_com = np.array(sl['v_com'])
    r_com = np.array(sl['r_com'])
    tau_w = np.array(sl['tau_w'])
    ph = np.array(sl['phase'], dtype=object)
    dock_t = [e['t'] for e in sl['dock_events']]

    # Robot mass — pulled from a hardcoded canonical value matching
    # ROBOT_TOTAL_MASS in robot_interface (71 kg at 1% mass ratio).
    m = 71.0

    # Reconstruct Ḣ_FF using the EXACT legacy_corrected formula:
    #   L_dot_est  = (L_com(k) − L_com(k−1)) / dt
    #   orbital    = r_com(k) × m · (v_com(k) − v_com(k−1)) / dt
    #   Ḣ_FF       = L_dot_est + orbital
    L_dot = np.zeros_like(L_com)
    dv = np.zeros_like(v_com)
    L_dot[1:] = (L_com[1:] - L_com[:-1]) / DT_QP
    dv[1:] = (v_com[1:] - v_com[:-1]) / DT_QP
    orbital = np.cross(r_com, m * dv)
    H_dot_ff = L_dot + orbital  # the AOCS feedforward, reconstructed

    H_dot_ff_norm = np.linalg.norm(H_dot_ff, axis=1)
    tau_w_norm = np.linalg.norm(tau_w, axis=1)
    tau_w_axis_max = np.abs(tau_w).max(axis=1)

    # Per-axis saturation detection (the runner used τ_w_max=2 Nm here).
    tau_clip = float(np.abs(tau_w).max())
    print(f'=== AOCS Ḣ feedforward source diagnostic ===')
    print(f'  subdir: {_subdir}')
    print(f'  τ_w peak per-axis: {tau_clip:.3f} Nm — interpreting as clip')
    sat_mask = (np.abs(tau_w) >= tau_clip - SAT_TOL).any(axis=1)
    n_sat = int(sat_mask.sum())
    print(f'  τ_w saturation: {n_sat}/{len(t)} ticks ({100*n_sat/len(t):.1f}%, '
          f'{n_sat * DT_QP:.2f} s)')
    print()

    # Classify saturation events by alignment with dock transitions.
    sat_t = t[sat_mask]
    if len(sat_t) > 0 and dock_t:
        dist_to_dock = np.array([min(abs(st - dt) for dt in dock_t)
                                 for st in sat_t])
        dock_aligned = (dist_to_dock <= DOCK_WINDOW).sum()
        swing_dist = (dist_to_dock > DOCK_WINDOW).sum()
        print(f'  Classification of {len(sat_t)} saturation ticks '
              f'(window: ±{DOCK_WINDOW*1000:.0f} ms around docks):')
        print(f'    dock-aligned (≤200 ms from a dock): {dock_aligned} '
              f'({100*dock_aligned/len(sat_t):.0f}%)')
        print(f'    swing-distributed (>200 ms):        {swing_dist} '
              f'({100*swing_dist/len(sat_t):.0f}%)')
    elif len(sat_t) == 0:
        print('  No saturation events to classify.')
    print()

    # Spike-shape analysis: cluster contiguous saturation ticks into events,
    # report median event duration.
    if n_sat > 0:
        # Find contiguous runs of saturation.
        sat_diff = np.diff(np.concatenate([[False], sat_mask, [False]]).astype(int))
        starts = np.where(sat_diff == 1)[0]
        ends = np.where(sat_diff == -1)[0]
        durations_ms = (ends - starts) * DT_QP * 1000
        print(f'  Saturation event durations (n={len(durations_ms)} events):')
        print(f'    min={durations_ms.min():.0f} ms, '
              f'median={np.median(durations_ms):.0f} ms, '
              f'max={durations_ms.max():.0f} ms')
        print(f'    events ≤30 ms (narrow spikes): '
              f'{int((durations_ms <= 30).sum())} / {len(durations_ms)}')
        print(f'    events >100 ms (broad sustained): '
              f'{int((durations_ms > 100).sum())} / {len(durations_ms)}')

    # Ḣ_FF spike alignment with docks
    print()
    H_dot_p95 = np.percentile(H_dot_ff_norm, 95)
    H_dot_peak_ticks = np.where(H_dot_ff_norm > 3 * H_dot_p95)[0]
    print(f'  ‖Ḣ_FF‖ stats: median {np.median(H_dot_ff_norm):.3f} Nm, '
          f'p95 {H_dot_p95:.3f} Nm, peak {H_dot_ff_norm.max():.3f} Nm')
    if len(H_dot_peak_ticks):
        H_peak_t = t[H_dot_peak_ticks]
        dock_aligned_H = sum(1 for tt in H_peak_t
                             if any(abs(tt - dt) <= DOCK_WINDOW for dt in dock_t))
        print(f'  ‖Ḣ_FF‖ peaks > 3×p95: {len(H_peak_t)} ticks; '
              f'{dock_aligned_H} dock-aligned ({100*dock_aligned_H/len(H_peak_t):.0f}%)')

    # --- figure ---
    fig, ax = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

    # Top: Ḣ feedforward norm
    ax[0].plot(t, H_dot_ff_norm, 'C0', lw=0.9, label='‖Ḣ_FF‖  (FD on L_com, v_com)')
    ax[0].set_ylabel('‖Ḣ_FF‖ [Nm]')
    ax[0].set_title('AOCS Ḣ feedforward (reconstructed) — '
                    f'{os.path.basename(_subdir)}')
    ax[0].set_yscale('symlog', linthresh=0.5)
    ax[0].legend(fontsize=8, loc='upper right')

    # Middle: τ_w_norm + saturation marks
    ax[1].plot(t, tau_w_axis_max, 'C3', lw=0.9, label='max axis |τ_w|')
    ax[1].axhline(tau_clip, color='r', ls='--', lw=0.7,
                  label=f'τ_w_max = {tau_clip:.2f} Nm')
    # Saturation ticks
    if n_sat > 0:
        ax[1].scatter(t[sat_mask], np.full(n_sat, tau_clip * 1.02),
                      s=4, color='red', marker='|',
                      label=f'saturated ({n_sat} ticks, {100*n_sat/len(t):.1f}%)')
    ax[1].set_ylabel('τ_w per-axis [Nm]')
    ax[1].legend(fontsize=8, loc='upper right')

    # Bottom: phase shading reference + dock markers context
    ax[2].plot(t, H_dot_ff_norm, 'C0', lw=0.8, alpha=0.6, label='‖Ḣ_FF‖')
    ax[2].plot(t, tau_w_axis_max, 'C3', lw=0.8, alpha=0.6, label='max |τ_w|')
    ax[2].set_ylabel('overlay [Nm]')
    ax[2].set_xlabel('t [s]')
    ax[2].set_yscale('symlog', linthresh=0.5)
    ax[2].legend(fontsize=8, loc='upper right')

    # Dock event markers across all panels
    for a in ax:
        for dt in dock_t:
            a.axvline(dt, color='k', ls=':', lw=0.7, alpha=0.55)
        ss = ph == 'SS'
        ymin, ymax = a.get_ylim()
        a.fill_between(t, ymin, ymax, where=ss, color='orange', alpha=0.05)
        a.grid(alpha=0.3)

    # Annotation legend for context
    ax[0].text(0.01, 0.95,
               'orange = SS swing phase\n'
               'dotted vertical = dock event',
               transform=ax[0].transAxes, fontsize=7,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    fig.tight_layout()
    fig.savefig(OUT, dpi=120)
    plt.close(fig)
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
