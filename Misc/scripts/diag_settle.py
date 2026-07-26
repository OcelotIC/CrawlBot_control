"""Diagnostic: post-traversal settle — measure irreversible per-traversal drift.

After the last dock, both arms are welded and the robot is kinematically
stationary. By conservation about the structure CoM (initial L_total/s = 0):

    L_robot/s(t) + I_s · ω_s(t) + h_w(t) = 0

If the robot is truly at rest in the inertial frame, L_robot/s = 0, so
I_s · ω_s + h_w = 0; the AOCS desat term drives h_w → 0, which by
conservation forces ω_s → 0. The structure ATTITUDE (∫ω_s dt) accumulates
the drift that's irreversible — that's the per-traversal net rotation
that scales linearly with N traversals.

This script separates:
  - the **transient** rotation (peaks during motion, reversed by AOCS)
  - the **net** rotation (locked in after settle, accumulates over N steps)

Pre-req: a canonical run with sufficiently long --settle_seconds
(default 20s is one PD time constant; 120s gives ~3-4 time constants).

Usage:
  PYTHONPATH=. python3 Misc/scripts/diag_settle.py [subdir]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_subdir = sys.argv[1] if len(sys.argv) > 1 else 'diag_cooperative_arms'
LOG = os.path.join(_root, 'results', _subdir, 'sim_log.json')
OUT = os.path.join(_root, 'results', _subdir, 'settle.png')


def quat_angle_deg(qa, qb):
    """Angle between two unit quaternions (wxyz) [deg]."""
    d = abs(float(np.dot(qa, qb)))
    return float(np.degrees(2.0 * np.arccos(min(d, 1.0))))


def main():
    sl = json.load(open(LOG))
    t = np.array(sl['t'])
    ph = np.array(sl['phase'], dtype=object)
    omega_s = np.array(sl['omega_s'])      # (N, 3) rad/s, structure body frame
    hw = np.array(sl['hw'])                # (N, 3) Nms
    sq = np.array(sl['struct_quat'])       # (N, 4) wxyz structure attitude
    dock_t = [e['t'] for e in sl['dock_events']]
    if not dock_t:
        print('  no dock events — cannot identify settle window')
        return
    t_last_dock = max(dock_t)

    # Settle window = [t_last_dock, t_end]
    settle_mask = t >= t_last_dock
    t_set = t[settle_mask]
    omega_set = omega_s[settle_mask]
    hw_set = hw[settle_mask]
    sq_set = sq[settle_mask]

    print('=== Post-traversal settle ===')
    print(f'  last dock at t = {t_last_dock:.2f}s')
    print(f'  settle window: t = [{t_set[0]:.2f}, {t_set[-1]:.2f}]s '
          f'= {t_set[-1] - t_set[0]:.2f}s')

    # Run-wide max (the transient peak)
    om_peak_run = np.linalg.norm(omega_s, axis=1).max()
    hw_peak_run = np.linalg.norm(hw, axis=1).max()
    # Attitude vs t=0 (the irreversible accumulated rotation)
    att_run = np.array([quat_angle_deg(sq[i], sq[0]) for i in range(len(sq))])
    att_peak_run = att_run.max()

    # Settle-window stats
    om_norm_set = np.linalg.norm(omega_set, axis=1)
    hw_norm_set = np.linalg.norm(hw_set, axis=1)
    att_set = np.array([quat_angle_deg(sq_set[i], sq[0])
                        for i in range(len(sq_set))])

    print()
    print('  TRANSIENT (whole run, peak):')
    print(f'    ‖ω_s‖ peak: {om_peak_run * 1000:.2f} mrad/s')
    print(f'    ‖h_w‖ peak: {hw_peak_run:.3f} Nms')
    print(f'    |attitude| peak: {att_peak_run:.3f}° (vs t=0)')
    print()
    print('  SETTLE WINDOW evolution:')
    print(f'    ‖ω_s‖ at settle start: {om_norm_set[0] * 1000:.2f} mrad/s')
    print(f'    ‖ω_s‖ at settle end:   {om_norm_set[-1] * 1000:.2f} mrad/s '
          f'({"decayed" if om_norm_set[-1] < om_norm_set[0] else "GREW"} '
          f'by {abs(om_norm_set[-1] - om_norm_set[0]) * 1000:.2f} mrad/s)')
    print(f'    ‖h_w‖ at settle start: {hw_norm_set[0]:.3f} Nms')
    print(f'    ‖h_w‖ at settle end:   {hw_norm_set[-1]:.3f} Nms '
          f'({"decayed" if hw_norm_set[-1] < hw_norm_set[0] else "GREW"})')
    print(f'    |attitude| at settle start: {att_set[0]:.3f}°')
    print(f'    |attitude| at settle end:   {att_set[-1]:.3f}° '
          f'(net per-traversal drift = THIS)')
    print()
    # The key paper-relevant numbers
    reversible = att_peak_run - att_set[-1]
    net = att_set[-1]
    pct_irreversible = 100 * net / max(att_peak_run, 1e-9)
    print(f'  → Reversible (AOCS-recovered) attitude rotation: '
          f'{reversible:.3f}°')
    print(f'  → IRREVERSIBLE net per-traversal rotation:       '
          f'{net:.3f}°  ({pct_irreversible:.0f}% of transient peak)')
    print(f'    Extrapolated to N=1000 traversals: {net * 1000:.1f}°')
    print(f'    (>5° spec budget if >{int(5/max(net, 1e-9))} traversals)')

    # --- figure ---
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for dt in dock_t:
        for a in ax:
            a.axvline(dt, color='k', ls=':', lw=0.6, alpha=0.4)
    ss = ph == 'SS'

    def shade(a):
        ymin, ymax = a.get_ylim()
        a.fill_between(t, ymin, ymax, where=ss, color='orange', alpha=0.06)
        a.axvspan(t_last_dock, t[-1], color='lightgreen', alpha=0.10,
                  label='_settle')
        a.grid(alpha=0.3)

    om_norm = np.linalg.norm(omega_s, axis=1)
    hw_norm = np.linalg.norm(hw, axis=1)

    ax[0].plot(t, om_norm * 1000, 'C0', lw=1.0)
    ax[0].set_ylabel('‖ω_s‖ [mrad/s]')
    ax[0].set_title(f'Post-traversal settle — last dock at t={t_last_dock:.1f}s '
                    f'(settle window in green)')

    ax[1].plot(t, hw_norm, 'C2', lw=1.0)
    ax[1].axhline(5.0, color='r', ls='--', lw=0.7, label='hw_max=5 Nms')
    ax[1].set_ylabel('‖h_w‖ [Nms]')
    ax[1].legend(fontsize=8)

    att = np.array([quat_angle_deg(sq[i], sq[0]) for i in range(len(sq))])
    ax[2].plot(t, att, 'C3', lw=1.0, label='|attitude| vs t=0')
    ax[2].axhline(att_set[-1], color='gray', ls='--', lw=0.7,
                  label=f'net = {att_set[-1]:.3f}°')
    ax[2].set_ylabel('|attitude| [deg]')
    ax[2].set_xlabel('t [s]')
    ax[2].legend(fontsize=8)

    for a in ax:
        shade(a)

    fig.tight_layout()
    fig.savefig(OUT, dpi=120)
    plt.close(fig)
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
