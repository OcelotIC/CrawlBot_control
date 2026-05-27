"""Group B diagnostic: momentum / AOCS health.

Does the traversal keep the free-floating platform managed, or is
"docking" hiding momentum saturation / attitude drift? Checks the
spec thresholds (HANDOFF M0.3):
  hw_saturation_ratio_peak  < 1.0   (|hw|/hw_max)
  hw_saturation_ratio_rms   < 0.7
  platform_rotation_total   < 5.0 deg
  platform_omega_peak       < 2.0 deg/s
  tau_w_peak_ratio          < 1.0   (|tau_w|/tau_w_max)

Curves: hw (3 axes) vs +-box; |hw|/hw_max; structure attitude drift
from start [deg]; |omega_s| [deg/s]; tau_w (3 axes) vs +-box; |L_com|.
Phase shading (DS/SS) + dock lines.

Usage:
  PYTHONPATH=. MUJOCO_GL=osmesa python3 scripts/diag_momentum_aocs.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(_root, 'results', 'diag_cooperative_arms', 'sim_log.json')
OUT = os.path.join(_root, 'results', 'diag_cooperative_arms', 'momentum_aocs.png')

HW_MAX = 5.0      # Nms (spec/config)
TAUW_MAX = 5.0    # Nm


def quat_angle_deg(qa, qb):
    d = abs(float(np.dot(qa, qb)))
    return np.degrees(2.0 * np.arccos(min(d, 1.0)))


def main():
    sl = json.load(open(LOG))
    t = np.array(sl['t'])
    ph = np.array(sl['phase'], dtype=object)
    sidx = np.array(sl['step_idx'])
    hw = np.array(sl['hw'])                       # (N,3) structure wheel momentum
    tauw = np.array(sl['tau_w'])                  # (N,3) wheel torque cmd
    oms = np.array(sl['omega_s'])                 # (N,3) structure ang vel (rad/s)
    Lc = np.array(sl['L_com'])                    # (N,3) robot centroidal ang mom
    sq = np.array(sl['struct_quat'])              # (N,4) structure attitude wxyz
    dock_t = [e['t'] for e in sl['dock_events']]

    hw_n = np.linalg.norm(hw, axis=1)
    tauw_n = np.linalg.norm(tauw, axis=1)
    oms_deg = np.degrees(np.linalg.norm(oms, axis=1))
    Lc_n = np.linalg.norm(Lc, axis=1)
    drift_deg = np.array([quat_angle_deg(sq[i], sq[0]) for i in range(len(sq))])

    # --- metrics + thresholds ---
    metrics = {
        'hw_sat_peak':       (hw_n.max() / HW_MAX, 1.0),
        'hw_sat_rms':        (np.sqrt(np.mean((hw_n / HW_MAX) ** 2)), 0.7),
        'platform_rot_total_deg': (drift_deg.max(), 5.0),
        'platform_omega_peak_deg_s': (oms_deg.max(), 2.0),
        'tau_w_peak_ratio':  (tauw_n.max() / TAUW_MAX, 1.0),
    }

    print('=== Group B: momentum / AOCS health ===')
    print(f"{'metric':>28} {'value':>10} {'thresh':>8} {'PASS?':>6}")
    for k, (v, th) in metrics.items():
        print(f"{k:>28} {v:>10.3f} {th:>8.2f} {str(v < th):>6}")
    print(f"\n  peak |hw| = {hw_n.max():.2f} Nms (box +-{HW_MAX}); "
          f"peak |tau_w| = {tauw_n.max():.2f} Nm (box +-{TAUW_MAX})")
    print(f"  |L_com| peak = {Lc_n.max():.3f}  mean = {Lc_n.mean():.3f} Nms")
    print(f"  platform net rotation start->end = {drift_deg[-1]:.2f} deg (peak {drift_deg.max():.2f})")

    # --- figure ---
    fig, ax = plt.subplots(5, 1, figsize=(12, 13), sharex=True)
    ss = ph == 'SS'
    def shade(a):
        a.fill_between(t, *a.get_ylim(), where=ss, color='orange', alpha=0.06)
        for dt in dock_t:
            a.axvline(dt, color='k', ls=':', lw=0.6, alpha=0.5)

    for i, c in enumerate('xyz'):
        ax[0].plot(t, hw[:, i], lw=0.9, label=f'hw_{c}')
    ax[0].axhline(HW_MAX, color='r', ls='--', lw=0.8); ax[0].axhline(-HW_MAX, color='r', ls='--', lw=0.8)
    ax[0].set_ylabel('hw [Nms]'); ax[0].legend(fontsize=7, ncol=3, loc='upper right')
    ax[0].set_title(f'Structure RWA momentum (peak sat {metrics["hw_sat_peak"][0]*100:.0f}% of box)')

    ax[1].plot(t, drift_deg, 'C4', lw=1.1)
    ax[1].axhline(5.0, color='r', ls='--', lw=0.8, label='5 deg spec')
    ax[1].set_ylabel('platform attitude\ndrift from t0 [deg]'); ax[1].legend(fontsize=7)

    ax[2].plot(t, oms_deg, 'C0', lw=1.0)
    ax[2].axhline(2.0, color='r', ls='--', lw=0.8, label='2 deg/s spec')
    ax[2].set_ylabel('|omega_s| [deg/s]'); ax[2].legend(fontsize=7)

    for i, c in enumerate('xyz'):
        ax[3].plot(t, tauw[:, i], lw=0.9, label=f'tau_w_{c}')
    ax[3].axhline(TAUW_MAX, color='r', ls='--', lw=0.8); ax[3].axhline(-TAUW_MAX, color='r', ls='--', lw=0.8)
    ax[3].set_ylabel('tau_w [Nm]'); ax[3].legend(fontsize=7, ncol=3, loc='upper right')

    ax[4].plot(t, Lc_n, 'C2', lw=1.0)
    ax[4].set_ylabel('|L_com| [Nms]'); ax[4].set_xlabel('t [s]')

    for a in ax:
        shade(a); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT, dpi=120); plt.close(fig)
    print('\nwrote', OUT)


if __name__ == '__main__':
    main()
