"""Group C diagnostic: CoM->torso cascade band-aid footprint.

The live cascade is world-frame delta(q_current) + F-SAT rate clamp
(not the principled mapping). This checks its footprint:
  1. CoM-z standoff hold across ALL steps (target -0.35; only step 0 was
     verified before). Range = how flat the crawl height stays.
  2. Torso position tracking: |p_torso - p_torso_ref| (does the body
     follow the mapped+clamped reference, or lag/drag?).
  3. F-SAT footprint: per-tick |d r_b_ref| (rate the reference is allowed
     to move) + aggregate clip rate from sat_stats.txt.

Usage:
  PYTHONPATH=. MUJOCO_GL=osmesa python3 scripts/diag_cascade_health.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(_root, 'results', 'diag_cooperative_arms')
LOG = os.path.join(DIR, 'sim_log.json')
OUT = os.path.join(DIR, 'cascade_health.png')
Z_STANDOFF = -0.35


def main():
    sl = json.load(open(LOG))
    t = np.array(sl['t']); ph = np.array(sl['phase'], dtype=object)
    sidx = np.array(sl['step_idx'])
    rc = np.array(sl['r_com']); pt = np.array(sl['p_torso'])
    ptr = np.array(sl['p_torso_ref'])
    dock_t = [e['t'] for e in sl['dock_events']]

    com_z = rc[:, 2]
    torso_z = pt[:, 2]
    track_err = np.linalg.norm(pt - ptr, axis=1) * 1000.0     # mm, body vs mapped ref
    drbref = np.r_[0.0, np.linalg.norm(np.diff(ptr, axis=0), axis=1)] * 1000.0  # mm/tick

    print('=== Group C: cascade band-aid footprint ===')
    print(f'  CoM-z (target {Z_STANDOFF}): overall range '
          f'[{com_z.min():.3f}, {com_z.max():.3f}] = {(com_z.max()-com_z.min())*1000:.0f}mm')
    print(f'  {"step":>4} {"CoM-z[min,max]":>20} {"dev from -0.35[mm]":>18} '
          f'{"torso-trk peak[mm]":>18}')
    for s in range(5):
        m = (sidx == s) & (ph == 'SS')
        if not m.any():
            continue
        cz = com_z[m]
        print(f'  {s:>4} [{cz.min():+.3f},{cz.max():+.3f}]      '
              f'{np.max(np.abs(cz - Z_STANDOFF))*1000:>14.0f}     '
              f'{track_err[m].max():>14.1f}')
    print()
    print(f'  torso position tracking |p_torso - p_torso_ref|: '
          f'peak {track_err.max():.1f}mm, mean {track_err.mean():.1f}mm')
    print(f'  F-SAT r_b_ref increment |d r_b_ref|: peak {drbref.max():.2f}mm/tick, '
          f'mean {drbref.mean():.3f}mm/tick')
    sat_path = os.path.join(DIR, 'sat_stats.txt')
    if os.path.exists(sat_path):
        print('  --- sat_stats.txt ---')
        for ln in open(sat_path).read().splitlines():
            if ln.strip():
                print('   ', ln.strip())

    # --- figure ---
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    ss = ph == 'SS'
    def deco(a):
        a.fill_between(t, *a.get_ylim(), where=ss, color='orange', alpha=0.06)
        for dt in dock_t:
            a.axvline(dt, color='k', ls=':', lw=0.6, alpha=0.5)
        a.grid(alpha=0.3)

    ax[0].plot(t, com_z, 'C0', lw=1.1, label='CoM-z actual')
    ax[0].plot(t, torso_z, 'C1', lw=0.9, alpha=0.7, label='torso-z')
    ax[0].axhline(Z_STANDOFF, color='green', ls='--', lw=0.9, label='standoff -0.35')
    ax[0].axhline(0.025, color='gray', ls=':', lw=0.8, label='beam top +0.025')
    ax[0].set_ylabel('z [m]'); ax[0].legend(fontsize=7, ncol=2)
    ax[0].set_title('CoM-z standoff hold (crawl height) + torso-z vs beam')

    ax[1].plot(t, track_err, 'C3', lw=1.0)
    ax[1].set_ylabel('|p_torso - p_torso_ref|\n[mm]')
    ax[1].set_title('Torso position tracking (body vs mapped+clamped reference)')

    ax[2].plot(t, drbref, 'C2', lw=0.8)
    ax[2].set_ylabel('|d r_b_ref| [mm/tick]')
    ax[2].set_title('F-SAT footprint: per-tick mapped-reference increment')
    ax[2].set_xlabel('t [s]')
    for a in ax:
        deco(a)
    fig.tight_layout(); fig.savefig(OUT, dpi=120); plt.close(fig)
    print('\nwrote', OUT)


if __name__ == '__main__':
    main()
