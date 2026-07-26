"""Visualize the step-0 failure under the constant-CoM-z standoff.

Plots, over step-0 SS:
  - CoM-z actual vs reference (shows the body overshooting the -0.35
    standoff up to -0.21)
  - swing gripper distance to anchor + EE error (near-dock at t~3.6s
    then regrows as the body drifts up)
  - |lambda_qp| contact-wrench norm (force spike)
  - torso clearance to the beam

Usage:
  PYTHONPATH=. MUJOCO_GL=osmesa python3 Misc/scripts/diag_step0_standoff_fail.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG = os.path.join(_root, 'Misc', 'runs', 'diag_cooperative_arms', 'sim_log.json')
OUT = os.path.join(_root, 'Misc', 'runs', 'diag_cooperative_arms',
                   'step0_standoff_fail.png')
Z_TARGET = -0.35
BEAM_HALF_Z = 0.025


def beam_clear_mm(p):
    h = np.array([2.4, 0.4, 0.025])
    d = np.abs(p) - h
    out = np.linalg.norm(np.maximum(d, 0))
    ins = min(np.max(d), 0.0)
    return (out + ins) * 1000.0


def main():
    sl = json.load(open(LOG))
    t = np.array(sl['t']); sidx = np.array(sl['step_idx'])
    ph = np.array(sl['phase'], dtype=object)
    rc = np.array(sl['r_com']); rcr = np.array(sl['r_com_ref'])
    pt = np.array(sl['p_torso'])
    lqn = np.array(sl['lambda_qp_norm'])
    dgs = np.array(sl['d_grip_swing']); eep = np.array(sl['e_ee_pos'])

    m = (sidx == 0) & (ph == 'SS')
    tt = t[m] - t[m][0]
    clr = np.array([beam_clear_mm(p) for p in pt[m]])

    fig, ax = plt.subplots(4, 1, figsize=(11, 12), sharex=True)

    ax[0].plot(tt, rc[m][:, 2], 'C0', lw=1.4, label='CoM-z actual')
    ax[0].plot(tt, rcr[m][:, 2], 'C0--', lw=1.0, label='CoM-z reference')
    ax[0].axhline(Z_TARGET, color='green', ls=':', lw=1.0,
                  label=f'standoff target {Z_TARGET}')
    ax[0].annotate('body OVERSHOOTS up to -0.21\n(target -0.35)',
                   xy=(tt[-1], rc[m][-1, 2]), xytext=(tt[-1]*0.5, -0.27),
                   arrowprops=dict(arrowstyle='->', color='red'), color='red')
    ax[0].set_ylabel('CoM-z [m]'); ax[0].legend(fontsize=8, loc='lower right')
    ax[0].grid(alpha=0.3); ax[0].set_title('Step-0 failure under constant CoM-z standoff (-0.35)')

    ax[1].plot(tt, dgs[m]*1000, 'C1', lw=1.4, label='gripper→anchor d [mm]')
    ax[1].plot(tt, eep[m]*1000, 'C3', lw=1.0, label='EE error [mm]')
    ax[1].axhline(5.0, color='green', ls=':', lw=1.0, label='5mm dock gate')
    kmin = np.argmin(dgs[m])
    ax[1].annotate(f'nearly docks ({dgs[m][kmin]*1000:.1f}mm)\nthen regrows',
                   xy=(tt[kmin], dgs[m][kmin]*1000),
                   xytext=(tt[kmin]+1.5, 200),
                   arrowprops=dict(arrowstyle='->', color='red'), color='red')
    ax[1].set_ylabel('distance [mm]'); ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3); ax[1].set_yscale('log')

    ax[2].plot(tt, lqn[m], 'C2', lw=1.2, label='|lambda_qp| (contact wrench)')
    ax[2].axhline(25.0, color='orange', ls=':', lw=1.0, label='preplanner f_max=25')
    ax[2].set_ylabel('|lambda_qp| [N]'); ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)

    ax[3].plot(tt, clr, 'C4', lw=1.2, label='torso→beam clearance [mm]')
    ax[3].axhline(0.0, color='red', ls=':', lw=1.0)
    ax[3].set_ylabel('clearance [mm]'); ax[3].set_xlabel('time since SS entry [s]')
    ax[3].legend(fontsize=8); ax[3].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT, dpi=120)
    print('wrote', OUT)
    print(f'peak |lambda_qp| = {lqn[m].max():.1f} N at t+{tt[lqn[m].argmax()]:.2f}s')
    print(f'CoM-z: start {rc[m][0,2]:+.3f} -> end {rc[m][-1,2]:+.3f} (ref {Z_TARGET}); '
          f'overshoot {abs(rc[m][-1,2]-Z_TARGET)*1000:.0f}mm above target')
    print(f'gripper min approach = {dgs[m].min()*1000:.1f}mm (gate 5mm) at t+{tt[np.argmin(dgs[m])]:.2f}s')


if __name__ == '__main__':
    main()
