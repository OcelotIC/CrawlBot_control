"""Dock-quality diagnostic: position + orientation + RESIDUAL VELOCITY.

The dock gate is purely positional (d < weld_radius AND ori < thresh) —
no velocity term. A gripper at 1mm still moving fast welds anyway, and
the weld's inelastic impact projection then injects momentum (candidate
source of recoil + contact-force spikes). So a *clean* dock needs small
position error AND small relative velocity.

For each step this plots, over the SS approach:
  - d_grip(t)  [mm, log]  with 5mm (current) and 1mm (proposed) gate lines
  - |v_ee|(t)  [mm/s]     residual EE linear velocity (structure frame =
                          relative to the static anchor)
  - ori(t)     [deg]      EE orientation error  + 5deg/1deg lines
  - |omega_ee|(t) [deg/s] residual EE angular velocity
and annotates the value AT the dock instant.

Prints a summary table: at dock, would each step pass the tightened
1mm/1deg gate, and what is the residual |v_ee| / |omega_ee|.

Usage:
  PYTHONPATH=. MUJOCO_GL=osmesa python3 Misc/scripts/diag_dock_quality.py
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
OUT = os.path.join(_root, 'Misc', 'runs', 'diag_cooperative_arms', 'dock_quality.png')


def main():
    sl = json.load(open(LOG))
    t = np.array(sl['t']); sidx = np.array(sl['step_idx'])
    ph = np.array(sl['phase'], dtype=object)
    sw = np.array(sl['swing_arm'], dtype=object)
    dgs = np.array(sl['d_grip_swing'])
    eeo = np.array(sl['e_ee_ori'])          # EE ori err vs ref (~anchor at dock), deg
    vea = np.array(sl['v_ee_a']); veb = np.array(sl['v_ee_b'])
    oea = np.array(sl['omega_ee_a']); oeb = np.array(sl['omega_ee_b'])
    dock_t = {e['step']: e['t'] for e in sl['dock_events']}

    steps = [s for s in range(6) if ((sidx == s) & (ph == 'SS')).any()]
    fig, axes = plt.subplots(len(steps), 2, figsize=(13, 2.4 * len(steps)))
    if len(steps) == 1:
        axes = axes[None, :]

    rows = []
    for r, s in enumerate(steps):
        m = (sidx == s) & (ph == 'SS')
        tt = t[m] - t[m][0]
        arm = sw[m][0]
        vee = (vea if arm == 'a' else veb)[m]
        oee = (oea if arm == 'a' else oeb)[m]
        v_norm = np.linalg.norm(vee, axis=1) * 1000.0          # mm/s
        w_norm = np.degrees(np.linalg.norm(oee, axis=1))       # deg/s
        d_mm = dgs[m] * 1000.0
        ori = eeo[m]

        # dock instant index
        kd = None
        if s in dock_t:
            kd = int(np.argmin(np.abs(t[m] - dock_t[s])))

        # --- position panel ---
        ax = axes[r, 0]; ax2 = ax.twinx()
        ax.plot(tt, np.maximum(d_mm, 1e-3), 'C0', lw=1.2, label='d_grip [mm]')
        ax.axhline(5.0, color='orange', ls='--', lw=0.8); ax.axhline(1.0, color='red', ls=':', lw=0.8)
        ax2.plot(tt, v_norm, 'C2', lw=1.0, alpha=0.8, label='|v_ee| [mm/s]')
        ax.set_yscale('log'); ax.set_ylabel(f's{s} d_grip[mm]\n(5/1 lines)', color='C0')
        ax2.set_ylabel('|v_ee| [mm/s]', color='C2')
        if kd is not None:
            ax.axvline(tt[kd], color='k', ls=':', lw=0.8)
            ax.annotate(f'dock: d={d_mm[kd]:.2f}mm\n|v|={v_norm[kd]:.1f}mm/s',
                        xy=(tt[kd], max(d_mm[kd], 1e-2)), xytext=(tt[kd]*0.3+0.2, 50),
                        fontsize=7, color='k',
                        arrowprops=dict(arrowstyle='->', color='k', lw=0.6))
        ax.grid(alpha=0.3, which='both')
        if r == 0: ax.set_title('Position approach: d_grip + residual |v_ee|')

        # --- orientation panel ---
        ax = axes[r, 1]; ax2 = ax.twinx()
        ax.plot(tt, np.maximum(ori, 1e-3), 'C1', lw=1.2, label='ori [deg]')
        ax.axhline(5.0, color='orange', ls='--', lw=0.8); ax.axhline(1.0, color='red', ls=':', lw=0.8)
        ax2.plot(tt, w_norm, 'C3', lw=1.0, alpha=0.8, label='|w_ee| [deg/s]')
        ax.set_yscale('log'); ax.set_ylabel(f's{s} ori[deg]\n(5/1 lines)', color='C1')
        ax2.set_ylabel('|w_ee| [deg/s]', color='C3')
        if kd is not None:
            ax.axvline(tt[kd], color='k', ls=':', lw=0.8)
            ax.annotate(f'ori={ori[kd]:.2f}deg\n|w|={w_norm[kd]:.1f}deg/s',
                        xy=(tt[kd], max(ori[kd], 1e-2)), xytext=(tt[kd]*0.3+0.2, 20),
                        fontsize=7, color='k',
                        arrowprops=dict(arrowstyle='->', color='k', lw=0.6))
        ax.grid(alpha=0.3, which='both')
        if r == 0: ax.set_title('Orientation approach: ori + residual |w_ee|')
        axes[r, 1].set_xlabel('t since SS entry [s]') if r == len(steps)-1 else None
        axes[r, 0].set_xlabel('t since SS entry [s]') if r == len(steps)-1 else None

        if kd is not None:
            rows.append((s, arm, d_mm[kd], ori[kd], v_norm[kd], w_norm[kd],
                         d_mm.min(), (d_mm < 1.0).any()))

    fig.tight_layout(); fig.savefig(OUT, dpi=120); plt.close(fig)
    print('wrote', OUT)
    print()
    print('=== dock-quality summary (values AT the dock instant) ===')
    print(f"{'step':>4} {'arm':>3} {'d[mm]':>7} {'ori[deg]':>8} {'|v_ee|[mm/s]':>12} "
          f"{'|w_ee|[deg/s]':>13} {'d_min[mm]':>9} {'ever<1mm?':>9} {'pass 1mm/1deg?':>14}")
    for (s, arm, d, ori, v, w, dmin, ever1) in rows:
        passes = (d < 1.0) and (ori < 1.0)
        print(f"{s:>4} {arm:>3} {d:>7.2f} {ori:>8.2f} {v:>12.1f} {w:>13.1f} "
              f"{dmin:>9.2f} {str(bool(ever1)):>9} {str(passes):>14}")


if __name__ == '__main__':
    main()
