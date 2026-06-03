"""Diagnostic: tracking errors for the swing EE and the torso.

For the canonical run (or any opt-in mode), plots:
  - Swing-EE: position error, orientation error, linear speed (actual vs
    reference), angular speed (actual vs reference). The "swing" arm changes
    each step (b/a/b/a/b); we pick the currently-swinging arm per tick.
  - Torso: same four traces.

Reference velocities are not logged; computed here by central finite
difference on p_ee_ref / q_ee_ref / p_torso_ref / q_torso_ref. Reference
angular speeds are extracted from the relative quaternion between consecutive
reference orientations:
    Δq = q_ref(t+1) * conj(q_ref(t))
    |ω_ref| ≈ 2·arccos(|Δq_w|) / dt

Single figure: 2 columns (EE | torso) × 4 rows (pos err / ori err / linear
speed / angular speed).

Usage:
  PYTHONPATH=. python3 scripts/diag_tracking.py [subdir]
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
    'diag_cooperative_arms_legacy_pid_numerical_Kt36.3_Kw355.4')
LOG = os.path.join(_root, 'results', _subdir, 'sim_log.json')
OUT = os.path.join(_root, 'results', _subdir, 'tracking.png')

DT = 0.01  # dt_qp


def _quat_norm_vec(q):
    """Force quaternion canonical sign (q_w >= 0). q is (N, 4) wxyz."""
    q = q.copy()
    q[q[:, 0] < 0] *= -1
    return q


def _quat_mul(qa, qb):
    """Hamilton product, wxyz convention. qa, qb: (N, 4). Returns (N, 4)."""
    wa, xa, ya, za = qa[:, 0], qa[:, 1], qa[:, 2], qa[:, 3]
    wb, xb, yb, zb = qb[:, 0], qb[:, 1], qb[:, 2], qb[:, 3]
    return np.stack([
        wa * wb - xa * xb - ya * yb - za * zb,
        wa * xb + xa * wb + ya * zb - za * yb,
        wa * yb - xa * zb + ya * wb + za * xb,
        wa * zb + xa * yb - ya * xb + za * wb,
    ], axis=1)


def _quat_conj(q):
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def _omega_from_quat_diff(q):
    """Reference angular speed magnitude via consecutive-quaternion diff.

    For unit quaternions q(t), q(t+dt) representing small rotations:
        Δq = q(t+1) * conj(q(t))
        ω ≈ 2 · vec(Δq) / dt  (small-angle)
        |ω| ≈ 2 · arccos(|Δq.w|) / dt  (geodesic distance)
    """
    q = _quat_norm_vec(q)
    dq = _quat_mul(q[1:], _quat_conj(q[:-1]))
    # Canonicalize each step so dot product with previous tick is positive.
    flip = dq[:, 0] < 0
    dq[flip] *= -1
    dq_w = np.clip(np.abs(dq[:, 0]), 0.0, 1.0)
    omega_mag = 2.0 * np.arccos(dq_w) / DT  # rad/s
    # Pad to match input length
    return np.concatenate([[0.0], omega_mag])


def _speed_from_pos_diff(p):
    """Reference linear speed by central difference of position. p: (N, 3)."""
    v = np.zeros(len(p))
    v[1:-1] = np.linalg.norm(p[2:] - p[:-2], axis=1) / (2 * DT)
    v[0] = v[1]
    v[-1] = v[-2]
    return v


def main():
    sl = json.load(open(LOG))
    t = np.array(sl['t'])
    ph = np.array(sl['phase'], dtype=object)
    swing_arm = np.array(sl['swing_arm'])
    dock_t = [e['t'] for e in sl['dock_events']]

    # EE-side
    # d_grip_swing = distance from the swing EE to its TARGET dock anchor
    # (the hotdock the EE is converging to). This is the "tracking" signal
    # the user wants: it shows the EE approaching the dock, decreasing
    # monotonically through the swing rather than the noisier reference-
    # trajectory error e_ee_pos.
    d_grip_swing = np.array(sl['d_grip_swing'])
    e_ee_ori = np.array(sl['e_ee_ori'])
    p_ee_ref = np.array(sl['p_ee_ref'])
    q_ee_ref = np.array(sl['q_ee_ref'])
    v_ee_a = np.array(sl['v_ee_a'])
    v_ee_b = np.array(sl['v_ee_b'])
    omega_ee_a = np.array(sl['omega_ee_a'])
    omega_ee_b = np.array(sl['omega_ee_b'])

    # Pick swinging arm's velocity at each tick.
    sw_a = (swing_arm == 'a')
    v_ee = np.where(sw_a[:, None], v_ee_a, v_ee_b)
    omega_ee = np.where(sw_a[:, None], omega_ee_a, omega_ee_b)
    v_ee_norm = np.linalg.norm(v_ee, axis=1) * 1000.0  # mm/s
    omega_ee_norm = np.linalg.norm(omega_ee, axis=1)   # rad/s

    v_ee_ref_norm = _speed_from_pos_diff(p_ee_ref) * 1000.0  # mm/s
    omega_ee_ref_norm = _omega_from_quat_diff(q_ee_ref)      # rad/s

    # Torso-side
    e_torso_pos = np.array(sl['e_torso_pos']) * 1000.0  # mm
    e_torso_ori = np.array(sl['e_torso_ori'])           # deg already
    p_torso_ref = np.array(sl['p_torso_ref'])
    q_torso_ref = np.array(sl['q_torso_ref'])
    v_torso = np.array(sl['v_torso'])
    omega_torso = np.array(sl['omega_torso'])
    v_torso_norm = np.linalg.norm(v_torso, axis=1) * 1000.0  # mm/s
    omega_torso_norm = np.linalg.norm(omega_torso, axis=1)   # rad/s

    v_torso_ref_norm = _speed_from_pos_diff(p_torso_ref) * 1000.0
    omega_torso_ref_norm = _omega_from_quat_diff(q_torso_ref)

    print('=== Tracking diagnostic ===')
    print(f'  subdir: {_subdir}')
    print(f'  duration: {t[-1]:.1f} s, {len(t)} ticks @ dt={DT}s')
    print(f'  dock events at: {[f"{x:.2f}" for x in dock_t]}\n')

    # SS-only peaks (when tracking actually matters)
    ss = ph == 'SS'
    print(f'  SS-phase peak tracking errors:')
    print(f'    swing EE→target  : peak {d_grip_swing[ss].max() * 1000:.2f} mm, '
          f'median {np.median(d_grip_swing[ss]) * 1000:.2f} mm, '
          f'at-dock {d_grip_swing[ss].min() * 1000:.2f} mm (min)')
    print(f'    swing EE  ori (vs ref): peak {e_ee_ori[ss].max():.3f} deg, '
          f'median {np.median(e_ee_ori[ss]):.3f} deg')
    print(f'    torso     pos (vs ref): peak {e_torso_pos[ss].max():.2f} mm, '
          f'median {np.median(e_torso_pos[ss]):.2f} mm')
    print(f'    torso     ori (vs ref): peak {e_torso_ori[ss].max():.3f} deg, '
          f'median {np.median(e_torso_ori[ss]):.3f} deg')
    print()
    print(f'  SS-phase peak speeds:')
    print(f'    swing EE  v:    actual peak {v_ee_norm[ss].max():.1f} mm/s, '
          f'ref peak {v_ee_ref_norm[ss].max():.1f} mm/s')
    print(f'    swing EE  ω:    actual peak {np.degrees(omega_ee_norm[ss].max()):.2f}°/s, '
          f'ref peak {np.degrees(omega_ee_ref_norm[ss].max()):.2f}°/s')
    print(f'    torso     v:    actual peak {v_torso_norm[ss].max():.1f} mm/s, '
          f'ref peak {v_torso_ref_norm[ss].max():.1f} mm/s')
    print(f'    torso     ω:    actual peak {np.degrees(omega_torso_norm[ss].max()):.3f}°/s, '
          f'ref peak {np.degrees(omega_torso_ref_norm[ss].max()):.3f}°/s')

    # --- figure ---
    fig, ax = plt.subplots(4, 2, figsize=(14, 11), sharex=True)
    fig.suptitle(f'Tracking diagnostic — {os.path.basename(_subdir)}',
                 fontsize=11, y=0.995)

    def deco(a, ss_mask):
        ymin, ymax = a.get_ylim()
        a.fill_between(t, ymin, ymax, where=ss_mask, color='orange', alpha=0.06)
        for dt_ev in dock_t:
            a.axvline(dt_ev, color='k', ls=':', lw=0.6, alpha=0.5)
        a.grid(alpha=0.3)

    # Column 0: swing EE
    # Position panel uses d_grip_swing — distance from the swing EE to its
    # target dock anchor. This converges toward zero as the EE reaches the
    # hotdock (the "tracking-to-target" view).
    #
    # NOTE: d_grip_swing is logged every tick but is only meaningful during
    # SS (when an arm is actually swinging). During the trailing-DS settle
    # the runner passes target_anchor=0 as a placeholder (sim_loop.py:1929),
    # producing a constant ~4 m sentinel-like value (distance from the
    # welded gripper at the far-rail anchor to anchor_0 on the opposite
    # end). We mask SS-only here so the curve shows the actual convergence.
    d_grip_ss = np.where(ss, d_grip_swing * 1000, np.nan)
    ax[0, 0].plot(t, d_grip_ss, 'C0', lw=0.9)
    ax[0, 0].set_ylabel('d_grip [mm]')
    ax[0, 0].set_title(f'Swing EE → target dock  (SS-only; min '
                       f'{d_grip_swing[ss].min()*1000:.2f} mm at dock, '
                       f'peak {d_grip_swing[ss].max()*1000:.0f} mm)')

    ax[1, 0].plot(t, e_ee_ori, 'C0', lw=0.9)
    ax[1, 0].set_ylabel('Δori [deg]')
    ax[1, 0].set_title(f'Swing EE orientation error  (SS-peak '
                       f'{e_ee_ori[ss].max():.2f}°)')

    ax[2, 0].plot(t, v_ee_norm, 'C3', lw=0.9, label='actual')
    ax[2, 0].plot(t, v_ee_ref_norm, 'C0', lw=0.9, alpha=0.7, ls='--',
                  label='reference')
    ax[2, 0].set_ylabel('|v| [mm/s]')
    ax[2, 0].set_title('Swing EE linear speed: actual vs reference')
    ax[2, 0].legend(fontsize=8, loc='upper right')
    # Cap y-axis so FD artifacts on the reference don't crush the actual.
    ax[2, 0].set_ylim(0, max(v_ee_norm.max() * 1.2,
                              np.percentile(v_ee_ref_norm[ss], 95) * 1.2))

    ax[3, 0].plot(t, np.degrees(omega_ee_norm), 'C3', lw=0.9, label='actual')
    ax[3, 0].plot(t, np.degrees(omega_ee_ref_norm), 'C0', lw=0.9,
                  alpha=0.7, ls='--', label='reference')
    ax[3, 0].set_ylabel('|ω| [deg/s]')
    ax[3, 0].set_title('Swing EE angular speed')
    ax[3, 0].set_xlabel('t [s]')
    ax[3, 0].legend(fontsize=8, loc='upper right')
    ax[3, 0].set_ylim(0, max(np.degrees(omega_ee_norm).max() * 1.2,
                              np.degrees(np.percentile(omega_ee_ref_norm[ss], 95)) * 1.2))

    # Column 1: torso
    ax[0, 1].plot(t, e_torso_pos, 'C2', lw=0.9)
    ax[0, 1].set_ylabel('|Δp| [mm]')
    ax[0, 1].set_title(f'Torso position error  (SS-peak '
                       f'{e_torso_pos[ss].max():.1f} mm)')

    ax[1, 1].plot(t, e_torso_ori, 'C2', lw=0.9)
    ax[1, 1].set_ylabel('Δori [deg]')
    ax[1, 1].set_title(f'Torso orientation error  (SS-peak '
                       f'{e_torso_ori[ss].max():.2f}°)')

    ax[2, 1].plot(t, v_torso_norm, 'C3', lw=0.9, label='actual')
    ax[2, 1].plot(t, v_torso_ref_norm, 'C2', lw=0.9, alpha=0.7, ls='--',
                  label='reference')
    ax[2, 1].set_ylabel('|v| [mm/s]')
    ax[2, 1].set_title('Torso linear speed: actual vs reference')
    ax[2, 1].legend(fontsize=8, loc='upper right')
    ax[2, 1].set_ylim(0, max(v_torso_norm.max() * 1.2,
                              np.percentile(v_torso_ref_norm[ss], 95) * 1.2))

    ax[3, 1].plot(t, np.degrees(omega_torso_norm), 'C3', lw=0.9, label='actual')
    ax[3, 1].plot(t, np.degrees(omega_torso_ref_norm), 'C2', lw=0.9,
                  alpha=0.7, ls='--', label='reference')
    ax[3, 1].set_ylabel('|ω| [deg/s]')
    ax[3, 1].set_title('Torso angular speed')
    ax[3, 1].set_xlabel('t [s]')
    ax[3, 1].legend(fontsize=8, loc='upper right')
    ax[3, 1].set_ylim(0, max(np.degrees(omega_torso_norm).max() * 1.2,
                              np.degrees(np.percentile(omega_torso_ref_norm[ss], 95)) * 1.2))

    for row in ax:
        for a in row:
            deco(a, ss)

    fig.tight_layout()
    fig.savefig(OUT, dpi=120)
    plt.close(fig)
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
