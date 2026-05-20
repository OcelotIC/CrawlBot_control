"""Reference angular velocity diagnostic.

Computes per-tick omega_ref = d(q_torso_ref)/dt from the logged
q_torso_ref quaternions (wxyz, MuJoCo convention) using the
finite-difference relative-rotation log map:

  q_rel  = q_ref[t+dt] * q_ref[t]^{-1}
  axis,θ = log(q_rel)              (shortest rotation chosen)
  omega  = axis * θ / dt

Compares against omega_torso (actual) per-tick. Predicts:
  step 2 peak |omega_ref| ~130 deg/s vs |omega_actual| ~5 deg/s
  step 1 peak |omega_ref| ~20-40 deg/s, comparable actual.

Usage:
  PYTHONPATH=. MUJOCO_GL=osmesa python3 scripts/diag_omega_ref.py \
      results/diag_cooperative_arms/
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def quat_conj_wxyz(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul_wxyz(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def omega_from_quat_seq(q_seq, t):
    """Compute omega(t) from a quaternion sequence using central
    differences for interior ticks and forward/backward for ends."""
    N = q_seq.shape[0]
    omega = np.zeros((N, 3))
    for i in range(N):
        if i == 0:
            qa, qb, dt = q_seq[0], q_seq[1], t[1] - t[0]
        elif i == N - 1:
            qa, qb, dt = q_seq[-2], q_seq[-1], t[-1] - t[-2]
        else:
            qa, qb, dt = q_seq[i-1], q_seq[i+1], t[i+1] - t[i-1]
        q_rel = quat_mul_wxyz(qb, quat_conj_wxyz(qa))
        if q_rel[0] < 0:
            q_rel = -q_rel
        w = np.clip(q_rel[0], -1.0, 1.0)
        v = q_rel[1:]
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            omega[i] = 0.0
        else:
            theta = 2.0 * np.arctan2(nv, w)
            omega[i] = (v / nv) * theta / dt
    return omega


def step_slice(sl, step_idx, phase='SS'):
    sidx = np.array(sl['step_idx'])
    ph = np.array(sl['phase'], dtype=object)
    return (sidx == step_idx) & (ph == phase)


def analyze_step(sl, step_idx):
    mask = step_slice(sl, step_idx)
    if not mask.any():
        return None
    t = np.array(sl['t'])[mask]
    trel = t - t[0]
    q_ref = np.array(sl['q_torso_ref'])[mask]
    q_act = np.array(sl['q_torso'])[mask]
    omega_act = np.array(sl['omega_torso'])[mask]
    e_ori_deg = np.degrees(np.array(sl['e_torso_ori'])[mask])

    omega_ref = omega_from_quat_seq(q_ref, t)
    omega_ref_num = omega_from_quat_seq(q_act, t)

    return dict(
        step_idx=step_idx, t=t, trel=trel,
        omega_ref=omega_ref,
        omega_ref_norm=np.linalg.norm(omega_ref, axis=1),
        omega_act=omega_act,
        omega_act_norm=np.linalg.norm(omega_act, axis=1),
        omega_act_num=omega_ref_num,
        omega_act_num_norm=np.linalg.norm(omega_ref_num, axis=1),
        e_ori_deg=e_ori_deg,
    )


def main(results_dir: str):
    with open(os.path.join(results_dir, 'sim_log.json')) as f:
        sl = json.load(f)
    s1 = analyze_step(sl, 1)
    s2 = analyze_step(sl, 2)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    for s, color, label in ((s1, 'C0', 's1'), (s2, 'C1', 's2')):
        if s is None:
            continue
        ax = axes[0]
        ax.plot(s['trel'], np.degrees(s['omega_ref_norm']), color=color, lw=1.1,
                label=f'{label}: ||ω_ref|| (planner)')
        ax.plot(s['trel'], np.degrees(s['omega_act_norm']), color=color, lw=0.9,
                linestyle='--', label=f'{label}: ||ω_actual||')

        ax = axes[1]
        ratio = (s['omega_ref_norm']
                 / np.maximum(s['omega_act_norm'], np.radians(0.5)))
        ax.plot(s['trel'], ratio, color=color, lw=1.0,
                label=f'{label}: ||ω_ref||/||ω_actual||')

        ax = axes[2]
        ax.plot(s['trel'], s['e_ori_deg'], color=color, lw=1.1,
                label=f'{label}: e_torso_ori [deg]')

    axes[0].set_ylabel('angular speed [deg/s]')
    axes[0].set_title('Planner reference vs achieved torso angular speed')
    axes[0].set_yscale('log')
    axes[1].set_ylabel('ratio (log)')
    axes[1].set_yscale('log')
    axes[1].axhline(1.0, color='red', linestyle=':', alpha=0.4, lw=0.7)
    axes[1].set_title(
        'Reference-over-actual angular speed ratio '
        '(>>1 = reference outruns torso)')
    axes[2].set_ylabel('e_torso_ori [deg]')
    axes[2].set_title('Tracking error (accumulates when ratio >> 1)')
    axes[2].set_xlabel('time since SS entry [s]')

    for ax in axes:
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Reference angular velocity (ω_ref) — {results_dir}')
    fig.tight_layout()
    png_path = os.path.join(results_dir, 'omega_ref.png')
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f'Wrote {png_path}')

    for s in (s1, s2):
        if s is None:
            continue
        print(f'\n=== Step {s["step_idx"]} ===')
        print(f'  duration: {s["trel"][-1]:.2f} s')
        print(f'  peak ||ω_ref||       = {np.degrees(s["omega_ref_norm"].max()):7.2f} deg/s'
              f'  at t+{s["trel"][s["omega_ref_norm"].argmax()]:.2f} s')
        print(f'  peak ||ω_actual||    = {np.degrees(s["omega_act_norm"].max()):7.2f} deg/s'
              f'  at t+{s["trel"][s["omega_act_norm"].argmax()]:.2f} s')
        print(f'  median ||ω_ref||     = {np.degrees(np.median(s["omega_ref_norm"])):7.2f} deg/s')
        print(f'  median ||ω_actual||  = {np.degrees(np.median(s["omega_act_norm"])):7.2f} deg/s')
        active = s["omega_ref_norm"] > np.radians(2.0)
        if active.any():
            ratio = (s["omega_ref_norm"][active]
                     / np.maximum(s["omega_act_norm"][active], np.radians(0.5)))
            print(f'  median ratio (where ω_ref > 2°/s): {np.median(ratio):6.2f}×')
            print(f'  peak ratio:                         {np.max(ratio):6.2f}×')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir')
    args = parser.parse_args()
    main(args.results_dir)
