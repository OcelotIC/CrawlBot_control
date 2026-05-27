"""Wrench balance / inertial recruitment diagnostic (zero-g).

Whole-robot Newton-2 in microgravity:
    F_contact_total = m_total * a_com_world

If this closes per-tick, the sim is consistent. The diagnostic
quantity is then the per-arm "inertial recruitment force":

    F_recruitment = m_total * a_com - m_torso * a_torso
                  = m_armA * a_armA_com + m_armB * a_armB_com
                  = "net force from arm CoM accelerations"

Hypothesis (recruitment): in failed step 2, the QP keeps CoM
roughly stationary (per pre-planner) while accelerating the torso
hard; the only way to satisfy both is by counter-accelerating the
arms. F_recruitment should be large and anti-aligned with
m_torso * a_torso.

This script computes the closure and the recruitment force per tick
without Pinocchio — uses only fields already logged in sim_log.json
(v_com, v_torso, lambda_qp) plus mass constants from the URDF.

Usage:
  PYTHONPATH=. MUJOCO_GL=osmesa python3 scripts/diag_wrench_balance.py \
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


M_TOTAL = 71.056
M_TORSO = 40.000
M_ARM = 15.528
M_ARMS_TOTAL = 2 * M_ARM


def step_slice(sl, step_idx, phase='SS'):
    sidx = np.array(sl['step_idx'])
    ph = np.array(sl['phase'], dtype=object)
    return (sidx == step_idx) & (ph == phase)


def numerical_derivative(x, t):
    """Central difference with edge fallback. x: (N,3), t: (N,)."""
    return np.gradient(x, t, axis=0)


def analyze_step(sl, step_idx):
    mask = step_slice(sl, step_idx)
    if not mask.any():
        return None
    t = np.array(sl['t'])[mask]
    trel = t - t[0]

    v_com = np.array(sl['v_com'])[mask]
    v_torso = np.array(sl['v_torso'])[mask]
    omega_torso = np.array(sl['omega_torso'])[mask]
    lam = np.array(sl['lambda_qp'])[mask]
    swing = np.array(sl['swing_arm'], dtype=object)[mask]

    F_a = lam[:, 0:3]
    F_b = lam[:, 6:9]
    # Stance is whichever isn't swinging
    is_swing_b = swing == 'b'
    F_contact = np.where(is_swing_b[:, None], F_a, F_b)

    a_com = numerical_derivative(v_com, t)
    a_torso = numerical_derivative(v_torso, t)
    alpha_torso = numerical_derivative(omega_torso, t)

    F_total_inertial = M_TOTAL * a_com  # zero-g Newton-2
    F_torso_inertial = M_TORSO * a_torso
    F_recruitment = F_total_inertial - F_torso_inertial

    closure_residual = F_contact - F_total_inertial

    # Magnitudes
    F_contact_norm = np.linalg.norm(F_contact, axis=1)
    F_total_inertial_norm = np.linalg.norm(F_total_inertial, axis=1)
    F_torso_inertial_norm = np.linalg.norm(F_torso_inertial, axis=1)
    F_recruitment_norm = np.linalg.norm(F_recruitment, axis=1)
    closure_norm = np.linalg.norm(closure_residual, axis=1)

    # Anti-alignment: cos(angle) between F_recruitment and -F_torso_inertial.
    # +1 means perfect counter-reaction (arms push opposite to torso accel).
    eps = 1e-6
    cos_anti = -np.einsum('ij,ij->i', F_recruitment, F_torso_inertial) / np.maximum(
        F_recruitment_norm * F_torso_inertial_norm, eps)

    return dict(
        step_idx=step_idx, t=t, trel=trel, swing_arm=swing[0],
        F_contact=F_contact, F_total_inertial=F_total_inertial,
        F_torso_inertial=F_torso_inertial, F_recruitment=F_recruitment,
        F_contact_norm=F_contact_norm,
        F_total_inertial_norm=F_total_inertial_norm,
        F_torso_inertial_norm=F_torso_inertial_norm,
        F_recruitment_norm=F_recruitment_norm,
        closure_norm=closure_norm,
        cos_anti=cos_anti,
        a_com=a_com, a_torso=a_torso, alpha_torso=alpha_torso,
    )


def main(results_dir: str):
    with open(os.path.join(results_dir, 'sim_log.json')) as f:
        sl = json.load(f)

    s1 = analyze_step(sl, 1)
    s2 = analyze_step(sl, 2)

    fig, axes = plt.subplots(5, 1, figsize=(12, 14))

    for s, color, label in ((s1, 'C0', 's1'), (s2, 'C1', 's2')):
        if s is None:
            continue
        ax = axes[0]
        ax.plot(s['trel'], s['F_contact_norm'], color=color, lw=1.1,
                label=f'{label}: ||F_contact||')
        ax.plot(s['trel'], s['F_total_inertial_norm'], color=color, lw=0.9,
                linestyle='--',
                label=f'{label}: m_total·||a_com||')

        ax = axes[1]
        ax.plot(s['trel'], s['closure_norm'], color=color, lw=1.0,
                label=f'{label}: ||F_contact - m_total·a_com||')

        ax = axes[2]
        ax.plot(s['trel'], s['F_torso_inertial_norm'], color=color, lw=1.1,
                label=f'{label}: m_torso·||a_torso||')
        ax.plot(s['trel'], s['F_recruitment_norm'], color=color, lw=0.9,
                linestyle='--',
                label=f'{label}: ||F_recruitment||')

        ax = axes[3]
        ax.plot(s['trel'], s['cos_anti'], color=color, lw=1.0,
                label=f'{label}: cos∠(F_recruit, -F_torso_inert)')

        ax = axes[4]
        ax.plot(s['trel'], np.linalg.norm(s['a_torso'], axis=1), color=color,
                lw=1.1, label=f'{label}: ||a_torso_actual|| [m/s²]')

    axes[0].set_ylabel('Force [N]')
    axes[0].set_title('Global linear closure (should overlap in zero-g)')
    axes[1].set_ylabel('Residual [N]')
    axes[1].set_title('Closure residual ||F_contact − m_total·a_com|| (sanity)')
    axes[2].set_ylabel('Force [N]')
    axes[2].set_title('Torso inertial demand vs arm recruitment force')
    axes[3].set_ylabel('cos angle')
    axes[3].set_ylim(-1.1, 1.1)
    axes[3].axhline(1.0, color='red', linestyle=':', alpha=0.4, lw=0.7)
    axes[3].set_title(
        'Recruitment anti-alignment with torso accel '
        '(+1 = pure counter-reaction)')
    axes[4].set_ylabel('a_torso [m/s²]')
    axes[4].set_title('Achieved torso linear acceleration')
    axes[4].set_xlabel('time since SS entry [s]')

    for ax in axes:
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Wrench balance / inertial recruitment — {results_dir}')
    fig.tight_layout()
    png_path = os.path.join(results_dir, 'wrench_balance.png')
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f'Wrote {png_path}')

    for s in (s1, s2):
        if s is None:
            continue
        print(f'\n=== Step {s["step_idx"]} (swing arm {s["swing_arm"]}) ===')
        print(f'  peak ||a_torso_actual||   = {np.linalg.norm(s["a_torso"], axis=1).max():.3f} m/s²')
        print(f'  peak ||a_com_actual||     = {np.linalg.norm(s["a_com"], axis=1).max():.3f} m/s²')
        print(f'  peak ||F_contact||        = {s["F_contact_norm"].max():.2f} N')
        print(f'  peak m_total·||a_com||    = {s["F_total_inertial_norm"].max():.2f} N')
        print(f'  median closure residual   = {np.median(s["closure_norm"]):.2f} N')
        print(f'  peak m_torso·||a_torso||  = {s["F_torso_inertial_norm"].max():.2f} N')
        print(f'  peak ||F_recruitment||    = {s["F_recruitment_norm"].max():.2f} N')
        # Where torso is accel'ing significantly, what fraction of the
        # torso force is from arm recruitment vs contact?
        active = s["F_torso_inertial_norm"] > 5.0  # > 5N filter
        if active.any():
            ratio = (s["F_recruitment_norm"][active] /
                     np.maximum(s["F_torso_inertial_norm"][active], 1e-3))
            print(f'  median ||F_recruit||/||F_torso_inert|| (where torso>5N): '
                  f'{np.median(ratio):.2f}')
            print(f'  median cos(F_recruit, -F_torso_inert) (where torso>5N): '
                  f'{np.median(s["cos_anti"][active]):.2f}')

    if s1 is not None and s2 is not None:
        print('\n=== Step 2 / Step 1 peak ratios ===')
        for k in ('F_contact_norm', 'F_total_inertial_norm',
                  'F_torso_inertial_norm', 'F_recruitment_norm'):
            r = s2[k].max() / max(s1[k].max(), 1e-9)
            print(f'  {k:>25s}: {r:6.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir')
    args = parser.parse_args()
    main(args.results_dir)
