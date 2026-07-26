"""Swing Cartesian velocity diagnostic.

For a given SS step, computes:
  v_ee_ref(t)   = numerical d(p_ee_ref)/dt  (commanded EE Cartesian velocity)
  v_ee_actual   = sim_log v_ee_a/b           (achieved EE Cartesian velocity)
  dq_swing      = joint velocity of the swing arm                  (rad/s)
  gain(t)       = ||dq_swing|| / ||v_ee_ref||                       (rad/s per m/s)
                  — empirical Jacobian gain at this configuration.

If gain ≫ 1/σ_min_planned, the closed-loop has drifted into a more
ill-conditioned region than the IK predicted. If ||v_ee_ref|| is
modest but ||dq_swing|| is huge, the runaway is amplification from
the small singular direction, not an aggressive reference.

Compares step 1 (success) and step 2 (fail).

Usage:
  PYTHONPATH=. MUJOCO_GL=osmesa python3 scripts/diag_swing_cartvel.py \
      Misc/runs/diag_cooperative_arms/
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def step_slice(sl, step_idx):
    sidx = np.array(sl['step_idx'])
    phase = np.array(sl['phase'], dtype=object)
    return (sidx == step_idx) & (phase == 'SS')


def analyze_step(sl, step_idx):
    mask = step_slice(sl, step_idx)
    if not mask.any():
        return None
    t = np.array(sl['t'])[mask]
    t0 = t[0]
    trel = t - t0
    p_ee_ref = np.array(sl['p_ee_ref'])[mask]
    p_ee = np.array(sl['p_ee'])[mask]
    v_ee_a = np.array(sl['v_ee_a'])[mask]
    v_ee_b = np.array(sl['v_ee_b'])[mask]
    qvel_a = np.array(sl['qvel_joints_a'])[mask]
    qvel_b = np.array(sl['qvel_joints_b'])[mask]
    swing_arm = np.array(sl['swing_arm'], dtype=object)[mask][0]
    tau = np.array(sl['tau'])[mask]

    dt = float(np.mean(np.diff(t))) if len(t) > 1 else 0.01
    v_ee_ref = np.gradient(p_ee_ref, dt, axis=0)
    v_ee_ref_norm = np.linalg.norm(v_ee_ref, axis=1)
    v_ee_act = v_ee_b if swing_arm == 'b' else v_ee_a
    v_ee_act_norm = np.linalg.norm(v_ee_act, axis=1)
    p_ee_err = np.linalg.norm(p_ee - p_ee_ref, axis=1)
    dq_swing = qvel_b if swing_arm == 'b' else qvel_a
    dq_stance = qvel_a if swing_arm == 'b' else qvel_b
    dq_swing_norm = np.linalg.norm(dq_swing, axis=1)
    dq_stance_norm = np.linalg.norm(dq_stance, axis=1)

    eps = 1e-6
    gain_ref = dq_swing_norm / np.maximum(v_ee_ref_norm, eps)
    gain_act = dq_swing_norm / np.maximum(v_ee_act_norm, eps)

    tau_swing = tau[:, 7:14] if swing_arm == 'b' else tau[:, :7]
    tau_stance = tau[:, :7] if swing_arm == 'b' else tau[:, 7:14]
    tau_stance_max = np.abs(tau_stance).max(axis=1)
    tau_swing_max = np.abs(tau_swing).max(axis=1)

    return dict(
        step_idx=step_idx, swing_arm=swing_arm, t_abs=t, trel=trel,
        p_ee_ref=p_ee_ref, p_ee=p_ee,
        v_ee_ref=v_ee_ref, v_ee_act=v_ee_act,
        v_ee_ref_norm=v_ee_ref_norm, v_ee_act_norm=v_ee_act_norm,
        p_ee_err=p_ee_err,
        dq_swing_norm=dq_swing_norm, dq_stance_norm=dq_stance_norm,
        gain_ref=gain_ref, gain_act=gain_act,
        tau_stance_max=tau_stance_max, tau_swing_max=tau_swing_max,
    )


def plot_step(ax_row, s, color, label_prefix):
    if s is None:
        return
    trel = s['trel']
    ax_row[0].plot(trel, s['v_ee_ref_norm'] * 1000, color=color,
                   linestyle='--', lw=0.9, label=f'{label_prefix}: |v_ref|')
    ax_row[0].plot(trel, s['v_ee_act_norm'] * 1000, color=color, lw=1.1,
                   label=f'{label_prefix}: |v_actual|')
    ax_row[1].plot(trel, s['dq_swing_norm'], color=color, lw=1.1,
                   label=f'{label_prefix}: |dq_swing|')
    ax_row[1].plot(trel, s['dq_stance_norm'], color=color, lw=0.9,
                   linestyle='--', alpha=0.7,
                   label=f'{label_prefix}: |dq_stance|')
    ax_row[2].plot(trel, s['gain_ref'], color=color, lw=1.1,
                   label=f'{label_prefix}: dq/v_ref')
    ax_row[2].axhline(1.0 / 0.16, color='red', linestyle=':', alpha=0.4,
                      lw=0.7)
    ax_row[3].plot(trel, s['tau_stance_max'], color=color, lw=1.1,
                   label=f'{label_prefix}: τ_stance max')
    ax_row[3].plot(trel, s['tau_swing_max'], color=color, lw=0.9,
                   linestyle='--', alpha=0.7,
                   label=f'{label_prefix}: τ_swing max')


def main(results_dir: str):
    with open(os.path.join(results_dir, 'sim_log.json')) as f:
        sl = json.load(f)

    s1 = analyze_step(sl, 1)
    s2 = analyze_step(sl, 2)

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=False)
    plot_step([axes[0], axes[1], axes[2], axes[3]], s1, 'C0', 's1')
    plot_step([axes[0], axes[1], axes[2], axes[3]], s2, 'C1', 's2')

    axes[0].set_ylabel('EE Cartesian speed [mm/s]')
    axes[0].set_yscale('log')
    axes[0].legend(fontsize=8, loc='best')

    axes[1].set_ylabel('|dq_arm| [rad/s]')
    axes[1].legend(fontsize=8, loc='best')

    axes[2].set_ylabel('empirical gain |dq|/|v_ref|  [rad/s per m/s]')
    axes[2].set_yscale('log')
    axes[2].legend(fontsize=8, loc='best')

    axes[3].set_ylabel('max joint |τ| [Nm]')
    axes[3].axhline(20, color='red', linestyle='--', alpha=0.5)
    axes[3].legend(fontsize=8, loc='best')

    for ax in axes:
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('time since SS entry [s]')
    fig.suptitle(f'Swing Cartesian velocity diagnostic — {results_dir}')
    fig.tight_layout()
    png_path = os.path.join(results_dir, 'swing_cartvel.png')
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f'Wrote {png_path}')

    for s in (s1, s2):
        if s is None:
            continue
        print(f'\n=== Step {s["step_idx"]} (swing arm {s["swing_arm"]}) ===')
        print(f'  duration: {s["trel"][-1]:.2f} s ({len(s["trel"])} ticks)')
        print(f'  peak |v_ee_ref| = {s["v_ee_ref_norm"].max() * 1000:8.2f} mm/s'
              f'  at t+{s["trel"][s["v_ee_ref_norm"].argmax()]:.2f} s')
        print(f'  peak |v_ee_act| = {s["v_ee_act_norm"].max() * 1000:8.2f} mm/s'
              f'  at t+{s["trel"][s["v_ee_act_norm"].argmax()]:.2f} s')
        print(f'  peak |dq_swing| = {s["dq_swing_norm"].max():8.3f} rad/s'
              f'  at t+{s["trel"][s["dq_swing_norm"].argmax()]:.2f} s')
        print(f'  peak |dq_stance|= {s["dq_stance_norm"].max():8.3f} rad/s')
        ref_mask = s['v_ee_ref_norm'] > 1e-3
        if ref_mask.any():
            print(f'  peak gain (dq/v_ref) [where v_ref > 1mm/s]: '
                  f'{s["gain_ref"][ref_mask].max():8.2f}'
                  f'  (1/σ_min_planned = {1/0.16:.2f} for s2, '
                  f'1/0.18 = {1/0.18:.2f} for s1)')
        print(f'  peak |τ_stance| = {s["tau_stance_max"].max():.2f} Nm')
        print(f'  peak |τ_swing|  = {s["tau_swing_max"].max():.2f} Nm')

    if s1 is not None and s2 is not None:
        ratios = {
            '|v_ee_ref|': s2['v_ee_ref_norm'].max() / max(
                s1['v_ee_ref_norm'].max(), 1e-9),
            '|v_ee_act|': s2['v_ee_act_norm'].max() / max(
                s1['v_ee_act_norm'].max(), 1e-9),
            '|dq_swing|': s2['dq_swing_norm'].max() / max(
                s1['dq_swing_norm'].max(), 1e-9),
            'gain_ref (peak)': (
                s2['gain_ref'][s2['v_ee_ref_norm'] > 1e-3].max() /
                max(s1['gain_ref'][s1['v_ee_ref_norm'] > 1e-3].max(), 1e-9)
            ),
            '|τ_stance|': s2['tau_stance_max'].max() / max(
                s1['tau_stance_max'].max(), 1e-9),
        }
        print(f'\n=== Step 2 / Step 1 peak ratios ===')
        for k, v in ratios.items():
            print(f'  {k:>22s}: {v:8.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir')
    args = parser.parse_args()
    main(args.results_dir)
