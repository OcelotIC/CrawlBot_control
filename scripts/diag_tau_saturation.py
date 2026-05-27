"""Per-joint torque saturation trace.

Tests whether stance-arm joint torque saturation precedes torso
tracking error growth in step 2 (failed step). Pattern would
confirm the "demand exceeds actuator authority" mechanism: the
QP commands more contact force than the stance arm can deliver
within tau_max = 20 Nm, joint torque saturates, contact force
falls below demand, torso accel undershoots, error accumulates.

Plot layout:
  row 0 — all 7 stance arm joint torques over time (step 1)
  row 1 — all 7 stance arm joint torques over time (step 2)
  row 2 — all 7 swing arm joint torques over time (step 2)
  row 3 — torso pos/ori error and ||a_torso_actual|| overlay

For each step computes:
  - saturation onset time (first tick where any |tau_i| >= 0.95*tau_max)
  - which joints saturate
  - saturated-fraction (% of ticks with at least one joint saturated)
  - relative timing vs e_torso_ori first crossing 5deg, 20deg

Usage:
  PYTHONPATH=. MUJOCO_GL=osmesa python3 scripts/diag_tau_saturation.py \
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


TAU_MAX = 20.0
SAT_THRESHOLD = 0.95  # |tau|/tau_max above this is "saturated"


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
    tau = np.array(sl['tau'])[mask]            # (N, 14)
    swing = np.array(sl['swing_arm'], dtype=object)[mask][0]
    e_pos = np.array(sl['e_torso_pos'])[mask]
    e_ori_rad = np.array(sl['e_torso_ori'])[mask]
    e_ori_deg = np.degrees(e_ori_rad)
    v_torso = np.array(sl['v_torso'])[mask]
    a_torso = np.gradient(v_torso, t, axis=0)
    a_torso_norm = np.linalg.norm(a_torso, axis=1)

    if swing == 'a':
        tau_swing = tau[:, 0:7]
        tau_stance = tau[:, 7:14]
    else:
        tau_swing = tau[:, 7:14]
        tau_stance = tau[:, 0:7]

    abs_swing = np.abs(tau_swing)
    abs_stance = np.abs(tau_stance)
    sat_swing = abs_swing >= SAT_THRESHOLD * TAU_MAX  # (N, 7)
    sat_stance = abs_stance >= SAT_THRESHOLD * TAU_MAX

    any_sat_stance = sat_stance.any(axis=1)
    any_sat_swing = sat_swing.any(axis=1)
    first_sat_stance = (np.argmax(any_sat_stance) if any_sat_stance.any()
                        else None)
    first_sat_swing = (np.argmax(any_sat_swing) if any_sat_swing.any()
                       else None)

    err5 = e_ori_deg >= 5.0
    err20 = e_ori_deg >= 20.0
    first_err5 = np.argmax(err5) if err5.any() else None
    first_err20 = np.argmax(err20) if err20.any() else None

    return dict(
        step_idx=step_idx, swing_arm=swing, t=t, trel=trel,
        tau_swing=tau_swing, tau_stance=tau_stance,
        sat_swing=sat_swing, sat_stance=sat_stance,
        any_sat_stance=any_sat_stance, any_sat_swing=any_sat_swing,
        first_sat_stance=first_sat_stance, first_sat_swing=first_sat_swing,
        first_err5=first_err5, first_err20=first_err20,
        e_pos=e_pos, e_ori_deg=e_ori_deg,
        a_torso_norm=a_torso_norm,
    )


def plot_arm_tau(ax, trel, tau_arm, title, mark_indices=None):
    for j in range(tau_arm.shape[1]):
        ax.plot(trel, tau_arm[:, j], lw=0.9, label=f'j{j+1}')
    ax.axhline(TAU_MAX, color='red', linestyle='--', alpha=0.5, lw=0.7)
    ax.axhline(-TAU_MAX, color='red', linestyle='--', alpha=0.5, lw=0.7)
    ax.axhline(SAT_THRESHOLD * TAU_MAX, color='orange', linestyle=':',
               alpha=0.4, lw=0.7)
    ax.axhline(-SAT_THRESHOLD * TAU_MAX, color='orange', linestyle=':',
               alpha=0.4, lw=0.7)
    ax.set_ylabel(title + '\nτ [Nm]')
    ax.legend(fontsize=7, loc='upper right', ncol=7)
    ax.grid(True, alpha=0.3)
    if mark_indices is not None:
        for label, idx in mark_indices.items():
            if idx is None:
                continue
            ax.axvline(trel[idx], linestyle=':', alpha=0.5,
                       label=label, color='k', lw=0.8)


def main(results_dir: str):
    with open(os.path.join(results_dir, 'sim_log.json')) as f:
        sl = json.load(f)

    s1 = analyze_step(sl, 1)
    s2 = analyze_step(sl, 2)

    fig, axes = plt.subplots(4, 1, figsize=(13, 14))

    if s1 is not None:
        plot_arm_tau(axes[0], s1['trel'], s1['tau_stance'],
                     f's1 STANCE (arm {"b" if s1["swing_arm"]=="a" else "a"})')
    if s2 is not None:
        marks_st = {'1st sat (stance)': s2['first_sat_stance'],
                    '|e_ori|=5°': s2['first_err5'],
                    '|e_ori|=20°': s2['first_err20']}
        plot_arm_tau(axes[1], s2['trel'], s2['tau_stance'],
                     f's2 STANCE (arm {"b" if s2["swing_arm"]=="a" else "a"})',
                     marks_st)
        marks_sw = {'1st sat (swing)': s2['first_sat_swing'],
                    '|e_ori|=5°': s2['first_err5']}
        plot_arm_tau(axes[2], s2['trel'], s2['tau_swing'],
                     f's2 SWING (arm {s2["swing_arm"]})', marks_sw)

        ax = axes[3]
        ax.plot(s2['trel'], s2['e_ori_deg'], color='C3', lw=1.1,
                label='s2: e_torso_ori [deg]')
        ax.plot(s2['trel'], s2['e_pos'] * 1000, color='C2', lw=0.9,
                linestyle='--', label='s2: e_torso_pos [mm]')
        ax2 = ax.twinx()
        ax2.plot(s2['trel'], s2['a_torso_norm'], color='C0', lw=0.9,
                 alpha=0.7, label='s2: ||a_torso|| [m/s²]')
        # mark first stance saturation
        if s2['first_sat_stance'] is not None:
            ax.axvline(s2['trel'][s2['first_sat_stance']], color='orange',
                       linestyle=':', alpha=0.7,
                       label=f'1st stance τ sat @ t+{s2["trel"][s2["first_sat_stance"]]:.2f}s')
        ax.set_ylabel('error: ori [deg], pos [mm]', color='C3')
        ax2.set_ylabel('||a_torso|| [m/s²]', color='C0')
        ax.legend(fontsize=8, loc='upper left')
        ax2.legend(fontsize=8, loc='upper right')
        ax.set_xlabel('time since SS entry [s]')
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Per-joint τ saturation trace — {results_dir}')
    fig.tight_layout()
    png_path = os.path.join(results_dir, 'tau_saturation.png')
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f'Wrote {png_path}')

    for s in (s1, s2):
        if s is None:
            continue
        print(f'\n=== Step {s["step_idx"]} (swing {s["swing_arm"]}) ===')
        print(f'  duration: {s["trel"][-1]:.2f} s ({len(s["trel"])} ticks)')
        print(f'  peak |τ_stance| per joint: '
              f'{np.abs(s["tau_stance"]).max(axis=0).round(2).tolist()}')
        print(f'  peak |τ_swing|  per joint: '
              f'{np.abs(s["tau_swing"]).max(axis=0).round(2).tolist()}')
        sat_st = s['sat_stance'].any(axis=0)
        sat_sw = s['sat_swing'].any(axis=0)
        print(f'  stance joints that EVER saturate (≥{SAT_THRESHOLD*TAU_MAX:.0f}Nm): '
              f'{[i+1 for i, b in enumerate(sat_st) if b]}')
        print(f'  swing  joints that EVER saturate (≥{SAT_THRESHOLD*TAU_MAX:.0f}Nm): '
              f'{[i+1 for i, b in enumerate(sat_sw) if b]}')
        print(f'  ticks with ANY stance joint saturated: '
              f'{s["any_sat_stance"].sum()} / {len(s["trel"])} '
              f'({100*s["any_sat_stance"].mean():.1f}%)')
        print(f'  ticks with ANY swing  joint saturated: '
              f'{s["any_sat_swing"].sum()} / {len(s["trel"])} '
              f'({100*s["any_sat_swing"].mean():.1f}%)')
        if s['first_sat_stance'] is not None:
            print(f'  first stance saturation: t+{s["trel"][s["first_sat_stance"]]:.2f}s')
        if s['first_err5'] is not None:
            print(f'  first e_ori >= 5°:       t+{s["trel"][s["first_err5"]]:.2f}s')
        if s['first_err20'] is not None:
            print(f'  first e_ori >= 20°:      t+{s["trel"][s["first_err20"]]:.2f}s')
        if (s['first_sat_stance'] is not None
                and s['first_err5'] is not None):
            delta = s["trel"][s['first_err5']] - s["trel"][s['first_sat_stance']]
            print(f'  Δ (e_ori5° − τ_sat):     {delta:+.2f}s '
                  f'({"sat FIRST" if delta > 0 else "err FIRST"})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir')
    args = parser.parse_args()
    main(args.results_dir)
