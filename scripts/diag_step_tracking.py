"""Per-tick closed-loop tracking diagnostic.

Loads sim_log.json from a results dir and plots key tracking errors
and state magnitudes vs time, with shaded step boundaries. Designed
to localize *when* and *which channel* the closed-loop diverges from
the NMPC reference during a failing step (compared with successful
steps on the same run).

Inputs are the pre-computed per-tick channels in sim_log.json:
  e_torso_pos, e_torso_ori, e_ee_pos, e_ee_ori, e_com  — residuals
  lambda_qp / lambda_ref / tau / hw                     — actuation
  qvel_joints_a / qvel_joints_b                         — joint rates
  qp_ok / nmpc_ok                                       — solver flags

Output (in the given results dir):
  step_tracking.png  — 8-panel time-series with step shading
  console table      — peak per-channel for each step + step2/step1 ratios

Usage:
  PYTHONPATH=. MUJOCO_GL=osmesa python3 scripts/diag_step_tracking.py \
      results/diag_cooperative_arms/
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(results_dir: str):
    with open(os.path.join(results_dir, 'sim_log.json')) as f:
        sl = json.load(f)

    t = np.array(sl['t'])
    sidx = np.array(sl['step_idx'])
    phase = np.array(sl['phase'], dtype=object)

    e_torso_pos = np.array(sl['e_torso_pos'])
    e_torso_ori = np.array(sl['e_torso_ori'])
    e_ee_pos = np.array(sl['e_ee_pos'])
    e_ee_ori = np.array(sl['e_ee_ori'])
    e_com = np.array(sl['e_com'])
    lambda_norm = np.array(sl['lambda_qp_norm'])
    lambda_ref_norm = np.array(sl['lambda_ref_norm'])
    L_com_norm = np.array(sl['L_com_norm'])
    tau = np.array(sl['tau'])
    tau_max = np.abs(tau).max(axis=1)
    tau_norm = np.linalg.norm(tau, axis=1)
    hw = np.array(sl['hw'])
    hw_norm = np.linalg.norm(hw, axis=1)
    qvel_a = np.array(sl['qvel_joints_a'])
    qvel_b = np.array(sl['qvel_joints_b'])
    qvel_a_norm = np.linalg.norm(qvel_a, axis=1)
    qvel_b_norm = np.linalg.norm(qvel_b, axis=1)
    qp_ok = np.array(sl['qp_ok'])
    nmpc_ok = np.array(sl['nmpc_ok'])

    step_bounds = []
    n_steps = int(sidx.max()) + 1
    for s in range(n_steps):
        ss_mask = (sidx == s) & (phase == 'SS')
        if ss_mask.any():
            step_bounds.append(
                (float(t[ss_mask][0]), float(t[ss_mask][-1]), s))

    aborted = {ev['step_idx']: ev for ev in sl.get('aborted_steps', [])}

    fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)

    def _shade(ax):
        for t0, t1, s in step_bounds:
            color = 'red' if s in aborted else 'green'
            ax.axvspan(t0, t1, alpha=0.06, color=color)
            ax.axvline(t0, alpha=0.3, color='gray', lw=0.5)
        ymin, ymax = ax.get_ylim()
        for t0, t1, s in step_bounds:
            ax.text((t0 + t1) / 2, ymax * 0.92 if ymax > 0 else ymax * 1.08,
                    f's{s}', ha='center', va='top', fontsize=8,
                    color=('darkred' if s in aborted else 'darkgreen'))

    axes[0, 0].plot(t, e_torso_pos * 1000, label='torso', lw=1)
    axes[0, 0].plot(t, e_ee_pos * 1000, label='EE', lw=1)
    axes[0, 0].set_ylabel('|pos err| [mm]')
    axes[0, 0].set_yscale('log')
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(t, np.degrees(e_torso_ori), label='torso', lw=1)
    axes[0, 1].plot(t, np.degrees(e_ee_ori), label='EE', lw=1)
    axes[0, 1].set_ylabel('|ori err| [deg]')
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(t, e_com * 1000, label='|e_com|', lw=1)
    axes[1, 0].plot(t, L_com_norm, label='|L_com|', lw=1)
    axes[1, 0].set_ylabel('CoM err [mm] / |L_com|')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(t, lambda_norm, label='λ_qp', lw=1)
    axes[1, 1].plot(t, lambda_ref_norm, label='λ_ref', linestyle='--', lw=1)
    axes[1, 1].set_ylabel('|λ| [N]')
    axes[1, 1].legend(fontsize=8)

    axes[2, 0].plot(t, tau_max, label='max |τ_joint|', lw=1)
    axes[2, 0].plot(t, tau_norm, label='|τ|', lw=1)
    axes[2, 0].axhline(20.0, color='red', linestyle='--', alpha=0.5,
                      label='τ_max=20')
    axes[2, 0].set_ylabel('τ [Nm]')
    axes[2, 0].legend(fontsize=8)

    axes[2, 1].plot(t, hw_norm, label='|hw|', lw=1)
    axes[2, 1].axhline(5.0, color='red', linestyle='--', alpha=0.5,
                      label='hw_max=5')
    axes[2, 1].set_ylabel('|hw| [Nms]')
    axes[2, 1].legend(fontsize=8)

    axes[3, 0].plot(t, qvel_a_norm, label='arm A |dq|', lw=1)
    axes[3, 0].plot(t, qvel_b_norm, label='arm B |dq|', lw=1)
    axes[3, 0].set_ylabel('|dq_arm| [rad/s]')
    axes[3, 0].legend(fontsize=8)

    axes[3, 1].plot(t, qp_ok.astype(int), label='qp_ok', lw=1)
    axes[3, 1].plot(t, nmpc_ok.astype(int), label='nmpc_ok',
                    linestyle='--', lw=1)
    axes[3, 1].set_ylim(-0.1, 1.3)
    axes[3, 1].set_ylabel('solver_ok')
    axes[3, 1].legend(fontsize=8)

    for ax in axes.flat:
        _shade(ax)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel('time [s]')
    axes[-1, 1].set_xlabel('time [s]')
    fig.suptitle(f'Per-tick closed-loop tracking — {results_dir}')
    fig.tight_layout()
    png_path = os.path.join(results_dir, 'step_tracking.png')
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f'Wrote {png_path}')

    channels = [
        ('e_torso_pos [mm]', e_torso_pos * 1000),
        ('e_torso_ori [deg]', np.degrees(e_torso_ori)),
        ('e_ee_pos [mm]', e_ee_pos * 1000),
        ('e_ee_ori [deg]', np.degrees(e_ee_ori)),
        ('e_com [mm]', e_com * 1000),
        ('|lambda_qp| [N]', lambda_norm),
        ('|tau| max [Nm]', tau_max),
        ('|hw| [Nms]', hw_norm),
        ('|dq_arm_a| [rad/s]', qvel_a_norm),
        ('|dq_arm_b| [rad/s]', qvel_b_norm),
    ]
    print('\nPer-step peaks (SS phase, sim_log per-tick):')
    header = f"{'channel':>22s}"
    for _, _, s in step_bounds:
        header += f"  {'s'+str(s):>10s}"
    if len(step_bounds) >= 2:
        header += f"  {'s2/s1':>7s}"
    print(header)
    for name, arr in channels:
        row = f'{name:>22s}'
        peaks = []
        for _, _, s in step_bounds:
            mask = (sidx == s) & (phase == 'SS')
            peak = float(arr[mask].max()) if mask.any() else float('nan')
            peaks.append(peak)
            row += f'  {peak:>10.3f}'
        if len(peaks) >= 3 and peaks[1] > 0:
            row += f'  {peaks[2]/peaks[1]:>7.2f}'
        print(row)

    print('\nAborted steps:')
    for s, ev in aborted.items():
        print(f'  step {s}: {ev}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir',
                        help='Path to results dir containing sim_log.json')
    args = parser.parse_args()
    main(args.results_dir)
