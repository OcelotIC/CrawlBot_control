"""Step-N deep-dive: why does τ saturate, and which joint goes first?

Slices a single SS step from the baseline run and produces an 8-panel
zoom plot:
  row 0:  per-joint τ for arm A (left) and arm B (right) — 7 lines each
  row 1:  per-joint dq for arm A (left) and arm B (right)
  row 2:  τ summaries (per-arm norm + max) | dq norms per arm
  row 3:  torso-acceleration reference vs QP-achieved (NMPC step_log) |
          contact-wrench norms (lambda_qp vs lambda_ref) per arm

Also prints:
  - SS window duration and tick count
  - τ-saturation onset (first tick where any joint |τ| ≥ 19.9 Nm)
  - dq runaway onset (first tick where |dq_B| > 1.0 rad/s)
  - Temporal ordering: which fires first
  - Per-joint τ_peak and dq_peak tables for both arms

This localizes the dominant joint(s), the temporal ordering, and
whether the saturation is driven by an aggressive NMPC reference
or by a QP feedback loop.

Usage:
  PYTHONPATH=. MUJOCO_GL=osmesa python3 Misc/scripts/diag_step_zoom.py \
      Misc/runs/diag_cooperative_arms/ --step_idx 2
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(results_dir: str, step_idx: int = 2):
    with open(os.path.join(results_dir, 'sim_log.json')) as f:
        sl = json.load(f)
    pl = None
    pl_path = os.path.join(results_dir, 'step_log.json')
    if os.path.exists(pl_path):
        with open(pl_path) as f:
            pl = json.load(f)

    t_all = np.array(sl['t'])
    sidx_all = np.array(sl['step_idx'])
    phase_all = np.array(sl['phase'], dtype=object)
    mask = (sidx_all == step_idx) & (phase_all == 'SS')
    if not mask.any():
        print(f'No SS data for step {step_idx}.')
        return

    t_abs = t_all[mask]
    t = t_abs - t_abs[0]

    tau = np.array(sl['tau'])[mask]
    qvel_a = np.array(sl['qvel_joints_a'])[mask]
    qvel_b = np.array(sl['qvel_joints_b'])[mask]
    lam = np.array(sl['lambda_qp'])[mask]
    lam_ref = np.array(sl['lambda_ref'])[mask]
    swing_arm = np.array(sl['swing_arm'], dtype=object)[mask]
    swing_letter = swing_arm[0] if len(swing_arm) else '?'

    tau_abs = np.abs(tau)
    tau_max_per_tick = tau_abs.max(axis=1)
    sat_mask = tau_max_per_tick >= 19.9
    t_sat_onset = float(t[sat_mask][0]) if sat_mask.any() else None

    dq_a_norm = np.linalg.norm(qvel_a, axis=1)
    dq_b_norm = np.linalg.norm(qvel_b, axis=1)
    dq_swing_norm = dq_b_norm if swing_letter == 'b' else dq_a_norm
    runaway_mask = dq_swing_norm > 1.0
    t_runaway_onset = float(t[runaway_mask][0]) if runaway_mask.any() else None

    dom_joints = []
    for i in range(tau.shape[1]):
        if tau_abs[:, i].max() >= 19.9:
            first_idx = int(np.where(tau_abs[:, i] >= 19.9)[0][0])
            dom_joints.append((i, float(t[first_idx]),
                               float(tau_abs[:, i].max())))
    dom_joints.sort(key=lambda x: x[1])

    a_des = a_qp = t_step = None
    if pl is not None:
        step_t = np.array([e['t'] for e in pl])
        sel = (step_t >= t_abs[0]) & (step_t <= t_abs[-1])
        if sel.any():
            idxs = np.where(sel)[0]
            a_des = np.array([pl[i]['a_torso_des'] for i in idxs])
            a_qp = np.array([pl[i]['a_torso_qp'] for i in idxs])
            t_step = step_t[sel] - t_abs[0]

    fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
    fig.suptitle(f'Step {step_idx} SS deep-dive — swing arm = {swing_letter}')

    cmap_a = plt.cm.Blues(np.linspace(0.3, 0.95, 7))
    cmap_b = plt.cm.Oranges(np.linspace(0.3, 0.95, 7))

    for i in range(7):
        axes[0, 0].plot(t, tau[:, i], lw=0.9, color=cmap_a[i], label=f'A{i}')
    axes[0, 0].axhline(20, color='red', linestyle='--', alpha=0.4)
    axes[0, 0].axhline(-20, color='red', linestyle='--', alpha=0.4)
    axes[0, 0].set_ylabel('arm A τ [Nm]')
    axes[0, 0].legend(fontsize=7, ncol=2, loc='best')

    for i in range(7):
        axes[0, 1].plot(t, tau[:, 7 + i], lw=0.9, color=cmap_b[i],
                        label=f'B{i}')
    axes[0, 1].axhline(20, color='red', linestyle='--', alpha=0.4)
    axes[0, 1].axhline(-20, color='red', linestyle='--', alpha=0.4)
    axes[0, 1].set_ylabel('arm B τ [Nm]')
    axes[0, 1].legend(fontsize=7, ncol=2, loc='best')

    for i in range(7):
        axes[1, 0].plot(t, qvel_a[:, i], lw=0.9, color=cmap_a[i],
                        label=f'A{i}')
    axes[1, 0].set_ylabel('arm A dq [rad/s]')
    axes[1, 0].legend(fontsize=7, ncol=2, loc='best')

    for i in range(7):
        axes[1, 1].plot(t, qvel_b[:, i], lw=0.9, color=cmap_b[i],
                        label=f'B{i}')
    axes[1, 1].set_ylabel('arm B dq [rad/s]')
    axes[1, 1].legend(fontsize=7, ncol=2, loc='best')

    axes[2, 0].plot(t, tau_max_per_tick, lw=1.5, color='black',
                    label='max |τ|')
    axes[2, 0].plot(t, np.linalg.norm(tau[:, :7], axis=1), lw=1,
                    color='C0', label='|τ_A|')
    axes[2, 0].plot(t, np.linalg.norm(tau[:, 7:], axis=1), lw=1,
                    color='C1', label='|τ_B|')
    axes[2, 0].axhline(20, color='red', linestyle='--', alpha=0.4,
                      label='τ_max')
    if t_sat_onset is not None:
        axes[2, 0].axvline(t_sat_onset, color='red', alpha=0.6, lw=1,
                           label=f'τ-sat onset {t_sat_onset:.2f}s')
    axes[2, 0].set_ylabel('τ summaries [Nm]')
    axes[2, 0].legend(fontsize=8)

    axes[2, 1].plot(t, dq_a_norm, lw=1, color='C0', label='|dq_A|')
    axes[2, 1].plot(t, dq_b_norm, lw=1, color='C1', label='|dq_B|')
    if t_runaway_onset is not None:
        axes[2, 1].axvline(t_runaway_onset, color='red', alpha=0.6, lw=1,
                           label=f'|dq_swing|>1 at {t_runaway_onset:.2f}s')
    if t_sat_onset is not None:
        axes[2, 1].axvline(t_sat_onset, color='black', alpha=0.4, lw=0.7,
                           linestyle=':', label=f'τ-sat {t_sat_onset:.2f}s')
    axes[2, 1].set_ylabel('|dq_arm| [rad/s]')
    axes[2, 1].legend(fontsize=8)

    if a_des is not None and t_step is not None:
        for k, lbl in enumerate(['x', 'y', 'z']):
            axes[3, 0].plot(t_step, a_des[:, k], lw=0.9, linestyle='--',
                            alpha=0.6, label=f'a_des[{lbl}]')
            axes[3, 0].plot(t_step, a_qp[:, k], lw=1.0, alpha=0.85,
                            label=f'a_qp[{lbl}]')
        axes[3, 0].set_ylabel('torso lin accel [m/s²]')
        axes[3, 0].legend(fontsize=7, ncol=2)
    else:
        axes[3, 0].text(0.5, 0.5, 'step_log.json unavailable',
                        ha='center', va='center')

    axes[3, 1].plot(t, np.linalg.norm(lam[:, :6], axis=1), lw=1,
                    color='C0', label='|λ_qp_A|')
    axes[3, 1].plot(t, np.linalg.norm(lam[:, 6:], axis=1), lw=1,
                    color='C1', label='|λ_qp_B|')
    axes[3, 1].plot(t, np.linalg.norm(lam_ref[:, :6], axis=1), lw=0.8,
                    linestyle='--', alpha=0.6, color='C0',
                    label='|λ_ref_A|')
    axes[3, 1].plot(t, np.linalg.norm(lam_ref[:, 6:], axis=1), lw=0.8,
                    linestyle='--', alpha=0.6, color='C1',
                    label='|λ_ref_B|')
    axes[3, 1].set_ylabel('|λ| [N]')
    axes[3, 1].legend(fontsize=7)

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel(f'time since step {step_idx} SS entry [s]')
    axes[-1, 1].set_xlabel(f'time since step {step_idx} SS entry [s]')
    fig.tight_layout()

    png_path = os.path.join(results_dir, f'step{step_idx}_zoom.png')
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f'Wrote {png_path}')

    print(f'\n=== Step {step_idx} SS timing ===')
    print(f'  window: [{t_abs[0]:.2f}, {t_abs[-1]:.2f}] s, '
          f'duration {t[-1]:.2f} s, {len(t)} ticks, swing arm = {swing_letter}')
    if t_sat_onset is not None:
        print(f'  τ saturation onset:   t+{t_sat_onset:.3f} s')
    else:
        print(f'  τ saturation onset:   never reached 19.9 Nm')
    if t_runaway_onset is not None:
        print(f'  |dq_swing|>1 onset:   t+{t_runaway_onset:.3f} s')
    else:
        print(f'  |dq_swing|>1 onset:   never crossed 1.0 rad/s')
    if t_sat_onset is not None and t_runaway_onset is not None:
        d = t_runaway_onset - t_sat_onset
        order = ('τ-sat FIRST (then dq runaway)' if d > 0
                 else 'dq runaway FIRST (then τ-sat)')
        print(f'  ordering: {order}, Δ = {d * 1000:+.1f} ms')

    print(f'\n=== First joints to hit |τ| ≥ 19.9 Nm (sorted by time) ===')
    for jid, t_first, peak in dom_joints[:7]:
        arm = 'A' if jid < 7 else 'B'
        ji = jid if jid < 7 else jid - 7
        print(f'  arm {arm} joint {ji}: first at t+{t_first:.3f} s, '
              f'peak |τ|={peak:.2f} Nm')
    if not dom_joints:
        print('  (no joints saturated)')

    print(f'\n=== Per-joint peaks ===')
    print(f'  arm A τ_peak [Nm]:  '
          f'{np.abs(tau[:, :7]).max(axis=0).round(2).tolist()}')
    print(f'  arm B τ_peak [Nm]:  '
          f'{np.abs(tau[:, 7:]).max(axis=0).round(2).tolist()}')
    print(f'  arm A dq_peak [rad/s]: '
          f'{np.abs(qvel_a).max(axis=0).round(2).tolist()}')
    print(f'  arm B dq_peak [rad/s]: '
          f'{np.abs(qvel_b).max(axis=0).round(2).tolist()}')

    if a_des is not None:
        print(f'\n=== Reference acceleration profile ===')
        a_des_norm = np.linalg.norm(a_des[:, :3], axis=1)
        a_qp_norm = np.linalg.norm(a_qp[:, :3], axis=1)
        print(f'  max |a_des linear| = {a_des_norm.max():.4f} m/s² '
              f'at t+{float(t_step[a_des_norm.argmax()]):.2f} s')
        print(f'  max |a_qp  linear| = {a_qp_norm.max():.4f} m/s² '
              f'at t+{float(t_step[a_qp_norm.argmax()]):.2f} s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir')
    parser.add_argument('--step_idx', type=int, default=2)
    args = parser.parse_args()
    main(args.results_dir, args.step_idx)
