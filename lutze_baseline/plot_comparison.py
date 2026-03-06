"""Comparison plots for Lutze baseline vs MPC controller.

Generates journal-quality figures comparing both controllers on the same
single-step locomotion scenario.

Usage:
    # Plot single run:
    python -m lutze_baseline.plot_comparison sim_lutze_log.json

    # Compare both:
    python -m lutze_baseline.plot_comparison sim_lutze_log.json sim_torso6d_log.json
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt


def _load(path):
    with open(path) as f:
        log = json.load(f)
    for k in log:
        if isinstance(log[k], list) and len(log[k]) > 0:
            if isinstance(log[k][0], list):
                log[k] = np.array(log[k])
    return log


def plot_results(log, save_path='sim_lutze_results.png', title='Lutze Baseline'):
    """Plot results from a single simulation run."""
    t = np.array(log['t'])
    pt = np.array(log['p_torso'])
    pt_ref = np.array(log['p_torso_ref'])
    db = np.array(log['d_grip_b'])
    da = np.array(log['d_grip_a'])
    tau = np.array(log['tau'])
    sp = np.array(log['struct_pos'])
    e_torso = np.array(log['e_torso_pos'])

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14)

    # 1. Torso x-position
    ax = axes[0, 0]
    ax.plot(t, pt[:, 0] * 100, 'b-', label='actual')
    ax.plot(t, pt_ref[:, 0] * 100, 'r--', label='reference')
    ax.set_ylabel('Torso x [cm]')
    ax.legend()
    ax.set_title('Torso Advancement')
    ax.grid(True, alpha=0.3)

    # 2. Gripper B distance
    ax = axes[0, 1]
    ax.semilogy(t, db * 1000, 'b-')
    ax.axhline(5.0, color='r', ls='--', label='5mm dock')
    ax.set_ylabel('d_grip_b [mm]')
    ax.legend()
    ax.set_title('Gripper B → Anchor Distance')
    ax.grid(True, alpha=0.3)

    # 3. Torso tracking error
    ax = axes[1, 0]
    ax.plot(t, e_torso * 100, 'b-')
    ax.set_ylabel('||e_torso|| [cm]')
    ax.set_title('Torso Tracking Error')
    ax.grid(True, alpha=0.3)

    # 4. Joint torques
    ax = axes[1, 1]
    for i in range(tau.shape[1]):
        ax.plot(t, tau[:, i], alpha=0.5, lw=0.8)
    ax.axhline(10, color='r', ls='--', alpha=0.5)
    ax.axhline(-10, color='r', ls='--', alpha=0.5)
    ax.set_ylabel('τ [Nm]')
    ax.set_title('Joint Torques')
    ax.grid(True, alpha=0.3)

    # 5. Structure drift
    ax = axes[2, 0]
    sp_rel = (sp - sp[0]) * 100
    ax.plot(t, sp_rel[:, 0], label='x')
    ax.plot(t, sp_rel[:, 1], label='y')
    ax.plot(t, sp_rel[:, 2], label='z')
    ax.set_ylabel('Struct drift [cm]')
    ax.set_xlabel('Time [s]')
    ax.legend()
    ax.set_title('Structure CoM Drift')
    ax.grid(True, alpha=0.3)

    # 6. Angular momentum (if available)
    ax = axes[2, 1]
    if 'L_com' in log:
        L = np.array(log['L_com'])
        L_norm = np.linalg.norm(L, axis=1)
        ax.plot(t[:len(L_norm)], L_norm, 'b-')
        ax.axhline(5.0, color='r', ls='--', label='L_max=5 Nms')
        ax.set_ylabel('|L_com| [Nms]')
        ax.legend()
    ax.set_xlabel('Time [s]')
    ax.set_title('Robot Angular Momentum')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_comparison(log_lutze, log_mpc, save_path='comparison.png'):
    """Side-by-side comparison of Lutze baseline vs MPC controller."""
    t_l = np.array(log_lutze['t'])
    t_m = np.array(log_mpc['t'])

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Lutze Baseline vs MPC Controller', fontsize=14)

    # 1. Gripper distance
    ax = axes[0, 0]
    ax.semilogy(t_l, np.array(log_lutze['d_grip_b']) * 1000, 'b-', label='Lutze')
    ax.semilogy(t_m, np.array(log_mpc['d_grip_b']) * 1000, 'r--', label='MPC')
    ax.axhline(5.0, color='k', ls=':', alpha=0.5, label='5mm dock')
    ax.set_ylabel('d_grip_b [mm]')
    ax.legend()
    ax.set_title('Docking Convergence')
    ax.grid(True, alpha=0.3)

    # 2. Torso advancement
    ax = axes[0, 1]
    pt_l = np.array(log_lutze['p_torso'])
    pt_m = np.array(log_mpc['p_torso'])
    ax.plot(t_l, (pt_l[:, 0] - pt_l[0, 0]) * 100, 'b-', label='Lutze')
    ax.plot(t_m, (pt_m[:, 0] - pt_m[0, 0]) * 100, 'r--', label='MPC')
    ax.set_ylabel('Torso Δx [cm]')
    ax.legend()
    ax.set_title('Torso Advancement')
    ax.grid(True, alpha=0.3)

    # 3. Joint torques (max per timestep)
    ax = axes[1, 0]
    ax.plot(t_l, log_lutze['tau_max_joint'], 'b-', label='Lutze')
    ax.plot(t_m, log_mpc['tau_max_joint'], 'r--', label='MPC')
    ax.axhline(10, color='k', ls=':', alpha=0.5)
    ax.set_ylabel('max|τ_i| [Nm]')
    ax.legend()
    ax.set_title('Peak Joint Torque')
    ax.grid(True, alpha=0.3)

    # 4. Structure drift
    ax = axes[1, 1]
    sp_l = np.array(log_lutze['struct_pos'])
    sp_m = np.array(log_mpc['struct_pos'])
    drift_l = np.linalg.norm(sp_l - sp_l[0], axis=1) * 100
    drift_m = np.linalg.norm(sp_m - sp_m[0], axis=1) * 100
    ax.plot(t_l, drift_l, 'b-', label='Lutze')
    ax.plot(t_m, drift_m, 'r--', label='MPC')
    ax.set_ylabel('Struct drift [cm]')
    ax.legend()
    ax.set_title('Structure CoM Drift')
    ax.grid(True, alpha=0.3)

    # 5. Angular momentum
    ax = axes[2, 0]
    if 'L_com' in log_lutze:
        L_l = np.linalg.norm(np.array(log_lutze['L_com']), axis=1)
        ax.plot(t_l[:len(L_l)], L_l, 'b-', label='Lutze')
    if 'L_com' in log_mpc:
        L_m = np.linalg.norm(np.array(log_mpc['L_com']), axis=1)
        ax.plot(t_m[:len(L_m)], L_m, 'r--', label='MPC')
    ax.axhline(5.0, color='k', ls=':', alpha=0.5, label='L_max=5 Nms')
    ax.set_ylabel('|L_com| [Nms]')
    ax.set_xlabel('Time [s]')
    ax.legend()
    ax.set_title('Robot Angular Momentum')
    ax.grid(True, alpha=0.3)

    # 6. Torso tracking error
    ax = axes[2, 1]
    ax.plot(t_l, np.array(log_lutze['e_torso_pos']) * 100, 'b-', label='Lutze')
    ax.plot(t_m, np.array(log_mpc['e_torso_pos']) * 100, 'r--', label='MPC')
    ax.set_ylabel('||e_torso|| [cm]')
    ax.set_xlabel('Time [s]')
    ax.legend()
    ax.set_title('Torso Tracking Error')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python -m lutze_baseline.plot_comparison <log1.json> [log2.json]")
        sys.exit(1)

    log1 = _load(sys.argv[1])
    if len(sys.argv) >= 3:
        log2 = _load(sys.argv[2])
        plot_comparison(log1, log2)
    else:
        plot_results(log1)
