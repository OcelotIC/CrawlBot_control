#!/usr/bin/env python3
"""R7 — Generate publication-quality figures for the VISPA paper.

Produces:
    1. fig1_single_step_comparison.pdf  — Lutze vs MPC (single step)
    2. fig2_multistep_locomotion.pdf    — 3-step closed-loop results
    3. fig3_momentum_comparison.pdf     — Angular momentum across controllers

Requires:
    - sim_torso6d_log.json   (single-step MPC)
    - sim_lutze_log.json     (single-step Lutze baseline)
    - r6_multistep_log.json  (multi-step MPC)

Usage:
    MUJOCO_GL=disabled python run_r7_figures.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ─── Publication style ────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'lines.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

COL_MPC = '#d62728'    # red
COL_LUTZE = '#1f77b4'  # blue
COL_MULTI = '#2ca02c'  # green
COL_LIMIT = '#333333'  # dark gray for constraints


def load_json(path):
    with open(path) as f:
        d = json.load(f)
    return d


# ═════════════════════════════════════════════════════════════════════
#  Figure 1 — Single-step comparison: Lutze vs MPC
# ═════════════════════════════════════════════════════════════════════

def fig1_single_step(log_mpc, log_lutze, save_path='fig1_single_step_comparison.pdf'):
    t_m = np.array(log_mpc['t'])
    t_l = np.array(log_lutze['t'])

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.5))

    # (a) Docking convergence
    ax = axes[0, 0]
    ax.semilogy(t_l, np.array(log_lutze['d_grip_b']) * 1000,
                color=COL_LUTZE, label='Lutze')
    ax.semilogy(t_m, np.array(log_mpc['d_grip_b']) * 1000,
                color=COL_MPC, ls='--', label='CNMPC')
    ax.axhline(5.0, color=COL_LIMIT, ls=':', lw=0.8, label='Dock threshold (5 mm)')
    ax.set_ylabel('Gripper distance [mm]')
    ax.set_title('(a) Docking convergence')
    ax.legend(loc='upper right')
    ax.set_ylim([1, 500])

    # (b) Angular momentum
    ax = axes[0, 1]
    L_l = np.linalg.norm(np.array(log_lutze['L_com']), axis=1)
    L_m = np.linalg.norm(np.array(log_mpc['L_com']), axis=1)
    ax.plot(t_l[:len(L_l)], L_l, color=COL_LUTZE, label='Lutze')
    ax.plot(t_m[:len(L_m)], L_m, color=COL_MPC, ls='--', label='CNMPC')
    ax.axhline(5.0, color=COL_LIMIT, ls=':', lw=0.8, label='$L_{max}$ = 5 Nms')
    ax.fill_between([t_m[0], max(t_l[-1], t_m[-1])], 0, 5.0,
                    alpha=0.05, color='green')
    ax.set_ylabel('$\\|\\mathbf{L}_{com}\\|$ [Nms]')
    ax.set_title('(b) Angular momentum')
    ax.legend(loc='upper right')

    # (c) Structure drift
    ax = axes[1, 0]
    sp_l = np.array(log_lutze['struct_pos'])
    sp_m = np.array(log_mpc['struct_pos'])
    drift_l = np.linalg.norm(sp_l - sp_l[0], axis=1) * 100
    drift_m = np.linalg.norm(sp_m - sp_m[0], axis=1) * 100
    ax.plot(t_l, drift_l, color=COL_LUTZE, label='Lutze')
    ax.plot(t_m, drift_m, color=COL_MPC, ls='--', label='CNMPC')
    ax.set_ylabel('Structure drift [cm]')
    ax.set_xlabel('Time [s]')
    ax.set_title('(c) Structure drift')
    ax.legend(loc='upper left')

    # (d) Torso tracking error
    ax = axes[1, 1]
    ax.plot(t_l, np.array(log_lutze['e_torso_pos']) * 100,
            color=COL_LUTZE, label='Lutze')
    ax.plot(t_m, np.array(log_mpc['e_torso_pos']) * 100,
            color=COL_MPC, ls='--', label='CNMPC')
    ax.set_ylabel('Torso error [cm]')
    ax.set_xlabel('Time [s]')
    ax.set_title('(d) Torso tracking error')
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.savefig(save_path.replace('.pdf', '.png'))
    print(f'  Saved: {save_path}')
    plt.close()


# ═════════════════════════════════════════════════════════════════════
#  Figure 2 — Multi-step locomotion (3 steps)
# ═════════════════════════════════════════════════════════════════════

def fig2_multistep(log, save_path='fig2_multistep_locomotion.pdf'):
    t = np.array(log['t'])
    ph = log['phase']
    Lcom = np.array(log['L_com'])
    Lnorm = np.array(log['L_com_norm'])
    d = np.array(log['d_grip_swing'])
    tau = np.array(log['tau'])
    sp = np.array(log['struct_pos'])
    euler = np.array(log['struct_euler_deg'])
    dock_events = log['dock_events']

    L_max = 5.0
    tau_max = 10.0

    def shade(ax):
        """Phase shading: blue=DS, red=EXT."""
        for i in range(len(t)):
            if ph[i] == 'DS':
                ax.axvspan(t[i] - 0.04, t[i] + 0.04,
                           alpha=0.08, color='blue')
            elif ph[i] == 'EXT':
                ax.axvspan(t[i] - 0.04, t[i] + 0.04,
                           alpha=0.08, color='red')
        # Phase transition lines
        for i in range(1, len(ph)):
            if ph[i] != ph[i - 1]:
                ax.axvline(t[i], color='gray', ls=':', alpha=0.4, lw=0.5)

    fig, axes = plt.subplots(4, 1, figsize=(7.0, 8.0), sharex=True)

    # (a) Docking convergence
    ax = axes[0]
    shade(ax)
    ax.semilogy(t, d * 100, color=COL_MPC, lw=1.5)
    ax.axhline(0.5, color='green', ls='--', lw=1.0,
               label='Dock threshold (5 mm)')
    for ev in dock_events:
        ax.axvline(ev['t'], color='green', ls='-', lw=1.5, alpha=0.4)
        ax.annotate(f'{ev["d_mm"]:.1f} mm',
                    xy=(ev['t'], 0.5), xytext=(ev['t'] + 0.5, 0.15),
                    fontsize=7, color='green',
                    arrowprops=dict(arrowstyle='->', color='green', lw=0.8))
    ax.set_ylabel('EE $\\rightarrow$ anchor [cm]')
    ax.set_title('(a) End-effector docking distance')
    ax.set_ylim([0.1, 200])
    ax.legend(loc='upper right', fontsize=7)

    # (b) Angular momentum
    ax = axes[1]
    shade(ax)
    ax.plot(t, Lcom[:, 0], color='C0', lw=0.8, alpha=0.7, label='$L_x$')
    ax.plot(t, Lcom[:, 1], color='C2', lw=0.8, alpha=0.7, label='$L_y$')
    ax.plot(t, Lcom[:, 2], color='C1', lw=0.8, alpha=0.7, label='$L_z$')
    ax.plot(t, Lnorm, 'k-', lw=1.5, label='$\\|\\mathbf{L}_{com}\\|$')
    ax.axhline(L_max, color=COL_LIMIT, ls='--', lw=1.0)
    ax.axhline(-L_max, color=COL_LIMIT, ls='--', lw=1.0)
    ax.fill_between(t, -L_max, L_max, alpha=0.04, color='green')
    ax.set_ylabel('$\\mathbf{L}_{com}$ [Nms]')
    ax.set_title('(b) Robot angular momentum')
    ax.legend(loc='upper right', ncol=4, fontsize=7)

    # (c) Joint torques
    ax = axes[2]
    shade(ax)
    for j in range(6):
        ax.plot(t, tau[:, j], '-', color='C0', alpha=0.25, lw=0.6)
    for j in range(6, 12):
        ax.plot(t, tau[:, j], '-', color='C1', alpha=0.25, lw=0.6)
    ax.plot(t, np.max(np.abs(tau), axis=1), 'k-', lw=1.2,
            label='max $|\\tau_i|$')
    ax.axhline(tau_max, color=COL_LIMIT, ls='--', lw=1.0)
    ax.axhline(-tau_max, color=COL_LIMIT, ls='--', lw=1.0)
    # Arm labels
    ax.plot([], [], '-', color='C0', lw=1.5, label='Arm A')
    ax.plot([], [], '-', color='C1', lw=1.5, label='Arm B')
    ax.set_ylabel('$\\tau$ [Nm]')
    ax.set_title('(c) Joint torques')
    ax.legend(loc='upper right', ncol=3, fontsize=7)

    # (d) Structure attitude
    ax = axes[3]
    shade(ax)
    ax.plot(t, euler[:, 0], color='C3', lw=1.0, label='Roll')
    ax.plot(t, euler[:, 1], color='C2', lw=1.0, label='Pitch')
    ax.plot(t, euler[:, 2], color='C0', lw=1.0, label='Yaw')
    ax.set_ylabel('Angle [deg]')
    ax.set_xlabel('Time [s]')
    ax.set_title('(d) Structure attitude disturbance')
    ax.legend(loc='upper left', ncol=3, fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.savefig(save_path.replace('.pdf', '.png'))
    print(f'  Saved: {save_path}')
    plt.close()


# ═════════════════════════════════════════════════════════════════════
#  Figure 3 — Momentum comparison across controllers
# ═════════════════════════════════════════════════════════════════════

def fig3_momentum(log_mpc, log_lutze, log_multi,
                  save_path='fig3_momentum_comparison.pdf'):
    t_m = np.array(log_mpc['t'])
    t_l = np.array(log_lutze['t'])
    t_multi = np.array(log_multi['t'])

    L_m = np.linalg.norm(np.array(log_mpc['L_com']), axis=1)
    L_l = np.linalg.norm(np.array(log_lutze['L_com']), axis=1)
    L_multi = np.array(log_multi['L_com_norm'])

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

    # (a) Single-step comparison
    ax = axes[0]
    ax.plot(t_l[:len(L_l)], L_l, color=COL_LUTZE, label='Lutze')
    ax.plot(t_m[:len(L_m)], L_m, color=COL_MPC, ls='--', label='CNMPC')
    ax.axhline(5.0, color=COL_LIMIT, ls=':', lw=0.8)
    ax.fill_between([0, max(t_l[-1], t_m[-1])], 0, 5.0,
                    alpha=0.05, color='green')
    ax.set_ylabel('$\\|\\mathbf{L}_{com}\\|$ [Nms]')
    ax.set_xlabel('Time [s]')
    ax.set_title('(a) Single step')
    ax.legend(loc='upper right')
    ax.set_ylim(bottom=0)

    # (b) Multi-step
    ax = axes[1]
    ph = log_multi['phase']
    for i in range(len(t_multi)):
        if ph[i] == 'DS':
            ax.axvspan(t_multi[i] - 0.04, t_multi[i] + 0.04,
                       alpha=0.08, color='blue')
        elif ph[i] == 'EXT':
            ax.axvspan(t_multi[i] - 0.04, t_multi[i] + 0.04,
                       alpha=0.08, color='red')
    ax.plot(t_multi, L_multi, color=COL_MULTI, lw=1.2)
    ax.axhline(5.0, color=COL_LIMIT, ls=':', lw=0.8,
               label='$L_{max}$ = 5 Nms')
    for ev in log_multi['dock_events']:
        ax.axvline(ev['t'], color='green', ls='-', lw=1.0, alpha=0.4)
    ax.set_ylabel('$\\|\\mathbf{L}_{com}\\|$ [Nms]')
    ax.set_xlabel('Time [s]')
    ax.set_title('(b) Multi-step (3 steps)')
    ax.legend(loc='upper right')
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.savefig(save_path.replace('.pdf', '.png'))
    print(f'  Saved: {save_path}')
    plt.close()


# ═════════════════════════════════════════════════════════════════════
#  Summary table (LaTeX-ready)
# ═════════════════════════════════════════════════════════════════════

def print_latex_table(log_mpc, log_lutze, log_multi):
    L_m = np.linalg.norm(np.array(log_mpc['L_com']), axis=1)
    L_l = np.linalg.norm(np.array(log_lutze['L_com']), axis=1)
    L_multi = np.array(log_multi['L_com_norm'])

    sp_m = np.array(log_mpc['struct_pos'])
    sp_l = np.array(log_lutze['struct_pos'])
    sp_multi = np.array(log_multi['struct_pos'])
    drift_m = np.linalg.norm(sp_m[-1] - sp_m[0]) * 100
    drift_l = np.linalg.norm(sp_l[-1] - sp_l[0]) * 100
    drift_multi = np.linalg.norm(sp_multi[-1] - sp_multi[0]) * 100

    n_dock_multi = len(log_multi['dock_events'])
    dock_dists = [e['d_mm'] for e in log_multi['dock_events']]

    tau_m = max(log_mpc['tau_max_joint'])
    tau_l = max(log_lutze['tau_max_joint'])
    tau_multi = max(log_multi['tau_max_joint'])

    nmpc_fail = sum(1 for x in log_multi['nmpc_ok'] if not x)
    n_total = len(log_multi['nmpc_ok'])

    print('\n' + '=' * 70)
    print('  LaTeX Table (copy-paste ready)')
    print('=' * 70)
    print(r'\begin{table}[t]')
    print(r'\centering')
    print(r'\caption{Simulation results comparison.}')
    print(r'\label{tab:results}')
    print(r'\begin{tabular}{lccc}')
    print(r'\toprule')
    print(r'Metric & Lutze & CNMPC (1-step) & CNMPC (3-step) \\')
    print(r'\midrule')
    print(f'Docks achieved & 1/1 & 1/1 & {n_dock_multi}/3 \\\\')
    print(f'Dock distance [mm] & {min(np.array(log_lutze["d_grip_b"]))*1000:.1f}'
          f' & {min(np.array(log_mpc["d_grip_b"]))*1000:.1f}'
          f' & {", ".join(f"{d:.1f}" for d in dock_dists)} \\\\')
    print(f'Peak $\\|\\mathbf{{L}}_{{com}}\\|$ [Nms] & {max(L_l):.2f}'
          f' & {max(L_m):.2f} & {max(L_multi):.2f} \\\\')
    print(f'Mean $\\|\\mathbf{{L}}_{{com}}\\|$ [Nms] & {np.mean(L_l):.2f}'
          f' & {np.mean(L_m):.2f} & {np.mean(L_multi):.2f} \\\\')
    print(f'Peak $|\\tau_i|$ [Nm] & {tau_l:.1f}'
          f' & {tau_m:.1f} & {tau_multi:.1f} \\\\')
    print(f'Structure drift [cm] & {drift_l:.1f}'
          f' & {drift_m:.1f} & {drift_multi:.1f} \\\\')
    print(f'NMPC infeasibility [\\%] & --- & ---'
          f' & {nmpc_fail/n_total*100:.1f} \\\\')
    print(f'Sim duration [s] & {np.array(log_lutze["t"])[-1]:.1f}'
          f' & {np.array(log_mpc["t"])[-1]:.1f}'
          f' & {np.array(log_multi["t"])[-1]:.1f} \\\\')
    print(r'\bottomrule')
    print(r'\end{tabular}')
    print(r'\end{table}')


# ═════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('=' * 70)
    print('  R7 — Generating publication figures')
    print('=' * 70)

    log_mpc = load_json('sim_torso6d_log.json')
    log_lutze = load_json('sim_lutze_log.json')
    log_multi = load_json('r6_multistep_log.json')

    fig1_single_step(log_mpc, log_lutze)
    fig2_multistep(log_multi)
    fig3_momentum(log_mpc, log_lutze, log_multi)
    print_latex_table(log_mpc, log_lutze, log_multi)

    print('\n' + '=' * 70)
    print('  R7 complete. Figures:')
    print('    - fig1_single_step_comparison.pdf/.png')
    print('    - fig2_multistep_locomotion.pdf/.png')
    print('    - fig3_momentum_comparison.pdf/.png')
    print('=' * 70)
