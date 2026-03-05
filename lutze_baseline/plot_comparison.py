"""Comparative plots: Lutze baseline vs MPC controller.

Generates a 6-panel figure comparing logged data from both controllers.
Accepts JSON log files from sim_lutze.py and the MPC's simulation_loop.

Usage:
    python -m lutze_baseline.plot_comparison sim_lutze_log.json sim_mpc_log.json
"""

import argparse
import json
import sys
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def load_log(path):
    """Load a simulation log JSON."""
    with open(path) as f:
        return json.load(f)


def plot_comparison(log_lutze, log_mpc=None, L_max=5.0, tau_max=10.0,
                    weld_radius=0.005, save_path=None):
    """Generate 6-panel comparison figure.

    Parameters
    ----------
    log_lutze : dict - Lutze baseline log.
    log_mpc : dict or None - MPC log (omit for Lutze-only plot).
    L_max : float - Angular momentum limit [Nms].
    tau_max : float - Joint torque limit [Nm].
    weld_radius : float - Dock threshold [m].
    save_path : str or None - Save path for figure.
    """
    assert plt is not None, "matplotlib required"

    t_l = np.array(log_lutze['t'])
    phase_l = log_lutze['phase']

    def shade(ax, t, phase):
        for i in range(len(t)):
            if phase[i] == 'DS':
                ax.axvspan(t[i] - 0.04, t[i] + 0.04,
                           alpha=0.08, color='blue')
            elif phase[i] == 'EXT':
                ax.axvspan(t[i] - 0.04, t[i] + 0.04,
                           alpha=0.08, color='red')

    fig, axes = plt.subplots(6, 1, figsize=(14, 24), sharex=True)
    has_mpc = log_mpc is not None

    title = 'Lutze Baseline'
    if has_mpc:
        title += ' vs MPC'
    title += f' | $L_{{max}}$={L_max} Nms, $\\tau_j$={tau_max} Nm'
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # ── Panel 1: Gripper-to-anchor distance ──────────────────────────────
    ax = axes[0]
    shade(ax, t_l, phase_l)
    d_l = np.array(log_lutze['d_grip_swing']) * 100  # cm
    ax.semilogy(t_l, d_l, 'r-', lw=2.5, label='Lutze')
    if has_mpc:
        t_m = np.array(log_mpc['t'])
        d_m = np.array(log_mpc['d_grip_swing']) * 100
        ax.semilogy(t_m, d_m, 'b-', lw=2, alpha=0.7, label='MPC')
    ax.axhline(weld_radius * 100, color='g', ls='--', lw=2,
               label=f'dock {weld_radius*1000:.0f}mm')
    ax.set_ylabel('Distance [cm] (log)')
    ax.set_title('① EE → Anchor Distance')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim([0.1, 200])

    # ── Panel 2: CoM tracking ────────────────────────────────────────────
    ax = axes[1]
    shade(ax, t_l, phase_l)
    rcom_l = np.array(log_lutze['r_com'])
    rcom_ref_l = np.array(log_lutze['r_com_ref'])
    ecom_l = np.array(log_lutze['e_com']) * 100

    ax.plot(t_l, rcom_l[:, 0] * 100, 'r-', lw=2, label='Lutze CoM_x')
    ax.plot(t_l, rcom_ref_l[:, 0] * 100, 'r--', lw=1.5, alpha=0.5,
            label='ref')
    ax.plot(t_l, ecom_l, 'k-', lw=1.5, label='||e_com|| Lutze')
    if has_mpc:
        t_m = np.array(log_mpc['t'])
        ecom_m = np.array(log_mpc['e_com']) * 100
        ax.plot(t_m, ecom_m, 'b-', lw=1.5, alpha=0.7,
                label='||e_com|| MPC')
    ax.set_ylabel('[cm]')
    ax.set_title('② CoM Tracking')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Angular momentum ────────────────────────────────────────
    ax = axes[2]
    shade(ax, t_l, phase_l)
    Lcom_l = np.array(log_lutze['L_com'])
    Lnorm_l = np.array(log_lutze['L_com_norm'])

    ax.plot(t_l, Lcom_l[:, 0], 'r-', lw=1.2, alpha=0.6, label='$L_x$ Lutze')
    ax.plot(t_l, Lcom_l[:, 1], 'g-', lw=1.2, alpha=0.6, label='$L_y$')
    ax.plot(t_l, Lcom_l[:, 2], 'b-', lw=1.2, alpha=0.6, label='$L_z$')
    ax.plot(t_l, Lnorm_l, 'r-', lw=2.5, label='$||L||$ Lutze')

    if has_mpc:
        t_m = np.array(log_mpc['t'])
        Lnorm_m = np.array(log_mpc['L_com_norm'])
        ax.plot(t_m, Lnorm_m, 'b-', lw=2.5, label='$||L||$ MPC')

    ax.axhline(L_max, color='r', ls='--', lw=2)
    ax.axhline(-L_max, color='r', ls='--', lw=2)
    ax.fill_between(t_l, -L_max, L_max, alpha=0.05, color='green')
    ax.set_ylabel('[Nms]')
    ax.set_title('③ Robot Angular Momentum')
    ax.legend(fontsize=9, ncol=3)
    ax.grid(True, alpha=0.3)

    # ── Panel 4: Joint torques ───────────────────────────────────────────
    ax = axes[3]
    shade(ax, t_l, phase_l)
    tau_l = np.array(log_lutze['tau'])
    tau_max_l = np.array(log_lutze['tau_max_joint'])

    for j in range(min(tau_l.shape[1], 12)):
        c = 'C0' if j < 6 else 'C1'
        ax.plot(t_l, tau_l[:, j], '-', color=c, alpha=0.2, lw=0.8)
    ax.plot(t_l, tau_max_l, 'r-', lw=2, label='max |τ| Lutze')

    if has_mpc:
        t_m = np.array(log_mpc['t'])
        tau_max_m = np.array(log_mpc['tau_max_joint'])
        ax.plot(t_m, tau_max_m, 'b-', lw=2, label='max |τ| MPC')

    ax.axhline(tau_max, color='r', ls='--', lw=1.5)
    ax.axhline(-tau_max, color='r', ls='--', lw=1.5)
    ax.set_ylabel('[Nm]')
    ax.set_title('④ Joint Torques')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Panel 5: Structure drift ─────────────────────────────────────────
    ax = axes[4]
    shade(ax, t_l, phase_l)
    sp_l = np.array(log_lutze['struct_pos'])
    sd_l = np.linalg.norm(sp_l - sp_l[0], axis=1) * 100
    ax.plot(t_l, sd_l, 'r-', lw=2, label='Lutze')

    if has_mpc:
        t_m = np.array(log_mpc['t'])
        sp_m = np.array(log_mpc['struct_pos'])
        sd_m = np.linalg.norm(sp_m - sp_m[0], axis=1) * 100
        ax.plot(t_m, sd_m, 'b-', lw=2, label='MPC')

    ax.set_ylabel('[cm]')
    ax.set_title('⑤ Structure Translation Drift')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Panel 6: Structure rotation ──────────────────────────────────────
    ax = axes[5]
    shade(ax, t_l, phase_l)
    euler_l = np.array(log_lutze['struct_euler_deg'])
    ax.plot(t_l, euler_l[:, 0], 'r-', lw=1.5, alpha=0.7, label='roll Lutze')
    ax.plot(t_l, euler_l[:, 1], 'g-', lw=1.5, alpha=0.7, label='pitch')
    ax.plot(t_l, euler_l[:, 2], 'b-', lw=1.5, alpha=0.7, label='yaw')
    ax.plot(t_l, np.max(np.abs(euler_l), axis=1), 'r-', lw=2.5,
            label='max |angle| Lutze')

    if has_mpc:
        t_m = np.array(log_mpc['t'])
        euler_m = np.array(log_mpc['struct_euler_deg'])
        ax.plot(t_m, np.max(np.abs(euler_m), axis=1), 'b-', lw=2.5,
                label='max |angle| MPC')

    ax.set_ylabel('[deg]')
    ax.set_xlabel('Time [s]')
    ax.set_title('⑥ Structure Orientation (Euler)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Compare Lutze baseline vs MPC simulation logs')
    parser.add_argument('lutze_log', help='Lutze baseline log JSON')
    parser.add_argument('mpc_log', nargs='?', default=None,
                        help='MPC log JSON (optional)')
    parser.add_argument('--output', default='comparison.png',
                        help='Output figure path')
    parser.add_argument('--L-max', type=float, default=5.0)
    parser.add_argument('--tau-max', type=float, default=10.0)
    args = parser.parse_args()

    log_lutze = load_log(args.lutze_log)
    log_mpc = load_log(args.mpc_log) if args.mpc_log else None

    plot_comparison(log_lutze, log_mpc,
                    L_max=args.L_max, tau_max=args.tau_max,
                    save_path=args.output)


if __name__ == '__main__':
    main()
