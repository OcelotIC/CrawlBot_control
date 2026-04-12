"""9-panel diagnostic plot for simulation results.

Separated from the simulation loop so plotting can be done offline
from saved SimLog JSON files without importing MuJoCo/Pinocchio.
"""

import numpy as np


def plot_simulation(log, save_path=None, cfg=None):
    """Generate 9-panel diagnostic plot.

    Parameters
    ----------
    log : SimLog
        Simulation log data.
    cfg : SimConfig, optional
        Config for limit lines. Uses defaults if None.
    save_path : str, optional
        If provided, saves the figure to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t = np.array(log.t)
    pt = np.array(log.p_torso)
    pt_ref = np.array(log.p_torso_ref)
    d = np.array(log.d_grip_swing)
    tau = np.array(log.tau)
    ecom = np.array(log.e_com)
    rcom = np.array(log.r_com)
    rcom_ref = np.array(log.r_com_ref)
    Lcom = np.array(log.L_com)
    Lnorm = np.array(log.L_com_norm)
    sp = np.array(log.struct_pos)
    euler = np.array(log.struct_euler_deg)
    ph = log.phase

    L_max = cfg.L_max if cfg else 5.0
    tw = cfg.tau_w_max if cfg else 2.0
    wr = cfg.weld_radius if cfg else 0.005
    tm = cfg.tau_max if cfg else 10.0

    def shade(ax):
        """Shade DS (blue) phases; SS is unshaded. (M7: two-phase.)"""
        for i in range(len(t)):
            if ph[i] == 'DS':
                ax.axvspan(t[i] - .04, t[i] + .04, alpha=.08, color='blue')
        for i in range(1, len(ph)):
            if ph[i] != ph[i - 1]:
                ax.axvline(t[i], color='gray', ls=':', alpha=.5)

    fig, axes = plt.subplots(9, 1, figsize=(14, 36), sharex=True)
    nd = len(log.dock_events)
    fig.suptitle(
        f'VISPA — $L_{{max}}$={L_max}, $\\tau_w$={tw} Nm, '
        f'$\\tau_j$={tm} Nm | {nd} dock(s)',
        fontsize=14, fontweight='bold')

    # 1 — EE distance to anchor
    ax = axes[0]; shade(ax)
    ax.semilogy(t, d * 100, 'r-', lw=2.5, label='||grip-anchor||')
    ax.axhline(wr * 100, color='g', ls='--', lw=2, label=f'seuil {wr * 1000:.0f}mm')
    for ev in log.dock_events:
        ax.axvline(ev['t'], color='green', ls='-', lw=2, alpha=.4)
    ax.set_ylabel('Distance [cm] (log)')
    ax.set_title('1. Distance EE - ancre')
    ax.legend(fontsize=9); ax.grid(True, alpha=.3, which='both')
    ax.set_ylim([0.1, 200])

    # 2 — Torso X advance
    ax = axes[1]; shade(ax)
    ax.plot(t, pt[:, 0] * 100, 'r-', lw=2.5, label='torse x')
    ax.plot(t, pt_ref[:, 0] * 100, 'r--', lw=1.5, alpha=.5, label='ref')
    ax.set_ylabel('[cm]'); ax.set_title('2. Avancement torse')
    ax.legend(fontsize=9); ax.grid(True, alpha=.3)

    # 3 — CoM tracking
    ax = axes[2]; shade(ax)
    ax.plot(t, rcom[:, 0] * 100, 'r-', lw=2, label='CoM x')
    ax.plot(t, rcom_ref[:, 0] * 100, 'r--', lw=1.5, alpha=.6, label='ref')
    ax.plot(t, ecom * 100, 'k-', lw=2, label='||e_com||')
    ax.set_ylabel('[cm]'); ax.set_title('3. Suivi CoM')
    ax.legend(fontsize=9); ax.grid(True, alpha=.3)

    # 4 — Angular momentum
    ax = axes[3]; shade(ax)
    ax.plot(t, Lcom[:, 0], 'r-', lw=1.5, alpha=.7, label='$L_x$')
    ax.plot(t, Lcom[:, 1], 'g-', lw=1.5, alpha=.7, label='$L_y$')
    ax.plot(t, Lcom[:, 2], 'b-', lw=1.5, alpha=.7, label='$L_z$')
    ax.plot(t, Lnorm, 'k-', lw=2.5, label='$||L||$')
    ax.axhline(L_max, color='r', ls='--', lw=2)
    ax.axhline(-L_max, color='r', ls='--', lw=2)
    ax.fill_between(t, -L_max, L_max, alpha=.05, color='green')
    ax.set_ylabel('[Nms]'); ax.set_title('4. Moment cinetique robot')
    ax.legend(fontsize=9, ncol=3); ax.grid(True, alpha=.3)

    # 5 — Joint torques
    ax = axes[4]; shade(ax)
    for j in range(6):
        ax.plot(t, tau[:, j], '-', color='C0', alpha=.3, lw=1)
    for j in range(6, 12):
        ax.plot(t, tau[:, j], '-', color='C1', alpha=.3, lw=1)
    ax.plot(t, np.max(np.abs(tau), axis=1), 'k-', lw=2, label='max |tau|')
    ax.axhline(tm, color='r', ls='--', lw=1.5)
    ax.axhline(-tm, color='r', ls='--', lw=1.5)
    ax.set_ylabel('[Nm]'); ax.set_title('5. Couples articulaires')
    ax.legend(fontsize=9); ax.grid(True, alpha=.3)

    # 6 — Structure drift
    ax = axes[5]; shade(ax)
    sd = np.linalg.norm(sp - sp[0], axis=1) * 100
    ax.plot(t, sd, 'k-', lw=2)
    ax.set_ylabel('[cm]'); ax.set_title('6. Derive structure (translation)')
    ax.grid(True, alpha=.3)

    # 7 — Structure rotation
    ax = axes[6]; shade(ax)
    ax.plot(t, euler[:, 0], 'r-', lw=1.5, label='roll')
    ax.plot(t, euler[:, 1], 'g-', lw=1.5, label='pitch')
    ax.plot(t, euler[:, 2], 'b-', lw=1.5, label='yaw')
    ax.plot(t, np.max(np.abs(euler), axis=1), 'k-', lw=2, label='max |angle|')
    ax.set_ylabel('[deg]'); ax.set_title('7. Orientation structure (Euler)')
    ax.legend(fontsize=9); ax.grid(True, alpha=.3)

    # 8 — Torso position error
    e_pos_vec = (pt - pt_ref) * 100
    e_pos_norm = np.array(log.e_torso_pos) * 100
    ax = axes[7]; shade(ax)
    ax.plot(t, e_pos_vec[:, 0], 'r-', lw=1.2, alpha=.7, label='$e_x$')
    ax.plot(t, e_pos_vec[:, 1], 'g-', lw=1.2, alpha=.7, label='$e_y$')
    ax.plot(t, e_pos_vec[:, 2], 'b-', lw=1.2, alpha=.7, label='$e_z$')
    ax.plot(t, e_pos_norm, 'k-', lw=2.5, label='$\\|e_{pos}\\|$')
    ax.set_ylabel('[cm]'); ax.set_title('8. Erreur tracking torso — position')
    ax.legend(fontsize=9, ncol=4); ax.grid(True, alpha=.3)

    # 9 — Torso orientation error
    e_ori = np.array(log.e_torso_ori) if log.e_torso_ori else np.zeros(len(t))
    ax = axes[8]; shade(ax)
    ax.plot(t, e_ori, 'b-', lw=2.5)
    ax.set_ylabel('[deg]'); ax.set_xlabel('Time [s]')
    ax.set_title('9. Erreur tracking torso — orientation (angle geodesique)')
    ax.grid(True, alpha=.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
