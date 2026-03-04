"""Visualization plots.

Translated from plot_satellite_rotation.m, plot_contact_wrenches.m,
plot_tracking_error.m, plot_momentum_saturation.m.
"""

import os
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def plot_satellite_rotation(results_no_opt, results_opt, exp_id, env,
                             save_dir='results/figures'):
    """Plot satellite Euler angles (alpha, beta, gamma)."""
    if not HAS_MPL:
        print("  matplotlib not available, skipping rotation plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    labels = [('alpha', r'$\alpha$ [deg]', 'Roll Angle'),
              ('beta', r'$\beta$ [deg]', 'Pitch Angle (Critical)'),
              ('gamma', r'$\gamma$ [deg]', 'Yaw Angle')]

    for ax, (key, ylabel, title) in zip(axes, labels):
        ax.plot(results_no_opt['t'], np.rad2deg(results_no_opt[key]),
                'r--', linewidth=1.5, label='Non-optimized')
        ax.plot(results_opt['t'], np.rad2deg(results_opt[key]),
                'b-', linewidth=1.5, label='Optimized')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        ax.set_xlim([0, results_no_opt['t'][-1]])

    # Beta annotation
    beta_max_no = np.rad2deg(np.max(np.abs(results_no_opt['beta'])))
    beta_max_opt = np.rad2deg(np.max(np.abs(results_opt['beta'])))
    axes[1].text(0.5, 0.95, f'|beta|_max no-opt: {beta_max_no:.3f} deg',
                 transform=axes[1].transAxes, fontsize=9, color='r')
    axes[1].text(0.5, 0.85, f'|beta|_max opt: {beta_max_opt:.4f} deg',
                 transform=axes[1].transAxes, fontsize=9, color='b')

    fig.suptitle(f'Experiment #{exp_id}: Satellite Attitude Angles',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()

    path = os.path.join(save_dir, f'exp{exp_id}_rotation.png')
    _ensure_dir(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Rotation plot saved: {path}")


def plot_contact_wrenches(results_no_opt, results_opt, exp_id,
                           save_dir='results/figures'):
    """Plot contact forces and moments."""
    if not HAS_MPL:
        print("  matplotlib not available, skipping wrenches plot")
        return

    fig, axes = plt.subplots(2, 1, figsize=(15, 8))

    # Forces
    axes[0].plot(results_no_opt['t'],
                 np.linalg.norm(results_no_opt['Fc'][:3, :], axis=0),
                 'r--', linewidth=1.5, label='No-opt ||F||')
    axes[0].plot(results_opt['t'],
                 np.linalg.norm(results_opt['Fc'][:3, :], axis=0),
                 'b-', linewidth=1.5, label='Opt ||F||')
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Contact Force [N]')
    axes[0].set_title('Contact Force Magnitude')
    axes[0].legend()
    axes[0].grid(True)

    # Moments
    axes[1].plot(results_no_opt['t'],
                 np.linalg.norm(results_no_opt['Fc'][3:, :], axis=0),
                 'r--', linewidth=1.5, label=r'No-opt ||$\tau$||')
    axes[1].plot(results_opt['t'],
                 np.linalg.norm(results_opt['Fc'][3:, :], axis=0),
                 'b-', linewidth=1.5, label=r'Opt ||$\tau$||')
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Contact Moment [Nm]')
    axes[1].set_title('Contact Moment Magnitude')
    axes[1].legend()
    axes[1].grid(True)

    fig.suptitle(f'Experiment #{exp_id}: Contact Wrenches',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()

    path = os.path.join(save_dir, f'exp{exp_id}_wrenches.png')
    _ensure_dir(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Wrenches plot saved: {path}")


def plot_tracking_error(results_no_opt, results_opt, exp_id,
                         save_dir='results/figures'):
    """Plot robot CoM tracking error."""
    if not HAS_MPL:
        print("  matplotlib not available, skipping tracking plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    error_no = np.linalg.norm(results_no_opt['pos_error'], axis=0)
    error_opt = np.linalg.norm(results_opt['pos_error'], axis=0)

    # Norm
    axes[0].plot(results_no_opt['t'], error_no * 1000,
                 'r--', linewidth=1.5, label='Non-optimized')
    axes[0].plot(results_opt['t'], error_opt * 1000,
                 'b-', linewidth=1.5, label='Optimized')
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Position Error [mm]')
    axes[0].set_title('CoM Tracking Error')
    axes[0].legend()
    axes[0].grid(True)

    rms_no = np.sqrt(np.mean(error_no**2))
    rms_opt = np.sqrt(np.mean(error_opt**2))
    axes[0].text(0.6, 0.9, f'RMS no-opt: {rms_no*1000:.2f} mm',
                 transform=axes[0].transAxes, fontsize=9, color='r')
    axes[0].text(0.6, 0.8, f'RMS opt: {rms_opt*1000:.2f} mm',
                 transform=axes[0].transAxes, fontsize=9, color='b')

    # Per axis (optimized)
    axes[1].plot(results_opt['t'], results_opt['pos_error'][0, :] * 1000,
                 'r-', linewidth=1.2, label='e_x')
    axes[1].plot(results_opt['t'], results_opt['pos_error'][1, :] * 1000,
                 'g-', linewidth=1.2, label='e_y')
    axes[1].plot(results_opt['t'], results_opt['pos_error'][2, :] * 1000,
                 'b-', linewidth=1.2, label='e_z')
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Position Error [mm]')
    axes[1].set_title('Tracking Error per Axis (Optimized)')
    axes[1].legend()
    axes[1].grid(True)

    fig.suptitle(f'Experiment #{exp_id}: Tracking Performance',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()

    path = os.path.join(save_dir, f'exp{exp_id}_tracking.png')
    _ensure_dir(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Tracking plot saved: {path}")


def plot_momentum_saturation(results_no_opt, results_opt, exp_id, env,
                              save_dir='results/figures'):
    """Plot reaction wheel momentum and saturation analysis."""
    if not HAS_MPL:
        print("  matplotlib not available, skipping momentum plot")
        return

    h_max = env['RW']['h_total_max']

    h_norm_no = np.linalg.norm(results_no_opt['h_RW'], axis=0)
    h_norm_opt = np.linalg.norm(results_opt['h_RW'], axis=0)

    sat_no = h_norm_no / h_max * 100
    sat_opt = h_norm_opt / h_max * 100

    fig, axes = plt.subplots(2, 1, figsize=(15, 8))

    # Momentum
    axes[0].plot(results_no_opt['t'], h_norm_no, 'r--', linewidth=1.5,
                 label='No-opt ||h_RW||')
    axes[0].plot(results_opt['t'], h_norm_opt, 'b-', linewidth=1.5,
                 label='Opt ||h_RW||')
    axes[0].axhline(y=h_max, color='k', linestyle='--', linewidth=2, label='h_max')
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Angular Momentum [Nms]')
    axes[0].set_title('Reaction Wheel Momentum Storage')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_ylim([0, h_max * 1.1])

    # Saturation %
    axes[1].plot(results_no_opt['t'], sat_no, 'r--', linewidth=1.5,
                 label='Non-optimized')
    axes[1].plot(results_opt['t'], sat_opt, 'b-', linewidth=1.5,
                 label='Optimized')
    axes[1].axhline(y=100, color='k', linestyle='--', linewidth=2, label='Saturation')
    axes[1].axhline(y=90, color='r', linestyle=':', linewidth=1.5, label='Critical (90%)')
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Saturation [%]')
    axes[1].set_title('RW Saturation Level')
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_ylim([0, 110])

    sat_max_no = np.max(sat_no)
    sat_max_opt = np.max(sat_opt)
    axes[1].text(0.05, 0.9, f'Max sat. no-opt: {sat_max_no:.1f}%',
                 transform=axes[1].transAxes, fontsize=10, color='r', fontweight='bold')
    axes[1].text(0.05, 0.8, f'Max sat. opt: {sat_max_opt:.1f}%',
                 transform=axes[1].transAxes, fontsize=10, color='b', fontweight='bold')

    fig.suptitle(f'Experiment #{exp_id}: Reaction Wheel Saturation Analysis',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()

    path = os.path.join(save_dir, f'exp{exp_id}_momentum_saturation.png')
    _ensure_dir(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Momentum saturation plot saved: {path}")
