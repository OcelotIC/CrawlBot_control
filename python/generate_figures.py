"""Generate paper results figures for all 3 experiments.

Produces for each experiment (opt vs no-opt):
  - Satellite orientation (alpha, beta, gamma)
  - Satellite angular velocity (omega_x, omega_y, omega_z)
  - Angular momentum breakdown (h_satellite, h_robot, h_RW, h_total)
"""

import sys
import os
import warnings
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from python.spart.robot_model import urdf2robot
from python.config.environment import setup_environment
from python.config.trajectories import setup_trajectories
from python.control.controller import LutzeQPController
from python.dynamics.system_state import compute_system_state
from python.dynamics.angular_momentum import compute_angular_momentum
from python.simulation.integrator import integrate_dynamics
from python.spart.attitude import quat_angles321


def run_experiment(robot, env, controller, exp_id, use_optimization):
    """Run full simulation, returning extended results with omega and momentum."""
    traj, sim_params = setup_trajectories(exp_id)
    N = len(traj['t'])

    state = {
        'q_base': traj['pos'][:, 0].copy(),
        'quat_base': np.array([0.0, 0.0, 0.0, 1.0]),
        'qm': np.deg2rad([15, -30, 60, 0, -30, 0, -15, -30, 60, 0, -30, 0]),
        'omega_base': np.zeros(3),
        'u_base': np.zeros(3),
        'um': np.zeros(robot['n_q']),
        'omega_satellite': np.zeros(3),
        'quat_satellite': np.array([0.0, 0.0, 0.0, 1.0]),
        'h_RW_stored': np.zeros(3),
    }

    results = {
        't': traj['t'],
        'alpha': np.zeros(N),
        'beta': np.zeros(N),
        'gamma': np.zeros(N),
        'omega_sat': np.zeros((3, N)),
        'h_satellite': np.zeros((3, N)),
        'h_robot': np.zeros((3, N)),
        'h_RW': np.zeros((3, N)),
        'h_total': np.zeros((3, N)),
        'Fc': np.zeros((6, N)),
        'pos_error': np.zeros((3, N)),
    }

    mode = "OPT" if use_optimization else "NO-OPT"
    print(f"  Exp #{exp_id} [{mode}]: running {N} steps...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in range(N - 1):
            sys = compute_system_state(state, robot, env)
            Fc, tau = controller.solve(sys, traj, k, use_optimization)
            state = integrate_dynamics(state, tau, Fc, sys, robot,
                                       sim_params['dt'], env)

            # Euler angles from satellite quaternion
            angles = quat_angles321(state['quat_satellite'])
            results['alpha'][k] = angles[0]
            results['beta'][k] = angles[1]
            results['gamma'][k] = angles[2]

            # Satellite angular velocity
            results['omega_sat'][:, k] = state['omega_satellite']

            # Angular momentum breakdown
            # Build a combined state dict for momentum computation
            mom_state = {**state, **sys}
            h_total, h_sat, h_rob, h_rw = compute_angular_momentum(mom_state, robot, env)
            results['h_satellite'][:, k] = h_sat
            results['h_robot'][:, k] = h_rob
            results['h_RW'][:, k] = state['h_RW_stored']
            results['h_total'][:, k] = h_total

            results['Fc'][:, k] = Fc
            results['pos_error'][:, k] = traj['pos'][:, k] - state['q_base']

            if (k + 1) % 1000 == 0:
                print(f"    {100*(k+1)/N:.0f}%")

    # Copy last
    for key in ['alpha', 'beta', 'gamma']:
        results[key][-1] = results[key][-2]
    for key in ['omega_sat', 'h_satellite', 'h_robot', 'h_RW', 'h_total', 'Fc', 'pos_error']:
        results[key][:, -1] = results[key][:, -2]

    print(f"  Exp #{exp_id} [{mode}]: done.")
    return results


def plot_orientation(res_no, res_opt, exp_id, save_dir):
    """Satellite Euler angles (alpha, beta, gamma)."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    t = res_no['t']

    labels = [('alpha', r'$\alpha$ (Roll)'),
              ('beta', r'$\beta$ (Pitch)'),
              ('gamma', r'$\gamma$ (Yaw)')]

    for ax, (key, lbl) in zip(axes, labels):
        ax.plot(t, np.rad2deg(res_no[key]), 'r--', lw=1.5, label='Non-optimized')
        ax.plot(t, np.rad2deg(res_opt[key]), 'b-', lw=1.5, label='Optimized')
        ax.set_ylabel(f'{lbl} [deg]')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        max_no = np.rad2deg(np.max(np.abs(res_no[key])))
        max_opt = np.rad2deg(np.max(np.abs(res_opt[key])))
        ax.set_title(f'{lbl}: max |no-opt|={max_no:.4f} deg, max |opt|={max_opt:.4f} deg',
                     fontsize=10)

    axes[-1].set_xlabel('Time [s]')
    fig.suptitle(f'Experiment #{exp_id} — Satellite Orientation', fontsize=14, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(save_dir, f'exp{exp_id}_orientation.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_angular_velocity(res_no, res_opt, exp_id, save_dir):
    """Satellite angular velocity components."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    t = res_no['t']
    comp_labels = [r'$\omega_x$', r'$\omega_y$', r'$\omega_z$']

    for i, (ax, lbl) in enumerate(zip(axes, comp_labels)):
        ax.plot(t, np.rad2deg(res_no['omega_sat'][i, :]), 'r--', lw=1.5, label='Non-optimized')
        ax.plot(t, np.rad2deg(res_opt['omega_sat'][i, :]), 'b-', lw=1.5, label='Optimized')
        ax.set_ylabel(f'{lbl} [deg/s]')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time [s]')
    fig.suptitle(f'Experiment #{exp_id} — Satellite Angular Velocity', fontsize=14, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(save_dir, f'exp{exp_id}_angular_velocity.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_momentum(res_no, res_opt, exp_id, save_dir):
    """Angular momentum breakdown: h_satellite, h_robot, h_RW, h_total."""
    fig, axes = plt.subplots(4, 2, figsize=(16, 14), sharex=True)
    t = res_no['t']

    mom_keys = ['h_satellite', 'h_robot', 'h_RW', 'h_total']
    mom_labels = ['Satellite $h_{sat}$', 'Robot $h_{rob}$',
                  'Reaction Wheels $h_{RW}$', 'Total $h_{total}$']
    comp = ['x', 'y', 'z']

    for row, (key, lbl) in enumerate(zip(mom_keys, mom_labels)):
        # Non-optimized (left column)
        for c in range(3):
            axes[row, 0].plot(t, res_no[key][c, :], lw=1.2, label=f'{comp[c]}')
        axes[row, 0].set_ylabel(f'{lbl} [Nms]')
        axes[row, 0].legend(loc='upper right', fontsize=8)
        axes[row, 0].grid(True, alpha=0.3)
        if row == 0:
            axes[row, 0].set_title('Non-optimized', fontsize=12, fontweight='bold')

        # Optimized (right column)
        for c in range(3):
            axes[row, 1].plot(t, res_opt[key][c, :], lw=1.2, label=f'{comp[c]}')
        axes[row, 1].set_ylabel(f'{lbl} [Nms]')
        axes[row, 1].legend(loc='upper right', fontsize=8)
        axes[row, 1].grid(True, alpha=0.3)
        if row == 0:
            axes[row, 1].set_title('Optimized', fontsize=12, fontweight='bold')

    axes[-1, 0].set_xlabel('Time [s]')
    axes[-1, 1].set_xlabel('Time [s]')
    fig.suptitle(f'Experiment #{exp_id} — Angular Momentum Breakdown',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(save_dir, f'exp{exp_id}_momentum.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_momentum_norm(res_no, res_opt, exp_id, env, save_dir):
    """Momentum norms comparison (opt vs no-opt on same plot)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    t = res_no['t']

    mom_keys = ['h_satellite', 'h_robot', 'h_RW', 'h_total']
    mom_labels = ['Satellite $||h_{sat}||$', 'Robot $||h_{rob}||$',
                  'RW $||h_{RW}||$', 'Total $||h_{total}||$']

    for idx, (key, lbl) in enumerate(zip(mom_keys, mom_labels)):
        ax = axes[idx // 2, idx % 2]
        norm_no = np.linalg.norm(res_no[key], axis=0)
        norm_opt = np.linalg.norm(res_opt[key], axis=0)
        ax.plot(t, norm_no, 'r--', lw=1.5, label='Non-optimized')
        ax.plot(t, norm_opt, 'b-', lw=1.5, label='Optimized')
        ax.set_ylabel(f'{lbl} [Nms]')
        ax.set_xlabel('Time [s]')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_title(lbl, fontsize=11)

        if key == 'h_RW':
            h_max = env['RW']['h_total_max']
            ax.axhline(y=h_max, color='k', ls='--', lw=1.5, label=f'h_max={h_max} Nms')
            ax.legend(loc='upper right')

    fig.suptitle(f'Experiment #{exp_id} — Momentum Norms Comparison',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(save_dir, f'exp{exp_id}_momentum_norms.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


if __name__ == '__main__':
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')
    os.makedirs(save_dir, exist_ok=True)

    # Setup
    print("Loading robot model...")
    urdf_path = os.path.join(os.path.dirname(__file__), '..', 'URDF_models', 'MAR_DualArm_6DoF.urdf')
    robot, _ = urdf2robot(urdf_path)
    env = setup_environment()
    controller = LutzeQPController(env)

    for exp_id in [1, 2, 3]:
        print(f"\n{'='*60}")
        print(f"EXPERIMENT #{exp_id}")
        print(f"{'='*60}")

        res_no = run_experiment(robot, env, controller, exp_id, use_optimization=False)
        res_opt = run_experiment(robot, env, controller, exp_id, use_optimization=True)

        print(f"\n  Generating figures...")
        plot_orientation(res_no, res_opt, exp_id, save_dir)
        plot_angular_velocity(res_no, res_opt, exp_id, save_dir)
        plot_momentum(res_no, res_opt, exp_id, save_dir)
        plot_momentum_norm(res_no, res_opt, exp_id, env, save_dir)

        # Print summary metrics
        beta_max_no = np.rad2deg(np.max(np.abs(res_no['beta'])))
        beta_max_opt = np.rad2deg(np.max(np.abs(res_opt['beta'])))
        omega_max_no = np.rad2deg(np.max(np.linalg.norm(res_no['omega_sat'], axis=0)))
        omega_max_opt = np.rad2deg(np.max(np.linalg.norm(res_opt['omega_sat'], axis=0)))
        h_rw_max_no = np.max(np.linalg.norm(res_no['h_RW'], axis=0))
        h_rw_max_opt = np.max(np.linalg.norm(res_opt['h_RW'], axis=0))

        print(f"\n  --- Experiment #{exp_id} Summary ---")
        print(f"  Max |beta|  : no-opt={beta_max_no:.4f} deg, opt={beta_max_opt:.4f} deg")
        print(f"  Max |omega| : no-opt={omega_max_no:.4f} deg/s, opt={omega_max_opt:.4f} deg/s")
        print(f"  Max |h_RW|  : no-opt={h_rw_max_no:.4f} Nms, opt={h_rw_max_opt:.4f} Nms")
        if beta_max_no > 1e-10:
            improv = (beta_max_no - beta_max_opt) / beta_max_no * 100
            print(f"  Beta improvement: {improv:.1f}%")

    print(f"\nAll figures saved in: {os.path.abspath(save_dir)}")
