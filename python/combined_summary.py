#!/usr/bin/env python3
"""Generate combined summary table and comparison figure for all 3 experiments.

Loads saved results from run_benchmark.py and produces:
  1. A text summary table (printed + saved to results/summary.txt)
  2. A combined 3x3 figure comparing all experiments
"""

import os
import sys
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_results(data_dir, exp_id):
    """Load saved experiment results."""
    path_no = os.path.join(data_dir, f'exp{exp_id}_no_opt.pkl')
    path_opt = os.path.join(data_dir, f'exp{exp_id}_opt.pkl')
    with open(path_no, 'rb') as f:
        res_no = pickle.load(f)
    with open(path_opt, 'rb') as f:
        res_opt = pickle.load(f)
    return res_no, res_opt


def compute_metrics(res_no, res_opt):
    """Extract key metrics from a pair of results."""
    m = {}
    for label, res in [('no_opt', res_no), ('opt', res_opt)]:
        m[f'alpha_max_{label}'] = np.rad2deg(np.max(np.abs(res['alpha'])))
        m[f'beta_max_{label}'] = np.rad2deg(np.max(np.abs(res['beta'])))
        m[f'gamma_max_{label}'] = np.rad2deg(np.max(np.abs(res['gamma'])))
        m[f'F_rms_{label}'] = np.sqrt(np.mean(np.sum(res['Fc'][0:3, :]**2, axis=0)))
        m[f'F_max_{label}'] = np.max(np.linalg.norm(res['Fc'][0:3, :], axis=0))
        m[f'err_rms_{label}'] = np.sqrt(np.mean(np.sum(res['pos_error']**2, axis=0))) * 1000
        m[f'err_max_{label}'] = np.max(np.linalg.norm(res['pos_error'], axis=0)) * 1000
        m[f'h_rw_max_{label}'] = np.max(np.linalg.norm(res['h_RW'], axis=0))

    # Improvement percentages
    for angle in ['alpha', 'beta', 'gamma']:
        no_val = m[f'{angle}_max_no_opt']
        opt_val = m[f'{angle}_max_opt']
        if abs(no_val) > 1e-10:
            m[f'{angle}_improvement'] = (no_val - opt_val) / no_val * 100
        else:
            m[f'{angle}_improvement'] = 0.0
    return m


def print_summary_table(all_metrics, descriptions):
    """Print and return a formatted summary table."""
    lines = []
    sep = '=' * 90
    lines.append(sep)
    lines.append('COMBINED RESULTS — Lutze et al. (2023) Three Benchmark Experiments')
    lines.append(sep)
    lines.append('')

    for exp_id in [1, 2, 3]:
        m = all_metrics[exp_id]
        desc = descriptions[exp_id]
        lines.append(f'Experiment #{exp_id}: {desc}')
        lines.append('-' * 70)
        lines.append(f'  {"Metric":<30} {"Non-Optimized":>15} {"Optimized":>15} {"Improvement":>15}')
        lines.append(f'  {"-"*30} {"-"*15} {"-"*15} {"-"*15}')

        lines.append(f'  {"Max |alpha| [deg]":<30} {m["alpha_max_no_opt"]:>15.4f} {m["alpha_max_opt"]:>15.4f} {m["alpha_improvement"]:>14.1f}%')
        lines.append(f'  {"Max |beta| [deg]":<30} {m["beta_max_no_opt"]:>15.4f} {m["beta_max_opt"]:>15.4f} {m["beta_improvement"]:>14.1f}%')
        lines.append(f'  {"Max |gamma| [deg]":<30} {m["gamma_max_no_opt"]:>15.4f} {m["gamma_max_opt"]:>15.4f} {m["gamma_improvement"]:>14.1f}%')
        lines.append(f'  {"RMS ||F_c|| [N]":<30} {m["F_rms_no_opt"]:>15.1f} {m["F_rms_opt"]:>15.1f}')
        lines.append(f'  {"Max ||F_c|| [N]":<30} {m["F_max_no_opt"]:>15.1f} {m["F_max_opt"]:>15.1f}')
        lines.append(f'  {"RMS tracking error [mm]":<30} {m["err_rms_no_opt"]:>15.2f} {m["err_rms_opt"]:>15.2f}')
        lines.append(f'  {"Max tracking error [mm]":<30} {m["err_max_no_opt"]:>15.2f} {m["err_max_opt"]:>15.2f}')
        lines.append('')

    # Cross-experiment summary
    lines.append(sep)
    lines.append('CROSS-EXPERIMENT SUMMARY')
    lines.append(sep)
    lines.append(f'  {"Experiment":<35} {"beta improvement":>20}')
    lines.append(f'  {"-"*35} {"-"*20}')
    for exp_id in [1, 2, 3]:
        m = all_metrics[exp_id]
        desc = descriptions[exp_id]
        lines.append(f'  Exp #{exp_id}: {desc:<30} {m["beta_improvement"]:>19.1f}%')
    lines.append('')

    text = '\n'.join(lines)
    print(text)
    return text


def plot_combined_figure(all_results, descriptions, save_path):
    """3x3 grid: rows = experiments, cols = orientation / wrenches / tracking."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    for row, exp_id in enumerate([1, 2, 3]):
        res_no, res_opt = all_results[exp_id]
        t = res_no['t']

        # Col 0: Beta angle
        ax = axes[row, 0]
        ax.plot(t, np.rad2deg(res_no['beta']), 'r--', lw=1.2, label='Non-optimized')
        ax.plot(t, np.rad2deg(res_opt['beta']), 'b-', lw=1.2, label='Optimized')
        ax.set_ylabel(r'$\beta$ [deg]')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if row == 0:
            ax.set_title('Satellite Pitch Angle', fontweight='bold')

        # Col 1: Contact force norm
        ax = axes[row, 1]
        F_no = np.linalg.norm(res_no['Fc'][:3, :], axis=0)
        F_opt = np.linalg.norm(res_opt['Fc'][:3, :], axis=0)
        ax.plot(t, F_no, 'r--', lw=1.0, label='Non-optimized')
        ax.plot(t, F_opt, 'b-', lw=1.0, label='Optimized')
        ax.set_ylabel('||F_contact|| [N]')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if row == 0:
            ax.set_title('Contact Force Norm', fontweight='bold')

        # Col 2: Tracking error
        ax = axes[row, 2]
        err_no = np.linalg.norm(res_no['pos_error'], axis=0) * 1000
        err_opt = np.linalg.norm(res_opt['pos_error'], axis=0) * 1000
        ax.plot(t, err_no, 'r--', lw=1.0, label='Non-optimized')
        ax.plot(t, err_opt, 'b-', lw=1.0, label='Optimized')
        ax.set_ylabel('Tracking error [mm]')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if row == 0:
            ax.set_title('CoM Tracking Error', fontweight='bold')

        # Row label
        axes[row, 0].annotate(
            f'Exp #{exp_id}\n{descriptions[exp_id]}',
            xy=(0, 0.5), xycoords='axes fraction',
            xytext=(-70, 0), textcoords='offset points',
            fontsize=9, fontweight='bold',
            ha='right', va='center', rotation=0,
        )

    for ax in axes[-1, :]:
        ax.set_xlabel('Time [s]')

    fig.suptitle('Lutze et al. (2023) — Comparative Results: 3 Experiments',
                 fontsize=16, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0.08, 0, 1, 0.96])
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nCombined figure saved: {save_path}")


if __name__ == '__main__':
    project_root = os.path.join(os.path.dirname(__file__), '..')
    data_dir = os.path.join(project_root, 'results', 'data')
    fig_dir = os.path.join(project_root, 'results', 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    descriptions = {
        1: 'Straight line crossing CoM',
        2: 'Offset straight line',
        3: 'Circular arc (20 deg)',
    }

    all_metrics = {}
    all_results = {}

    for exp_id in [1, 2, 3]:
        try:
            res_no, res_opt = load_results(data_dir, exp_id)
            all_metrics[exp_id] = compute_metrics(res_no, res_opt)
            all_results[exp_id] = (res_no, res_opt)
        except FileNotFoundError:
            print(f"WARNING: Results for experiment {exp_id} not found. "
                  f"Run 'python python/run_benchmark.py {exp_id}' first.")

    if len(all_metrics) < 3:
        print("Not all experiments available. Run missing ones first.")
        sys.exit(1)

    # Summary table
    text = print_summary_table(all_metrics, descriptions)

    summary_path = os.path.join(project_root, 'results', 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(text)
    print(f"\nSummary saved: {summary_path}")

    # Combined figure
    plot_combined_figure(all_results, descriptions,
                         os.path.join(fig_dir, 'combined_comparison.png'))
