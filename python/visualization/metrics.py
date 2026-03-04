"""Performance metrics computation.

Translated from compute_performance_metrics.m.
"""

import numpy as np


def compute_performance_metrics(results_no_opt, results_opt, env):
    """Print quantitative comparison metrics.

    Parameters
    ----------
    results_no_opt : dict – Non-optimized simulation results.
    results_opt : dict – Optimized simulation results.
    env : dict – Environment configuration.
    """
    print("+" + "-" * 57 + "+")
    print("| METRIC                    | NO-OPT    | OPTIMIZED     |")
    print("+" + "-" * 57 + "+")

    # Rotation
    alpha_max_no = np.rad2deg(np.max(np.abs(results_no_opt['alpha'])))
    alpha_max_opt = np.rad2deg(np.max(np.abs(results_opt['alpha'])))
    print(f"| Max |alpha| [deg]         | {alpha_max_no:8.4f}  | {alpha_max_opt:8.4f}      |")

    beta_max_no = np.rad2deg(np.max(np.abs(results_no_opt['beta'])))
    beta_max_opt = np.rad2deg(np.max(np.abs(results_opt['beta'])))
    print(f"| Max |beta| [deg]          | {beta_max_no:8.4f}  | {beta_max_opt:8.4f}      |")

    gamma_max_no = np.rad2deg(np.max(np.abs(results_no_opt['gamma'])))
    gamma_max_opt = np.rad2deg(np.max(np.abs(results_opt['gamma'])))
    print(f"| Max |gamma| [deg]         | {gamma_max_no:8.4f}  | {gamma_max_opt:8.4f}      |")

    print("+" + "-" * 57 + "+")

    # Contact forces
    F_no = np.linalg.norm(results_no_opt['Fc'][:3, :], axis=0)
    F_opt = np.linalg.norm(results_opt['Fc'][:3, :], axis=0)

    F_rms_no = np.sqrt(np.mean(F_no**2))
    F_rms_opt = np.sqrt(np.mean(F_opt**2))
    print(f"| RMS ||F_c|| [N]           | {F_rms_no:8.1f}  | {F_rms_opt:8.1f}      |")

    F_max_no = np.max(F_no)
    F_max_opt = np.max(F_opt)
    print(f"| Max ||F_c|| [N]           | {F_max_no:8.1f}  | {F_max_opt:8.1f}      |")

    print("+" + "-" * 57 + "+")

    # Tracking
    err_no = np.linalg.norm(results_no_opt['pos_error'], axis=0)
    err_opt = np.linalg.norm(results_opt['pos_error'], axis=0)

    err_rms_no = np.sqrt(np.mean(err_no**2)) * 1000
    err_rms_opt = np.sqrt(np.mean(err_opt**2)) * 1000
    print(f"| RMS tracking error [mm]   | {err_rms_no:8.2f}  | {err_rms_opt:8.2f}      |")

    err_max_no = np.max(err_no) * 1000
    err_max_opt = np.max(err_opt) * 1000
    print(f"| Max tracking error [mm]   | {err_max_no:8.2f}  | {err_max_opt:8.2f}      |")

    print("+" + "-" * 57 + "+")

    # RW Saturation
    h_max = env['RW']['h_total_max']
    h_no = np.linalg.norm(results_no_opt['h_RW'], axis=0)
    h_opt = np.linalg.norm(results_opt['h_RW'], axis=0)

    sat_max_no = np.max(h_no) / h_max * 100
    sat_max_opt = np.max(h_opt) / h_max * 100
    print(f"| Max RW saturation [%]     | {sat_max_no:8.1f}  | {sat_max_opt:8.1f}      |")

    sat_avg_no = np.mean(h_no) / h_max * 100
    sat_avg_opt = np.mean(h_opt) / h_max * 100
    print(f"| Avg RW saturation [%]     | {sat_avg_no:8.1f}  | {sat_avg_opt:8.1f}      |")

    print("+" + "-" * 57 + "+")

    # Warnings
    if sat_max_no > 90 or sat_max_opt > 90:
        print("\n  WARNING: Critical RW saturation detected!")
        print("   -> Desaturation maneuver required")

    # Improvement
    if beta_max_no > 1e-10:
        improvement = (beta_max_no - beta_max_opt) / beta_max_no * 100
        print(f"\n  beta improvement: {improvement:.1f}%")
        if improvement < 50:
            print("  Limited improvement suggests trajectory crosses satellite CoM")
