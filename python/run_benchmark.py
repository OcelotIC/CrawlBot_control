#!/usr/bin/env python3
"""Main benchmark script for Lutze et al. (2023).

Reproduction of IEEE Aerospace Conference 2023 paper:
"Optimization of multi-arm robot locomotion to reduce satellite disturbances
 during in-orbit assembly"

Usage:
    python -m python.run_benchmark [experiment_id]
    python python/run_benchmark.py [experiment_id]
"""

import os
import sys
import time
import pickle
import numpy as np

# Add parent dir to path for module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.spart.robot_model import urdf2robot
from python.config.environment import setup_environment
from python.config.trajectories import setup_trajectories
from python.control.controller import LutzeQPController
from python.simulation.experiment import simulate_experiment
from python.visualization.plots import (
    plot_satellite_rotation, plot_contact_wrenches,
    plot_tracking_error, plot_momentum_saturation,
)
from python.visualization.metrics import compute_performance_metrics


def run_benchmark(experiment_id=1):
    """Run the full benchmark for a given experiment.

    Parameters
    ----------
    experiment_id : int – 1, 2, or 3.
    """
    print("=" * 56)
    print("   BENCHMARK: Lutze et al. (2023) — Python Translation")
    print("=" * 56)
    print()

    # Resolve URDF path relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    urdf_path = os.path.join(project_root, 'URDF_models', 'MAR_DualArm_6DoF.urdf')

    # Load robot
    print("-> Loading robot model...")
    robot, robot_keys = urdf2robot(urdf_path)
    print(f"  Robot: {6 + robot['n_q']} DoF total")
    print()

    # Environment
    print("-> Setup environment...")
    env = setup_environment()
    print("  Config loaded")
    print()

    # Trajectory
    print(f"-> Generating trajectory (Exp #{experiment_id})...")
    traj, sim_params = setup_trajectories(experiment_id)
    print(f"  {traj['description']}")
    print()

    # Controller
    controller = LutzeQPController(env)

    # Simulation 1: Non-optimized
    print("=" * 56)
    print(" SIMULATION 1: NON-OPTIMIZED")
    print("=" * 56)
    t_start = time.time()
    results_no_opt = simulate_experiment(env, robot, traj, sim_params,
                                          controller, False)
    print(f"Done in {time.time() - t_start:.2f}s")
    print()

    # Simulation 2: Optimized
    print("=" * 56)
    print(" SIMULATION 2: OPTIMIZED (QP)")
    print("=" * 56)
    t_start = time.time()
    results_opt = simulate_experiment(env, robot, traj, sim_params,
                                       controller, True)
    print(f"Done in {time.time() - t_start:.2f}s")
    print()

    # Plots
    print("-> Generating figures...")
    fig_dir = os.path.join(project_root, 'results', 'figures')
    plot_satellite_rotation(results_no_opt, results_opt, experiment_id, env,
                             save_dir=fig_dir)
    plot_contact_wrenches(results_no_opt, results_opt, experiment_id,
                           save_dir=fig_dir)
    plot_tracking_error(results_no_opt, results_opt, experiment_id,
                         save_dir=fig_dir)
    plot_momentum_saturation(results_no_opt, results_opt, experiment_id, env,
                              save_dir=fig_dir)
    print("  4 figures saved")
    print()

    # Metrics
    print("=" * 56)
    print(" PERFORMANCE METRICS")
    print("=" * 56)
    compute_performance_metrics(results_no_opt, results_opt, env)

    # Save results
    data_dir = os.path.join(project_root, 'results', 'data')
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, f'exp{experiment_id}_no_opt.pkl'), 'wb') as f:
        pickle.dump(results_no_opt, f)
    with open(os.path.join(data_dir, f'exp{experiment_id}_opt.pkl'), 'wb') as f:
        pickle.dump(results_opt, f)

    print()
    print("BENCHMARK COMPLETE")


if __name__ == '__main__':
    exp_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    run_benchmark(exp_id)
