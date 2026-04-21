"""Single entry point for the diagnostic suite.

Usage:
    from crawlbot.diagnostics import run_diagnostics
    metrics = run_diagnostics(sim_log, 'results/my_run/', cfg=sim_cfg)
"""

import os

from .metrics import compute_metrics, print_metrics, save_metrics_csv
from .plots import generate_plots


def run_diagnostics(log, output_dir, cfg=None, thresholds=None,
                    model=None, data=None):
    """Run the full diagnostic pipeline.

    1. Compute metrics and print summary table
    2. Save metrics CSV
    3. Generate all 8 figures
    4. Render MuJoCo snapshots (if model/data provided)
    5. Return metrics dict

    Args:
        log: SimLog instance
        output_dir: directory for all outputs
        cfg: SimConfig (optional, for thresholds)
        thresholds: override dict for metric thresholds
        model: mujoco.MjModel (optional, for snapshot rendering)
        data: mujoco.MjData (optional, for snapshot rendering)

    Returns:
        dict of {metric_name: (value, threshold, pass_bool)}
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Compute metrics
    results = compute_metrics(log, cfg=cfg, thresholds=thresholds)

    # 2. Print summary
    print('\n' + '=' * 67)
    print('  DIAGNOSTIC METRICS')
    print('=' * 67)
    print_metrics(results)

    n_pass = sum(1 for _, _, p in results.values() if p is True)
    n_fail = sum(1 for _, _, p in results.values() if p is False)
    n_skip = sum(1 for _, _, p in results.values() if p == 'SKIP')
    n_warn = sum(1 for _, _, p in results.values() if p == 'WARN')
    n_info = sum(1 for _, _, p in results.values() if p == 'INFO')
    print(f'\nSummary: {n_pass} PASS, {n_fail} FAIL, {n_skip} SKIP, '
          f'{n_warn} WARN, {n_info} INFO')
    print('=' * 67)

    # 3. Save metrics CSV
    csv_path = os.path.join(output_dir, 'metrics.csv')
    save_metrics_csv(results, csv_path)

    # 4. Generate plots
    generate_plots(log, output_dir, cfg=cfg)

    # 5. Render snapshots (if rendering is available)
    if model is not None and data is not None:
        from .snapshots import capture_snapshots
        capture_snapshots(model, data, log, output_dir)
        # Regenerate fig8 grid after snapshots are rendered
        from .plots import _fig8_snapshots
        _fig8_snapshots(log, output_dir, dpi=150)

    return results
