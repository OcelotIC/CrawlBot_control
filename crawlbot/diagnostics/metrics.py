"""Compute scalar summary metrics from SimLog time series.

Usage:
    from crawlbot.diagnostics.metrics import compute_metrics
    results = compute_metrics(log, cfg)
"""

import csv
import numpy as np
from typing import Dict, Tuple, Optional

DEFAULT_THRESHOLDS = {
    # Tracking
    'torso_pos_err_peak_mm': 10.0,
    'torso_ori_err_peak_deg': 5.0,
    'ee_pos_err_at_dock_mm': 5.0,
    'ee_ori_err_at_dock_deg': 5.0,
    'com_tracking_err_rms_mm': 15.0,
    # Momentum & AOCS
    'hw_saturation_ratio_peak': 1.0,
    'hw_saturation_ratio_rms': 0.7,
    'platform_rotation_total_deg': 5.0,
    'platform_omega_peak_deg_s': 2.0,
    'tau_w_peak_ratio': 1.0,
    # Energy & passivity
    'passivity_violations': 0,
    # NMPC health
    'nmpc_solve_rate_50ms': 0.95,
    'nmpc_infeasibility_rate': 0.02,
}


def _to_np(lst):
    """Convert list of scalars or arrays to numpy array."""
    if len(lst) == 0:
        return np.array([])
    return np.array(lst, dtype=float)


def _to_np2d(lst, cols=3):
    """Convert list of (cols,) arrays to (N, cols)."""
    if len(lst) == 0:
        return np.empty((0, cols))
    return np.array(lst, dtype=float).reshape(-1, cols)


def _available(lst):
    """Check if a log field has meaningful data."""
    return len(lst) > 0


def compute_metrics(log, cfg=None, thresholds=None) -> Dict[str, Tuple]:
    """Compute scalar summary metrics from a SimLog.

    Returns dict of {metric_name: (value, threshold, pass_bool)}.
    Metrics whose required data is missing are marked SKIPPED.
    """
    th = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        th.update(thresholds)

    hw_max = 5.0
    tau_w_max = 5.0
    if cfg is not None:
        hw_max = float(np.max(np.abs(cfg.hw_max)))
        tau_w_max = float(cfg.tau_w_max)

    results = {}

    def add(name, value, skip=False):
        t = th.get(name, np.inf)
        if skip or value is None:
            results[name] = (None, t, 'SKIP')
        else:
            # For 'nmpc_solve_rate_50ms': higher is better (value >= threshold)
            # For 'nmpc_infeasibility_rate': lower is better (value <= threshold)
            # For 'passivity_violations': lower is better (value <= threshold)
            if name in ('nmpc_solve_rate_50ms',):
                passed = value >= t
            else:
                passed = value <= t
            results[name] = (value, t, passed)

    # --- Torso tracking ---
    if _available(log.e_torso_pos):
        e_tp = _to_np(log.e_torso_pos)
        add('torso_pos_err_peak_mm', float(np.max(e_tp)) * 1000)
    else:
        add('torso_pos_err_peak_mm', None, skip=True)

    if _available(log.e_torso_ori):
        e_to = _to_np(log.e_torso_ori)
        add('torso_ori_err_peak_deg', float(np.max(e_to)))
    else:
        add('torso_ori_err_peak_deg', None, skip=True)

    # --- EE error at dock ---
    if _available(log.dock_events) and _available(log.e_ee_pos):
        t_arr = _to_np(log.t)
        ee_pos = _to_np(log.e_ee_pos)
        ee_ori = _to_np(log.e_ee_ori)
        worst_pos = 0.0
        worst_ori = 0.0
        for ev in log.dock_events:
            t_dock = ev['t']
            idx = int(np.argmin(np.abs(t_arr - t_dock)))
            worst_pos = max(worst_pos, float(ee_pos[idx]))
            worst_ori = max(worst_ori, float(ee_ori[idx]))
        add('ee_pos_err_at_dock_mm', worst_pos * 1000)
        add('ee_ori_err_at_dock_deg', worst_ori)
    else:
        add('ee_pos_err_at_dock_mm', None, skip=True)
        add('ee_ori_err_at_dock_deg', None, skip=True)

    # --- CoM tracking (SS phases only) ---
    if _available(log.e_com) and _available(log.phase):
        e_c = _to_np(log.e_com)
        ph = log.phase
        ss_mask = np.array([p == 'SS' for p in ph])
        if np.any(ss_mask):
            rms = float(np.sqrt(np.mean(e_c[ss_mask] ** 2)))
            add('com_tracking_err_rms_mm', rms * 1000)
        else:
            add('com_tracking_err_rms_mm', None, skip=True)
    else:
        add('com_tracking_err_rms_mm', None, skip=True)

    # --- Momentum & AOCS ---
    if _available(log.hw_physical):
        hw_arr = _to_np2d(log.hw_physical)
        hw_norms = np.linalg.norm(hw_arr, axis=1)
        add('hw_saturation_ratio_peak', float(np.max(hw_norms)) / hw_max)
        add('hw_saturation_ratio_rms',
            float(np.sqrt(np.mean((hw_norms / hw_max) ** 2))))
    else:
        add('hw_saturation_ratio_peak', None, skip=True)
        add('hw_saturation_ratio_rms', None, skip=True)

    if _available(log.struct_euler_deg):
        euler = _to_np2d(log.struct_euler_deg)
        delta = np.diff(euler, axis=0)
        total_rot = float(np.sum(np.linalg.norm(delta, axis=1)))
        add('platform_rotation_total_deg', total_rot)
    else:
        add('platform_rotation_total_deg', None, skip=True)

    if _available(log.omega_s):
        om = _to_np2d(log.omega_s)
        om_deg = np.degrees(np.linalg.norm(om, axis=1))
        add('platform_omega_peak_deg_s', float(np.max(om_deg)))
    else:
        add('platform_omega_peak_deg_s', None, skip=True)

    if _available(log.tau_w):
        tw = _to_np2d(log.tau_w)
        peak_ratio = float(np.max(np.abs(tw))) / tau_w_max
        add('tau_w_peak_ratio', peak_ratio)
    else:
        add('tau_w_peak_ratio', None, skip=True)

    # --- Energy & passivity ---
    add('passivity_violations', None, skip=True)  # requires passivity_lhs (not yet logged)

    # --- NMPC health ---
    if _available(log.nmpc_time_ms):
        times = _to_np(log.nmpc_time_ms)
        n = len(times)
        if n > 0:
            add('nmpc_solve_rate_50ms', float(np.sum(times < 50)) / n)
        else:
            add('nmpc_solve_rate_50ms', None, skip=True)
    else:
        add('nmpc_solve_rate_50ms', None, skip=True)

    if _available(log.nmpc_status):
        status = _to_np(log.nmpc_status)
        n = len(status)
        if n > 0:
            add('nmpc_infeasibility_rate', float(np.sum(status == 2)) / n)
        else:
            add('nmpc_infeasibility_rate', None, skip=True)
    elif _available(log.nmpc_ok):
        ok = np.array(log.nmpc_ok, dtype=bool)
        n = len(ok)
        if n > 0:
            add('nmpc_infeasibility_rate', float(np.sum(~ok)) / n)
        else:
            add('nmpc_infeasibility_rate', None, skip=True)
    else:
        add('nmpc_infeasibility_rate', None, skip=True)

    return results


def print_metrics(results: Dict[str, Tuple], file=None):
    """Print metrics table to console."""
    lines = []
    lines.append(f"{'Metric':<35s} {'Value':>10s} {'Threshold':>10s} {'Status':>8s}")
    lines.append('-' * 67)
    for name, (val, thresh, passed) in results.items():
        if val is None:
            v_str = '  ---'
        else:
            v_str = f'{val:10.4f}'
        t_str = f'{thresh:10.4f}' if np.isfinite(thresh) else '   ---'
        if passed == 'SKIP':
            s_str = '  SKIP'
        elif passed:
            s_str = '  PASS'
        else:
            s_str = '**FAIL'
        lines.append(f'{name:<35s} {v_str} {t_str} {s_str}')
    text = '\n'.join(lines)
    print(text, file=file)
    return text


def save_metrics_csv(results: Dict[str, Tuple], path: str):
    """Save metrics to CSV."""
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'value', 'threshold', 'status'])
        for name, (val, thresh, passed) in results.items():
            status = 'SKIP' if passed == 'SKIP' else ('PASS' if passed else 'FAIL')
            w.writerow([name,
                        '' if val is None else f'{val:.6f}',
                        f'{thresh:.6f}' if np.isfinite(thresh) else '',
                        status])
