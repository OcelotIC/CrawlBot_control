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
    # Per-phase peaks (SS = useful locomotion, DS = hold/recovery)
    'torso_ori_peak_deg_SS': 5.0,
    'torso_ori_peak_deg_DS': 5.0,
    'torso_ori_peak_deg_global': 5.0,
    'torso_pos_peak_mm_SS': 10.0,
    'torso_pos_peak_mm_DS': 10.0,
    'torso_pos_peak_mm_global': 10.0,
    'ee_pos_peak_mm_SS': 5.0,
    'ee_pos_peak_mm_DS': 5.0,
    'ee_pos_peak_mm_global': 5.0,
    'ee_ori_peak_deg_SS': 5.0,
    'ee_ori_peak_deg_DS': 5.0,
    'ee_ori_peak_deg_global': 5.0,
    'hw_sat_peak_SS': 1.0,
    'hw_sat_peak_DS': 1.0,
    'hw_sat_peak_global': 1.0,
    'tau_peak_Nm_SS': 20.0,
    'tau_peak_Nm_DS': 20.0,
    'tau_peak_Nm_global': 20.0,
    'tau_w_peak_ratio_SS': 1.0,
    'tau_w_peak_ratio_DS': 1.0,
    'tau_w_peak_ratio_global': 1.0,
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


def _quat_geodesic_deg(q1, q2):
    """Geodesic angle between two unit quaternions (convention-agnostic)."""
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    dot = float(np.clip(abs(np.dot(q1, q2)), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def compute_metrics(log, cfg=None, thresholds=None) -> Dict[str, Tuple]:
    """Compute scalar summary metrics from a SimLog.

    Returns dict of {metric_name: (value, threshold, pass_bool)}.
    Metrics whose required data is missing are marked SKIPPED.
    """
    th = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        th.update(thresholds)

    hw_max = 5.0
    tau_w_max = 2.5
    if cfg is not None:
        hw_max = float(np.max(np.abs(cfg.hw_max)))
        tau_w_max = float(cfg.tau_w_max)

    results = {}

    def add(name, value, skip=False, status=None):
        t = th.get(name, np.inf)
        if skip or value is None:
            results[name] = (None, t, 'SKIP')
        elif status is not None:
            # Caller overrides status (e.g., 'WARN' or 'INFO').
            results[name] = (value, t, status)
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

    # ------------------------------------------------------------------
    # Per-phase peaks + abort-boundary instrumentation.
    # Rationale: a global np.max(|.|) conflates the useful locomotion
    # window (phase == 'SS') with the hold/recovery window (phase == 'DS')
    # that follows a dock abort. The _global entries are preserved for
    # backward compatibility; when len(aborted_steps) > 0 they are flagged
    # 'WARN' because they may be polluted by post-abort divergence.
    # ------------------------------------------------------------------
    if _available(log.phase):
        ph = np.array(log.phase)
        ss_mask = (ph == 'SS')
        ds_mask = (ph == 'DS')
    else:
        ss_mask = None
        ds_mask = None

    aborted = _available(log.aborted_steps) and len(log.aborted_steps) > 0

    def _peak_abs_1d(arr, mask):
        if arr is None or mask is None or not np.any(mask):
            return None
        return float(np.max(np.abs(arr[mask])))

    def _peak_norm_rows(arr2d, mask):
        """Peak of per-row L2 norm over masked samples (for hw, tau_w 2D arrays)."""
        if arr2d is None or mask is None or not np.any(mask):
            return None
        return float(np.max(np.linalg.norm(arr2d[mask], axis=1)))

    def _peak_abs_all(arr2d, mask):
        """Peak of |arr2d| over masked rows and all columns (for joint torques)."""
        if arr2d is None or mask is None or not np.any(mask):
            return None
        return float(np.max(np.abs(arr2d[mask])))

    def _add_triple(base, v_ss, v_ds, v_global, scale=1.0):
        for suffix, v in (('_SS', v_ss), ('_DS', v_ds), ('_global', v_global)):
            name = base + suffix
            if v is None:
                add(name, None, skip=True)
            else:
                val = v * scale
                if suffix == '_global' and aborted:
                    add(name, val, status='WARN')
                else:
                    add(name, val)

    # Torso orientation [deg]
    if _available(log.e_torso_ori):
        e_to = _to_np(log.e_torso_ori)
        _add_triple('torso_ori_peak_deg',
                    _peak_abs_1d(e_to, ss_mask),
                    _peak_abs_1d(e_to, ds_mask),
                    float(np.max(np.abs(e_to))))
    else:
        _add_triple('torso_ori_peak_deg', None, None, None)

    # Torso position [m -> mm]
    if _available(log.e_torso_pos):
        e_tp = _to_np(log.e_torso_pos)
        _add_triple('torso_pos_peak_mm',
                    _peak_abs_1d(e_tp, ss_mask),
                    _peak_abs_1d(e_tp, ds_mask),
                    float(np.max(np.abs(e_tp))),
                    scale=1000.0)
    else:
        _add_triple('torso_pos_peak_mm', None, None, None)

    # EE position [m -> mm]
    if _available(log.e_ee_pos):
        e_ep = _to_np(log.e_ee_pos)
        _add_triple('ee_pos_peak_mm',
                    _peak_abs_1d(e_ep, ss_mask),
                    _peak_abs_1d(e_ep, ds_mask),
                    float(np.max(np.abs(e_ep))),
                    scale=1000.0)
    else:
        _add_triple('ee_pos_peak_mm', None, None, None)

    # EE orientation [deg]
    if _available(log.e_ee_ori):
        e_eo = _to_np(log.e_ee_ori)
        _add_triple('ee_ori_peak_deg',
                    _peak_abs_1d(e_eo, ss_mask),
                    _peak_abs_1d(e_eo, ds_mask),
                    float(np.max(np.abs(e_eo))))
    else:
        _add_triple('ee_ori_peak_deg', None, None, None)

    # Reaction-wheel momentum saturation ratio [-]
    if _available(log.hw_physical):
        hw_arr = _to_np2d(log.hw_physical)
        _add_triple('hw_sat_peak',
                    _peak_norm_rows(hw_arr, ss_mask),
                    _peak_norm_rows(hw_arr, ds_mask),
                    float(np.max(np.linalg.norm(hw_arr, axis=1))),
                    scale=1.0 / hw_max)
    else:
        _add_triple('hw_sat_peak', None, None, None)

    # Joint torque peak [Nm]
    if _available(log.tau):
        tau_cols = len(np.atleast_1d(log.tau[0]))
        tau_arr = _to_np2d(log.tau, cols=tau_cols)
        _add_triple('tau_peak_Nm',
                    _peak_abs_all(tau_arr, ss_mask),
                    _peak_abs_all(tau_arr, ds_mask),
                    float(np.max(np.abs(tau_arr))))
    else:
        _add_triple('tau_peak_Nm', None, None, None)

    # Reaction-wheel torque peak ratio [-]
    if _available(log.tau_w):
        tw_arr = _to_np2d(log.tau_w)
        _add_triple('tau_w_peak_ratio',
                    _peak_abs_all(tw_arr, ss_mask),
                    _peak_abs_all(tw_arr, ds_mask),
                    float(np.max(np.abs(tw_arr))),
                    scale=1.0 / tau_w_max)
    else:
        _add_triple('tau_w_peak_ratio', None, None, None)

    # Abort-boundary instrumentation fields (status='INFO' — not pass/fail).
    if ss_mask is not None and ds_mask is not None \
            and np.any(ss_mask) and np.any(ds_mask) \
            and _available(log.e_torso_ori) and _available(log.q_torso_ref):
        ss_idx = np.where(ss_mask)[0]
        ds_idx = np.where(ds_mask)[0]
        i_ss_last = int(ss_idx[-1])
        i_ds_first = int(ds_idx[0])
        e_to_arr = _to_np(log.e_torso_ori)
        q_tr_arr = _to_np2d(log.q_torso_ref, cols=4)
        add('ss_end_torso_ori_deg',
            float(abs(e_to_arr[i_ss_last])), status='INFO')
        add('ds_entry_torso_ori_deg',
            float(abs(e_to_arr[i_ds_first])), status='INFO')
        add('q_torso_ref_ss_to_ds_jump_deg',
            _quat_geodesic_deg(q_tr_arr[i_ss_last], q_tr_arr[i_ds_first]),
            status='INFO')
    else:
        add('ss_end_torso_ori_deg', None, skip=True)
        add('ds_entry_torso_ori_deg', None, skip=True)
        add('q_torso_ref_ss_to_ds_jump_deg', None, skip=True)

    return results


def _status_str(passed):
    """Canonical short string for a metric status."""
    if passed == 'SKIP':
        return 'SKIP'
    if passed == 'INFO':
        return 'INFO'
    if passed == 'WARN':
        return 'WARN'
    return 'PASS' if passed else 'FAIL'


def print_metrics(results: Dict[str, Tuple], file=None):
    """Print metrics table to console."""
    lines = []
    name_w = max(35, max((len(k) for k in results.keys()), default=35))
    lines.append(f"{'Metric':<{name_w}s} {'Value':>12s} {'Threshold':>12s} {'Status':>8s}")
    lines.append('-' * (name_w + 12 + 12 + 8 + 3))
    for name, (val, thresh, passed) in results.items():
        if val is None:
            v_str = '  ---'
        else:
            v_str = f'{val:12.4f}'
        t_str = f'{thresh:12.4f}' if np.isfinite(thresh) else '   ---'
        if passed == 'SKIP':
            s_str = '  SKIP'
        elif passed == 'INFO':
            s_str = '  INFO'
        elif passed == 'WARN':
            s_str = '  WARN'
        elif passed:
            s_str = '  PASS'
        else:
            s_str = '**FAIL'
        suffix = ''
        if name.endswith('_global') and passed == 'WARN':
            suffix = '   [WARN: may include post-abort divergence]'
        lines.append(f'{name:<{name_w}s} {v_str} {t_str} {s_str}{suffix}')
    text = '\n'.join(lines)
    print(text, file=file)
    return text


def save_metrics_csv(results: Dict[str, Tuple], path: str):
    """Save metrics to CSV."""
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'value', 'threshold', 'status'])
        for name, (val, thresh, passed) in results.items():
            w.writerow([name,
                        '' if val is None else f'{val:.6f}',
                        f'{thresh:.6f}' if np.isfinite(thresh) else '',
                        _status_str(passed)])
