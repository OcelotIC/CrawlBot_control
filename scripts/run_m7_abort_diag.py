#!/usr/bin/env python3
"""M7 post-abort DS diagnostic decomposition — four orthogonal runs.

Per docs/architecture/M7_DS_DIAGNOSTIC_EXPERIMENTS.md. Executes R1/R2/R3/R4
on the v21 single-step baseline config, toggling at most one of the three
`diag_*_on_abort` flags per run. No other config change, no architectural fix.

Runs:
    R1_baseline       — all three flags False
    R2_freeze_ref     — diag_freeze_torso_ref_on_abort=True  (H_DS2)
    R3_single_contact — diag_force_single_contact_on_abort=True  (H_DS1)
    R4_no_passivity   — diag_disable_passivity_on_abort=True  (H_DS3)

Outputs:
    results/M7_abort_diag/R1_baseline/
    results/M7_abort_diag/R2_freeze_ref/
    results/M7_abort_diag/R3_single_contact/
    results/M7_abort_diag/R4_no_passivity/
    results/M7_abort_diag/summary.md
"""
from __future__ import annotations

import os
import sys

_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')

import numpy as np

import scripts.run_m7_single_step as r
from crawlbot.simulation.logging import SimLog
from crawlbot.diagnostics.metrics import compute_metrics


RESULTS_ROOT = 'results/M7_abort_diag'


def _make_v21_config():
    """v21 baseline config — exactly matches scripts/run_m7_v21_preplanner_cruise.py."""
    cfg = r._make_m7_config()
    cfg.preplanner_a_cruise_max = 0.01       # [m/s²] cruise accel cap
    cfg.preplanner_cruise_ramp_frac = 0.2    # 20/60/20 split
    return cfg


RUN_SPECS = [
    # (tag, subdir, flag_attr or None)
    ('R1_baseline',       'R1_baseline',       None),
    ('R2_freeze_ref',     'R2_freeze_ref',     'diag_freeze_torso_ref_on_abort'),
    ('R3_single_contact', 'R3_single_contact', 'diag_force_single_contact_on_abort'),
    ('R4_no_passivity',   'R4_no_passivity',   'diag_disable_passivity_on_abort'),
]


def _run_one(tag, subdir, flag):
    out_dir = os.path.join(RESULTS_ROOT, subdir)
    cfg = _make_v21_config()
    if flag is not None:
        setattr(cfg, flag, True)
    print("\n" + "#" * 78)
    print(f"#  {tag}  (flag: {flag or '<none>'})")
    print("#" * 78)
    r.run_case(tag, out_dir, n_steps=1, config=cfg)
    return out_dir


def _safe(val, prec=4):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return '—'
    return f'{val:.{prec}f}'


def _ds_mask_arrays(log):
    ph = np.array(log.phase)
    ds_mask = (ph == 'DS')
    ss_mask = (ph == 'SS')
    return ss_mask, ds_mask


def _tau_sat_fraction(tau_arr, mask, tau_sat_threshold=0.99 * 20.0):
    """Fraction of masked ticks that have max|tau| >= threshold."""
    if not np.any(mask):
        return None
    sub = tau_arr[mask]
    if sub.size == 0:
        return None
    per_tick_max = np.max(np.abs(sub), axis=1)
    return float(np.mean(per_tick_max >= tau_sat_threshold))


def _row_for_run(subdir):
    sim_log_path = os.path.join(RESULTS_ROOT, subdir, 'sim_log.json')
    log = SimLog.load(sim_log_path)
    metrics = compute_metrics(log)

    def _fetch(key):
        entry = metrics.get(key)
        if entry is None:
            return None
        return entry[0]

    ss_peak_ori = _fetch('torso_ori_peak_deg_SS')
    q_ref_jump = _fetch('q_torso_ref_ss_to_ds_jump_deg')
    ds_entry = _fetch('ds_entry_torso_ori_deg')
    ds_peak_ori = _fetch('torso_ori_peak_deg_DS')

    ss_mask, ds_mask = _ds_mask_arrays(log)

    tau_arr = np.array(log.tau, dtype=float) if len(log.tau) else np.empty((0, 14))
    tau_w_arr = np.array(log.tau_w, dtype=float) if len(log.tau_w) else np.empty((0, 3))

    if np.any(ds_mask) and tau_arr.shape[0] == ds_mask.size:
        max_tau_ds = float(np.max(np.abs(tau_arr[ds_mask])))
        tau_sat_frac_ds = _tau_sat_fraction(tau_arr, ds_mask, 0.99 * 20.0)
    else:
        max_tau_ds = None
        tau_sat_frac_ds = None

    if np.any(ds_mask) and tau_w_arr.shape[0] == ds_mask.size:
        tau_w_sat_frac_ds = _tau_sat_fraction(tau_w_arr, ds_mask, 0.99 * 5.0)
    else:
        tau_w_sat_frac_ds = None

    docked = len(log.dock_events) > 0
    return {
        'run': subdir,
        'SS peak ori [°]': _safe(ss_peak_ori),
        'q_ref jump [°]':  _safe(q_ref_jump),
        'DS entry ori [°]': _safe(ds_entry),
        'DS peak ori [°]':  _safe(ds_peak_ori),
        'max|τ_q|_DS [Nm]': _safe(max_tau_ds, prec=3),
        'τ_q sat frac DS':  _safe(tau_sat_frac_ds, prec=3),
        'τ_w sat frac DS':  _safe(tau_w_sat_frac_ds, prec=3),
        'dock?': 'yes' if docked else 'no',
    }


def _verify_invariants(rows, baseline_name='R1_baseline'):
    """Invariant check per M7_DS_DIAGNOSTIC_EXPERIMENTS.md §5.

    Returns list of violation strings (empty if all pass).
    """
    baseline = next((r_ for r_ in rows if r_['run'] == baseline_name), None)
    if baseline is None:
        return ['baseline row not found']

    # Pull baseline numeric values (re-parse the formatted strings).
    def _num(s):
        return None if s == '—' else float(s)

    base_ss = _num(baseline['SS peak ori [°]'])

    violations = []
    for row in rows:
        ss = _num(row['SS peak ori [°]'])
        if base_ss is None or ss is None:
            violations.append(f"{row['run']}: SS peak ori missing")
            continue
        if abs(ss - base_ss) > 0.01:
            violations.append(
                f"{row['run']}: SS peak ori = {ss:.4f}°, "
                f"baseline = {base_ss:.4f}°, Δ={abs(ss - base_ss):.4f}° > 0.01°")

    # Bit-for-bit checks on ee_pos_peak_SS, ss_end_torso_ori_deg,
    # preplanner_T_steps[0], abort d_mm/ori_deg: read them fresh from
    # each sim_log and compare against baseline.
    base_log = SimLog.load(os.path.join(RESULTS_ROOT, baseline_name, 'sim_log.json'))
    base_metrics = compute_metrics(base_log)
    base_ee_ss = base_metrics['ee_pos_peak_mm_SS'][0]
    base_ss_end = base_metrics['ss_end_torso_ori_deg'][0]
    base_tstep = base_log.preplanner_T_steps[0] if base_log.preplanner_T_steps else None
    base_ab = base_log.aborted_steps[0] if base_log.aborted_steps else None

    for row in rows:
        if row['run'] == baseline_name:
            continue
        log = SimLog.load(os.path.join(RESULTS_ROOT, row['run'], 'sim_log.json'))
        m = compute_metrics(log)
        ee_ss = m['ee_pos_peak_mm_SS'][0]
        ss_end = m['ss_end_torso_ori_deg'][0]
        tstep = log.preplanner_T_steps[0] if log.preplanner_T_steps else None
        ab = log.aborted_steps[0] if log.aborted_steps else None

        if ee_ss is None or base_ee_ss is None or ee_ss != base_ee_ss:
            violations.append(
                f"{row['run']}: ee_pos_peak_SS = {ee_ss!r} vs baseline {base_ee_ss!r}")
        if ss_end is None or base_ss_end is None or ss_end != base_ss_end:
            violations.append(
                f"{row['run']}: ss_end_torso_ori_deg = {ss_end!r} vs baseline {base_ss_end!r}")
        if tstep != base_tstep:
            violations.append(
                f"{row['run']}: preplanner_T_steps[0] = {tstep!r} vs baseline {base_tstep!r}")
        if (ab is None) != (base_ab is None):
            violations.append(f"{row['run']}: abort presence differs from baseline")
        elif ab is not None:
            for k in ('d_mm', 'ori_deg'):
                if ab.get(k) != base_ab.get(k):
                    violations.append(
                        f"{row['run']}: abort {k} = {ab.get(k)!r} "
                        f"vs baseline {base_ab.get(k)!r}")

    return violations


def _render_markdown(rows):
    cols = ['run', 'SS peak ori [°]', 'q_ref jump [°]', 'DS entry ori [°]',
            'DS peak ori [°]', 'max|τ_q|_DS [Nm]', 'τ_q sat frac DS',
            'τ_w sat frac DS', 'dock?']
    lines = ['| ' + ' | '.join(cols) + ' |',
             '|' + '|'.join(['---'] * len(cols)) + '|']
    for r_ in rows:
        lines.append('| ' + ' | '.join(str(r_[c]) for c in cols) + ' |')
    return '\n'.join(lines)


def _render_text(rows):
    cols = ['run', 'SS peak ori [°]', 'q_ref jump [°]', 'DS entry ori [°]',
            'DS peak ori [°]', 'max|τ_q|_DS [Nm]', 'τ_q sat frac DS',
            'τ_w sat frac DS', 'dock?']
    widths = {c: max(len(c), max(len(str(r_[c])) for r_ in rows)) for c in cols}
    header = ' | '.join(c.ljust(widths[c]) for c in cols)
    sep = '-+-'.join('-' * widths[c] for c in cols)
    lines = [header, sep]
    for r_ in rows:
        lines.append(' | '.join(str(r_[c]).ljust(widths[c]) for c in cols))
    return '\n'.join(lines)


def main():
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    for tag, subdir, flag in RUN_SPECS:
        _run_one(tag, subdir, flag)

    rows = [_row_for_run(subdir) for _, subdir, _ in RUN_SPECS]

    violations = _verify_invariants(rows)

    print('\n' + '=' * 110)
    print('M7 post-abort DS diagnostic decomposition — summary')
    print('=' * 110)
    print(_render_text(rows))

    summary_path = os.path.join(RESULTS_ROOT, 'summary.md')
    with open(summary_path, 'w') as f:
        f.write('# M7 post-abort DS diagnostic decomposition — summary\n\n')
        f.write('Four runs per `docs/architecture/M7_DS_DIAGNOSTIC_EXPERIMENTS.md`.\n')
        f.write('Each run toggles at most one `diag_*_on_abort` flag; all other\n')
        f.write('parameters identical to the v21 baseline. `τ_q` saturation threshold\n')
        f.write('is `0.99 · tau_max = 19.8 Nm`; `τ_w` saturation threshold is\n')
        f.write('`0.99 · tau_w_max = 4.95 Nm`. No interpretation included.\n\n')
        f.write(_render_markdown(rows))
        f.write('\n\n')
        if violations:
            f.write('## Invariant violations\n\n')
            for v in violations:
                f.write(f'- {v}\n')
        else:
            f.write('## Invariants\n\n')
            f.write('All SS-side invariants hold (SS peak ori within 0.01° of baseline;\n')
            f.write('`ee_pos_peak_mm_SS`, `ss_end_torso_ori_deg`, `preplanner_T_steps[0]`,\n')
            f.write('abort `d_mm` / `ori_deg` bit-for-bit identical to baseline).\n')

    print()
    print(f'Saved: {summary_path}')
    if violations:
        print('\nINVARIANT VIOLATIONS:')
        for v in violations:
            print(f'  - {v}')
        print('\nPer spec §5: stop and report; do not interpret DS numbers.')
        sys.exit(1)


if __name__ == '__main__':
    main()
