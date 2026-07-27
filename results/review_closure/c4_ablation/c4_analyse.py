#!/usr/bin/env python3
"""C4 — metric suite across the three ablation arms.

Reads results/review_closure/c4_ablation/{none,rate,full}/ (sim_log.json +
the --solver-diag fulldiag CSV) and writes c4_metrics.json alongside.

Conventions fixed here on purpose, because C1.6 and C2.3 both found a metric in
this family being misread:

  * ATTITUDE is reported over the TRAVERSAL WINDOW — up to and including the
    last single-support tick — and separately over the full log. On the
    canonical those differ (0.367 vs 0.535 deg) because theta_s keeps rising
    through the 20 s terminal settle. Labelled, never conflated.
  * SATURATION is reported under BOTH conventions (any-axis-per-tick and
    axis-sample) and broken down by phase, because C2.3 found the previously
    cited figure matched neither.
  * TRAVERSAL TIME is reported next to the dock-gate refusal counts, because
    C1.6 established the difference is dominated by refused first approaches,
    not by a slower plan.
"""
import csv
import json
import os
import statistics as st

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
BASE = os.path.join(ROOT, 'results', 'review_closure', 'c4_ablation')
ARMS = ('none', 'rate', 'full')
CAP = 2.5
HW_BOX = 5.0


def load(arm):
    """CSV + the dock channels, preferring the committed meta over the raw log.

    `sim_log.json` is ~8.8 MB per arm and is NOT committed — it is regenerable
    with one command (see C4_ABLATION.md §Artifacts). Everything this analysis
    needs from it (`dock_events`, `dock_gate_trace`, `aborted_steps`) is
    mirrored into `*_fulldiag_meta.json` by the C2.1 exporter change, so the
    committed artifacts are self-sufficient. The raw log is used when present.
    """
    d = os.path.join(BASE, arm)
    csvp = os.path.join(d, f'c4_{arm}_fulldiag.csv')
    rows = list(csv.DictReader(open(csvp))) if os.path.exists(csvp) else []
    raw = os.path.join(d, 'sim_log.json')
    if os.path.exists(raw):
        with open(raw) as fh:
            return json.load(fh), rows
    with open(os.path.join(d, f'c4_{arm}_fulldiag_meta.json')) as fh:
        m = json.load(fh)
    return {'dock_events': m.get('dock_events', []),
            'dock_gate_trace': m.get('dock_gate_trace', []),
            'aborted_steps': m.get('aborted_steps', [])}, rows


def col(rows, k):
    return np.array([float(r[k]) if r[k] not in ('', 'nan') else np.nan
                     for r in rows])


def vec(rows, stem, unit='Nm'):
    """(n,3) array; blank/nan cells -> NaN.

    `Hdot_s_planned_ss_*` is deliberately blank outside SS — `lambda_ref` is
    NaN when the NMPC is bypassed, and the exporter marks that rather than
    fabricating a value. Reading it as 0.0 would silently drag every peak
    downward, so blanks become NaN and the callers mask them.
    """
    def f(x):
        return float(x) if x not in ('', 'nan', 'None') else np.nan
    return np.array([[f(r[f'{stem}_{a}_{unit}']) for a in 'xyz']
                     for r in rows])


def analyse(arm):
    sl, rows = load(arm)
    out = {'arm': arm}
    ph = np.array([r['phase'] for r in rows])
    si = np.array([int(r['step_index']) for r in rows])
    t = col(rows, 't_s')

    # ── completion ──
    dev = sl.get('dock_events', [])
    out['n_docked'] = len(dev)
    out['aborts'] = sl.get('aborted_steps', [])
    out['docks'] = [{'step': e['step'], 't': e['t'], 'd_mm': e['d_mm'],
                     'ori_deg': e['ori_deg'], 'twist': e['twist'],
                     'twist_lin_norm': e.get('twist_lin_norm'),
                     'twist_ang_norm': e.get('twist_ang_norm')} for e in dev]

    # ── envelope: per-step peak per-axis |Hdot_s|inf, planned and realized ──
    plan = vec(rows, 'Hdot_s_planned_ss')
    real = vec(rows, 'Hdot_s_realized_cont')
    ss = ph == 'SS'
    out['env_planned_per_step'] = []
    out['env_realized_per_step'] = []
    for k in range(6):
        m = ss & (si == k)
        p = plan[m]
        p = p[np.isfinite(p).all(1)] if m.any() else np.zeros((0, 3))
        out['env_planned_per_step'].append(
            round(float(np.abs(p).max()), 4) if len(p) else None)
        rr = real[m]
        rr = rr[np.isfinite(rr).all(1)] if m.any() else np.zeros((0, 3))
        out['env_realized_per_step'].append(
            round(float(np.abs(rr).max()), 4) if len(rr) else None)

    # ── attitude: traversal window vs full log ──
    th = np.abs(np.nan_to_num(vec(rows, 'theta_s', 'deg')))
    last_ss = int(np.where(ss)[0].max()) if ss.any() else len(rows) - 1
    out['theta_s_peak_traversal_deg'] = round(float(th[:last_ss + 1].max()), 4)
    out['theta_s_peak_fulllog_deg'] = round(float(th.max()), 4)
    out['traversal_window_end_s'] = round(float(t[last_ss]), 2)

    # ── storage ──
    hw = np.abs(np.nan_to_num(vec(rows, 'hw', 'Nms')))
    out['hw_peak_axis_Nms'] = round(float(hw.max()), 4)
    out['hw_box_fraction_pct'] = round(100 * float(hw.max()) / HW_BOX, 2)
    out['hw_ticks_over_box'] = int((hw > HW_BOX).any(1).sum())

    # ── saturation, both conventions + per phase ──
    pre = np.nan_to_num(vec(rows, 'aocs_tau_w_preclip'))
    sat = np.abs(pre) > CAP + 1e-9
    n = len(rows)
    out['sat_tick_any_axis_pct'] = round(100 * float(sat.any(1).mean()), 3)
    out['sat_axis_sample_pct'] = round(100 * float(sat.sum()) / (3 * n), 3)
    out['sat_by_phase_pct'] = {
        p: round(100 * float(sat[ph == p].any(1).mean()), 3)
        for p in ('SS', 'DS_terminal', 'DS_interstep') if (ph == p).any()}
    out['tau_w_peak_demand_Nm'] = round(float(np.abs(pre).max()), 4)

    # ── solver ──
    it = np.array([int(r['nmpc_iterations']) for r in rows])
    act = ph != 'DS_interstep'
    out['nmpc_solves'] = int(act.sum())
    out['nmpc_iters_total'] = int(it[act].sum())
    out['nmpc_iters_median'] = float(np.median(it[act]))
    out['nmpc_iters_mean'] = round(float(it[act].mean()), 3)

    # ── traversal time + the dock-gate trace behind it ──
    out['t_end_s'] = round(float(t[-1]), 2)
    out['last_dock_s'] = dev[-1]['t'] if dev else None
    per_step = {}
    for k in range(6):
        m = ss & (si == k)
        if m.any():
            per_step[k] = round(float(t[m].max() - t[m].min()), 2)
    out['ss_duration_per_step_s'] = per_step

    trace = sl.get('dock_gate_trace', [])
    refused = [r for r in trace if not r['fired']]
    pose_valid_refused = [r for r in refused
                          if r['d_mm'] < 5.0 and r['ori_deg'] < 5.0]
    out['gate_evals'] = len(trace)
    out['gate_refused'] = len(refused)
    out['gate_refused_pose_valid'] = len(pose_valid_refused)
    out['gate_refused_pose_valid_detail'] = [
        {'step': r['step'], 't': r['t'], 'd_mm': r['d_mm'],
         'ori_deg': r['ori_deg'], 'twist': r['twist'],
         'blocked_by': 'twist' if r['twist'] >= 0.05 else 'unknown'}
        for r in pose_valid_refused]
    by_step = {}
    for r in refused:
        s = r['step']
        b = by_step.setdefault(s, {'total': 0, 'pos': 0, 'ori': 0, 'twist': 0})
        b['total'] += 1
        if r['d_mm'] >= 5.0:
            b['pos'] += 1
        elif r['ori_deg'] >= 5.0:
            b['ori'] += 1
        else:
            b['twist'] += 1
    out['gate_refusals_by_step'] = {str(k): v for k, v in sorted(by_step.items())}
    return out


def main():
    # Key arm discovery on the COMMITTED artifact, not the uncommitted raw log.
    res = {a: analyse(a) for a in ARMS if
           os.path.exists(os.path.join(BASE, a, f'c4_{a}_fulldiag.csv'))}
    with open(os.path.join(BASE, 'c4_metrics.json'), 'w') as fh:
        json.dump(res, fh, indent=2)

    have = [a for a in ARMS if a in res]
    W = 14

    def line(label, fn):
        print(f'  {label:34s}' + ''.join(f'{str(fn(res[a])):>{W}}'
                                         for a in have))

    print('=' * (36 + W * len(have)))
    print('C4 — three-way envelope ablation')
    print(f"  {'':34s}" + ''.join(f'{a:>{W}}' for a in have))
    print('=' * (36 + W * len(have)))
    print('COMPLETION')
    line('docked steps / 6', lambda r: r['n_docked'])
    line('aborts', lambda r: len(r['aborts']))
    print('ENVELOPE  peak per-axis |Hdot_s|inf [Nm]')
    for k in range(6):
        line(f'  step {k}  planned',
             lambda r, k=k: r['env_planned_per_step'][k])
    for k in range(6):
        line(f'  step {k}  realized',
             lambda r, k=k: r['env_realized_per_step'][k])
    print('ATTITUDE')
    line('theta_s peak, TRAVERSAL [deg]',
         lambda r: r['theta_s_peak_traversal_deg'])
    line('theta_s peak, full log [deg]',
         lambda r: r['theta_s_peak_fulllog_deg'])
    print('STORAGE')
    line('|h_w| peak per-axis [Nms]', lambda r: r['hw_peak_axis_Nms'])
    line('  as % of the +-5 box', lambda r: r['hw_box_fraction_pct'])
    line('  ticks outside the box', lambda r: r['hw_ticks_over_box'])
    print('SATURATION at the AOCS cap (+-2.5)')
    line('any-axis, per tick [%]', lambda r: r['sat_tick_any_axis_pct'])
    line('axis-sample [%]', lambda r: r['sat_axis_sample_pct'])
    for p in ('SS', 'DS_interstep', 'DS_terminal'):
        line(f'  [{p}] any-axis [%]',
             lambda r, p=p: r['sat_by_phase_pct'].get(p, '-'))
    line('peak |demand| pre-clip [Nm]', lambda r: r['tau_w_peak_demand_Nm'])
    print('SOLVER')
    line('NMPC solves', lambda r: r['nmpc_solves'])
    line('IPOPT iters, total', lambda r: r['nmpc_iters_total'])
    line('IPOPT iters, median', lambda r: r['nmpc_iters_median'])
    line('IPOPT iters, mean', lambda r: r['nmpc_iters_mean'])
    print('TRAVERSAL TIME  (read with the gate trace below, not alone)')
    line('log end [s]', lambda r: r['t_end_s'])
    line('last dock [s]', lambda r: r['last_dock_s'])
    line('gate evaluations', lambda r: r['gate_evals'])
    line('  refused', lambda r: r['gate_refused'])
    line('  refused while POSE-VALID', lambda r: r['gate_refused_pose_valid'])
    print()
    for a in have:
        r = res[a]
        print(f'  [{a}] SS duration per step: {r["ss_duration_per_step_s"]}')
        print(f'  [{a}] refusals by step (pos/ori/twist): '
              f'{r["gate_refusals_by_step"]}')
        for d in r['gate_refused_pose_valid_detail']:
            print(f'        POSE-VALID REFUSED  step {d["step"]} t={d["t"]:.2f} '
                  f'd={d["d_mm"]:.3f}mm ori={d["ori_deg"]:.3f} '
                  f'twist={d["twist"]:.6f} -> {d["blocked_by"]}')
    print()
    print(f"wrote {os.path.join(BASE, 'c4_metrics.json')}")


if __name__ == '__main__':
    main()
