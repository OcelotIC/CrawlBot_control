"""Aggregate baseline (margin=5s, existing) + sweep (10s, 15s) side-by-side.

Reads the three result directories and emits a single comparison report.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

import numpy as np  # noqa: E402


RUNS = [
    ('A', 5,  os.path.join(_root, 'Misc', 'runs', 'diag_nmpc_warm_start_5step')),
    ('B', 10, os.path.join(_root, 'Misc', 'runs', 'diag_nmpc_margin_sweep',
                           'margin_10s')),
    ('C', 15, os.path.join(_root, 'Misc', 'runs', 'diag_nmpc_margin_sweep',
                           'margin_15s')),
]


def _parse_label(label: str):
    if not label:
        return -1, '?'
    parts = label.split('/')
    try:
        step = int(parts[0].replace('step', ''))
    except Exception:
        step = -1
    phase = parts[1] if len(parts) >= 2 else '?'
    return step, phase


def _aggregate(step_log):
    groups: dict = {}
    for e in step_log:
        key = _parse_label(e.get('label', ''))
        groups.setdefault(key, []).append(e)
    out = {}
    for key, entries in groups.items():
        iters = np.array([e['iter'] for e in entries])
        times = np.array([e['time_ms'] for e in entries])
        failed = sum(1 for e in entries
                     if 'Succeeded' not in e['status']
                     and 'Acceptable' not in e['status'])
        statuses = Counter(e['status'] for e in entries)
        out[key] = {
            'N': len(entries),
            'iter_min': int(iters.min()),
            'iter_max': int(iters.max()),
            'iter_p95': int(np.percentile(iters, 95)),
            't_max': float(times.max()),
            't_mean': float(times.mean()),
            'fail': failed,
            'status': ', '.join(f"{s}:{c}" for s, c in statuses.most_common()),
        }
    return out


def _step_outcomes(out_dir, n_steps=5):
    """Read sim_log.json and extract per-step outcome metrics."""
    sim_log_path = os.path.join(out_dir, 'sim_log.json')
    if not os.path.exists(sim_log_path):
        return None
    with open(sim_log_path) as f:
        sl = json.load(f)
    t = np.array(sl.get('t', []))
    phase = np.array(sl.get('phase', []), dtype=object)
    step_idx_arr = np.array(sl.get('step_idx', []))
    p_torso = np.array(sl.get('p_torso', [])) if sl.get('p_torso') else None
    e_ee_pos_raw = sl.get('e_ee_pos', [])
    e_ee_pos = np.array(e_ee_pos_raw) if e_ee_pos_raw else None

    dock_by_step = {ev.get('step', None): ev for ev in sl.get('dock_events', [])}
    abort_by_step = {ab['step_idx']: ab for ab in sl.get('aborted_steps', [])}
    rows = []
    for s in range(n_steps):
        mask_ss = (step_idx_arr == s) & (phase == 'SS')
        if not mask_ss.any():
            rows.append({'step': s, 'outcome': 'NO_DATA',
                         'd_mm': None, 'ori_deg': None,
                         'recoil_mm': None, 'ee_err_mm': None})
            continue
        if p_torso is not None and p_torso.size:
            p0 = p_torso[mask_ss][0]
            disp = np.linalg.norm(p_torso[mask_ss] - p0, axis=1)
            recoil_mm = float(disp.max() * 1000.0)
        else:
            recoil_mm = None
        if e_ee_pos is not None and e_ee_pos.size:
            ee_err = e_ee_pos[mask_ss]
            if ee_err.ndim == 1:
                peak_ee_mm = float(np.abs(ee_err).max() * 1000.0)
            else:
                peak_ee_mm = float(np.linalg.norm(ee_err, axis=1).max() * 1000.0)
        else:
            peak_ee_mm = None
        if s in abort_by_step:
            ab = abort_by_step[s]
            outcome = ab['reason'].upper()
            d_mm = ab.get('d_mm', None)
            ori_deg = ab.get('ori_deg', None)
        elif s in dock_by_step:
            ev = dock_by_step[s]
            outcome = 'DOCK'
            d_mm = ev.get('d_mm', None)
            ori_deg = ev.get('ori_deg', None)
        else:
            outcome = 'UNKNOWN'
            d_mm = ori_deg = None
        rows.append({'step': s, 'outcome': outcome,
                     'd_mm': d_mm, 'ori_deg': ori_deg,
                     'recoil_mm': recoil_mm, 'ee_err_mm': peak_ee_mm})
    return rows


def _fmt(x, w, fmt='.1f', dash='--'):
    if x is None:
        return f"{dash:>{w}s}"
    return f"{x:>{w}{fmt}}"


def main():
    print('=' * 88)
    print('  t_ss_margin parametric sweep — side-by-side report')
    print('  Config: T15-FK reference architecture, 5 steps, start=(2,2)')
    print('=' * 88)

    # Load all three.
    aggregates = {}
    outcomes = {}
    for tag, m, d in RUNS:
        sl_path = os.path.join(d, 'nmpc_step_log.json')
        if not os.path.exists(sl_path):
            print(f"\n  [Run {tag} margin={m}s] MISSING: {sl_path}")
            aggregates[tag] = None
            outcomes[tag] = None
            continue
        with open(sl_path) as f:
            step_log = json.load(f)
        aggregates[tag] = _aggregate(step_log)
        outcomes[tag] = _step_outcomes(d)

    # Per-step outcome table — main result.
    print('\n--- Per-step outcomes ---')
    header = (f"{'Step':>4s} | "
              f"{'Run A (5s)':>30s} | {'Run B (10s)':>30s} | {'Run C (15s)':>30s}")
    print(header)
    print('-' * len(header))
    for s in range(5):
        cells = []
        for tag in 'ABC':
            rows = outcomes[tag]
            if rows is None:
                cells.append(f"{'MISSING':>30s}")
                continue
            r = next((x for x in rows if x['step'] == s), None)
            if r is None:
                cells.append(f"{'NO_DATA':>30s}")
                continue
            out = r['outcome'][:14]
            d = f"{r['d_mm']:.1f}mm" if r['d_mm'] is not None else '--'
            o = f"{r['ori_deg']:.1f}°" if r['ori_deg'] is not None else '--'
            cells.append(f"{out:>14s} d={d:>7s} ori={o:>6s}")
        print(f"{s:>4d} | " + ' | '.join(cells))

    # Step 2 detail (the key step).
    print('\n--- Step 2 detail (peak torso recoil / peak EE error during SS) ---')
    header = (f"{'Run':>4s} {'margin':>8s} {'outcome':>22s} "
              f"{'d [mm]':>8s} {'ori [°]':>8s} {'recoil [mm]':>12s} "
              f"{'EE err [mm]':>12s}")
    print(header)
    print('-' * len(header))
    for tag, m, _ in RUNS:
        if outcomes[tag] is None:
            print(f"{tag:>4s} {m:>8.1f} {'MISSING':>22s}")
            continue
        r = next((x for x in outcomes[tag] if x['step'] == 2), None)
        if r is None:
            print(f"{tag:>4s} {m:>8.1f} {'NO_DATA':>22s}")
            continue
        d = f"{r['d_mm']:.2f}" if r['d_mm'] is not None else '--'
        o = f"{r['ori_deg']:.2f}" if r['ori_deg'] is not None else '--'
        recoil = (f"{r['recoil_mm']:.1f}"
                  if r['recoil_mm'] is not None else '--')
        ee = (f"{r['ee_err_mm']:.1f}"
              if r['ee_err_mm'] is not None else '--')
        print(f"{tag:>4s} {m:>8.1f} {r['outcome']:>22s} {d:>8s} {o:>8s} "
              f"{recoil:>12s} {ee:>12s}")

    # NMPC per-(step, phase) — abbreviated, focus on SS phases.
    print('\n--- NMPC aggregate per (step, phase) ---')
    all_keys = set()
    for tag in 'ABC':
        if aggregates[tag] is not None:
            all_keys.update(aggregates[tag].keys())
    keys = sorted(all_keys)
    header = (f"{'(step,phase)':<14s} | "
              f"{'A iter_max/p95':>16s} {'A fails':>8s} | "
              f"{'B iter_max/p95':>16s} {'B fails':>8s} | "
              f"{'C iter_max/p95':>16s} {'C fails':>8s}")
    print(header)
    print('-' * len(header))
    for key in keys:
        step, phase = key
        cells = []
        for tag in 'ABC':
            agg = aggregates[tag]
            if agg is None or key not in agg:
                cells.append(f"{'--':>16s} {'--':>8s}")
                continue
            row = agg[key]
            cells.append(f"{row['iter_max']:>4d}/{row['iter_p95']:>4d}"
                         f"{'':>7s} {row['fail']:>8d}")
        print(f"step{step:02d}/{phase:<5s}  | " + ' | '.join(cells))

    # Save the report.
    out_path = os.path.join(_root, 'Misc', 'runs', 'diag_nmpc_margin_sweep',
                            'sweep_report.txt')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        # re-emit by calling main steps inline would duplicate code — instead
        # just save the printed output via tee from the caller. As a hook,
        # also dump the JSON-friendly aggregate.
        pass
    with open(out_path.replace('.txt', '.json'), 'w') as f:
        json.dump({
            tag: {
                'margin_s': m,
                'aggregates': {f"{k[0]}/{k[1]}": v
                               for k, v in (aggregates[tag] or {}).items()},
                'outcomes': outcomes[tag],
            } for (tag, m, _) in RUNS
        }, f, indent=2, default=str)
    print(f"\nSummary JSON: {out_path.replace('.txt', '.json')}")


if __name__ == '__main__':
    main()
