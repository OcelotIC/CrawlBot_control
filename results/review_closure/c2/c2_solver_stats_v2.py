#!/usr/bin/env python3
"""C2 deliverable 2/3 — the C3.4 table re-run with TRUE per-QP-solve statistics.

Replaces C3.4's `qp_time_ms / 10` upper bound (which contained ten Pinocchio
updates, ten AOCS evaluations, ten mj_step calls and the logging) with the QP's
own `solve_time_ms`, and answers deliverable 3: with the physics removed, is
the controller-only QP cost inside the 10 ms tick budget?

Reads  results/review_closure/c2/c25_c2_fulldiag.csv      (--solver-diag export)
       results/review_closure/c2/c25_c2_fulldiag_meta.json (pre-planner + host)
Writes results/review_closure/c2/c2_solver_stats_v2.json
"""
import csv
import json
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
HERE = os.path.join(ROOT, 'results', 'review_closure', 'c2')
CSV = os.path.join(HERE, 'c25_c2_fulldiag.csv')
META = os.path.join(HERE, 'c25_c2_fulldiag_meta.json')

DT_QP_MS = 10.0        # cfg.dt_qp  = 0.01 s
DT_NMPC_MS = 100.0     # cfg.dt_nmpc = 0.1 s


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return float('nan')
    k = (len(xs) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def dist(xs, unit=''):
    if not xs:
        return None
    return {'n': len(xs), 'min': min(xs), 'median': st.median(xs),
            'mean': st.fmean(xs), 'p95': pct(xs, 95), 'p99': pct(xs, 99),
            'max': max(xs), 'unit': unit}


def main():
    rows = list(csv.DictReader(open(CSV)))
    meta = json.load(open(META))

    grp = {'SS': [], 'DS_terminal': [], 'DS_interstep': []}
    for r in rows:
        grp[r['phase']].append(r)

    # per-SOLVE mean time, reconstructed from the per-tick aggregate
    solve_mean, solve_max, iters, blocks = [], [], [], []
    per_phase = {}
    n_solves_total = 0
    for ph, rs in grp.items():
        sm, sx, it = [], [], []
        for r in rs:
            n = int(r['qp_n_solves'])
            if n == 0:
                continue                       # documented sentinel row
            n_solves_total += n
            sm.append(float(r['qp_solve_ms_sum']) / n)
            sx.append(float(r['qp_solve_ms_max']))
            it.append(int(r['qp_iter_sum']) / n)
        per_phase[ph] = {
            'n_ticks': len(rs), 'n_solves': sum(int(r['qp_n_solves'])
                                                for r in rs),
            'per_solve_mean_ms': dist(sm, 'ms'),
            'per_tick_worst_solve_ms': dist(sx, 'ms'),
            'per_solve_iters': dist(it, 'iters'),
        }
        solve_mean += sm
        solve_max += sx
        iters += it
    for r in rows:
        if int(r['qp_n_solves']) > 0:
            blocks.append(float(r['qp_time_ms']))

    # budget: the binding question is the WORST single solve in a tick
    worst = [float(r['qp_solve_ms_max']) for r in rows
             if int(r['qp_n_solves']) > 0]
    over = [w for w in worst if w > DT_QP_MS]
    # controller-only per-NMPC-tick cost = sum of QP solves + NMPC solve
    ctrl = []
    for r in rows:
        if int(r['qp_n_solves']) == 0:
            continue
        ctrl.append(float(r['qp_solve_ms_sum']) + float(r['nmpc_time_ms']))
    ctrl_over = [c for c in ctrl if c > DT_NMPC_MS]

    # QP share of the measured WBC block
    share = []
    for r in rows:
        b = float(r['qp_time_ms'])
        if b > 0:
            share.append(100.0 * float(r['qp_solve_ms_sum']) / b)

    res = {
        'source_csv': os.path.relpath(CSV, ROOT),
        'host': meta.get('host', {}),
        'library_versions': meta.get('library_versions', {}),
        'n_ticks': len(rows),
        'n_qp_solves_recorded': n_solves_total,
        'per_solve_mean_ms': dist(solve_mean, 'ms'),
        'per_tick_worst_solve_ms': dist(worst, 'ms'),
        'per_solve_iters': dist(iters, 'iters'),
        'wbc_block_ms': dist(blocks, 'ms'),
        'qp_share_of_block_pct': dist(share, '%'),
        'controller_only_per_nmpc_tick_ms': dist(ctrl, 'ms'),
        'budget': {
            'dt_qp_ms': DT_QP_MS, 'dt_nmpc_ms': DT_NMPC_MS,
            'ticks_with_a_solve_over_dt_qp': len(over),
            'ticks_measured': len(worst),
            'worst_single_solve_ms': max(worst) if worst else None,
            'controller_only_over_dt_nmpc': len(ctrl_over),
        },
        'per_phase': per_phase,
        'outcomes': {
            'qp_status_worst_histogram': {},
            'qp_n_failed_total': sum(int(r['qp_n_failed']) for r in rows),
            'nmpc_status_str_histogram': {},
            'sentinel_rows': sum(1 for r in rows
                                 if int(r['qp_n_solves']) == 0),
        },
        'preplanner_stats': meta.get('preplanner_stats', []),
    }
    for r in rows:
        h = res['outcomes']['qp_status_worst_histogram']
        h[r['qp_status_worst']] = h.get(r['qp_status_worst'], 0) + 1
        h2 = res['outcomes']['nmpc_status_str_histogram']
        h2[r['nmpc_status_str']] = h2.get(r['nmpc_status_str'], 0) + 1

    pp = res['preplanner_stats']
    if pp:
        res['preplanner_summary'] = {
            'n_solves': len(pp),
            'solve_ms': dist([p['solve_ms'] for p in pp], 'ms'),
            'iter_count': dist([p['iter_count'] for p in pp], 'iters'),
            'all_success': all(p['success'] for p in pp),
            'statuses': sorted({p['status'] for p in pp}),
            'total_solve_ms': sum(p['solve_ms'] for p in pp),
        }

    with open(os.path.join(HERE, 'c2_solver_stats_v2.json'), 'w') as fh:
        json.dump(res, fh, indent=2)

    def row(label, d):
        if d is None:
            return f'  {label:36s}   -- none --'
        return (f"  {label:36s} n={d['n']:5d}  med {d['median']:8.3f}  "
                f"mean {d['mean']:8.3f}  p95 {d['p95']:8.3f}  "
                f"p99 {d['p99']:8.3f}  max {d['max']:9.3f} [{d['unit']}]")

    print('=' * 104)
    print('C2 — solver statistics with TRUE per-QP-solve timing')
    h = res['host']
    print(f"  host: {h.get('cpu_model', '?')} | {h.get('cpu_count_logical')} "
          f"logical CPUs | {h.get('mem_total', '?')} | {h.get('platform', '?')}")
    print(f"  {res['n_qp_solves_recorded']} QP solves recorded over "
          f"{res['n_ticks']} ticks "
          f"({res['outcomes']['sentinel_rows']} sentinel rows)")
    print('=' * 104)
    print(row('QP solve, per-solve mean', res['per_solve_mean_ms']))
    print(row('QP solve, worst in tick', res['per_tick_worst_solve_ms']))
    print(row('qpOASES iters per solve', res['per_solve_iters']))
    print(row('WBC block (for contrast)', res['wbc_block_ms']))
    print(row('QP share of the WBC block', res['qp_share_of_block_pct']))
    print(row('controller-only per NMPC tick', res['controller_only_per_nmpc_tick_ms']))
    print()
    b = res['budget']
    print(f"BUDGET  dt_qp = {b['dt_qp_ms']:.0f} ms:  "
          f"ticks whose WORST solve exceeded it: "
          f"{b['ticks_with_a_solve_over_dt_qp']} / {b['ticks_measured']}   "
          f"(worst single solve {b['worst_single_solve_ms']:.3f} ms)")
    print(f"        dt_nmpc = {b['dt_nmpc_ms']:.0f} ms: controller-only "
          f"(all QP solves + NMPC) over budget: "
          f"{b['controller_only_over_dt_nmpc']} / {b['ticks_measured']}")
    print()
    print('OUTCOMES')
    print(f"  qp_status_worst : {res['outcomes']['qp_status_worst_histogram']}"
          '   (0 = backend reported success)')
    print(f"  qp_n_failed     : {res['outcomes']['qp_n_failed_total']}")
    print(f"  nmpc_status_str : {res['outcomes']['nmpc_status_str_histogram']}")
    print()
    print('PER PHASE')
    for ph in ('SS', 'DS_terminal', 'DS_interstep'):
        p = per_phase[ph]
        print(f"  [{ph}]  {p['n_ticks']} ticks, {p['n_solves']} solves")
        print(row('    per-solve mean', p['per_solve_mean_ms']))
        print(row('    iters per solve', p['per_solve_iters']))
    print()
    if 'preplanner_summary' in res:
        s = res['preplanner_summary']
        print('COARSE PRE-PLANNER (IPOPT NLP, once per step) — never persisted before')
        print(row('    solve time', s['solve_ms']))
        print(row('    iterations', s['iter_count']))
        print(f"    {s['n_solves']} solves, all_success={s['all_success']}, "
              f"statuses={s['statuses']}, total {s['total_solve_ms']:.1f} ms")
        for p in res['preplanner_stats']:
            print(f"      step T={p['T_step']:.3f}s  {p['solve_ms']:8.1f} ms  "
                  f"{p['iter_count']:4d} iters  cost {p['cost']:.4f}  "
                  f"{p['status']}")
    print()
    print(f"wrote {os.path.join(HERE, 'c2_solver_stats_v2.json')}")


if __name__ == '__main__':
    main()
