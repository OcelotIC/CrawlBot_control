#!/usr/bin/env python3
"""C3.4 — solver-stats table for the paper, from committed artifacts only.

No new physics run. Reads:
  results/j2_adjconv/c25_fulldiag.csv        committed canonical, managed
  results/j2_adjconv/u25_fulldiag.csv        committed canonical, unmanaged
  gate/_run/c25_replay_fulldiag.csv          C0 replay on THIS container
                                             (byte-identical on all 64
                                             non-timing columns; the two
                                             wall-clock columns differ by
                                             machine, which is the point)
  results/gate_run_scratch/sim_log.json      IPOPT return-status strings
                                             (present in the log, absent
                                             from the 66-column CSV)

Two measurement facts that govern every number below (verified in source,
see C3_4_SOLVER_STATS.md):

 1. `nmpc_time_ms` brackets sim_loop.py:2198-2256 — the IPOPT NMPC solve plus
    a little surrounding work. Fair proxy for "NMPC solve time".
 2. `qp_time_ms` brackets sim_loop.py:2316-2947, which contains
    `for qs in range(n_qp_per_nmpc)` (=10) at :2326. It is therefore the
    WBC BLOCK time for ten QP solves plus ten Pinocchio updates, ten AOCS
    evaluations, ten MuJoCo steps and the per-tick logging. It is NOT a QP
    solve time. Divide by 10 for a per-solve UPPER BOUND.
 3. On DS_interstep ticks the NMPC is bypassed and BOTH timers are written as
    0.0 sentinels (tick_logging.py:318-319), with `nmpc_ok=False` ("not run;
    not a failure") and `qp_ok=True` hardcoded. Those ticks must be excluded
    from every distribution or the median collapses to zero.

Writes results/review_closure/c3/c3_4_solver_stats.json
"""
import csv
import json
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(ROOT, 'results', 'review_closure', 'c3')

SOURCES = [
    ('C_committed', 'results/j2_adjconv/c25_fulldiag.csv',
     'committed canonical (managed) — timings from the ARTIFACT-GENERATION '
     'machine, hardware unrecorded'),
    ('U_committed', 'results/j2_adjconv/u25_fulldiag.csv',
     'committed canonical (unmanaged) — same, hardware unrecorded'),
    ('C_replay_here', 'gate/_run/c25_replay_fulldiag.csv',
     'C0 replay on THIS container — hardware known, physics byte-identical'),
]


def pct(xs, p):
    """Linear-interpolated percentile (numpy's default 'linear' method)."""
    xs = sorted(xs)
    if not xs:
        return float('nan')
    k = (len(xs) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def dist(xs):
    if not xs:
        return None
    return {
        'n': len(xs),
        'min': min(xs), 'median': st.median(xs), 'mean': st.fmean(xs),
        'p95': pct(xs, 95), 'p99': pct(xs, 99), 'max': max(xs),
        'stdev': st.pstdev(xs) if len(xs) > 1 else 0.0,
    }


def analyse(path):
    with open(os.path.join(ROOT, path)) as fh:
        rows = list(csv.DictReader(fh))
    # NMPC-active ticks: the NMPC actually ran. Bypass sentinel is exactly the
    # DS_interstep phase; assert that rather than trust it.
    active = [r for r in rows if r['phase'] != 'DS_interstep']
    bypass = [r for r in rows if r['phase'] == 'DS_interstep']
    assert all(float(r['nmpc_time_ms']) == 0.0 for r in bypass), \
        'a DS_interstep tick carries a non-zero NMPC timer'
    assert all(float(r['nmpc_time_ms']) > 0.0 for r in active), \
        'an NMPC-active tick carries a zero timer'

    by_phase = {}
    for r in active:
        by_phase.setdefault(r['phase'], []).append(r)

    nm = [float(r['nmpc_time_ms']) for r in active]
    qp = [float(r['qp_time_ms']) for r in active]
    it = [int(r['nmpc_iterations']) for r in active]

    return {
        'path': path,
        'n_rows': len(rows),
        'n_nmpc_active': len(active),
        'n_nmpc_bypassed': len(bypass),
        'phase_counts': {k: len(v) for k, v in
                         sorted(by_phase.items())},
        'nmpc_solve_ms': dist(nm),
        'wbc_block_ms': dist(qp),
        'wbc_per_qp_solve_ms_upper_bound': dist([x / 10.0 for x in qp]),
        'nmpc_iterations': dist(it),
        'nmpc_iterations_histogram': {
            str(k): it.count(k) for k in sorted(set(it))},
        'nmpc_ok_false_count': sum(1 for r in rows if r['nmpc_ok'] != '1'),
        'nmpc_ok_false_among_active': sum(
            1 for r in active if r['nmpc_ok'] != '1'),
        'qp_ok_false_count': sum(1 for r in rows if r['qp_ok'] != '1'),
        'qp_status_unrecorded_ticks': len(bypass),
        'nmpc_solve_ms_by_phase': {
            k: dist([float(r['nmpc_time_ms']) for r in v])
            for k, v in sorted(by_phase.items())},
        'wbc_block_ms_by_phase': {
            k: dist([float(r['qp_time_ms']) for r in v])
            for k, v in sorted(by_phase.items())},
        'nmpc_iterations_by_phase': {
            k: dist([int(r['nmpc_iterations']) for r in v])
            for k, v in sorted(by_phase.items())},
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {'sources': {}, 'notes': {}}
    for tag, path, note in SOURCES:
        res['sources'][tag] = analyse(path)
        res['notes'][tag] = note

    # IPOPT return-status strings — present in the sim_log, absent from the CSV
    slp = os.path.join(ROOT, 'results', 'gate_run_scratch', 'sim_log.json')
    if os.path.exists(slp):
        with open(slp) as fh:
            sl = json.load(fh)
        ss = sl.get('nmpc_status_str', [])
        counts = {}
        for s in ss:
            counts[s] = counts.get(s, 0) + 1
        res['ipopt_return_status_from_sim_log'] = counts
        res['ipopt_return_status_active_only'] = {
            k: v for k, v in counts.items() if k != 'NMPC_BYPASSED'}

    res['real_time_budget'] = {
        'dt_nmpc_ms': 100.0, 'dt_qp_ms': 10.0,
        'n_qp_per_nmpc': 10,
        'comment': ('the WBC block must fit in dt_nmpc = 100 ms and the NMPC '
                    'solve must fit alongside it for real-time operation; '
                    'these runs are NOT real-time and make no such claim'),
    }

    with open(os.path.join(OUT, 'c3_4_solver_stats.json'), 'w') as fh:
        json.dump(res, fh, indent=2)

    # ---------------- printout ----------------
    def row(name, d, unit='ms'):
        if d is None:
            return f"  {name:34s}      -- no samples --"
        return (f"  {name:34s} n={d['n']:5d}  med {d['median']:8.2f}  "
                f"mean {d['mean']:8.2f}  p95 {d['p95']:8.2f}  "
                f"p99 {d['p99']:8.2f}  max {d['max']:9.2f}  [{unit}]")

    print('=' * 100)
    print('C3.4 — solver statistics (committed artifacts; no new run)')
    print('=' * 100)
    for tag, path, note in SOURCES:
        a = res['sources'][tag]
        print()
        print(f'--- {tag} :: {path}')
        print(f'    {note}')
        print(f"    {a['n_rows']} rows = {a['n_nmpc_active']} NMPC-active "
              f"+ {a['n_nmpc_bypassed']} NMPC-bypassed   "
              f"phases {a['phase_counts']}")
        print(row('NMPC solve time', a['nmpc_solve_ms']))
        print(row('WBC block (10 QP + physics)', a['wbc_block_ms']))
        print(row('  -> per-QP-solve upper bound', a['wbc_per_qp_solve_ms_upper_bound']))
        print(row('NMPC IPOPT iterations', a['nmpc_iterations'], 'iters'))
        print(f"    iteration histogram: {a['nmpc_iterations_histogram']}")
        print(f"    nmpc_ok==False: {a['nmpc_ok_false_count']} total "
              f"({a['nmpc_ok_false_among_active']} among NMPC-active ticks)")
        print(f"    qp_ok==False:   {a['qp_ok_false_count']}   "
              f"(QP status NOT recorded on {a['qp_status_unrecorded_ticks']} "
              f"DS_interstep ticks)")

    if 'ipopt_return_status_from_sim_log' in res:
        print()
        print('IPOPT return status (from sim_log.json — NOT a CSV column):')
        for k, v in sorted(res['ipopt_return_status_from_sim_log'].items(),
                           key=lambda kv: -kv[1]):
            print(f"    {k:34s} {v}")

    print()
    print('Per-phase breakdown (C_committed):')
    a = res['sources']['C_committed']
    for ph in sorted(a['nmpc_solve_ms_by_phase']):
        print(f"  [{ph}]")
        print(row('    NMPC solve', a['nmpc_solve_ms_by_phase'][ph]))
        print(row('    WBC block', a['wbc_block_ms_by_phase'][ph]))
        print(row('    iterations', a['nmpc_iterations_by_phase'][ph], 'iters'))

    print()
    print(f"wrote {os.path.join(OUT, 'c3_4_solver_stats.json')}")


if __name__ == '__main__':
    main()
