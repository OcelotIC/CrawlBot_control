#!/usr/bin/env python3
"""Phase ABL-HDOT-2 STEP 0 — prove the gitignored raw logs ARE the committed runs.

For each run, compare the raw sim_log arrays (the SAME keys the committed CSV was
built from, per export_figure_data.py:127-267) against the committed tidy CSV,
tick-for-tick, and report the max abs deviation per quantity. Match to CSV write
precision (~1e-6 for %.6f, ~rel-1e-7 for %.6e) => the log IS the committed run.
Any quantity diverging beyond float tolerance => NOT the same run => STOP.

Idriss lifted the no-gitignored-logs rule for THIS extraction only. READ-ONLY;
no crawlbot/ change, no new sim.
Run: MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_ablhdot2_identity.py
"""
import csv
import json
import numpy as np

from scripts.export_figure_data import phase_per_tick

RUNS = [
    ('C', 'results/figC_qpcond/sim_log.json',
     'results/j2_canonical_revalidation/runfix_traversal.csv'),
    ('U', 'results/figU_rateoff/sim_log.json',
     'results/j2_ablation_envelope/runU_rateoff_traversal.csv'),
]

# committed CSV column (per axis stem) -> sim_log key (export_figure_data.py mapping)
VEC = [
    ('tauw', 'Nm', 'tau_w'),
    ('hw', 'Nms', 'hw_physical'),
    ('theta_s', 'deg', 'struct_euler_deg'),
    ('rcom', 'm', 'r_com'),
    ('torso_pos', 'm', 'p_torso'),
]
SCALAR = [
    ('torso_ori_err_deg', 'e_torso_ori'),
    ('swing_dist_m', 'd_grip_swing'),
]
TOL = 1e-4   # >> CSV rounding (~3e-6); << any real dynamics difference (>=1e-3)


def csv_col(rows, name):
    return np.array([float(r[name]) if r[name] not in ('', None) else np.nan for r in rows])


overall_ok = True
for tag, logp, csvp in RUNS:
    print('=' * 78)
    print(f'RUN {tag}:  log={logp}\n         csv={csvp}')
    sl = json.load(open(logp))
    rows = list(csv.DictReader(open(csvp)))
    nlog = len(sl['t'])
    ncsv = len(rows)
    print(f'  tick count: log={nlog}  csv={ncsv}  ->  {"MATCH" if nlog==ncsv else "MISMATCH"}')
    if nlog != ncsv:
        print('  !! tick-count mismatch -> NOT the same run -> STOP')
        overall_ok = False
        continue
    n = nlog
    run_ok = True
    print(f'  {"quantity":22s} {"max|dev|":>12s}   verdict')
    # vector quantities (xyz)
    for stem, unit, key in VEC:
        arr = np.asarray(sl[key], float)[:n]
        for j, a in enumerate('xyz'):
            col = csv_col(rows, f'{stem}_{a}_{unit}')
            dev = float(np.nanmax(np.abs(arr[:, j] - col)))
            ok = dev < TOL
            run_ok &= ok
            print(f'  {stem+"_"+a:22s} {dev:12.3e}   {"ok" if ok else "FAIL"}')
    # scalar quantities
    for stem, key in SCALAR:
        arr = np.asarray(sl[key], float)[:n]
        col = csv_col(rows, stem)
        dev = float(np.nanmax(np.abs(arr - col)))
        ok = dev < TOL
        run_ok &= ok
        print(f'  {stem:22s} {dev:12.3e}   {"ok" if ok else "FAIL"}')
    # step_index (exact int)
    sidx = np.asarray(sl['step_idx'], int)[:n]
    csv_sidx = np.array([int(r['step_index']) for r in rows])
    n_mis = int((sidx != csv_sidx).sum())
    print(f'  {"step_index":22s} {n_mis:12d}   {"ok" if n_mis==0 else "FAIL"} (mismatched ticks)')
    run_ok &= (n_mis == 0)
    # phase (exact string, via the export's own phase_per_tick)
    ph = np.array(phase_per_tick(sl))[:n]
    csv_ph = np.array([r['phase'] for r in rows])
    n_pmis = int((ph != csv_ph).sum())
    print(f'  {"phase":22s} {n_pmis:12d}   {"ok" if n_pmis==0 else "FAIL"} (mismatched ticks)')
    run_ok &= (n_pmis == 0)

    print(f'  --> RUN {tag}: {"IDENTITY CONFIRMED" if run_ok else "DIVERGENT -> STOP"}')
    overall_ok &= run_ok

print('=' * 78)
print(f'OVERALL: {"BOTH RUNS IDENTITY-CONFIRMED — safe to proceed" if overall_ok else "STOP — a run diverged"}')
