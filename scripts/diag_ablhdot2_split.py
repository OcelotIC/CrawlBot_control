#!/usr/bin/env python3
"""Phase ABL-HDOT-2 Step 2 — is U's SS-swing over-5 REAL swing demand or a near-weld
transient? Weld-window sensitivity + peak localization on the committed continuous
column. READ-ONLY (reads the augmented ablation timeseries; writes nothing).
Run: MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_ablhdot2_split.py
"""
import csv
import numpy as np

DATA = 'results/j2_ablation_envelope/ablation_data'

for tag in ('C', 'U'):
    rows = list(csv.DictReader(open(f'{DATA}/ablation_{tag}_timeseries.csv')))
    n = len(rows)
    t = np.array([float(r['t_s']) for r in rows])
    ph = np.array([r['phase'] for r in rows])                       # DS/SS
    phr = np.array([r['phase_raw'] for r in rows])
    cont = np.stack([[float(r[f'Hdot_s_realized_cont_{a}_Nm']) for a in 'xyz'] for r in rows])
    contabs = np.abs(cont)
    is_ss = ph == 'SS'
    bnd = np.where(is_ss[1:] != is_ss[:-1])[0]
    # distance (in ticks) from each tick to the nearest SS<->DS boundary
    bidx = np.concatenate([bnd, bnd + 1]) if len(bnd) else np.array([], int)
    dist = np.full(n, 10**6)
    for b in bidx:
        dist = np.minimum(dist, np.abs(np.arange(n) - b))

    print('=' * 74)
    print(f'RUN {tag}: SS-swing peak/axis vs weld-exclusion width W (ticks/side)')
    print(f'  {"W":>3} {"n_swing":>8} | {"peak_x":>7} {"peak_y":>7} {"peak_z":>7} | any>5?')
    for W in (0, 2, 3, 5, 10):
        weld = np.zeros(n, bool)
        for b in bnd:
            weld[max(0, b - W + 1):min(n, b + W + 1)] = True
        swing = is_ss & ~weld
        pk = contabs[swing].max(axis=0) if swing.any() else np.zeros(3)
        print(f'  {W:>3} {int(swing.sum()):>8} | {pk[0]:7.2f} {pk[1]:7.2f} {pk[2]:7.2f} | {(pk>5).any()}')

    # locate the SS z-peak (dominant axis) and its distance to nearest weld boundary
    ss_idx = np.where(is_ss)[0]
    zc = contabs[ss_idx, 2]
    k = ss_idx[zc.argmax()]
    print(f'  SS |Hdot_z| peak = {contabs[k,2]:.2f} at tick {k} (t={t[k]:.2f}, {phr[k]}), '
          f'{dist[k]} ticks from nearest SS<->DS boundary')
    lo, hi = max(0, k - 4), min(n, k + 5)
    print('   context (tick: phase_raw  |Hdot_cont| xyz  dist2boundary):')
    for i in range(lo, hi):
        star = ' <=peak' if i == k else ''
        print(f'     {i:4d}: {phr[i]:13s} [{contabs[i,0]:5.2f}{contabs[i,1]:6.2f}{contabs[i,2]:6.2f}]  d={dist[i]}{star}')
