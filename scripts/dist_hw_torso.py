#!/usr/bin/env python3
"""Read-only distributions for C5 h_w and C2 torso-pos on the committed canonical CSV.
h_w is the GATED quantity = inf-norm per tick = max(|hw_x|,|hw_y|,|hw_z|) (C5 gate <=4.5).
C2 = ||torso_pos - torso_pos_ref|| (Euclidean) on SS rows, in mm.
"""
import csv
import sys
import numpy as np

CSV = sys.argv[1] if len(sys.argv) > 1 else 'results/j2_figdata/runA_traversal.csv'
rows = list(csv.DictReader(open(CSV)))
n = len(rows)
ph = np.array([r['phase'] for r in rows])
step = np.array([int(r['step_index']) for r in rows])
t = np.array([float(r['t_s']) for r in rows])
hw = np.stack([[float(r['hw_x_Nms']), float(r['hw_y_Nms']), float(r['hw_z_Nms'])] for r in rows])
hw_inf = np.max(np.abs(hw), axis=1)            # the C5 gated inf-norm
hw_eucl = np.linalg.norm(hw, axis=1)           # for context
tp = np.stack([[float(r['torso_pos_x_m']), float(r['torso_pos_y_m']), float(r['torso_pos_z_m'])] for r in rows])
tpr = np.stack([[float(r['torso_pos_ref_x_m']), float(r['torso_pos_ref_y_m']), float(r['torso_pos_ref_z_m'])] for r in rows])
etp_mm = np.linalg.norm(tp - tpr, axis=1) * 1000.0
ss = ph == 'SS'

print(f'CSV {CSV}  n={n} ticks  phases={sorted(set(ph))}  steps={sorted(set(step.tolist()))}')
pct = lambda m: 100.0 * m.sum() / n

print('\n================ 1. h_w (C5)  — inf-norm = max|component| (the gated quantity) ================')
print('Occurrences (full run, of %d ticks):' % n)
for thr in (4.5, 4.7, 4.9):
    m = hw_inf > thr
    by_ph = {p: int(((ph == p) & m).sum()) for p in sorted(set(ph))}
    print(f'  ||h_w||inf > {thr}: {int(m.sum()):4d} ticks ({pct(m):.2f}%)   by phase {by_ph}')

k = int(np.argmax(hw_inf))
print(f'\nLocation of the max ||h_w||inf = {hw_inf[k]:.4f}:  tick={k}  t={t[k]:.2f}s  phase={ph[k]}  step={step[k]}')
print('  per-axis at peak: hw=[%.3f, %.3f, %.3f]  (Euclidean ||h_w||=%.3f)' % (hw[k,0], hw[k,1], hw[k,2], hw_eucl[k]))
lo, hi = max(0, k - 10), min(n, k + 11)
print('  +/-10 ticks around the peak (tick: t phase step | hw_inf):')
for i in range(lo, hi):
    mark = '  <== PEAK' if i == k else ''
    print(f'     {i:3d}: {t[i]:6.2f} {ph[i]:12s} s{step[i]} | {hw_inf[i]:.4f}{mark}')

print('\nDistribution of ||h_w||inf:')
for lbl, a in (('full-run', hw_inf), ('SS-only', hw_inf[ss]), ('DS-only', hw_inf[~ss])):
    print(f'  {lbl:9s}: median {np.median(a):.4f}  p95 {np.percentile(a,95):.4f}  '
          f'p99 {np.percentile(a,99):.4f}  max {a.max():.4f}')

print('\nPer-step ||h_w||inf (the bounded-vs-accumulating signal; crawl distance grows with step):')
print('  step | n_ticks | median |  p95  |  max  | #>4.5 | peak-phase | peak-t')
for s in sorted(set(step.tolist())):
    sel = step == s
    a = hw_inf[sel]
    ks = np.where(sel)[0][np.argmax(a)]
    print(f'   {s:2d}  | {int(sel.sum()):6d}  | {np.median(a):.3f}  | {np.percentile(a,95):.3f} | '
          f'{a.max():.3f} | {int((a>4.5).sum()):4d}  | {ph[ks]:11s}| {t[ks]:.2f}s')

print('\n================ 2. C2 torso-pos  — SS ||torso_pos - ref|| (mm) ================')
a = etp_mm[ss]
print(f'  SS distribution: median {np.median(a):.2f}  p95 {np.percentile(a,95):.2f}  '
      f'p99 {np.percentile(a,99):.2f}  max {a.max():.2f} mm  (n_SS={int(ss.sum())})')
kk = np.where(ss)[0][np.argmax(a)]
print(f'  Location of SS max {a.max():.2f} mm: tick={kk} t={t[kk]:.2f}s phase={ph[kk]} step={step[kk]}')
print('  Per-step SS ||e_torso_pos|| (mm):  step | n_SS | median | p95 | max')
for s in sorted(set(step[ss].tolist())):
    sel = ss & (step == s)
    if sel.sum() == 0:
        continue
    b = etp_mm[sel]
    print(f'     {s:2d} | {int(sel.sum()):4d} | {np.median(b):6.2f} | {np.percentile(b,95):6.2f} | {b.max():6.2f}')
