#!/usr/bin/env python3
"""Phase ABL-CMD — locate the tau_w saturation in the COMMITTED runU CSV.
Where do the |tau_w|>=5 ticks live (SS vs DS), and what is the tagged Ḣ_s there?
On DS the AOCS feedforward is -Ḣ_s_realized (committed); on SS it is the
kinematic -L_dot-orbital (NOT committed). This decides whether ANY faithful
reconstruction from committed data is possible. READ-ONLY, no commit.
"""
import csv
import numpy as np

import sys
CSV = sys.argv[1] if len(sys.argv) > 1 else 'results/j2_ablation_envelope/runU_rateoff_traversal.csv'
print(f'### {CSV}')
rows = list(csv.DictReader(open(CSV)))
def col(name):
    return np.array([float(r[name]) if r[name] != '' else np.nan for r in rows])

phase = np.array([r['phase'] for r in rows])
src = np.array([r['Hdot_s_source'] for r in rows])
tauw = np.stack([col(f'tauw_{a}_Nm') for a in 'xyz'], 1)
hdot = np.stack([col(f'Hdot_s_{a}_Nm') for a in 'xyz'], 1)
tauw_inf = np.abs(tauw).max(1)
hdot_inf = np.abs(hdot).max(1)

is_ss = np.array([p == 'SS' for p in phase])
is_ds = ~is_ss
sat = tauw_inf >= 5.0 - 1e-3
print(f'total ticks: {len(rows)}   SS: {is_ss.sum()}   DS: {is_ds.sum()}')
print(f'|tau_w|_inf peak = {tauw_inf.max():.3f}   saturated ticks (>=5): {sat.sum()} '
      f'({100*sat.mean():.2f}%)')
print(f'  of saturated: on SS = {(sat & is_ss).sum()}   on DS = {(sat & is_ds).sum()}')
print()
print('Hdot_s_source tag counts:', {s: int((src == s).sum()) for s in set(src)})
print(f'|Hdot_s|_inf peak overall = {np.nanmax(hdot_inf):.3f}')
for tag in ('planned', 'realized'):
    m = src == tag
    if m.any():
        print(f'  tag={tag:9s} ({m.sum()} ticks): |Hdot_s|_inf peak = '
              f'{np.nanmax(hdot_inf[m]):.3f}   ticks>5 = {int((hdot_inf[m] > 5).sum())}')
print()
# On the saturated ticks, compare |tauw| to |Hdot_s| (the DS feedforward proxy)
for label, m in (('saturated&DS', sat & is_ds), ('saturated&SS', sat & is_ss)):
    if m.any():
        print(f'{label}: n={m.sum()}  |tauw|_inf=[{tauw_inf[m].min():.2f},{tauw_inf[m].max():.2f}]  '
              f'|Hdot_s|_inf=[{np.nanmin(hdot_inf[m]):.2f},{np.nanmax(hdot_inf[m]):.2f}]  '
              f'src={sorted(set(src[m]))}')
