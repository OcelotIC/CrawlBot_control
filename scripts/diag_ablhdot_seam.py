#!/usr/bin/env python3
"""Phase ABL-HDOT — homogeneity / seam evidence from COMMITTED data only.

The current committed Hdot_s column is a SINGLE tagged column: PLANNED wrench
(lambda_ref) on SS, REALIZED wrench (lambda_qp) on DS. This script, using only
the committed tidy CSVs, shows:
  (1) continuity: how many ticks have an empty/finite Hdot_s (it is continuous in
      CELLS but the SS half is planned, the DS half realized — not homogeneous);
  (2) that every SS tick is tagged 'planned' and every DS tick 'realized'
      (so a *realized* value on SS is simply absent from committed data);
  (3) the value on both sides of the SS<->DS transitions — the jump there is a
      PLANNED-vs-REALIZED wrench-source change, illustrating why a homogeneous
      realized signal is needed (and cannot be built from committed columns).
READ-ONLY, committed CSVs only. Run:
  MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_ablhdot_seam.py <csv>
"""
import csv
import sys
import numpy as np

CSV = sys.argv[1] if len(sys.argv) > 1 else 'results/j2_ablation_envelope/runU_rateoff_traversal.csv'
print(f'### {CSV}')
rows = list(csv.DictReader(open(CSV)))
n = len(rows)


def fcol(name):
    return np.array([float(r[name]) if r[name] not in ('', None) else np.nan for r in rows])


phase = np.array([r['phase'] for r in rows])            # DS_interstep/DS_terminal/SS
src = np.array([r['Hdot_s_source'] for r in rows])       # planned/realized
hd = np.stack([fcol(f'Hdot_s_{a}_Nm') for a in 'xyz'], 1)
is_ss = phase == 'SS'
is_ds = ~is_ss

# (1) continuity of the committed tagged column
empty = np.isnan(hd).any(axis=1)
print(f'total ticks {n}  SS {is_ss.sum()}  DS {is_ds.sum()}')
print(f'tagged Hdot_s empty cells: {int(empty.sum())}/{n}  '
      f'(finite everywhere? {"YES" if empty.sum()==0 else "NO"})')

# (2) source tag vs phase
print('source-tag x phase:')
for ph_lab, m in (('SS', is_ss), ('DS', is_ds)):
    tags = {t: int(((src == t) & m).sum()) for t in ('planned', 'realized', '')}
    print(f'  {ph_lab:2s}: {tags}')
print('  => SS carries PLANNED wrench, DS carries REALIZED wrench. A REALIZED '
      'value on SS is NOT present in committed data.')

# (3) value on both sides of SS<->DS transitions
trans = np.where(phase[1:] != phase[:-1])[0]  # index i where phase[i]!=phase[i+1]
print(f'\nSS<->DS (and DS-subphase) transitions: {len(trans)}; showing first 6 SS<->DS seams:')
shown = 0
for i in trans:
    a_ss, b_ss = is_ss[i], is_ss[i + 1]
    if a_ss == b_ss:
        continue  # skip DS_interstep<->DS_terminal seams
    lhs = f'{phase[i]}/{src[i]}'
    rhs = f'{phase[i+1]}/{src[i+1]}'
    hl = np.abs(hd[i])
    hr = np.abs(hd[i + 1])
    jump = np.abs(hd[i + 1] - hd[i])
    print(f'  tick {i:4d}->{i+1:<4d}  {lhs:>20s} | {rhs:<20s}  '
          f"|Hd|L=[{hl[0]:.2f} {hl[1]:.2f} {hl[2]:.2f}]  "
          f"|Hd|R=[{hr[0]:.2f} {hr[1]:.2f} {hr[2]:.2f}]  "
          f"jump_inf={jump.max():.2f}")
    shown += 1
    if shown >= 6:
        break
print('  (the jump at each seam is a PLANNED->REALIZED wrench-source change, NOT a '
      'pure physical weld transient — the two sides use different wrenches.)')
