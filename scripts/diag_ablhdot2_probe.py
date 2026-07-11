#!/usr/bin/env python3
"""Phase ABL-HDOT-2 Step 1/2 groundwork (READ-ONLY): find the stance source for
both runs, VALIDATE that the reduction reproduces the committed DS-realized Ḣ_s,
and characterize the SS single-contact structure (swing arm wrench ~ 0?).

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_ablhdot2_probe.py
"""
import csv
import glob
import json
import os
import numpy as np
import mujoco

from scripts.export_figure_data import (load_anchors_struct_frame, phase_per_tick,
                                         MJCF)

model = mujoco.MjModel.from_xml_path(MJCF)

print('===== 1. all on-disk postproc_F3F4.csv + row counts (find C=1080, U=1112) =====')
for p in sorted(glob.glob('results/**/postproc_F3F4.csv', recursive=True)):
    try:
        nr = sum(1 for _ in open(p)) - 1
    except Exception:
        nr = -1
    flag = '  <-- C?' if nr == 1080 else ('  <-- U?' if nr == 1112 else '')
    print(f'  {nr:5d} rows  {p}{flag}')


def read_stance(pp_path, n):
    sa = np.full(n, -1, int); sb = np.full(n, -1, int)
    hd = np.full((n, 3), np.nan)
    with open(pp_path) as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= n:
                break
            try:
                sa[i] = int(row['stance_a_idx']); sb[i] = int(row['stance_b_idx'])
            except (ValueError, KeyError, TypeError):
                pass
            for j, a in enumerate('xyz'):
                try:
                    hd[i, j] = float(row.get(f'Hdot_s_{a}', ''))
                except (ValueError, TypeError):
                    hd[i, j] = np.nan
    return sa, sb, hd


def reduce_realized(lam, sa, sb, anchors_a, anchors_b):
    """Sigma_{active}(r_Ci x f_i + tau_i): include an arm only when its stance idx
    is valid (>=0). SS: swing arm idx=-1 -> skipped (its f~0)."""
    n = len(lam); out = np.zeros((n, 3))
    for i in range(n):
        L = lam[i]
        if 0 <= sa[i] < len(anchors_a):
            out[i] += np.cross(anchors_a[sa[i]], L[0:3]) + L[3:6]
        if 0 <= sb[i] < len(anchors_b):
            out[i] += np.cross(anchors_b[sb[i]], L[6:9]) + L[9:12]
    return out


def csv_col(rows, name):
    return np.array([float(r[name]) if r[name] not in ('', None) else np.nan for r in rows])


# ----- U: validate reduction vs committed DS-realized Hdot_s -----
print('\n===== 2. U: reduction (postproc stance) vs committed DS-realized Hdot_s =====')
slU = json.load(open('results/figU_rateoff/sim_log.json'))
nU = len(slU['t'])
lamU = np.asarray(slU['lambda_qp'], float)
phU = np.array(phase_per_tick(slU))
saU, sbU, hdU_pp = read_stance('results/figU_rateoff/postproc_F3F4.csv', nU)
aA, aB = load_anchors_struct_frame(model, np.asarray(slU['struct_pos'])[0],
                                   np.asarray(slU['struct_quat'])[0])
print(f'  n_anchors_a={len(aA)} n_anchors_b={len(aB)}')
redU = reduce_realized(lamU, saU, sbU, aA, aB)
rowsU = list(csv.DictReader(open('results/j2_ablation_envelope/runU_rateoff_traversal.csv')))
hdU_csv = np.stack([csv_col(rowsU, f'Hdot_s_{a}_Nm') for a in 'xyz'], 1)
srcU = np.array([r['Hdot_s_source'] for r in rowsU])
is_ds = np.array(['DS' in p for p in phU])
dsm = is_ds & (srcU == 'realized')
dev = np.abs(redU[dsm] - hdU_csv[dsm])
print(f'  DS ticks (realized): {dsm.sum()}   max|reduce - committed| = {np.nanmax(dev):.3e}')
print(f'  (should be ~1e-6 if the reduction + stance reproduce the committed DS values)')

# ----- SS single-contact structure -----
print('\n===== 3. U SS single-contact check (per tick: swing arm |f| ~ 0?) =====')
sarm = np.asarray(slU['swing_arm'])
fa = np.linalg.norm(lamU[:, 0:3], axis=1); fb = np.linalg.norm(lamU[:, 6:9], axis=1)
ss = phU == 'SS'
# on SS ticks, swing arm = sarm; its wrench should be ~0
sw_a = ss & (sarm == 'a')   # arm A swings -> |f_a| should be ~0
sw_b = ss & (sarm == 'b')   # arm B swings -> |f_b| should be ~0
print(f'  SS & swing=a ({sw_a.sum():3d} ticks): swing |f_a| max={fa[sw_a].max() if sw_a.any() else 0:.4f}  '
      f'stance |f_b| max={fb[sw_a].max() if sw_a.any() else 0:.4f}')
print(f'  SS & swing=b ({sw_b.sum():3d} ticks): swing |f_b| max={fb[sw_b].max() if sw_b.any() else 0:.4f}  '
      f'stance |f_a| max={fa[sw_b].max() if sw_b.any() else 0:.4f}')
# swing arm FULL wrench (f AND tau) must be ~0 on SS, else the 2-arm formula adds a spurious tau
ta = np.linalg.norm(lamU[:, 3:6], axis=1); tb = np.linalg.norm(lamU[:, 9:12], axis=1)
print(f'  swing FULL-wrench on SS: swing=a |tau_a|max={ta[sw_a].max() if sw_a.any() else 0:.4f}  '
      f'swing=b |tau_b|max={tb[sw_b].max() if sw_b.any() else 0:.4f}  (must be ~0)')
print('  stance_idx on SS ticks (swing arm idx irrelevant since its f=0):')
print(f'    SS&swing=a: stance_a_idx (swing) unique={sorted(set(saU[sw_a].tolist()))}  '
      f'stance_b_idx (stance) unique={sorted(set(sbU[sw_a].tolist()))}')
print(f'    SS&swing=b: stance_b_idx (swing) unique={sorted(set(sbU[sw_b].tolist()))}  '
      f'stance_a_idx (stance) unique={sorted(set(saU[sw_b].tolist()))}')

# ----- 4. C: find the stance source that reproduces committed C DS-realized Hdot_s -----
print('\n===== 4. C: which 1080-row postproc reproduces committed runfix DS Hdot_s? =====')
slC = json.load(open('results/figC_qpcond/sim_log.json'))
nC = len(slC['t'])
lamC = np.asarray(slC['lambda_qp'], float)
phC = np.array(phase_per_tick(slC))
aAC, aBC = load_anchors_struct_frame(model, np.asarray(slC['struct_pos'])[0],
                                     np.asarray(slC['struct_quat'])[0])
rowsC = list(csv.DictReader(open('results/j2_canonical_revalidation/runfix_traversal.csv')))
hdC_csv = np.stack([csv_col(rowsC, f'Hdot_s_{a}_Nm') for a in 'xyz'], 1)
srcC = np.array([r['Hdot_s_source'] for r in rowsC])
is_dsC = np.array(['DS' in p for p in phC])
dsmC = is_dsC & (srcC == 'realized')
cands = ['results/diag_cooperative_arms_legacy_pid_numerical/postproc_F3F4.csv',
         'results/figA_canon_fixed/postproc_F3F4.csv',
         'results/ssmom_phase1_baseline_main_dcda974/postproc_F3F4.csv']
for pp in cands:
    if not os.path.exists(pp):
        print(f'  {pp}: MISSING'); continue
    saC, sbC, _ = read_stance(pp, nC)
    redC = reduce_realized(lamC, saC, sbC, aAC, aBC)
    dev = np.abs(redC[dsmC] - hdC_csv[dsmC])
    print(f'  {os.path.basename(os.path.dirname(pp)):45s}  DS max|dev|={np.nanmax(dev):.3e}  '
          f'{"<== MATCH" if np.nanmax(dev) < 1e-3 else ""}')
