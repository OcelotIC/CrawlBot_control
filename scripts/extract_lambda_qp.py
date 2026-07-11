#!/usr/bin/env python3
"""Phase ABL-HDOT-2 STEP 1 — extract the realized contact wrench lambda_qp (+ the
validated per-tick stance indices) from the identity-confirmed raw sim_logs into
COMMITTED tidy CSVs, so the continuous realized Ḣ_s is reproducible from committed
data forever (the gitignored log is never needed again).

Idriss lifted the no-gitignored-logs rule for THIS extraction only. No crawlbot/
change, no new sim. Self-checks (before writing) that the reduction with the
extracted lambda_qp + stance reproduces the committed DS-realized Ḣ_s.

Sources (identity-confirmed in Step 0, PHASE_ABL_HDOT2_STEP0_IDENTITY.md):
  C: figC_qpcond/sim_log.json (1080) ; stance = figA_canon_fixed/postproc_F3F4.csv
     (validated: reproduces committed runfix DS Ḣ_s to 5e-7)
  U: figU_rateoff/sim_log.json (1112) ; stance = figU_rateoff/postproc_F3F4.csv

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/extract_lambda_qp.py
"""
import csv
import json
import os
import numpy as np
import mujoco

from scripts.export_figure_data import (load_anchors_struct_frame, phase_per_tick, MJCF)

OUT = 'results/j2_ablation_envelope/ablation_data'
model = mujoco.MjModel.from_xml_path(MJCF)

RUNS = [
    dict(tag='C', log='results/figC_qpcond/sim_log.json',
         postproc='results/figA_canon_fixed/postproc_F3F4.csv',
         committed_csv='results/j2_canonical_revalidation/runfix_traversal.csv'),
    dict(tag='U', log='results/figU_rateoff/sim_log.json',
         postproc='results/figU_rateoff/postproc_F3F4.csv',
         committed_csv='results/j2_ablation_envelope/runU_rateoff_traversal.csv'),
]


def read_stance(pp, n):
    sa = np.full(n, -1, int); sb = np.full(n, -1, int)
    with open(pp) as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= n:
                break
            try:
                sa[i] = int(row['stance_a_idx']); sb[i] = int(row['stance_b_idx'])
            except (ValueError, KeyError, TypeError):
                pass
    return sa, sb


def reduce_realized(lam, sa, sb, aA, aB):
    """Sigma_{active}(r_Ci x f_i + tau_i) — export_figure_data.py:190-191 form,
    applied on ALL ticks. On SS the swing arm's wrench is 0 (verified) so its term
    vanishes -> automatically single-contact."""
    n = len(lam); out = np.zeros((n, 3))
    for i in range(n):
        L = lam[i]
        if 0 <= sa[i] < len(aA):
            out[i] += np.cross(aA[sa[i]], L[0:3]) + L[3:6]
        if 0 <= sb[i] < len(aB):
            out[i] += np.cross(aB[sb[i]], L[6:9]) + L[9:12]
    return out


def csv_col(rows, name):
    return np.array([float(r[name]) if r[name] not in ('', None) else np.nan for r in rows])


meta = {'mjcf': MJCF, 'reduction': 'Hdot_s = Sigma_j(r_Cj x f_j + tau_j), structure-frame '
        'anchors (load_anchors_struct_frame), origin-referenced; same as '
        'export_figure_data.py:190-191, applied on ALL ticks', 'runs': {}}

for R in RUNS:
    sl = json.load(open(R['log']))
    n = len(sl['t'])
    t = np.asarray(sl['t'], float)
    lam = np.asarray(sl['lambda_qp'], float)
    ph = phase_per_tick(sl)
    sidx = np.asarray(sl['step_idx'], int)
    sp0 = np.asarray(sl['struct_pos'], float)[0]
    sq0 = np.asarray(sl['struct_quat'], float)[0]
    sa, sb = read_stance(R['postproc'], n)
    aA, aB = load_anchors_struct_frame(model, sp0, sq0)

    # self-check: reduction reproduces committed DS-realized Hdot_s
    red = reduce_realized(lam, sa, sb, aA, aB)
    rows = list(csv.DictReader(open(R['committed_csv'])))
    hd_csv = np.stack([csv_col(rows, f'Hdot_s_{a}_Nm') for a in 'xyz'], 1)
    src = np.array([r['Hdot_s_source'] for r in rows])
    dsm = np.array(['DS' in p for p in ph]) & (src == 'realized')
    dev = float(np.nanmax(np.abs(red[dsm] - hd_csv[dsm])))
    assert dev < 1e-3, f'{R["tag"]}: DS reduction dev {dev} too large — stance/log mismatch'
    print(f'{R["tag"]}: n={n}  DS self-check max|dev|={dev:.3e}  OK ({int(dsm.sum())} DS ticks)')

    # write committed lambda_qp CSV
    path = os.path.join(OUT, f'lambda_qp_{R["tag"]}.csv')
    hdr = (['t_s', 'tick', 'phase', 'step_index', 'stance_a_idx', 'stance_b_idx']
           + [f'lambda_qp_{k}' for k in range(12)])
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(hdr)
        for i in range(n):
            w.writerow([f'{t[i]:.6f}', i, ph[i], int(sidx[i]), int(sa[i]), int(sb[i])]
                       + [f'{lam[i, k]:.6e}' for k in range(12)])
    print(f'   wrote {path}  ({n} rows, 18 cols)')

    meta['runs'][R['tag']] = dict(
        source_log=R['log'], source_postproc=R['postproc'], committed_csv=R['committed_csv'],
        n_ticks=n, struct_pos0=sp0.tolist(), struct_quat0=sq0.tolist(),
        ds_selfcheck_max_dev=dev, lambda_qp_layout='[f_a(0:3), tau_a(3:6), f_b(6:9), tau_b(9:12)]',
        n_anchors=len(aA))

json.dump(meta, open(os.path.join(OUT, 'lambda_qp_meta.json'), 'w'), indent=2)
print(f'wrote {os.path.join(OUT, "lambda_qp_meta.json")}')
