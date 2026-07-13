#!/usr/bin/env python3
"""Consolidate every COPRIORITY weight-experiment run into one complete tidy CSV.

Reads all results/j2_adjconv/copri*result.json (identical schema, one summary per
weight vector) and emits one row per run with the full weight vector + every
diagnostic (feasibility, per-step at-weld docks, per-step realized Hdot_s,
saturation, e_com, theta_s, kappa_SS, h_w, solver status). Campaign order + a
short label + Addendum tag are annotated per run (metadata, keyed by filename).

Run: PYTHONPATH=. python3 scripts/diag_copri_sweep_export.py
"""
import csv
import json
import os

SRC = 'results/j2_adjconv'
OUT_CSV = 'results/j2_adjconv/copri_sweep_complete.csv'

# Campaign order + label + Addendum, keyed by result-json basename (ground truth
# filenames). This is annotation only; all numbers come from the JSON.
META = {
    'copri1000_result.json':          (1, 'A1', "1:1:1 base (torso=mom=EE=1000, hw 1e4, tq1)"),
    'copri_m2000_result.json':        (2, 'A1-hunt', "momentum-up 2000 (hunt)"),
    'copri_t2000_result.json':        (3, 'A1-hunt', "torso-up 2000 (hunt)"),
    'copri_m400_result.json':         (4, 'A1-hunt', "momentum-down 400 (hunt)"),
    'copri_hw800_result.json':        (5, 'A4-hunt', "hw-slack-down 800 (hunt)"),
    'copri_tq5_result.json':          (6, 'A5', "torque-up 5 -> DOCKS 6/6 (best recipe)"),
    'copri_h1000m500t300_result.json':(7, 'A6', "low co-priority torso=EE=300"),
    'copri_t300e1000_result.json':    (8, 'A7', "EE-restore 1000, torso 300, mom 500"),
    'copri_t300e1000m400_result.json':(9, 'A8', "mom-down 400 (momentum refuted, torso isolated)"),
}

WKEYS = ['alpha_torso_pose', 'alpha_ee', 'ss_alpha_mom', 'w_hw_slack',
         'alpha_posture', 'alpha_torque', 'alpha_wrench', 'alpha_reg']

HEADER = (['run_order', 'addendum', 'label', 'result_json',
           'torso', 'EE', 'momentum', 'hw_slack', 'posture', 'torque', 'wrench',
           'accel_reg', 'eps', 'span',
           'feasible_6of6', 'n_dock']
          + [f'dock{i}_mm' for i in range(6)]
          + ['worst_mm', 'margin_mm']
          + [f'hdot{i}_realized_Nm' for i in range(6)]
          + ['saturation', 'e_com_pk', 'theta_pk_deg', 'theta_settled_deg',
             'kappa_SS', 'lambda_min_LS', 'hw_peak_Nms', 'qp_fail', 'nmpc_fail'])


def _cell(x):
    return '' if x is None else x


rows = []
for fn, (order, add, label) in META.items():
    p = os.path.join(SRC, fn)
    if not os.path.exists(p):
        print(f'[warn] missing {p} — skipped')
        continue
    d = json.load(open(p))
    w = d['weights']
    docks = (d.get('at_weld_docks') or [None] * 6) + [None] * 6
    hdot = (d.get('ss_hdot_realized') or [None] * 6) + [None] * 6
    row = [order, add, label, fn,
           w[WKEYS[0]], w[WKEYS[1]], w[WKEYS[2]], w[WKEYS[3]],
           w[WKEYS[4]], w[WKEYS[5]], w[WKEYS[6]], w[WKEYS[7]],
           d.get('eps'), d.get('span'),
           d.get('feasible_6of6'), d.get('n_dock')]
    row += [_cell(docks[i]) for i in range(6)]
    row += [_cell(d.get('worst')), _cell(d.get('margin'))]
    row += [_cell(hdot[i]) for i in range(6)]
    row += [d.get('saturation'), d.get('e_com_pk'), d.get('theta_pk'),
            d.get('theta_settled'),
            round(d['kappa_SS'], 1) if d.get('kappa_SS') is not None else '',
            d.get('lambda_min_LS'), d.get('hw_peak'),
            d.get('qp_fail'), d.get('nmpc_fail')]
    rows.append((order, row))

rows.sort(key=lambda t: t[0])
with open(OUT_CSV, 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(HEADER)
    for _, row in rows:
        wr.writerow(row)

print(f'wrote {OUT_CSV}  ({len(rows)} runs)')
# echo a compact view for transcription into the md
print(f"\n{'#':>2} {'add':7} {'torso':>6} {'EE':>5} {'mom':>5} {'hw':>6} {'tq':>3} "
      f"{'dock':>5} {'worst':>6} {'sat':>8} {'kappa':>9} {'th_set':>6}  label")
for _, r in rows:
    idx = {h: i for i, h in enumerate(HEADER)}
    nd = r[idx['n_dock']]
    print(f"{r[0]:>2} {r[1]:7} {r[4]:>6g} {r[5]:>5g} {r[6]:>5g} {r[7]:>6g} {r[9]:>3g} "
          f"{str(nd)+'/6':>5} {str(r[idx['worst_mm']]):>6} {str(r[idx['saturation']]):>8} "
          f"{str(r[idx['kappa_SS']]):>9} {str(r[idx['theta_settled_deg']]):>6}  {r[2]}")
