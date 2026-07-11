#!/usr/bin/env python3
"""Phase ABL-HDOT-2 STEP 2 — continuous REALIZED Ḣ_s over SS+DS from committed data.

Reads the committed lambda_qp CSV + meta (Step 1) + committed MJCF, applies the
SAME reduction as export_figure_data.py:190-191 — Ḣ_s = Σ_j(r_Cj×f_j+τ_j),
structure-frame anchors, origin-referenced, same sign — on EVERY tick (the DS-only
guard removed; on SS the swing arm's wrench is 0 so its term vanishes). Adds
Hdot_s_realized_cont_{x,y,z}_Nm to the committed ablation timeseries CSVs.

VERIFY: continuity (0 empty), homogeneity (λ_qp only), DS cross-check
(cont == committed realized), seam value-continuity (≥3 seams; vs the old
planned→realized artifact), and the swing-vs-weld peak split. Committed data only;
no gitignored log, no crawlbot/ change, no new sim.
Run: MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/build_continuous_hdot.py
"""
import csv
import json
import numpy as np
import mujoco

from scripts.export_figure_data import load_anchors_struct_frame, MJCF

DATA = 'results/j2_ablation_envelope/ablation_data'
W_WELD = 2   # ticks on each side of an SS<->DS boundary treated as weld transient
model = mujoco.MjModel.from_xml_path(MJCF)
meta = json.load(open(f'{DATA}/lambda_qp_meta.json'))

RUNS = {'C': f'{DATA}/ablation_C_timeseries.csv', 'U': f'{DATA}/ablation_U_timeseries.csv'}
report = {}

for tag, tspath in RUNS.items():
    print('=' * 78)
    print(f'RUN {tag}')
    lam_rows = list(csv.DictReader(open(f'{DATA}/lambda_qp_{tag}.csv')))
    n = len(lam_rows)
    lam = np.array([[float(r[f'lambda_qp_{k}']) for k in range(12)] for r in lam_rows])
    sa = np.array([int(r['stance_a_idx']) for r in lam_rows])
    sb = np.array([int(r['stance_b_idx']) for r in lam_rows])
    ph = np.array([r['phase'] for r in lam_rows])            # DS_interstep/DS_terminal/SS
    t_lam = np.array([float(r['t_s']) for r in lam_rows])

    sp0 = np.array(meta['runs'][tag]['struct_pos0'])
    sq0 = np.array(meta['runs'][tag]['struct_quat0'])
    aA, aB = load_anchors_struct_frame(model, sp0, sq0)

    # --- reduction on ALL ticks (export_figure_data.py:190-191 form) ---
    cont = np.zeros((n, 3))
    for i in range(n):
        L = lam[i]
        if 0 <= sa[i] < len(aA):
            cont[i] += np.cross(aA[sa[i]], L[0:3]) + L[3:6]
        if 0 <= sb[i] < len(aB):
            cont[i] += np.cross(aB[sb[i]], L[6:9]) + L[9:12]

    # --- load committed timeseries; align by t_s ---
    ts_rows = list(csv.DictReader(open(tspath)))
    assert len(ts_rows) == n, f'{tag}: timeseries {len(ts_rows)} != lambda_qp {n}'
    t_ts = np.array([float(r['t_s']) for r in ts_rows])
    assert np.max(np.abs(t_ts - t_lam)) < 1e-6, f'{tag}: t_s misaligned'
    is_ss = np.array([r['phase'] == 'SS' for r in ts_rows])
    is_ds = ~is_ss

    def tscol(name):
        return np.array([float(r[name]) if r[name] not in ('', None) else np.nan for r in ts_rows])
    real_ds = np.stack([tscol(f'Hdot_s_realized_{a}_Nm') for a in 'xyz'], 1)   # DS-only committed
    plan_ss = np.stack([tscol(f'Hdot_s_planned_{a}_Nm') for a in 'xyz'], 1)     # SS-only committed

    # --- VERIFY (1) continuity + homogeneity ---
    n_empty = int(np.isnan(cont).any(axis=1).sum())
    print(f'  continuity: empty cells = {n_empty}/{n}  (homogeneous: lambda_qp on ALL ticks)')

    # --- VERIFY (2) DS cross-check: cont == committed realized on DS ---
    d = np.abs(cont[is_ds] - real_ds[is_ds])
    print(f'  DS cross-check: max|cont - committed_realized| = {np.nanmax(d):.3e}  '
          f'({int(is_ds.sum())} DS ticks)')

    # --- VERIFY (3) seam value-continuity (>=3 SS<->DS seams) ---
    bnd = np.where(is_ss[1:] != is_ss[:-1])[0]   # i where phase flips at i->i+1
    print(f'  seams (SS<->DS): {len(bnd)}; new continuous vs OLD tagged (planned|realized):')
    seams_out = []
    for i in bnd[:6]:
        # OLD tagged value each side = planned on SS, realized on DS
        old_l = plan_ss[i] if is_ss[i] else real_ds[i]
        old_r = plan_ss[i + 1] if is_ss[i + 1] else real_ds[i + 1]
        new_jump = float(np.abs(cont[i + 1] - cont[i]).max())
        old_jump = float(np.abs(old_r - old_l).max())
        lab = f'{ph[i]}->{ph[i+1]}'
        print(f'    t{t_ts[i]:6.2f} {lab:26s} contL=[{cont[i,0]:6.2f}{cont[i,1]:6.2f}{cont[i,2]:6.2f}] '
              f'contR=[{cont[i+1,0]:6.2f}{cont[i+1,1]:6.2f}{cont[i+1,2]:6.2f}]  '
              f'NEWjump={new_jump:5.2f}  OLDjump={old_jump:5.2f}')
        seams_out.append(dict(t=float(t_ts[i]), label=lab, new_jump=new_jump, old_jump=old_jump))

    # --- VERIFY (4) swing vs weld split ---
    weld = np.zeros(n, bool)
    for i in bnd:
        weld[max(0, i - W_WELD + 1):min(n, i + W_WELD + 1)] = True   # +-W_WELD around boundary
    swing = is_ss & ~weld
    settle = is_ds & ~weld
    contabs = np.abs(cont)

    def axpk(mask):
        return contabs[mask].max(axis=0) if mask.any() else np.zeros(3)
    pk_swing, pk_weld, pk_settle = axpk(swing), axpk(weld), axpk(settle)
    print(f'  SPLIT (W_weld={W_WELD} ticks/side of each SS<->DS boundary):')
    print(f'    SS-swing (excl weld) [{int(swing.sum())} tk]: peak/axis = '
          f'[{pk_swing[0]:.2f} {pk_swing[1]:.2f} {pk_swing[2]:.2f}]  >5? {(pk_swing>5).any()}')
    print(f'    weld window          [{int(weld.sum())} tk]: peak/axis = '
          f'[{pk_weld[0]:.2f} {pk_weld[1]:.2f} {pk_weld[2]:.2f}]  >5? {(pk_weld>5).any()}')
    print(f'    DS-settle (excl weld)[{int(settle.sum())} tk]: peak/axis = '
          f'[{pk_settle[0]:.2f} {pk_settle[1]:.2f} {pk_settle[2]:.2f}]  >5? {(pk_settle>5).any()}')
    gpk = contabs.max(axis=0); gi = contabs.max(axis=1).argmax()
    print(f'    overall peak/axis = [{gpk[0]:.2f} {gpk[1]:.2f} {gpk[2]:.2f}]  '
          f'at t={t_ts[gi]:.2f} ({ph[gi]}, {"weld" if weld[gi] else "swing" if is_ss[gi] else "settle"})')

    # --- write augmented timeseries CSV ---
    fns = list(ts_rows[0].keys())
    newcols = [f'Hdot_s_realized_cont_{a}_Nm' for a in 'xyz']
    with open(tspath, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(fns + newcols)
        for i, r in enumerate(ts_rows):
            w.writerow([r[k] for k in fns] + [f'{cont[i, j]:.6e}' for j in range(3)])
    print(f'  wrote {tspath}  (+3 cols Hdot_s_realized_cont_*)')

    report[tag] = dict(
        n=n, empty=n_empty, ds_crosscheck_max=float(np.nanmax(d)),
        peak_swing=pk_swing.tolist(), peak_weld=pk_weld.tolist(), peak_settle=pk_settle.tolist(),
        peak_overall=gpk.tolist(), peak_overall_t=float(t_ts[gi]),
        peak_overall_cat=("weld" if weld[gi] else "swing" if is_ss[gi] else "settle"),
        swing_over5=bool((pk_swing > 5).any()), weld_over5=bool((pk_weld > 5).any()),
        settle_over5=bool((pk_settle > 5).any()), seams=seams_out, w_weld=W_WELD)

json.dump(report, open(f'{DATA}/continuous_hdot_report.json', 'w'), indent=2)
print('=' * 78)
print(f'wrote {DATA}/continuous_hdot_report.json')
