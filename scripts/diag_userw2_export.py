#!/usr/bin/env python3
"""Phase USERW2-DATA — export the committed userw2 run (3871de4) timeseries.

READ-ONLY on the on-disk figC_userw2/sim_log.json (provenance-verified == committed
userweights_result2.json). NO re-run, NO canonical change.

Homogeneous realized Hdot_s: the SAME method as the validated canonical export
(export_figure_data.py FIX 2, lines 184-192) — origin-referenced
  Hdot_s = cross(r_Ca, f_a) + tau_a + cross(r_Cb, f_b) + tau_b
from realized lambda_qp + structure-frame anchors — but applied on ALL ticks (SS and
DS), not planned-on-SS. Per-tick weld levers r_Ca/r_Cb from the confirmed leapfrog:
during SS one arm swings with lambda==0 (bounded), so only the stance anchor matters;
during DS both arms are welded at STANCE_ANCHOR[k] / SWING_ANCHOR[k].

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_userw2_export.py
"""
import json
import csv
import numpy as np
import mujoco
from scripts.export_figure_data import load_anchors_struct_frame, phase_per_tick, MJCF

RUN = 'results/figC_userw2'
sl = json.load(open(f'{RUN}/sim_log.json'))
model = mujoco.MjModel.from_xml_path(MJCF)

# Confirmed leapfrog (matches the block structure: SS stance-arm force active, swing==0):
STANCE_ARM = ['a', 'b', 'a', 'b', 'a', 'b']
STANCE_ANCHOR = [2, 3, 3, 4, 4, 5]
SWING_ARM = ['b', 'a', 'b', 'a', 'b', 'a']
SWING_ANCHOR = [3, 3, 4, 4, 5, 5]

t = np.asarray(sl['t'], float)
ph = np.array(phase_per_tick(sl))           # 'SS' / 'DS'
si = np.asarray(sl['step_idx'], int)
lam = np.asarray(sl['lambda_qp'], float)     # [f_a, tau_a, f_b, tau_b]
tw = np.asarray(sl['tau_w'], float)
hw = np.asarray(sl['hw_physical'], float)
eul = np.asarray(sl['struct_euler_deg'], float)
dg = np.asarray(sl['d_grip_swing'], float)
n = len(t)
aA, aB = load_anchors_struct_frame(model, np.asarray(sl['struct_pos'])[0],
                                   np.asarray(sl['struct_quat'])[0])

Hdot = np.full((n, 3), np.nan)
for i in range(n):
    k = si[i]
    if k < 0 or k > 5:                       # initial DS: unloaded (f==0) -> 0
        Hdot[i] = np.zeros(3); continue
    if STANCE_ARM[k] == 'a':                 # steps 0,2,4: a stance, b swing/just-docked
        ra = aA[STANCE_ANCHOR[k]]; rb = aB[SWING_ANCHOR[k]]
    else:                                    # steps 1,3,5: b stance, a swing/just-docked
        rb = aB[STANCE_ANCHOR[k]]; ra = aA[SWING_ANCHOR[k]]
    fa, ta, fb, tb = lam[i, 0:3], lam[i, 3:6], lam[i, 6:9], lam[i, 9:12]
    Hdot[i] = np.cross(ra, fa) + ta + np.cross(rb, fb) + tb

# ── validation: SS per-step peak |Hdot| (max abs component) vs committed result2 ──
res2 = json.load(open('results/j2_adjconv/userweights_result2.json'))['hdot_s_pk']
ss_pk = []
for k in range(6):
    m = (si == k) & (ph == 'SS')
    ss_pk.append(round(float(np.max(np.abs(Hdot[m]))), 3) if m.any() else float('nan'))
print('validation  homogeneous SS peak |Hdot| per step:', ss_pk)
print('validation  committed result2 hdot_s_pk        :', res2)
print('validation  MATCH (<=0.01):',
      all(abs(a - b) <= 0.011 for a, b in zip(ss_pk, res2)))

# ── CSV ──
cols = ['t_s', 'phase', 'step_index',
        'Hdot_s_realized_cont_x_Nm', 'Hdot_s_realized_cont_y_Nm', 'Hdot_s_realized_cont_z_Nm',
        'Hdot_s_realized_norm_Nm',
        'tauw_x_Nm', 'tauw_y_Nm', 'tauw_z_Nm',
        'hw_x_Nms', 'hw_y_Nms', 'hw_z_Nms',
        'theta_s_x_deg', 'theta_s_y_deg', 'theta_s_z_deg',
        'd_grip_swing_mm']
out_csv = 'results/j2_adjconv/userw2_timeseries.csv'
with open(out_csv, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(cols)
    for i in range(n):
        w.writerow([f'{t[i]:.4f}', ph[i], si[i],
                    f'{Hdot[i,0]:.6e}', f'{Hdot[i,1]:.6e}', f'{Hdot[i,2]:.6e}',
                    f'{np.linalg.norm(Hdot[i]):.6e}',
                    f'{tw[i,0]:.6e}', f'{tw[i,1]:.6e}', f'{tw[i,2]:.6e}',
                    f'{hw[i,0]:.6e}', f'{hw[i,1]:.6e}', f'{hw[i,2]:.6e}',
                    f'{eul[i,0]:.6e}', f'{eul[i,1]:.6e}', f'{eul[i,2]:.6e}',
                    ('' if not np.isfinite(dg[i]) else f'{1000*dg[i]:.4f}')])
print(f'wrote {out_csv}  ({n} ticks)')

# ── analysis ──
CAP = 5.0
absH = np.abs(Hdot)
is_ss = np.array([p == 'SS' for p in ph])
is_ds = np.array([p.startswith('DS') for p in ph])
gmax = np.nanmax(absH); gi = int(np.nanargmax(np.nanmax(absH, axis=1)))
gax = 'xyz'[int(np.nanargmax(absH[gi]))]
over = int((np.nanmax(absH, axis=1) > CAP + 1e-6).sum())
print(f'\n=== Q1: global realized |Hdot_s| peak = {gmax:.3f} Nm  '
      f'(tick {gi}, t={t[gi]:.2f}s, phase {ph[gi]}, step {si[gi]}, axis {gax}) — cap {CAP} ===')
print(f'  ticks EXCEEDING cap (>{CAP}): {over}  -> box {"VIOLATED (artifact?)" if over else "enforced (never exceeds)"}')
print(' step | SS-peak(x/y/z) | SSmax | DS-peak(x/y/z) | DSmax | dock[mm] | Hdot@dock')
rows = []
for k in range(6):
    ssm = (si == k) & is_ss; dsm = (si == k) & is_ds
    ssp = np.max(absH[ssm], axis=0) if ssm.any() else np.zeros(3)
    dsp = np.max(absH[dsm], axis=0) if dsm.any() else np.zeros(3)
    idxss = np.where(ssm & np.isfinite(dg))[0]
    idock = idxss[int(np.argmin(dg[idxss]))]
    dock_mm = 1000 * dg[idock]; Hdock = np.max(np.abs(Hdot[idock]))
    ds_atcap = int((np.max(absH[dsm], axis=1) > CAP - 0.05).sum()) if dsm.any() else 0
    rows.append(dict(step=k, ss_peak=[round(x, 3) for x in ssp], ss_max=round(float(ssp.max()), 3),
                     ds_peak=[round(x, 3) for x in dsp], ds_max=round(float(dsp.max()), 3),
                     ds_ticks_at_cap=ds_atcap, dock_mm=round(float(dock_mm), 3),
                     Hdot_at_dock=round(float(Hdock), 3)))
    print(f'  {k}   | {ssp[0]:.2f}/{ssp[1]:.2f}/{ssp[2]:.2f} | {ssp.max():5.2f} '
          f'| {dsp[0]:.2f}/{dsp[1]:.2f}/{dsp[2]:.2f} | {dsp.max():5.2f} | {dock_mm:7.3f} | {Hdock:8.3f}')

near_cap = np.any(absH > CAP - 0.05, axis=1)
nc_ss = int((near_cap & is_ss).sum()); nc_ds = int((near_cap & is_ds).sum())
print(f'\n=== Q2: Hdot_s box |.|<= {CAP} activity (plateau/active?) ===')
print(f'  ticks with any |Hdot_s axis| >= {CAP-0.05}: {int(near_cap.sum())}/{n} '
      f'({nc_ss} in SS-swing, {nc_ds} in DS-settle) -> box {"ACTIVE" if near_cap.any() else "INACTIVE"}')
print(f'  SS-swing peak |Hdot_s| = {np.max(absH[is_ss]):.3f} (<= cap, HEADROOM); '
      f'DS-settle peak = {np.max(absH[is_ds]):.3f} (AT cap)')
tw_at_cap = np.max(np.abs(tw), axis=1) > CAP - 1e-3
print(f'  contrast tau_w (AOCS wheel clamp): {int(tw_at_cap.sum())}/{n} ticks at |tau_w|={CAP} '
      f'-> wheel clamp ACTIVE; hw_physical peak {np.max(np.linalg.norm(hw,axis=1)):.3f} Nms (env +-{CAP})')

json.dump({'run': 'figC_userw2 (committed 3871de4)', 'csv': out_csv,
           'ss_peak_validation': {'homogeneous': ss_pk, 'committed': res2},
           'global_peak_Nm': round(float(gmax), 3), 'cap_Nm': CAP,
           'per_step': rows,
           'Hdot_box_active': bool(near_cap.any()),
           'tauw_clamp_active_ticks': int(tw_at_cap.sum()),
           'hw_peak_Nms': round(float(np.max(np.linalg.norm(hw, axis=1))), 3)},
          open('results/j2_adjconv/userw2_envelope_analysis.json', 'w'), indent=2)
print('\nwrote results/j2_adjconv/userw2_envelope_analysis.json')
