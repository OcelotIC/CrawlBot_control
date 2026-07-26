#!/usr/bin/env python3
"""Phase PREPLAN-FIX Step 2 — closed-loop before/after: pre-fix canonical
(figC_qpcond = runfix@5ab2c91) vs fixed (figC_preplanfix). Realized about-origin
Ḣ_s (lambda_qp reduction), h_w, θ_s, docks, T_step. READ-ONLY on the two sim_logs.
Run: MUJOCO_GL=disabled PYTHONPATH=. python3 Misc/scripts/diag_preplanfix_compare.py
"""
import csv
import json
import numpy as np
from scripts.export_figure_data import load_anchors_struct_frame, phase_per_tick, MJCF
import mujoco

model = mujoco.MjModel.from_xml_path(MJCF)
STANCE_PP = 'Misc/runs/figA_canon_fixed/postproc_F3F4.csv'   # validated stance for the 1080-tick canonical gait

def read_stance(pp, n):
    sa = np.full(n, -1, int); sb = np.full(n, -1, int)
    with open(pp) as f:
        for i, r in enumerate(csv.DictReader(f)):
            if i >= n: break
            try: sa[i] = int(r['stance_a_idx']); sb[i] = int(r['stance_b_idx'])
            except Exception: pass
    return sa, sb

def realized_hdot(sl, sa, sb):
    lam = np.asarray(sl['lambda_qp'], float); n = len(lam)
    aA, aB = load_anchors_struct_frame(model, np.asarray(sl['struct_pos'])[0], np.asarray(sl['struct_quat'])[0])
    out = np.zeros((n, 3))
    for i in range(n):
        L = lam[i]
        if 0 <= sa[i] < len(aA): out[i] += np.cross(aA[sa[i]], L[0:3]) + L[3:6]
        if 0 <= sb[i] < len(aB): out[i] += np.cross(aB[sb[i]], L[6:9]) + L[9:12]
    return out

def planned_hdot(sl, sa, sb):
    lam = np.asarray(sl['lambda_ref'], float); n = len(lam)
    aA, aB = load_anchors_struct_frame(model, np.asarray(sl['struct_pos'])[0], np.asarray(sl['struct_quat'])[0])
    out = np.full((n, 3), np.nan)
    for i in range(n):
        L = lam[i]
        if not np.all(np.isfinite(L)): continue
        v = np.zeros(3)
        if 0 <= sa[i] < len(aA): v += np.cross(aA[sa[i]], L[0:3]) + L[3:6]
        if 0 <= sb[i] < len(aB): v += np.cross(aB[sb[i]], L[6:9]) + L[9:12]
        out[i] = v
    return out

runs = {'PRE': 'results/figC_qpcond/sim_log.json', 'FIX': 'results/figC_preplanfix/sim_log.json'}
data = {}
for tag, p in runs.items():
    sl = json.load(open(p)); n = len(sl['phase'])
    sa, sb = read_stance(STANCE_PP, n)
    ph = np.array(phase_per_tick(sl)); is_ss = ph == 'SS'
    hw = np.abs(np.asarray(sl['hw_physical'], float))
    th = np.asarray(sl['struct_euler_deg'], float)
    rhs = realized_hdot(sl, sa, sb); rabs = np.abs(rhs)
    # weld window: +-2 of SS<->DS boundary
    bnd = np.where(is_ss[1:] != is_ss[:-1])[0]
    weld = np.zeros(n, bool)
    for b in bnd: weld[max(0,b-1):min(n,b+3)] = True
    swing = is_ss & ~weld
    plan = planned_hdot(sl, sa, sb); pabs = np.abs(plan)
    de = sl.get('dock_events', [])
    docks = [round(float(e.get('d_mm', np.nan)), 3) for e in de]
    data[tag] = dict(
        n=n, ph=ph, sidx=np.asarray(sl['step_idx']),
        realized_peak=rabs.max(0), realized_swing_peak=(rabs[swing].max(0) if swing.any() else np.zeros(3)),
        planned_peak=np.nanmax(pabs, 0), hw_peak=hw.max(0), th_peak=np.linalg.norm(th,axis=1).max(),
        th_settled=float(np.linalg.norm(th[-1])), docks=docks, docks_max=max(docks))
# verify same gait (phase + step tick-for-tick)
same_ph = np.array_equal(data['PRE']['ph'], data['FIX']['ph'])
same_si = np.array_equal(data['PRE']['sidx'][:min(data['PRE']['n'],data['FIX']['n'])],
                         data['FIX']['sidx'][:min(data['PRE']['n'],data['FIX']['n'])])
print(f"gait identical? phase={same_ph} step_idx={same_si}  (justifies shared stance source)\n")
def row(k, f):
    print(f"  {k:34s} PRE={f(data['PRE'])}   FIX={f(data['FIX'])}")
print("=== closed-loop before/after ===")
row('realized |Hdot_s| peak/axis', lambda d: [round(x,3) for x in d['realized_peak']])
row('realized |Hdot_s| SS-swing peak', lambda d: [round(x,3) for x in d['realized_swing_peak']])
row('planned |Hdot_s|(NMPC ref) peak', lambda d: [round(x,3) for x in d['planned_peak']])
row('h_w peak/axis [Nms]', lambda d: [round(x,3) for x in d['hw_peak']])
row('theta_s peak (norm) [deg]', lambda d: round(d['th_peak'],4))
row('theta_s settled (norm) [deg]', lambda d: round(d['th_settled'],4))
row('docks [mm]', lambda d: d['docks'])
row('worst dock [mm]', lambda d: round(d['docks_max'],3))
