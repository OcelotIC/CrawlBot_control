#!/usr/bin/env python3
"""Phase 1e'-B — anatomy of the h_max=6 dock-abort (Task A/B), zero new solve.
Reads the on-disk sim_logs (h_max=5 = figC_qpcond, 6 = figC_hmax6, 7 = figC_hmax7).

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/analyze_hmax6_abort.py
"""
import json
from collections import Counter

import numpy as np

RUNS = [(5, 'results/figC_qpcond/sim_log.json'),
        (6, 'results/figC_hmax6/sim_log.json'),
        (7, 'results/figC_hmax7/sim_log.json')]
out = {}


def arr(x):
    return np.asarray(x, float)


for h, p in RUNS:
    sl = json.load(open(p))
    ph = np.array(sl['phase']); t = arr(sl['t']); n = len(t)
    ss = ph == 'SS'
    nstr = sl.get('nmpc_status_str', [])
    c = Counter(nstr)
    nsolve_fail = sum(1 for s in nstr if s not in ('NMPC_BYPASSED', 'Solve_Succeeded'))
    # dock gate trace (per-eval EE-to-anchor distance during swing-end + hold)
    dgt = sl.get('dock_gate_trace', [])
    d_mm = arr([e['d_mm'] for e in dgt]) if dgt else np.array([])
    tt = arr([e['t'] for e in dgt]) if dgt else np.array([])
    fired = sum(int(e.get('fired', 0)) for e in dgt)
    # swing EE tracking: |p_ee - p_ee_ref| over SS, and |e_ee_pos|
    p_ee = arr(sl['p_ee']); p_ref = arr(sl['p_ee_ref'])
    track = np.linalg.norm(p_ee - p_ref, axis=1) if p_ee.ndim == 2 else np.abs(p_ee - p_ref)
    eep = arr(sl['e_ee_pos']); eep = np.linalg.norm(eep, axis=1) if eep.ndim == 2 else np.abs(eep)
    # hold window: dock-trace time span
    hold_span = float(tt[-1] - tt[0]) if tt.size else 0.0
    rec = dict(
        ticks=n, ss_ticks=int(ss.sum()),
        nmpc_status=dict(c), nmpc_solve_failures=nsolve_fail,
        dock_evals=len(dgt), dock_fired=fired,
        dock_d_first_mm=round(float(d_mm[0]), 3) if d_mm.size else None,
        dock_d_min_mm=round(float(d_mm.min()), 3) if d_mm.size else None,
        dock_d_end_mm=round(float(d_mm[-1]), 3) if d_mm.size else None,
        hold_span_s=round(hold_span, 2),
        ee_track_ss_rms_mm=round(float(np.sqrt(np.mean(track[ss] ** 2)) * 1e3), 3) if ss.any() else None,
        ee_track_ss_max_mm=round(float(np.max(track[ss]) * 1e3), 3) if ss.any() else None,
    )
    out[h] = rec

print('=' * 92)
print('TASK A/B — h_max=6 dock-abort anatomy (zero new solve)')
print(f'  {"h_max":>5} | {"ticks":>5} {"SSt":>4} | {"NMPC solve-fail":>15} | {"dockEvals":>9} '
      f'{"fired":>5} | {"d first/min/end mm":>20} | {"hold s":>7} | {"EE-track SS rms/max mm":>22}')
for h in (5, 6, 7):
    r = out[h]
    print(f'  {h:>5} | {r["ticks"]:>5} {r["ss_ticks"]:>4} | {r["nmpc_solve_failures"]:>15} | '
          f'{r["dock_evals"]:>9} {r["dock_fired"]:>5} | '
          f'{str(r["dock_d_first_mm"])+"/"+str(r["dock_d_min_mm"])+"/"+str(r["dock_d_end_mm"]):>20} | '
          f'{r["hold_span_s"]:>7} | {str(r["ee_track_ss_rms_mm"])+"/"+str(r["ee_track_ss_max_mm"]):>22}')
print('\n  NMPC status distributions (BYPASSED = DS/init, not a solve failure):')
for h in (5, 6, 7):
    print(f'    h_max={h}: {out[h]["nmpc_status"]}  -> real IPOPT solve failures: {out[h]["nmpc_solve_failures"]}')

# dock trajectory shape for h_max=6 (does it monotonically fall then get cut, or plateau/drift?)
sl6 = json.load(open('results/figC_hmax6/sim_log.json'))
d6 = [round(e['d_mm'], 2) for e in sl6['dock_gate_trace']]
print(f'\n  h_max=6 dock d_mm trajectory over the hold (n={len(d6)}): first10={d6[:10]}  last10={d6[-10:]}')
print(f'    monotone? d[end]-d[start] = {d6[-1]-d6[0]:+.2f} mm  (positive => DRIFTS AWAY during hold)')

json.dump(out, open('results/j2_ablation_envelope/hmax6_abort_anatomy.json', 'w'), indent=2)
print('\nwrote results/j2_ablation_envelope/hmax6_abort_anatomy.json')
print('=' * 92)
