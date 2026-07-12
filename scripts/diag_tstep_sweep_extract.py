#!/usr/bin/env python3
"""Phase TSTEP-DIAG-ALL — extract the (step × scale) sweep table + collapse analysis.
READ-ONLY on the sweep sim_logs. Run after diag_tstep_sweep_run.py.
Run: MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_tstep_sweep_extract.py
"""
import json
import numpy as np
import mujoco
from scripts.export_figure_data import load_anchors_struct_frame, phase_per_tick, MJCF

model = mujoco.MjModel.from_xml_path(MJCF)
SCALES = [1.0, 1.1, 1.25, 1.5, 2.0, 2.5]
STEPS = [0, 1, 2, 3, 4, 5]
# stance arm + anchor index per step, from the canonical dock sequence
# (b@3,a@3,b@4,a@4,b@5,a@5 ⇒ stance holds a@2,b@3,a@3,b@4,a@4,b@5):
STANCE_ARM = ['a', 'b', 'a', 'b', 'a', 'b']
STANCE_ANCHOR = [2, 3, 3, 4, 4, 5]
M_ROBOT = 71.056


def step_metrics(sl, k):
    ph = np.array(phase_per_tick(sl)); si = np.asarray(sl['step_idx'])
    dg = np.asarray(sl['d_grip_swing'], float); rc = np.asarray(sl['r_com'], float)
    lam = np.asarray(sl['lambda_qp'], float)
    ts = np.asarray(sl['preplanner_T_steps'], float)
    m = si == k
    if not m.any():
        return None
    idx = np.where(m)[0]
    seg = dg[m]; seg = seg[np.isfinite(seg)]
    dock_mm = float(1000 * seg.min()) if len(seg) else float('nan')
    standoff = float(np.linalg.norm(rc[idx[0]]))
    distance = float(np.linalg.norm(rc[idx[-1]] - rc[idx[0]]))
    Tstep = float(ts[k]) if k < len(ts) else float('nan')
    # realized Hdot_s on step k (SS ticks): stance-arm contact reduction
    aA, aB = load_anchors_struct_frame(model, np.asarray(sl['struct_pos'])[0],
                                       np.asarray(sl['struct_quat'])[0])
    anch = (aA if STANCE_ARM[k] == 'a' else aB)[STANCE_ANCHOR[k]]
    off = 0 if STANCE_ARM[k] == 'a' else 6
    ss = m & (ph == 'SS')
    hpk = 0.0
    if ss.any():
        for i in np.where(ss)[0]:
            f = lam[i, off:off + 3]; tau = lam[i, off + 3:off + 6]
            hpk = max(hpk, float(np.max(np.abs(np.cross(anch, f) + tau))))
    return dict(T_step=round(Tstep, 3), dock_mm=round(dock_mm, 4),
                standoff=round(standoff, 3), distance=round(distance, 4),
                Hdot_s_pk=round(hpk, 3))


rows = []
for k in STEPS:
    for s in SCALES:
        p = f'results/figC_sw_s{k}_x{s:g}/sim_log.json'
        try:
            sl = json.load(open(p))
        except FileNotFoundError:
            rows.append(dict(step=k, scale=s, missing=True)); continue
        mtr = step_metrics(sl, k)
        feasible = (mtr is not None and k < len(sl.get('preplanner_T_steps', []))
                    and np.isfinite(mtr['T_step']))
        rows.append(dict(step=k, scale=s, feasible=bool(feasible), **(mtr or {})))

json.dump(rows, open('results/j2_adjconv/tstep_sweep.json', 'w'), indent=2)

print(f'{"step":>4} {"scale":>6} {"T_step":>8} {"dock_mm":>8} {"standoff":>9} '
      f'{"dist":>7} {"Hdot_pk":>8} {"feas":>5}')
for r in rows:
    if r.get('missing'):
        print(f'{r["step"]:>4} {r["scale"]:>6} {"MISSING":>8}'); continue
    print(f'{r["step"]:>4} {r["scale"]:>6.2f} {r["T_step"]:>8.3f} {r["dock_mm"]:>8.4f} '
          f'{r["standoff"]:>9.3f} {r["distance"]:>7.4f} {r["Hdot_s_pk"]:>8.3f} {str(r["feasible"]):>5}')

# ---- Q1 per-step: dock vs scale (→0 or floor?) ----
print('\n=== Q1: per-step dock_mm vs scale (does it -> 0 or plateau?) ===')
for k in STEPS:
    ks = [r for r in rows if r['step'] == k and not r.get('missing')]
    ks = sorted(ks, key=lambda r: r['scale'])
    d = [r['dock_mm'] for r in ks]
    print(f'  step{k}: ' + '  '.join(f'x{r["scale"]:g}:{r["dock_mm"]:.3f}' for r in ks)
          + f'   | drop {d[0]-d[-1]:.3f}mm over x1->x2.5')

# ---- Q2 baseline ranking + collapse ----
print('\n=== baseline (scale=1.0): dock vs standoff (rank) ===')
base = sorted([r for r in rows if r['scale'] == 1.0 and not r.get('missing')],
              key=lambda r: r['standoff'])
for r in base:
    print(f'  step{r["step"]}: standoff={r["standoff"]:.3f} dist={r["distance"]:.4f} '
          f'T_step={r["T_step"]:.3f} dock={r["dock_mm"]:.4f}')
