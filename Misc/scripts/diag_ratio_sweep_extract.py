#!/usr/bin/env python3
"""Phase EE-RATIO-GLOBAL — extract collateral metrics per torso:EE ratio config.
READ-ONLY. Run after diag_ratio_sweep_run.py.

Per config: 6 SS-docks + margin; theta_s peak/settled; realized |Hdot_s|pk x6
(same stance-contact method as diag_tstep_sweep_extract.py); torso-pose tracking
residual (e_torso_ori/pos) over swing; h_w peak; feasibility (docks/solve).

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 Misc/scripts/diag_ratio_sweep_extract.py
"""
import json
import numpy as np
import mujoco
from scripts.export_figure_data import load_anchors_struct_frame, phase_per_tick, MJCF

model = mujoco.MjModel.from_xml_path(MJCF)
STANCE_ARM = ['a', 'b', 'a', 'b', 'a', 'b']
STANCE_ANCHOR = [2, 3, 3, 4, 4, 5]

# (label, dir, torso, ee, ratio)
CONFIGS = [
    ('baseline 8:1', 'results/figC_sw_s5_x1', 24000, 3000, 8.0),
    ('A t12000 4:1', 'results/figC_ratio_t12000', 12000, 3000, 4.0),
    ('A t8000 2.67:1', 'results/figC_ratio_t8000', 8000, 3000, 2.667),
    ('A t6000 2:1', 'results/figC_ratio_t6000', 6000, 3000, 2.0),
    ('A t4000 1.33:1', 'results/figC_ratio_t4000', 4000, 3000, 1.333),
    ('A t3000 1:1', 'results/figC_ratio_t3000', 3000, 3000, 1.0),
    ('B ee6000 4:1', 'results/figC_ratio_ee6000', 24000, 6000, 4.0),
    ('B ee9000 2.67:1', 'results/figC_ratio_ee9000', 24000, 9000, 2.667),
]


def metrics(sl):
    si = np.asarray(sl['step_idx'])
    ph = np.array(phase_per_tick(sl))
    dg = np.asarray(sl['d_grip_swing'], float)
    lam = np.asarray(sl['lambda_qp'], float)
    eul = np.asarray(sl['struct_euler_deg'], float)
    e_tori = np.asarray(sl.get('e_torso_ori', []), float)
    e_tpos = np.asarray(sl.get('e_torso_pos', []), float)
    hwp = np.asarray(sl.get('hw_physical', sl.get('hw', [])), float)

    # 6 SS-docks
    docks = []
    for k in range(6):
        m = (si == k) & (ph == 'SS') & np.isfinite(dg)
        docks.append(float(1000 * dg[m].min()) if m.any() else float('nan'))
    worst = float(np.nanmax(docks))

    # theta_s (structure attitude): norm of euler; peak over run, settled = last tick
    th = np.linalg.norm(eul, axis=1)
    th_peak = float(np.max(th))
    th_settled = float(th[-1])
    th_maxcomp = float(np.max(np.abs(eul)))   # alt convention cross-check

    # realized |Hdot_s|pk per step (stance-contact reduction about struct origin)
    aA, aB = load_anchors_struct_frame(
        model, np.asarray(sl['struct_pos'])[0], np.asarray(sl['struct_quat'])[0])
    hdot = []
    for k in range(6):
        anch = (aA if STANCE_ARM[k] == 'a' else aB)[STANCE_ANCHOR[k]]
        off = 0 if STANCE_ARM[k] == 'a' else 6
        ss = (si == k) & (ph == 'SS')
        hpk = 0.0
        for i in np.where(ss)[0]:
            f = lam[i, off:off + 3]; tau = lam[i, off + 3:off + 6]
            hpk = max(hpk, float(np.max(np.abs(np.cross(anch, f) + tau))))
        hdot.append(round(hpk, 3))

    # torso-pose tracking residual during swing (SS ticks)
    ssall = (ph == 'SS')
    tori_pk = float(np.max(e_tori[ssall])) if e_tori.size and ssall.any() else float('nan')
    tpos_pk = float(1000 * np.max(e_tpos[ssall])) if e_tpos.size and ssall.any() else float('nan')

    hw_pk = float(np.max(np.linalg.norm(hwp, axis=1))) if hwp.size else float('nan')
    n_dock = len(sl.get('dock_events', []))
    qp_ok = bool(np.all(sl.get('qp_ok', [True]))); nmpc_ok = bool(np.all(sl.get('nmpc_ok', [True])))

    return dict(docks=[round(d, 4) for d in docks], worst=round(worst, 4),
                margin=round(5.0 - worst, 4),
                theta_s_peak=round(th_peak, 4), theta_s_settled=round(th_settled, 4),
                theta_s_maxcomp=round(th_maxcomp, 4),
                hdot_s_pk=hdot, torso_ori_resid_pk=round(tori_pk, 4),
                torso_pos_resid_pk_mm=round(tpos_pk, 4), hw_pk=round(hw_pk, 4),
                n_dock=n_dock, qp_ok=qp_ok, nmpc_ok=nmpc_ok,
                all6=bool(sum(1 for d in docks if d < 5.0) == 6))


rows = []
for (lab, d, torso, ee, ratio) in CONFIGS:
    try:
        sl = json.load(open(f'{d}/sim_log.json'))
    except FileNotFoundError:
        rows.append(dict(label=lab, torso=torso, ee=ee, ratio=ratio, missing=True))
        print(f'{lab:18s}  MISSING'); continue
    m = metrics(sl)
    rows.append(dict(label=lab, torso=torso, ee=ee, ratio=ratio, **m))

json.dump(rows, open('results/j2_adjconv/ratio_sweep.json', 'w'), indent=2)

print(f'\n{"config":18s} {"ratio":>5} | {"docks (6 SS closest-approach, mm)":^44} | {"worst":>6} {"marg":>6}')
print('-' * 100)
for r in rows:
    if r.get('missing'):
        print(f'{r["label"]:18s}  MISSING'); continue
    ds = ' '.join(f'{d:5.3f}' for d in r['docks'])
    print(f'{r["label"]:18s} {r["ratio"]:5.2f} | {ds} | {r["worst"]:6.3f} {r["margin"]:6.3f}  '
          f'{"ALL<5" if r["all6"] else "MISS!"}')

print(f'\n{"config":18s} | {"theta_s pk":>10} {"settled":>8} {"(maxcomp)":>9} | '
      f'{"torso_ori_pk":>12} {"tpos_pk_mm":>10} | {"hw_pk":>6} | feas')
print('-' * 100)
for r in rows:
    if r.get('missing'):
        continue
    print(f'{r["label"]:18s} | {r["theta_s_peak"]:10.4f} {r["theta_s_settled"]:8.4f} '
          f'{r["theta_s_maxcomp"]:9.4f} | {r["torso_ori_resid_pk"]:12.4f} '
          f'{r["torso_pos_resid_pk_mm"]:10.4f} | {r["hw_pk"]:6.3f} | '
          f'{r["n_dock"]}dk qp{"OK" if r["qp_ok"] else "X"}')

print(f'\n{"config":18s} | realized |Hdot_s|pk per step (must stay ~ canonical)')
print('-' * 100)
for r in rows:
    if r.get('missing'):
        continue
    print(f'{r["label"]:18s} | ' + '  '.join(f's{k}:{v:5.3f}' for k, v in enumerate(r['hdot_s_pk'])))

# analysis: worst-dock & theta_s-peak vs ratio; first ratio with worst<4.8
print('\n=== worst-dock & theta_s-peak vs torso:EE ratio (route A, lowering torso) ===')
routeA = [r for r in rows if not r.get('missing') and (r['ee'] == 3000)]
for r in sorted(routeA, key=lambda x: -x['ratio']):
    hit = 'DOCK<4.8 OK' if r['worst'] < 4.8 else ''
    th_ok = 'theta_s OK' if r['theta_s_peak'] < 0.6 else 'theta_s>0.6!'
    print(f'  ratio {r["ratio"]:5.2f} (torso {r["torso"]:5d}): worst={r["worst"]:.3f}mm '
          f'theta_pk={r["theta_s_peak"]:.3f} {hit} {th_ok}')
print('=== route B (raising EE) vs route A at matched ratio ===')
for r in rows:
    if r.get('missing'):
        continue
    if r['ratio'] in (4.0, 2.667):
        print(f'  {r["label"]:18s} ratio {r["ratio"]:.2f}: worst={r["worst"]:.3f}mm '
              f'theta_pk={r["theta_s_peak"]:.3f} torso_ori_pk={r["torso_ori_resid_pk"]:.3f}')
