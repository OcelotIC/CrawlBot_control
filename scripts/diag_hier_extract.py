#!/usr/bin/env python3
"""Phase HIER-STANDARD-SWEEP — extract per-config metrics.
READ-ONLY. Reads CONFIGS (label,dir,torso,ee); handles both new figC_hier_* (with
regcap.json for kappa) and reused figC_ratio_*/figC_sw_s5_x1 (kappa analytic).

Metrics: at-weld docks x6 (dock_events, NOT min-over-swing), worst/margin/feasible;
SS-swing saturation (realized Hdot_s SS peak per step + ticks at cap); theta_s pk/settled;
e_com peak (SS); kappa(H) SS.

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_hier_extract.py [sweep2]
"""
import sys
import json
import numpy as np
import mujoco
from scripts.export_figure_data import load_anchors_struct_frame, phase_per_tick, MJCF

model = mujoco.MjModel.from_xml_path(MJCF)
STANCE_ARM = ['a', 'b', 'a', 'b', 'a', 'b']; STANCE_ANCHOR = [2, 3, 3, 4, 4, 5]
CAP = 5.0

SWEEP1 = [
    ('ee3000  8.0:1 (canon)', 'results/figC_sw_s5_x1', 24000, 3000),
    ('ee4500  5.3:1', 'results/figC_hier_ee4500', 24000, 4500),
    ('ee6000  4.0:1', 'results/figC_ratio_ee6000', 24000, 6000),
    ('ee9000  2.67:1', 'results/figC_ratio_ee9000', 24000, 9000),
    ('ee12000 2.0:1', 'results/figC_hier_ee12000', 24000, 12000),
    ('ee24000 1.0:1', 'results/figC_hier_ee24000', 24000, 24000),
]
import os
SWEEP2 = []
_s2 = 'results/j2_adjconv/_hier_sweep2_configs.json'
if os.path.exists(_s2):
    SWEEP2 = [tuple(x) for x in json.load(open(_s2))]
CONFIGS = SWEEP1 + (SWEEP2 if len(sys.argv) > 1 and sys.argv[1] == 'sweep2' else [])


def metrics(d, torso, ee):
    try:
        sl = json.load(open(f'{d}/sim_log.json'))
    except FileNotFoundError:
        return None
    si = np.asarray(sl['step_idx']); ph = np.array(phase_per_tick(sl))
    is_ss = np.array([p == 'SS' for p in ph])
    lam = np.asarray(sl['lambda_qp'], float)
    eul = np.asarray(sl['struct_euler_deg'], float)
    ecom = np.asarray(sl.get('e_com', []), float)
    dev = {e['step']: e for e in sl.get('dock_events', [])}
    weld = [round(dev[k]['d_mm'], 3) if k in dev else float('nan') for k in range(6)]
    n_dock = len(dev); all6 = (n_dock == 6 and all(np.isfinite(weld)))
    worst = float(np.nanmax(weld)) if np.any(np.isfinite(weld)) else float('nan')
    # SS realized Hdot_s peak per step (stance-contact)
    aA, aB = load_anchors_struct_frame(model, np.asarray(sl['struct_pos'])[0],
                                       np.asarray(sl['struct_quat'])[0])
    ssh = []
    for k in range(6):
        anch = (aA if STANCE_ARM[k] == 'a' else aB)[STANCE_ANCHOR[k]]
        off = 0 if STANCE_ARM[k] == 'a' else 6
        m = (si == k) & is_ss; hpk = 0.0
        for i in np.where(m)[0]:
            hpk = max(hpk, float(np.max(np.abs(np.cross(anch, lam[i, off:off+3]) + lam[i, off+3:off+6]))))
        ssh.append(round(hpk, 3))
    ss_at_cap = int(sum(1 for v in ssh if v >= CAP - 0.05))     # steps whose SS peak saturates
    th = np.linalg.norm(eul, axis=1)
    ecom_pk = float(np.max(ecom[is_ss])) if ecom.size and is_ss.any() else float('nan')
    # kappa from regcap
    kap = None
    rp = f'{d}/regcap.json'
    if os.path.exists(rp):
        cap = np.array(json.load(open(rp)).get('cap', []), float)
        if len(cap):
            ss = cap[cap[:, 1] == 26]
            if len(ss):
                lm = np.median(ss[:, 2]); lM = np.median(ss[:, 3])
                kap = float((lM + 1e-6) / (lm + 1e-6))
    return dict(torso=torso, ee=ee, ratio=round(torso / ee, 2), weld=weld, worst=round(worst, 3),
                margin=round(5.0 - worst, 3), n_dock=n_dock, all6=all6,
                ss_hdot=ssh, ss_steps_saturating=ss_at_cap,
                sat_preserved=bool(ssh[2] >= 4.9 and ssh[4] >= 4.9),
                theta_pk=round(float(th.max()), 4), theta_settled=round(float(th[-1]), 4),
                e_com_pk=round(ecom_pk, 5), kappa_SS=kap)


rows = []
print(f'{"config":22s} {"torso:EE":>9} | {"at-weld docks x6 [mm]":^40} | {"worst":>6} {"marg":>6} {"feas":>5}')
print('-' * 110)
for (lab, d, torso, ee) in CONFIGS:
    m = metrics(d, torso, ee)
    if m is None:
        print(f'{lab:22s}  (run not present yet)'); rows.append(dict(label=lab, missing=True)); continue
    rows.append(dict(label=lab, **m))
    ws = ' '.join(f'{x:5.3f}' if np.isfinite(x) else ' TMO ' for x in m['weld'])
    print(f'{lab:22s} {m["ratio"]:7.2f}:1 | {ws} | {m["worst"]:6.3f} {m["margin"]:6.3f} '
          f'{("6/6" if m["all6"] else str(m["n_dock"])+"/6"):>5}')

print(f'\n{"config":22s} | {"SS Hdot_s peak per step (thesis: s2,s4 ~5)":^46} | sat? | theta_pk theta_set | e_com_pk | kappaSS')
print('-' * 120)
for r in rows:
    if r.get('missing'):
        continue
    hs = '  '.join(f'{v:4.2f}' for v in r['ss_hdot'])
    ks = f'{r["kappa_SS"]:.2e}' if r['kappa_SS'] else '  (analytic)'
    print(f'{r["label"]:22s} | {hs} | {"YES" if r["sat_preserved"] else "no ":>4} '
          f'| {r["theta_pk"]:7.4f} {r["theta_settled"]:7.4f} | {r["e_com_pk"]:8.5f} | {ks}')

json.dump(rows, open('results/j2_adjconv/hier_sweep.json', 'w'), indent=2)
print('\nwrote results/j2_adjconv/hier_sweep.json')
