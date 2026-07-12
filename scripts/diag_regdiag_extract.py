#!/usr/bin/env python3
"""Phase REG-DIAG — extract dock/theta_s/Hdot_s + kappa(eps) + well-posedness floor.
READ-ONLY. Run after diag_regdiag_sweep.py.

Per epsilon: 6 SS-docks, theta_s peak/settled, realized |Hdot_s|pk x6, feasibility,
qp_ok count; and from regcap.json: min lam_min(H_LS) over the run + kappa(H)=
(lam_max+eps)/(lam_min+eps) at representative ticks. Confirms dock-vs-eps flatness,
the kappa(eps) curve (blow-up as eps->0?), and step-2 eps-invariance.

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_regdiag_extract.py
"""
import json
import numpy as np
import mujoco
from scripts.export_figure_data import load_anchors_struct_frame, phase_per_tick, MJCF

model = mujoco.MjModel.from_xml_path(MJCF)
STANCE_ARM = ['a', 'b', 'a', 'b', 'a', 'b']
STANCE_ANCHOR = [2, 3, 3, 4, 4, 5]
# (tag, eps)
CONFIGS = [('1em4', 1e-4), ('1em6', 1e-6), ('1em8', 1e-8),
           ('1em10', 1e-10), ('1em12', 1e-12)]
BASELINE_DOCKS = [4.9401, 4.4054, 4.9039, 4.4358, 4.0453, 4.9999]  # figC_sw_s5_x1


def dock_theta_hdot(sl):
    si = np.asarray(sl['step_idx']); ph = np.array(phase_per_tick(sl))
    dg = np.asarray(sl['d_grip_swing'], float); lam = np.asarray(sl['lambda_qp'], float)
    eul = np.asarray(sl['struct_euler_deg'], float)
    docks = []
    for k in range(6):
        m = (si == k) & (ph == 'SS') & np.isfinite(dg)
        docks.append(round(float(1000 * dg[m].min()), 4) if m.any() else float('nan'))
    th = np.linalg.norm(eul, axis=1)
    aA, aB = load_anchors_struct_frame(
        model, np.asarray(sl['struct_pos'])[0], np.asarray(sl['struct_quat'])[0])
    hdot = []
    for k in range(6):
        anch = (aA if STANCE_ARM[k] == 'a' else aB)[STANCE_ANCHOR[k]]
        off = 0 if STANCE_ARM[k] == 'a' else 6
        ss = (si == k) & (ph == 'SS'); hpk = 0.0
        for i in np.where(ss)[0]:
            f = lam[i, off:off + 3]; tau = lam[i, off + 3:off + 6]
            hpk = max(hpk, float(np.max(np.abs(np.cross(anch, f) + tau))))
        hdot.append(round(hpk, 3))
    return (docks, round(float(th.max()), 4), round(float(th[-1]), 4), hdot,
            len(sl.get('dock_events', [])), int(np.sum(~np.asarray(sl.get('qp_ok', [True])))))


def kappa_reps(cap, eps):
    """kappa(H)=(lam_max+eps)/(lam_min+eps) at DS/SS-mid/SS-dock; min lam_min over run."""
    a = np.array(cap, float)
    if not len(a):
        return {}
    ss = a[a[:, 1] == 26]; ds = a[a[:, 1] == 32]
    out = {'min_lam_min_LS': float(a[:, 2].min()),
           'min_lam_min_SS': float(ss[:, 2].min()) if len(ss) else float('nan'),
           'min_lam_min_DS': float(ds[:, 2].min()) if len(ds) else float('nan')}
    def kap(row):
        lm, lM = row[2], row[3]
        return float((lM + eps) / (lm + eps))
    if len(ds):
        out['kappa_DS'] = kap(ds[0])
    if len(ss):
        out['kappa_SS_mid'] = kap(ss[len(ss) // 2])
        out['kappa_SS_dock'] = kap(ss[-1])
    return out


rows = []
for (tag, eps) in CONFIGS:
    d = f'results/figC_reg_{tag}'
    try:
        sl = json.load(open(f'{d}/sim_log.json'))
        cap = json.load(open(f'{d}/regcap.json'))['cap']
    except FileNotFoundError:
        rows.append(dict(tag=tag, eps=eps, missing=True)); continue
    docks, th_pk, th_set, hdot, ndk, nqpfail = dock_theta_hdot(sl)
    kp = kappa_reps(cap, eps)
    rows.append(dict(tag=tag, eps=eps, docks=docks, worst=round(float(np.nanmax(docks)), 4),
                     theta_pk=th_pk, theta_settled=th_set, hdot=hdot,
                     n_dock=ndk, n_qp_fail=nqpfail,
                     dock_vs_baseline_max=round(float(np.nanmax(
                         np.abs(np.array(docks) - np.array(BASELINE_DOCKS)))), 5), **kp))

json.dump(rows, open('results/j2_adjconv/reg_sweep.json', 'w'), indent=2)

print(f'{"eps":>7} | {"6 SS-docks [mm]":^40} | {"worst":>6} {"dvsbase":>8} | {"th_pk":>6} | {"kSSdock":>9} {"minLamLS":>10} | dk qpF')
print('-' * 118)
for r in rows:
    if r.get('missing'):
        print(f'{r["eps"]:7.0e} | MISSING'); continue
    ds = ' '.join(f'{x:5.3f}' for x in r['docks'])
    print(f'{r["eps"]:7.0e} | {ds} | {r["worst"]:6.3f} {r["dock_vs_baseline_max"]:8.5f} | '
          f'{r["theta_pk"]:6.4f} | {r.get("kappa_SS_dock",float("nan")):9.3e} '
          f'{r.get("min_lam_min_LS",float("nan")):10.4e} | {r["n_dock"]}  {r["n_qp_fail"]}')

print('\n=== realized |Hdot_s|pk per step (should be eps-invariant) ===')
for r in rows:
    if r.get('missing'):
        continue
    print(f'  eps={r["eps"]:7.0e}: ' + '  '.join(f's{k}:{v:5.3f}' for k, v in enumerate(r['hdot'])))

print('\n=== step-2 dock vs eps (momentum-saturated floor; expect ~4.90 flat) ===')
for r in rows:
    if r.get('missing'):
        continue
    print(f'  eps={r["eps"]:7.0e}: step2 dock = {r["docks"][2]:.4f} mm   '
          f'(Hdot_s s2 = {r["hdot"][2]:.3f})')

print('\n=== kappa(H) vs eps at representative ticks + well-posedness floor ===')
print(f'{"eps":>8} {"kappa_DS":>11} {"kappa_SS_mid":>13} {"kappa_SS_dock":>13} {"minLamMin(H_LS)":>16}')
for r in rows:
    if r.get('missing'):
        continue
    print(f'{r["eps"]:8.0e} {r.get("kappa_DS",float("nan")):11.3e} '
          f'{r.get("kappa_SS_mid",float("nan")):13.3e} {r.get("kappa_SS_dock",float("nan")):13.3e} '
          f'{r.get("min_lam_min_LS",float("nan")):16.4e}')
