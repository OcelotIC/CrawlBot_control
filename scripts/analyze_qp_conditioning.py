#!/usr/bin/env python3
"""Phase QP-cond analysis — Task A (QP conditioning) + Task B (command smoothness / feedback).
Reads the captured qp_cond_raw.json + the rerun sim_log (tau_q) + committed C CSV (planned vs
realized Hdot_s). Pure post-processing; ZERO solve. Writes qp_cond_summary.json + prints tables.

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/analyze_qp_conditioning.py
"""
import csv
import json

import numpy as np

RAW = 'results/j2_ablation_envelope/qp_cond_raw.json'
SIM = 'results/figC_qpcond/sim_log.json'
CCSV = 'results/j2_canonical_revalidation/runfix_traversal.csv'
OUT = 'results/j2_ablation_envelope/qp_cond_summary.json'


def q(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if a.size == 0:
        return dict(median=None, p90=None, max=None, n=0)
    return dict(median=round(float(np.median(a)), 1), p90=round(float(np.percentile(a, 90)), 1),
                max=round(float(np.max(a)), 1), n=int(a.size))


raw = json.load(open(RAW))
qp = raw['qp']
nmpc = raw['nmpc']
dt_n = raw['dt_nmpc']
out = {}

# ============================ TASK A ============================
print('=' * 84)
print('TASK A — QP cost-Hessian conditioning on canonical C (h_max=5), sampled every 10th solve')
condH = np.array([r['cond_H'] for r in qp], float)
phase = np.array([r['phase'] for r in qp])
cond_notorso = np.array([r['cond_no_torso'] for r in qp], float)
ss = phase == 'SS'; ds = phase == 'DS'
print(f'  samples={len(qp)}  (SS={int(ss.sum())}, DS={int(ds.sum())}, other={int((~ss&~ds).sum())})')
out['taskA'] = {
    'n_samples': len(qp),
    'cond_H_all': q(condH), 'cond_H_SS': q(condH[ss]), 'cond_H_DS': q(condH[ds]),
    'cond_H_no_torso_all': q(cond_notorso),
}
print('  cond(H) distribution [median / p90 / max]:')
for tag, a in [('ALL', condH), ('SS', condH[ss]), ('DS', condH[ds])]:
    s = q(a)
    print(f'    {tag:>3}: {s["median"]:>12,.0f} / {s["p90"]:>12,.0f} / {s["max"]:>12,.0f}  (n={s["n"]})')
s0, s1 = q(condH), q(cond_notorso)
print(f'  cond(H) with torso(α=24000) block REMOVED [median/p90/max]: '
      f'{s1["median"]:,.0f} / {s1["p90"]:,.0f} / {s1["max"]:,.0f}   '
      f'(vs full {s0["median"]:,.0f}) -> factor {s0["median"]/max(s1["median"],1):.1f}x')

# dominant task
from collections import Counter
domw = Counter(r['dom_task_byweight'] for r in qp)
domb = Counter(r['dom_task_byblock'] for r in qp)
print(f'  dominant task by WEIGHT: {dict(domw)}')
print(f'  dominant task by BLOCK spectral-norm: {dict(domb)}')
out['taskA']['dominant_by_weight'] = dict(domw)
out['taskA']['dominant_by_block'] = dict(domb)

# solver status distribution
st = Counter(r['status'] for r in qp)
nfail = sum(1 for r in qp if not r['success'])
print(f'  solver return_status distribution: {dict(st)}')
print(f'  solves not-success: {nfail}/{len(qp)}')
out['taskA']['status_distribution'] = dict(st)
out['taskA']['not_success'] = nfail

# spike at DS<->SS switches: cond of samples adjacent to a phase change vs phase medians
sw = []
for k in range(1, len(qp)):
    if qp[k]['phase'] != qp[k - 1]['phase'] and qp[k]['phase'] in ('SS', 'DS'):
        sw.append(condH[k])
med_all = np.nanmedian(condH)
print(f'  cond(H) at DS<->SS switch samples: median={np.nanmedian(sw) if sw else float("nan"):,.0f} '
      f'(n={len(sw)}) vs overall median {med_all:,.0f} -> '
      f'{(np.nanmedian(sw)/med_all if sw else float("nan")):.2f}x')
out['taskA']['switch_cond_median'] = (round(float(np.nanmedian(sw)), 1) if sw else None)

# ============================ TASK B ============================
print('\n' + '=' * 84)
print('TASK B — realized-command smoothness + feedback contamination on canonical C')

# --- B1: tau_q roughness from the rerun sim_log ---
sl = json.load(open(SIM))
tau = np.asarray(sl['tau'], float)               # (T, nu) joint+wheel torque command
ph = np.array(sl['phase']); t = np.asarray(sl['t'], float)
# arm joints only (first 14 of the command vector); guard if fewer
nj = min(14, tau.shape[1])
tau_j = tau[:, :nj]
dtau = np.diff(tau_j, axis=0)
ph_edge = ph[1:]                                  # phase of the later tick of each diff
ss_e = ph_edge == 'SS'; ds_e = ph_edge == 'DS'


def rough(d):
    return dict(rms_diff=round(float(np.sqrt(np.mean(d ** 2))), 4),
                tv_per_joint=round(float(np.mean(np.sum(np.abs(d), axis=0))), 3),
                max_abs_diff=round(float(np.max(np.abs(d))), 4))


print('  tau_q roughness (arm joints), successive-tick |Δτ|:')
print(f'    SS: {rough(dtau[ss_e])}')
print(f'    DS: {rough(dtau[ds_e])}')
# discontinuity at support switches
sw_ticks = np.where(ph[1:] != ph[:-1])[0]
sw_jump = np.max(np.abs(dtau[sw_ticks]), axis=1) if sw_ticks.size else np.array([])
typ = np.median(np.max(np.abs(dtau), axis=1))
print(f'    support-switch |Δτ|_max: median={np.median(sw_jump) if sw_jump.size else float("nan"):.4f} '
      f'(n={sw_ticks.size}) vs typical-tick median {typ:.4f} -> '
      f'{(np.median(sw_jump)/typ if sw_jump.size else float("nan")):.2f}x')
out['taskB'] = {'tau_q_SS': rough(dtau[ss_e]), 'tau_q_DS': rough(dtau[ds_e]),
                'tau_q_switch_jump_median': round(float(np.median(sw_jump)), 4) if sw_jump.size else None,
                'tau_q_typical_jump_median': round(float(typ), 4)}

# --- B2: planned (NMPC Hdot_s) vs realized (tau_w) from committed C CSV ---
r = list(csv.DictReader(open(CCSV)))
src = np.array([x['Hdot_s_source'] for x in r]); phc = np.array([x['phase'] for x in r])


def col(n):
    return np.array([float(x[n]) if x[n] != '' else np.nan for x in r])


Hd = np.stack([col('Hdot_s_x_Nm'), col('Hdot_s_y_Nm'), col('Hdot_s_z_Nm')], 1)   # planned on SS
tw = np.stack([col('tauw_x_Nm'), col('tauw_y_Nm'), col('tauw_z_Nm')], 1)          # realized wheel torque
ss_pl = (src == 'planned') & (phc == 'SS')
Hp = Hd[ss_pl]; Tr = tw[ss_pl]
# align sign (tau_w may be -Hdot_s): pick sign per axis by correlation
gap = np.zeros(3); hf_plan = np.zeros(3); hf_real = np.zeros(3)
for a in range(3):
    s = np.sign(np.nansum(Hp[:, a] * Tr[:, a])) or 1.0
    gap[a] = np.sqrt(np.nanmean((Hp[:, a] - s * Tr[:, a]) ** 2))
    hf_plan[a] = np.sqrt(np.nanmean(np.diff(Hp[:, a]) ** 2))
    hf_real[a] = np.sqrt(np.nanmean(np.diff(s * Tr[:, a]) ** 2))
print('\n  planned Ḣ_s (NMPC) vs realized Ḣ_s (wheel τ_w) on SS ticks, per axis [x,y,z]:')
print(f'    RMS gap        = {[round(float(v),3) for v in gap]} Nm')
print(f'    HF (succ-diff) planned={[round(float(v),4) for v in hf_plan]}  '
      f'realized={[round(float(v),4) for v in hf_real]}  '
      f'ratio={[round(float(hf_real[a]/max(hf_plan[a],1e-9)),1) for a in range(3)]}')
out['taskB']['Hdot_s_RMS_gap'] = [round(float(v), 3) for v in gap]
out['taskB']['Hdot_s_HF_planned'] = [round(float(v), 4) for v in hf_plan]
out['taskB']['Hdot_s_HF_realized'] = [round(float(v), 4) for v in hf_real]

# --- B3: k+1 feedback residual (NMPC predicted next state vs measured next input) ---
if len(nmpc) >= 2:
    r_in = np.array([e['r_in'] for e in nmpc]); v_in = np.array([e['v_in'] for e in nmpc])
    L_in = np.array([e['L_in'] for e in nmpc])
    r_pl = np.array([e['r_plan'] for e in nmpc]); v_pl = np.array([e['v_plan'] for e in nmpc])
    L_pl = np.array([e['L_plan'] for e in nmpc])
    # predicted[k] vs measured-input[k+1]
    res_r = np.linalg.norm(r_pl[:-1] - r_in[1:], axis=1)
    res_v = np.linalg.norm(v_pl[:-1] - v_in[1:], axis=1)
    res_L = np.linalg.norm(L_pl[:-1] - L_in[1:], axis=1)
    print('\n  k+1 feedback residual (NMPC predicted t+dt vs measured state at next NMPC tick):')
    print(f'    |Δr_com| RMS={np.sqrt(np.mean(res_r**2))*1e3:.3f} mm  max={res_r.max()*1e3:.3f} mm')
    print(f'    |Δv_com| RMS={np.sqrt(np.mean(res_v**2))*1e3:.4f} mm/s  max={res_v.max()*1e3:.4f} mm/s')
    print(f'    |ΔL_com| RMS={np.sqrt(np.mean(res_L**2)):.4f} Nms  max={res_L.max():.4f} Nms')
    out['taskB']['kp1_residual'] = dict(
        r_rms_mm=round(float(np.sqrt(np.mean(res_r**2))*1e3), 3), r_max_mm=round(float(res_r.max()*1e3), 3),
        v_rms_mmps=round(float(np.sqrt(np.mean(res_v**2))*1e3), 4),
        L_rms_Nms=round(float(np.sqrt(np.mean(res_L**2))), 4),
        L_max_Nms=round(float(res_L.max()), 4), n=len(nmpc))
    print(f'    (n={len(nmpc)} NMPC solves)')
    # correlation of the k+1 residual with tau_q roughness (resample per-tick |Δτ|max to
    # the NMPC-solve count and Pearson-correlate with |ΔL| residual)
    rough_tick = np.max(np.abs(dtau), axis=1)                      # (T-1,)
    idx = np.linspace(0, len(rough_tick) - 1, len(res_L)).astype(int)
    rough_rs = rough_tick[idx]
    if np.std(rough_rs) > 0 and np.std(res_L) > 0:
        rho = float(np.corrcoef(rough_rs, res_L)[0, 1])
    else:
        rho = float('nan')
    print(f'    corr(|ΔL| k+1 residual, τ_q roughness) Pearson ρ = {rho:.3f}')
    out['taskB']['kp1_residual']['corr_with_tauq_roughness'] = round(rho, 3)

json.dump(out, open(OUT, 'w'), indent=2)
print(f'\nwrote {OUT}')
print('=' * 84)
