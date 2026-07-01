#!/usr/bin/env python3
"""Phase 1e'-B addendum — does the ~1e6 QP conditioning EXPRESS at speed?
Like-for-like cond(H) / Ḣ_s-HF / k+1-residual at h_max=6 vs the h_max=5 baseline (step-0
filtered for fairness, since h_max=6 aborts at step 0). Pure post-processing. ZERO solve.

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/analyze_cond_at_speed.py
"""
import json

import numpy as np

R5 = json.load(open('results/j2_ablation_envelope/qp_cond_raw.json'))            # h_max=5 (every 10th)
R6 = json.load(open('results/j2_ablation_envelope/qp_cond_raw_figC_qpcond6.json'))  # h_max=6 (every solve)
S5 = json.load(open('results/figC_qpcond/sim_log.json'))
S6 = json.load(open('results/figC_qpcond6/sim_log.json'))
out = {}


def qd(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return dict(median=round(float(np.median(a)), 0), p90=round(float(np.percentile(a, 90)), 0),
                max=round(float(np.max(a)), 0), n=int(a.size)) if a.size else dict(n=0)


def step0(qp):
    return [r for r in qp if 'step00' in r['label']]


print('=' * 88)
print('ADDENDUM — cond(H) at speed (h_max=6 vs h_max=5), STEP-0 like-for-like')
for tag, qp in [('h5 step0', step0(R5['qp'])), ('h6 step0', step0(R6['qp']))]:
    c = np.array([r['cond_H'] for r in qp], float); ph = np.array([r['phase'] for r in qp])
    css = c[ph == 'SS']; cds = c[ph == 'DS']
    print(f'  {tag:>9}: ALL {qd(c)}')
    print(f'            SS  {qd(css)}   DS {qd(cds)}')
    st = set(r['status'] for r in qp); fails = sum(1 for r in qp if not r['success'])
    print(f'            status={st}  not-success={fails}/{len(qp)}')
out['cond_h5_step0_SS'] = qd(np.array([r['cond_H'] for r in step0(R5['qp']) if r['phase'] == 'SS']))
out['cond_h6_step0_SS'] = qd(np.array([r['cond_H'] for r in step0(R6['qp']) if r['phase'] == 'SS']))

# --- Ḣ_s HF: realized (tau_w) vs planned (NMPC ΔL/dt) succ-diff RMS on SS, per axis ---
print('\nḢ_s high-frequency content on SS (does realized carry HF absent from the plan?)')
dt_n = R5['dt_nmpc']


def hf_ratio(sim, nmpc, tag):
    ph = np.array(sim['phase']); ss = ph == 'SS'
    tw = np.asarray(sim['tau_w'], float)[ss]                    # realized wheel torque = realized Ḣ_s
    # planned Ḣ_s = NMPC ΔL/dt (angular-momentum rate the wheels must supply)
    L_in = np.array([e['L_in'] for e in nmpc]); L_pl = np.array([e['L_plan'] for e in nmpc])
    plan_rate = (L_pl - L_in) / dt_n
    hf_real = np.sqrt(np.mean(np.diff(tw, axis=0) ** 2, axis=0))
    hf_plan = np.sqrt(np.mean(np.diff(plan_rate, axis=0) ** 2, axis=0))
    ratio = [round(float(hf_real[a] / max(hf_plan[a], 1e-9)), 2) for a in range(3)]
    print(f'  {tag}: HF realized(τ_w)={[round(float(v),4) for v in hf_real]}  '
          f'HF planned(NMPC ΔL/dt)={[round(float(v),4) for v in hf_plan]}  ratio={ratio}')
    return dict(hf_real=[round(float(v), 4) for v in hf_real],
                hf_plan=[round(float(v), 4) for v in hf_plan], ratio=ratio)


out['hf_h5'] = hf_ratio(S5, R5['nmpc'], 'h5')
out['hf_h6'] = hf_ratio(S6, R6['nmpc'], 'h6')

# --- k+1 residual h6 vs h5 ---
print('\nk+1 feedback residual (NMPC predicted t+dt vs measured next state)')


def kp1(nmpc, tag):
    r_in = np.array([e['r_in'] for e in nmpc]); r_pl = np.array([e['r_plan'] for e in nmpc])
    v_in = np.array([e['v_in'] for e in nmpc]); v_pl = np.array([e['v_plan'] for e in nmpc])
    L_in = np.array([e['L_in'] for e in nmpc]); L_pl = np.array([e['L_plan'] for e in nmpc])
    rr = np.linalg.norm(r_pl[:-1] - r_in[1:], axis=1); vv = np.linalg.norm(v_pl[:-1] - v_in[1:], axis=1)
    LL = np.linalg.norm(L_pl[:-1] - L_in[1:], axis=1)
    d = dict(r_rms_mm=round(float(np.sqrt(np.mean(rr**2))*1e3), 3),
             v_rms_mmps=round(float(np.sqrt(np.mean(vv**2))*1e3), 3),
             L_rms_Nms=round(float(np.sqrt(np.mean(LL**2))), 4), n=len(nmpc))
    print(f'  {tag}: |Δr|={d["r_rms_mm"]}mm  |Δv|={d["v_rms_mmps"]}mm/s  |ΔL|={d["L_rms_Nms"]}Nms  (n={d["n"]})')
    return d


out['kp1_h5'] = kp1(R5['nmpc'], 'h5 (all steps)')
out['kp1_h6'] = kp1(R6['nmpc'], 'h6 (step0)')

json.dump(out, open('results/j2_ablation_envelope/cond_at_speed.json', 'w'), indent=2)
print('\nwrote results/j2_ablation_envelope/cond_at_speed.json')
print('=' * 88)
