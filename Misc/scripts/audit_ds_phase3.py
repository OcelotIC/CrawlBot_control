#!/usr/bin/env python3
"""DS inter-step audit — PHASE 3 (behaviour metrics for §VII). READ-ONLY.
Per-axis θ_s = log3(R_init^T R_now); SS-vs-DS decomposition; Δθ_k series; SS→DS
boundary (transition physics); τ_w saturation duty in SS; §6 reconciliation.
Phase-segmented on the authoritative `phase` label (Phase-2 method). No edits.
"""
import csv
import json
import os
import numpy as np
import pinocchio as pin

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
RES = os.path.join(ROOT, 'Misc', 'runs', 'p3b_gate_w24000_kp3')
TAU_W_MAX = 5.0
GUARD_S = 0.2


def contiguous(mask):
    segs, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            segs.append((i, j - 1)); i = j
        else:
            i += 1
    return segs


def main():
    sl = json.load(open(os.path.join(RES, 'sim_log.json')))
    rows = list(csv.DictReader(open(os.path.join(RES, 'postproc_F3F4.csv'))))
    t = np.array(sl['t']); ph = np.array(sl['phase'])
    tau_w = np.array(sl['tau_w']); om = np.array(sl['omega_s'])
    sq = np.array(sl['struct_quat'])           # (N,4) wxyz
    ptr = np.array(sl['p_torso_ref']); qtr = np.array(sl['q_torso_ref'])  # torso ref
    theta_f3 = np.array([float(r['theta_s_deg']) for r in rows])

    # θ_s vector per tick = log3(R_init^T R_now), deg
    R_init = pin.Quaternion(*sq[0]).toRotationMatrix()
    th = np.empty((len(t), 3))
    for k in range(len(t)):
        R = pin.Quaternion(*sq[k]).toRotationMatrix()
        th[k] = np.degrees(pin.log3(R_init.T @ R))
    thn = np.linalg.norm(th, axis=1)
    print('## PHASE 3 — behaviour. θ_s per-axis vs F3F4 norm sanity: '
          f'max|‖θ_vec‖−theta_s_deg|={np.max(np.abs(thn-theta_f3)):.3e}°\n')

    # ordered segments
    segs = sorted([('DS', a, b) for a, b in contiguous(ph == 'DS')] +
                  [('SS', a, b) for a, b in contiguous(ph == 'SS')], key=lambda s: s[1])
    ds = [s for s in segs if s[0] == 'DS']; ssg = [s for s in segs if s[0] == 'SS']
    n_ss = len(ssg)

    def fmt(v):
        return f'[{v[0]:+.3f},{v[1]:+.3f},{v[2]:+.3f}]|{np.linalg.norm(v):.3f}'

    # ---- item 1: per-step SS-vs-DS decomposition ----
    print('=== §5.4/item1 — per-step attitude: SS contribution vs inter-step-DS deposit (deg, [x,y,z]|norm) ===')
    print(f'{"step":>4} {"ΔθSS_k (swing)":>34} {"ΔθDS_k (inter-step settle)":>40}')
    ss_contrib, ds_contrib = [], []
    for k in range(n_ss):
        _, sa, sb = ssg[k]
        dssa, dsb = ds[k + 1][1], ds[k + 1][2]   # DS AFTER SS_k (k+1 since ds[0]=initial)
        terminal = (k == n_ss - 1)
        dss = th[sb] - th[sa]
        dds = th[dsb] - th[dssa]
        ss_contrib.append(dss); ds_contrib.append(dds)
        tag = ' (DS=TERMINAL settle)' if terminal else ''
        print(f'{k:>4} {fmt(dss):>34} {fmt(dds):>40}{tag}')
    inter_ds = ds_contrib[:-1]
    print(f'  inter-step DS deposit Σ (excl terminal): {fmt(sum(inter_ds))}')
    print(f'  SS deposit Σ (all steps):                {fmt(sum(ss_contrib))}')

    # ---- item 2: Δθ_k series (net per cycle = θ(end DS_k) − θ(end DS_{k−1})) ----
    print('\n=== item2 — Δθ_k net-per-cycle series (θ at end of each DS window, consecutive diffs) ===')
    ds_end_theta = [th[b] for (_, a, b) in ds]   # end of DS0(initial),DS1..DS4, DS5(terminal)
    print(f'  θ_s at end of initial DS: {fmt(ds_end_theta[0])}')
    for k in range(1, len(ds)):
        dq = ds_end_theta[k] - ds_end_theta[k - 1]
        tag = ' (→ terminal)' if k == len(ds) - 1 else ''
        print(f'  Δθ_{k-1} (cycle {k-1}): {fmt(dq)}{tag}')
    sig = np.sign([ (ds_end_theta[k]-ds_end_theta[k-1]) for k in range(1,len(ds)) ])
    print('  same-sign within-gait? (sign of per-axis increments, cycles 0..3 excl terminal):')
    for ax,nm in enumerate('xyz'):
        s=[np.sign(ds_end_theta[k][ax]-ds_end_theta[k-1][ax]) for k in range(1,len(ds)-1)]
        print(f'    {nm}: {s}')

    # ---- item 3: Σ Δθ_k vs θ_final (confounded — mid-decay caveat) ----
    theta_final = th[-1]
    sum_dq = ds_end_theta[-1] - ds_end_theta[0]   # telescopes
    print(f'\n=== item3 — Σ Δθ_k vs θ_s,final (CONFOUNDED: 20s settle ≪ 288s t_s) ===')
    print(f'  θ_s,final (after terminal settle) = {fmt(theta_final)}')
    print(f'  θ_s at end of initial DS          = {fmt(ds_end_theta[0])}')
    print(f'  Σ Δθ_k (initial→final, telescoped)= {fmt(sum_dq)}  [report only; NOT validation]')

    # ---- item 4: SS→DS boundary (transition physics) ----
    print('\n=== §5.5/item4 — SS→DS boundary: ref-jump, τ_w drop, θ̈_s spike (transition physics) ===')
    dt = float(np.median(np.diff(t)))
    omdot = np.gradient(om, dt, axis=0)
    thddot = np.linalg.norm(omdot, axis=1)
    print(f'{"step":>4} {"ref_jump_pos_mm":>15} {"ref_jump_ori_deg":>16} {"τ_w_drop_Nm":>12} {"θ̈s_spike_pk":>12} {"θ̈s_body_pk":>11}')
    for k in range(n_ss):
        _, sa, sb = ssg[k]
        dssa, dsb = ds[k + 1][1], ds[k + 1][2]
        # ref jump: torso ref at last SS tick vs first DS tick
        rjp = np.linalg.norm(ptr[dssa] - ptr[sb]) * 1000
        Rss = pin.Quaternion(*qtr[sb]).toRotationMatrix(); Rds = pin.Quaternion(*qtr[dssa]).toRotationMatrix()
        rjo = np.degrees(np.linalg.norm(pin.log3(Rss.T @ Rds)))
        tw_drop = float(np.linalg.norm(tau_w[sb]))   # τ_w just before DS (drops to 0)
        # θ̈_s spike in guard vs body of the DS window
        win = np.arange(dssa, dsb + 1); tw_win = t[win]
        guard = tw_win <= (t[dssa] + GUARD_S); body = ~guard
        sp = float(np.max(thddot[win][guard])) if guard.any() else float('nan')
        bd = float(np.max(thddot[win][body])) if body.any() else float('nan')
        print(f'{k:>4} {rjp:>15.1f} {rjo:>16.3f} {tw_drop:>12.3f} {sp:>12.4f} {bd:>11.4f}')

    # ---- item 5: τ_w saturation duty cycle in SS ----
    print('\n=== item5 — τ_w saturation duty in SS (frac ticks |τ_w|∞=5.0: K_θ clamped/open-loop) ===')
    for k in range(n_ss):
        _, sa, sb = ssg[k]
        twi = np.abs(tau_w[sa:sb + 1])
        sat = np.mean(twi.max(1) >= TAU_W_MAX - 1e-6) * 100
        print(f'  SS step{k}: {sat:.1f}% saturated  (n={sb-sa+1})')
    ssmask = ph == 'SS'
    print(f'  ALL SS: {np.mean(np.abs(tau_w[ssmask]).max(1)>=TAU_W_MAX-1e-6)*100:.1f}% saturated')

    # ---- item 7: §6 reconciliation numbers ----
    print('\n=== item7 — §6 reconciliation: intra-DS Δθ_s norm per inter-step window (vs 0.085° base / 0.4° §11) ===')
    for k in range(len(inter_ds)):
        print(f'  inter-step DS{k}: ‖Δθ_DS‖ = {np.linalg.norm(inter_ds[k]):.4f}°')
    print(f'  mean intra-DS ‖Δθ_DS‖ = {np.mean([np.linalg.norm(x) for x in inter_ds]):.4f}°')
    print(f'\nθ_s,final norm = {np.linalg.norm(theta_final):.3f}° (bounded; over 5 steps + 20s settle)')

    # persist
    json.dump(dict(
        ss_contrib=[x.tolist() for x in ss_contrib], ds_contrib=[x.tolist() for x in ds_contrib],
        ds_end_theta=[x.tolist() for x in ds_end_theta], theta_final=theta_final.tolist(),
    ), open(os.path.join(RES, 'ds_audit_phase3.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
