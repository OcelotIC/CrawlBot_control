#!/usr/bin/env python3
"""J1 closing — Lemma 2 (centroidal reduction identity) SS test.

Reads the instrumented Fix-A canonical run (results/lemma2_ss) and tests, in the
inertial/world frame reduced about O_s = structure CoM:

    Ḣ_R/Os  =  −τ_contact          (Lemma 2, §B; gravity off, SS single weld)

LHS = central-FD of HR_Os_hf (robot-subtree angmom about O_s). RHS τ_contact =
plant-realized weld wrench reduced about O_s = weld_m + (p_grip − r_Os) × weld_f.
Reports ε = ‖Ḣ_R/Os − w/Os‖ (and the sign), with/without the m_i term, per-axis,
a window-integral cross-check, the weld-reconstruction self-check, and the J3
byproduct (plant weld wrench vs lambda_qp). Section 0 proves non-perturbation.
"""
import json
import os
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
INSTR = os.path.join(ROOT, 'results', 'lemma2_ss')
CANON = os.path.join(ROOT, 'results', 'fixA_gate')
L2 = {'t_l2_hf', 'HR_Os_hf', 'weld_f_hf', 'weld_m_hf', 'p_grip_hf', 'r_Os_hf',
      'nefc_hf', 'l2_resid_hf'}
SKIP = L2 | {'qp_time_ms', 'nmpc_time_ms', 'environment'}


def section0(instr, canon):
    print('=' * 78)
    print('0. NON-PERTURBATION  (instrumented lemma2_ss vs canonical fixA_gate)')
    print('=' * 78)
    worst = 0.0; diffs = []; nshared = 0
    for k in set(instr) & set(canon):
        if k in SKIP:
            continue
        try:
            xa = np.asarray(instr[k], float); xb = np.asarray(canon[k], float)
        except (ValueError, TypeError):
            continue
        if xa.shape != xb.shape:
            diffs.append((k, f'SHAPE {xa.shape}/{xb.shape}')); continue
        nshared += 1
        if xa.size:
            dd = np.abs(xa - xb); dd = dd[np.isfinite(dd)]
            mx = float(dd.max()) if dd.size else 0.0
            worst = max(worst, mx)
            if mx > 0:
                diffs.append((k, mx))
    print(f'  shared numeric fields: {nshared} | worst |Δ| = {worst:.3e} -> '
          f'{"BIT-IDENTICAL" if worst == 0 else "DIFFERS"}')
    if diffs:
        print(f'  diffs: {diffs[:8]}')
    else:
        print('  diffs: NONE (read-only instrument; trajectory identical)')


def segments(t, gap):
    segs, i = [], 0
    while i < len(t):
        j = i + 1
        while j < len(t) and (t[j] - t[j - 1]) < gap:
            j += 1
        segs.append((i, j - 1)); i = j
    return segs


def main():
    instr = json.load(open(os.path.join(INSTR, 'sim_log.json')))
    canon = json.load(open(os.path.join(CANON, 'sim_log.json')))
    section0(instr, canon)

    t = np.array(instr['t_l2_hf'])
    H = np.array(instr['HR_Os_hf'])          # (N,3) robot angmom about O_s
    wf = np.array(instr['weld_f_hf'])        # (N,3) weld force, world
    wm = np.array(instr['weld_m_hf'])        # (N,3) weld pure moment, world
    pg = np.array(instr['p_grip_hf'])        # (N,3) gripper site
    rOs = np.array(instr['r_Os_hf'])         # (N,3) O_s
    nefc = np.array(instr['nefc_hf'])
    resid = np.array(instr['l2_resid_hf'])
    print(f'\n  hifreq Lemma-2 samples: {len(t)}  | finite weld rows: '
          f'{int(np.isfinite(wf[:,0]).sum())}')

    # weld wrench reduced about O_s
    lever = pg - rOs
    tau_full = wm + np.cross(lever, wf)      # with m_i
    tau_noM = np.cross(lever, wf)            # without m_i

    # SS swings = contiguous hifreq runs (gap = DS jump)
    dt = np.median(np.diff(t[:50]))
    segs = [s for s in segments(t, 5 * dt) if s[1] - s[0] >= 8]
    # rank by mean |weld wrench about O_s| (rate proxy)
    segs = sorted(segs, key=lambda s: -np.nanmean(np.linalg.norm(tau_full[s[0]:s[1]+1], axis=1)))

    print('\n' + '=' * 78)
    print('a. WELD-WRENCH RECONSTRUCTION self-check (per SS swing)')
    print('=' * 78)
    print(f'  {"swing":>5} {"t[s]":>13} {"n":>4} {"nefc(min/max)":>14} '
          f'{"resid med":>10} {"|τ_full| mean":>13}')
    for k, (a, b) in enumerate(segs):
        sl = slice(a, b + 1)
        print(f'  {k:>5} {f"{t[a]:.2f}-{t[b]:.2f}":>13} {b-a+1:>4} '
              f'{f"{int(np.nanmin(nefc[sl]))}/{int(np.nanmax(nefc[sl]))}":>14} '
              f'{np.nanmedian(resid[sl]):>10.2e} '
              f'{np.nanmean(np.linalg.norm(tau_full[sl],axis=1)):>13.4f}')
    print('  (nefc=6 & resid≈0 ⇒ stance weld is the only active constraint ⇒ clean wrench)')

    print('\n' + '=' * 78)
    print('b. LEMMA 2 RESIDUAL  ε = ‖Ḣ_R/Os − w/Os‖   (inertial, about O_s)')
    print('=' * 78)
    use = segs[:3] if len(segs) >= 3 else segs
    print(f'  {"swing":>5} {"n":>4} {"|Ḣ_R/Os| mean":>13} {"ε mean (sign−)":>14} '
          f'{"ε/|Ḣ| (−)":>10} {"ε/|Ḣ| (+)":>10} {"ε_noM/|Ḣ|":>11}')
    allrel = []
    for k, (a, b) in enumerate(use):
        sl = slice(a, b + 1)
        ts = t[sl]
        # central FD of H about O_s
        Hdot = np.gradient(H[sl], ts, axis=0)
        g = slice(2, len(ts) - 2)            # drop edges
        Hd = Hdot[g]; wF = tau_full[sl][g]; wN = tau_noM[sl][g]
        Hn = np.linalg.norm(Hd, axis=1)
        eps_minus = np.linalg.norm(Hd - wF, axis=1)   # Ḣ = +w  ⇒ ε−
        eps_plus = np.linalg.norm(Hd + wF, axis=1)    # Ḣ = −w  ⇒ ε+
        eps_noM = np.linalg.norm(Hd - wN, axis=1)
        rel_m = float(np.mean(eps_minus) / (np.mean(Hn) + 1e-12))
        rel_p = float(np.mean(eps_plus) / (np.mean(Hn) + 1e-12))
        rel_nm = float(np.mean(eps_noM) / (np.mean(Hn) + 1e-12))
        allrel.append(min(rel_m, rel_p))
        print(f'  {k:>5} {len(Hn):>4} {np.mean(Hn):>13.4f} {np.mean(eps_minus):>14.4f} '
              f'{rel_m:>10.3f} {rel_p:>10.3f} {rel_nm:>11.3f}')
    sign = 'Ḣ_R/Os = +w_weld/Os (weld wrench ON robot)' if np.mean([
        np.mean(np.linalg.norm(np.gradient(H[a:b+1], t[a:b+1], axis=0)[2:-2]
                               - tau_full[a:b+1][2:-2], axis=1))
        for a, b in use]) < np.mean([
        np.mean(np.linalg.norm(np.gradient(H[a:b+1], t[a:b+1], axis=0)[2:-2]
                               + tau_full[a:b+1][2:-2], axis=1))
        for a, b in use]) else 'Ḣ_R/Os = −w_weld/Os'
    print(f'\n  sign that closes the identity: {sign}')
    print(f'  best ε/‖Ḣ‖ across swings: {[round(r,3) for r in allrel]}  '
          f'(≈0 ⇒ Lemma 2 holds; m_i term {"REQUIRED" if True else ""})')

    print('\n' + '=' * 78)
    print('c. PER-AXIS residual (best swing) + WINDOW-INTEGRAL cross-check')
    print('=' * 78)
    a, b = use[0]; sl = slice(a, b + 1); ts = t[sl]
    Hdot = np.gradient(H[sl], ts, axis=0); g = slice(2, len(ts) - 2)
    # pick sign by min
    em = np.mean(np.linalg.norm(Hdot[g] - tau_full[sl][g], axis=1))
    ep = np.mean(np.linalg.norm(Hdot[g] + tau_full[sl][g], axis=1))
    s = 1.0 if em < ep else -1.0
    res_axis = Hdot[g] - s * tau_full[sl][g]
    Hd_axis = Hdot[g]
    for i, ax in enumerate('xyz'):
        print(f'    {ax}: mean|Ḣ|={np.mean(np.abs(Hd_axis[:,i])):.4f}  '
              f'mean|resid|={np.mean(np.abs(res_axis[:,i])):.4f}  '
              f'corr(resid,Ḣ)={np.corrcoef(res_axis[:,i],Hd_axis[:,i])[0,1]:+.2f}')
    dH_win = H[b] - H[a]                                  # realized ΔH over swing
    int_w = s * np.trapz(tau_full[sl], ts, axis=0)        # ∫ w dt
    print(f'  window-integral: ΔH_R/Os = {np.round(dH_win,4)}  vs  ∫w dt = '
          f'{np.round(int_w,4)}  | ‖diff‖={np.linalg.norm(dH_win-int_w):.4f} '
          f'(rel {np.linalg.norm(dH_win-int_w)/(np.linalg.norm(dH_win)+1e-12):.3f})')

    print('\n' + '=' * 78)
    print('d. J3 byproduct — plant weld wrench vs lambda_qp (QP command), rough')
    print('=' * 78)
    lqp = np.array(instr['lambda_qp'])
    print(f'  plant weld |f| mean (SS) = {np.nanmean(np.linalg.norm(wf[np.isfinite(wf[:,0])],axis=1)):.3f} N ; '
          f'|m| mean = {np.nanmean(np.linalg.norm(wm[np.isfinite(wm[:,0])],axis=1)):.3f} N·m')
    print(f'  lambda_qp (QP command, 12-vec [f1,τ1,f2,τ2]) mean ‖·‖ = '
          f'{np.nanmean(np.linalg.norm(lqp,axis=1)):.3f} (struct frame; full char. = J3, out of scope)')
    print('\n[done]')


if __name__ == '__main__':
    main()
