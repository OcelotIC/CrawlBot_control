#!/usr/bin/env python3
"""Dock-leak diagnostic — Part 1 analysis (confirm the mechanism).

Operates on the instrumented re-run results/dockleak_instrumented/ (per-tick
H_sys=subtree_angmom[0], n_weld, and a per-dock P0/P1/P2 attribution probe),
against the FROZEN canonical results/p3b_gate_w24000_kp3/.

Sections:
  0. Non-perturbation proof — instrumented vs canonical shared log fields.
  a. Localization — (coarse) H_sys conserved in SS, injected in DS; (fine)
     per-dock attribution H0->H1 (weld activation) / H1->H2 (impact velocity
     projection) / H2->H3 (gap-closing stabilization, H3=first DS tick).
  b. Gap regression — per-engagement |ΔH| vs dock gap (5 engagements).
  c. Direction — ΔH vs geometric candidates (gap vector, lever×gap couple).

READ-ONLY on the logs (no stepping, no re-run here). Metric is subtree_angmom[0],
the validated J1 A'.1 ground truth.
"""
import json
import os
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
INSTR = os.path.join(ROOT, 'results', 'dockleak_instrumented')
CANON = os.path.join(ROOT, 'results', 'p3b_gate_w24000_kp3')

# Fields that legitimately differ: the new instrument fields, and wall-clock
# timing (non-deterministic). Everything else must match bit-for-bit.
SKIP = {'H_sys', 'n_weld', 'dock_probe', 'qp_time_ms', 'nmpc_time_ms',
        't_ss_hifreq', 'tau_w_ss_hifreq', 'hw_ss_hifreq', 'environment'}


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v * 0.0


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 1e-12 and nb > 1e-12 else float('nan')


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


def section0_nonperturbation(instr, canon):
    print('=' * 78)
    print('0. NON-PERTURBATION PROOF  (instrumented vs frozen canonical)')
    print('=' * 78)
    worst = 0.0
    diffs = []
    nshared = 0
    for k in set(instr) & set(canon):
        if k in SKIP:
            continue
        try:
            xa = np.asarray(instr[k], float)
            xb = np.asarray(canon[k], float)
        except (ValueError, TypeError):
            continue
        if xa.shape != xb.shape:
            diffs.append((k, f'SHAPE {xa.shape}/{xb.shape}'))
            continue
        nshared += 1
        if xa.size:
            d = np.abs(xa - xb)
            d = d[np.isfinite(d)]
            m = float(d.max()) if d.size else 0.0
            worst = max(worst, m)
            if m > 0:
                diffs.append((k, round(m, 12)))
    verdict = 'BIT-IDENTICAL' if worst == 0 else 'DIFFERS'
    print(f'  shared numeric fields compared: {nshared}')
    print(f'  worst |Δ| over all shared fields: {worst:.3e}  ->  {verdict}')
    if diffs:
        print(f'  nonzero/diff fields: {diffs}')
    else:
        print('  nonzero/diff fields: NONE  (trajectory identical; instrument is read-only)')
    return worst == 0


def section_a_localization(sl):
    H = np.array(sl['H_sys']); nw = np.array(sl['n_weld'])
    ph = np.array(sl['phase']); t = np.array(sl['t'])
    n = len(H)
    print('\n' + '=' * 78)
    print('a. LOCALIZATION')
    print('=' * 78)

    # ---- coarse: per n_weld-segment ΔH (SS=1 weld, DS=2 welds) ----
    print('\n  (a.1) coarse — ΔH per contiguous n_weld segment:')
    print(f'  {"seg":>4} {"n_weld":>6} {"phase":>5} {"ticks":>9} {"t[s]":>12} '
          f'{"|H_start|":>9} {"|H_end|":>9} {"|ΔH_vec|":>9}')
    ss_inj = []; ds_inj = []
    for val in (1, 2):
        for (a, b) in contiguous(nw == val):
            dH = H[b] - H[a]
            tag = 'SS' if val == 1 else 'DS'
            (ss_inj if val == 1 else ds_inj).append(np.linalg.norm(dH))
    # ordered by time for readability
    segs_all = []
    i = 0
    while i < n:
        j = i
        while j < n and nw[j] == nw[i]:
            j += 1
        segs_all.append((i, j - 1))
        i = j
    for si, (a, b) in enumerate(segs_all):
        dH = H[b] - H[a]
        tag = 'SS' if nw[a] == 1 else 'DS'
        print(f'  {si:>4} {int(nw[a]):>6} {tag:>5} {f"{a}-{b}":>9} '
              f'{f"{t[a]:.2f}-{t[b]:.2f}":>12} {np.linalg.norm(H[a]):>9.4f} '
              f'{np.linalg.norm(H[b]):>9.4f} {np.linalg.norm(dH):>9.4f}')
    print(f'\n  SS segments (n_weld=1): max |ΔH_vec| = {max(ss_inj):.4f}  '
          f'(mean {np.mean(ss_inj):.4f}) — should be ≈0 (conserved)')
    print(f'  DS segments (n_weld=2): max |ΔH_vec| = {max(ds_inj):.4f}  '
          f'(mean {np.mean(ds_inj):.4f}) — carries the injection')

    # ---- fine: single-tick jump at each 1->2 transition ----
    docks = [i for i in range(1, n) if nw[i] == 2 and nw[i - 1] == 1]
    print('\n  (a.2) fine — the jump is concentrated in ONE logged tick at the'
          ' n_weld 1→2 (dock) transition:')
    print(f'  {"dock tick":>9} {"t[s]":>6} {"|H@i-1|":>8} {"|H@i|":>8} '
          f'{"|ΔH 1tick|":>10} {"|ΔH rest-of-DS|":>15}')
    for i in docks:
        # find end of this DS window
        j = i
        while j < n and nw[j] == 2:
            j += 1
        j -= 1
        dH_tick = np.linalg.norm(H[i] - H[i - 1])
        dH_rest = np.linalg.norm(H[j] - H[i])
        print(f'  {i:>9} {t[i]:>6.2f} {np.linalg.norm(H[i-1]):>8.4f} '
              f'{np.linalg.norm(H[i]):>8.4f} {dH_tick:>10.4f} {dH_rest:>15.4f}')
    return docks


def section_a_fine_probe(sl, docks):
    """Per-dock attribution from the P0/P1/P2 probe + H3 (first DS tick)."""
    H = np.array(sl['H_sys']); nw = np.array(sl['n_weld'])
    probes = sl.get('dock_probe', [])
    print('\n  (a.3) fine — per-dock sub-event attribution '
          '(H0=pre-activation, H1=weld active, H2=post-impact-projection, '
          'H3=first DS tick):')
    print(f'  {"step":>4} {"arm":>3} {"anch":>4} {"gap[mm]":>7} {"v_app":>7} '
          f'{"|H0→H1|":>8} {"|H1→H2|":>8} {"|H2→H3|":>8} {"|H0→H3|":>8} '
          f'{"dominant":>10}')
    rows = []
    for p, i in zip(probes, docks):
        H0 = np.array(p['H0']); H1 = np.array(p['H1']); H2 = np.array(p['H2'])
        H3 = H[i]  # first DS tick after this dock
        d01 = np.linalg.norm(H1 - H0)
        d12 = np.linalg.norm(H2 - H1)
        d23 = np.linalg.norm(H3 - H2)
        d03 = np.linalg.norm(H3 - H0)
        parts = {'weld-act': d01, 'impact-proj': d12, 'gap-stab': d23}
        dom = max(parts, key=parts.get)
        print(f'  {p["step"]:>4} {p["arm"]:>3} {p["anchor"]:>4} '
              f'{p["gap"]*1000:>7.3f} {p["approach_speed"]:>7.4f} '
              f'{d01:>8.4f} {d12:>8.4f} {d23:>8.4f} {d03:>8.4f} {dom:>10}')
        rows.append(dict(step=p['step'], H0=H0, H1=H1, H2=H2, H3=H3,
                         d01=d01, d12=d12, d23=d23, d03=d03,
                         gap=p['gap'], gap_vec=np.array(p['gap_vec']),
                         grip=np.array(p['grip_pos']), com=np.array(p['com']),
                         v=p['approach_speed']))
    tot01 = sum(r['d01'] for r in rows)
    tot12 = sum(r['d12'] for r in rows)
    tot23 = sum(r['d23'] for r in rows)
    print(f'\n  Σ|H0→H1| (weld activation)   = {tot01:.4f}  '
          '(constraint added, no Δqvel — expect ≈0)')
    print(f'  Σ|H1→H2| (impact projection) = {tot12:.4f}  '
          '(discrete arm-joint velocity reset)')
    print(f'  Σ|H2→H3| (gap stabilization) = {tot23:.4f}  '
          '(first DS step with the weld closing a finite gap)')
    return rows


def section_b_gap(rows):
    print('\n' + '=' * 78)
    print('b. GAP REGRESSION  (per-engagement injection vs dock gap)')
    print('=' * 78)
    gaps = np.array([r['gap'] * 1000 for r in rows])          # mm
    # injection vector for the engagement = H3 - H0 (whole single-tick jump)
    dH = np.array([np.linalg.norm(r['H3'] - r['H0']) for r in rows])
    dH12 = np.array([r['d12'] for r in rows])                 # impact part
    dH23 = np.array([r['d23'] for r in rows])                 # gap-stab part
    vs = np.array([r['v'] for r in rows])
    print(f'  {"step":>4} {"gap[mm]":>8} {"v_app[m/s]":>10} {"|ΔH|":>8} '
          f'{"|ΔH_impact|":>11} {"|ΔH_gapstab|":>12}')
    for r, g, d, di, dg, v in zip(rows, gaps, dH, dH12, dH23, vs):
        print(f'  {r["step"]:>4} {g:>8.3f} {v:>10.4f} {d:>8.4f} '
              f'{di:>11.4f} {dg:>12.4f}')
    # correlations (5 points — descriptive, Part 2 forces the gap deliberately)
    def safe_corr(x, y):
        return float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 1e-12 and np.std(y) > 1e-12 else float('nan')
    print(f'\n  corr(|ΔH|, gap)        = {safe_corr(dH, gaps):+.3f}')
    print(f'  corr(|ΔH|, v_approach) = {safe_corr(dH, vs):+.3f}')
    print(f'  corr(|ΔH_impact|, gap) = {safe_corr(dH12, gaps):+.3f}   '
          f'corr(|ΔH_impact|, v) = {safe_corr(dH12, vs):+.3f}')
    print(f'  corr(|ΔH_gapstab|,gap) = {safe_corr(dH23, gaps):+.3f}   '
          f'corr(|ΔH_gapstab|, v) = {safe_corr(dH23, vs):+.3f}')
    print('  NOTE: 5 engagements only — descriptive. The deliberate gap sweep'
          ' (forced gaps, zero-gap probe) is Part 2.')


def section_c_direction(rows):
    print('\n' + '=' * 78)
    print('c. DIRECTION  (ΔH vs geometric candidates)')
    print('=' * 78)
    print('  per engagement: ΔH = H3−H0 ; gap_vec = grip−anchor (world) ;')
    print('  lever = grip − r_com_sys ; couple = lever × gap_vec\n')
    print(f'  {"step":>4} {"ΔH_dir (unit)":>26} {"cos(ΔH,gap)":>12} '
          f'{"cos(ΔH,lever×gap)":>18} {"cos(ΔH,lever)":>14}')
    for r in rows:
        dH = r['H3'] - r['H0']
        gap = r['gap_vec']
        lever = r['grip'] - r['com']
        couple = np.cross(lever, gap)
        u = unit(dH)
        print(f'  {r["step"]:>4} [{u[0]:+.2f},{u[1]:+.2f},{u[2]:+.2f}]'
              f'{"":>9} {cos(dH, gap):>12.3f} {cos(dH, couple):>18.3f} '
              f'{cos(dH, lever):>14.3f}')
    print('\n  (interpretation printed in the report — the dominant sub-event'
          ' from (a.3) determines which force model "g×F" should reference.)')


def main():
    instr = json.load(open(os.path.join(INSTR, 'sim_log.json')))
    canon = json.load(open(os.path.join(CANON, 'sim_log.json')))
    print('Dock-leak Part 1 analysis')
    print(f'  instrumented: {INSTR}')
    print(f'  canonical   : {CANON}')
    print(f'  |H_sys[final]| = {np.linalg.norm(np.array(instr["H_sys"][-1])):.4f}'
          ' N·m·s  (A\'.1 ground truth: 0.2030)\n')
    section0_nonperturbation(instr, canon)
    docks = section_a_localization(instr)
    rows = section_a_fine_probe(instr, docks)
    section_b_gap(rows)
    section_c_direction(rows)
    print('\n[done]')


if __name__ == '__main__':
    main()
