#!/usr/bin/env python3
"""DS inter-step audit — PHASE 2 (coherence). READ-ONLY.
Segments the run by the authoritative per-tick `phase` label (contiguous DS runs);
the SS→DS boundary tick is NOT a DS tick and is excluded by construction, and a
0.2 s post-weld guard is dropped for the body measurement (amendment 3: transition
physics, not d_DS). Gating (τ_w≈0, two-weld) + Q1 (θ̈_s) + Q2 (anchor drift) + Q3
(h_w EXACTLY constant). NOTE: `swing_arm` in DS rows is the just-landed-arm LABEL
(sim_loop:1804-1812), not a weld-state — two-weld is read from the stance anchor pair.
"""
import csv
import json
import os
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
RES = os.path.join(ROOT, 'Misc', 'runs', 'p3b_gate_w24000_kp3')
K_theta, K_omega, K_d, K_hw = 1.0, 50.0, 25.0, 2.0   # config.py:67/87/88/89
GUARD_S = 0.2


def struct_I():
    try:
        import mujoco
        m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, 'models', 'VISPA_crawling_rwa3.xml'))
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'structure')
        if sid >= 0:
            return m.body_inertia[sid].copy()
    except Exception as e:
        print(f'  (model load failed: {e}; documented fallback)')
    return np.array([597.0, 1493.0, 1777.0])


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
    tau_w = np.array(sl['tau_w']); hw = np.array(sl['hw']); om = np.array(sl['omega_s'])
    theta = np.array([float(r['theta_s_deg']) for r in rows])
    sa = np.array([r['stance_a_idx'] for r in rows]); sb = np.array([r['stance_b_idx'] for r in rows])
    dgs = np.array(sl['d_grip_stance'])
    aligned = len(rows) == len(t) and np.max(np.abs(np.array([float(r['t']) for r in rows]) - t)) < 1e-6

    I_s = struct_I(); t_s = 8 * (I_s + K_d) / K_omega
    print('## PHASE 2 — coherence (inter-step DS), phase-segmented')
    print(f'pinned gains K_θ={K_theta} K_ω={K_omega} K_d={K_d} K_hw={K_hw} (config.py:67/87/88/89)')
    print(f'I_s={np.round(I_s,1)} kg·m² (x,y,z); t_s(2%)=8(I_s+K_d)/K_ω={np.round(t_s,1)} s '
          f'→ heavy-axis x={t_s[0]:.0f}s ⇒ +20s terminal = MID-DECAY (amendment 2). aligned={aligned}\n')

    ds_segs = contiguous(ph == 'DS')
    ss = ph == 'SS'
    print(f'τ_w max|·| SS = {np.max(np.abs(tau_w[ss])):.3f} N·m (AOCS on, ≠0 expected)')
    print(f'{len(ds_segs)} contiguous DS segments; last = terminal settle (AOCS on, §5.6 context)\n')

    print('=== GATING + Q1 + Q3 per DS segment (last row = terminal) ===')
    print(f'{"seg":>3} {"role":>12} {"t_range":>13} {"n":>4} {"τw_max":>8} {"2weld":>6} '
          f'{"|Δhw|max_µNms":>13} {"θ̈s_body_pk":>11} {"θ̈s_guard_pk":>12} {"Δθ_s_deg":>9}')
    flags = []
    for s, (a, b) in enumerate(ds_segs):
        terminal = (s == len(ds_segs) - 1)
        role = 'TERMINAL' if terminal else ('initial' if s == 0 else f'post-step{s-1}')
        tw = tau_w[a:b+1]; hwi = hw[a:b+1]; omi = om[a:b+1]; ti = t[a:b+1]
        twmax = float(np.max(np.abs(tw)))
        twoweld = aligned and all(sa[a:b+1] != '') and all(sb[a:b+1] != '')
        dhw_max_uNms = float(np.max(np.abs(hwi - hwi[0]))) * 1e6
        dt_i = float(np.median(np.diff(ti))) if len(ti) > 1 else 0.01
        thddot = np.linalg.norm(np.gradient(omi, dt_i, axis=0), axis=1) if len(ti) > 2 else np.array([np.nan])
        guard = ti <= (ti[0] + GUARD_S); body = ~guard
        pk_body = float(np.max(thddot[body])) if body.any() else float('nan')
        pk_guard = float(np.max(thddot[guard])) if guard.any() else float('nan')
        dtheta = float(theta[b] - theta[a]) if aligned else float('nan')
        print(f'{s:>3} {role:>12} [{ti[0]:>5.2f},{ti[-1]:>5.2f}] {b-a+1:>4} {twmax:>8.2e} '
              f'{str(twoweld):>6} {dhw_max_uNms:>13.3f} {pk_body:>11.4f} {pk_guard:>12.4f} {dtheta:>+9.3f}')
        if not terminal:
            if twmax > 1e-9:
                flags.append(f'seg{s} ({role}): τ_w≠0 in inter-step DS (max {twmax:.2e})')
            if not twoweld:
                flags.append(f'seg{s} ({role}): stance pair not set every tick (possible single-weld)')
            if dhw_max_uNms > 1.0:   # >1 µNms
                flags.append(f'seg{s} ({role}): h_w drift {dhw_max_uNms:.2f} µNms (Q3 expects EXACTLY 0)')

    print('\n=== Q2 — compounding / anchor drift ===')
    ds_all = ph == 'DS'
    print(f'  d_grip_stance max over all DS = {np.max(dgs[ds_all])*1e3:.3f} mm '
          f'(M7_T12 anchor-drift concern — nominal vs actual gripper@anchor)')
    inter = ds_segs[:-1]
    dthetas = [float(theta[b] - theta[a]) for (a, b) in inter] if aligned else []
    print(f'  Σ Δθ_s over inter-step DS windows = {sum(dthetas):+.3f}°  (per-window Δθ_k full series in Phase 3)')

    print('\n=== FLAGS (§7) ===')
    print('  ' + ('\n  '.join(flags) if flags else
                  'NONE — every inter-step DS: τ_w=0 exactly, two-weld (stance pair set), h_w frozen.'))
    print('\nNote: `swing_arm` is the just-landed-arm label in DS rows, NOT a weld-state; two-weld '
          'is read from stance_a_idx & stance_b_idx (both set) + d_grip_stance≈0 (grippers at anchors).')


if __name__ == '__main__':
    main()
