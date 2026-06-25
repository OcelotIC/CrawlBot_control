#!/usr/bin/env python3
"""Phase 3 GATE analysis — six criteria vs the 5bca42c canonical baseline,
two-regime torso-arrival read, and the 5-step 100 Hz QP-rate τ_w-sat (B15).

Reads phase3_wp + baseline postproc_metrics.json (same schema) + sim_log.json.
Criterion 6 (bit-identical OFF + test) is produced by run_phase3_off.sh and read
from results/phase3_off/OFF_RESULT.txt if present. Read-only; no re-tuning.
"""
import csv
import json
import os
import sys

import numpy as np
from crawlbot.core.robot_interface import RobotInterface

ROOT = os.path.join(os.path.dirname(__file__), '..')
RES = os.path.join(ROOT, 'results')
BASE = 'ssmom_phase1_baseline_main_dcda974'
WP = 'phase3_wp'
TAU_W_MAX = 5.0
HDOT_MAX = 5.0
GATE_MM = 5.0
OUT = os.path.join(RES, 'phase3_wp')

_robot = RobotInterface(os.path.join(ROOT, 'models', 'VISPA_crawling_fixed.urdf'),
                        gravity='zero')
_zv = np.zeros(_robot.model.nv)


def fk_torso(q):
    return _robot.update(np.array(q, float), _zv).oMf_torso.translation.copy()


def jget(m, *ks):
    for k in ks:
        m = m[k]
    return m


def load_csv(d):
    return list(csv.DictReader(open(os.path.join(RES, d, 'postproc_F3F4.csv'))))


def main():
    bm = json.load(open(os.path.join(RES, BASE, 'postproc_metrics.json')))
    pm = json.load(open(os.path.join(RES, WP, 'postproc_metrics.json')))
    pl = json.load(open(os.path.join(RES, WP, 'sim_log.json')))
    bl = json.load(open(os.path.join(RES, BASE, 'sim_log.json')))
    rows = load_csv(WP)
    ss_rows = [r for r in rows if r['mask_ss'] == '1']
    steps = sorted(set(int(r['step_idx']) for r in ss_rows))

    R = {}   # results

    # ---- Criterion 1: docking ----
    base_d = [e['d_mm_sim_log'] for e in bm['d_ee_at_docks_mm']]
    wp_d = [e['d_mm_sim_log'] for e in pm['d_ee_at_docks_mm']]
    base_marg = [GATE_MM - d for d in base_d]
    wp_marg = [GATE_MM - d for d in wp_d]
    docks_ok = len(wp_d) == 5 and all(d <= GATE_MM + 1e-9 for d in wp_d)
    per_step_marg_ok = [wm >= bmg - 1e-9 for wm, bmg in zip(wp_marg, base_marg)]
    R['c1'] = dict(base_d=base_d, wp_d=wp_d, base_marg=base_marg, wp_marg=wp_marg,
                   docks_ok=docks_ok, per_step_marg_ok=per_step_marg_ok,
                   worst_marg_wp=min(wp_marg), worst_marg_base=min(base_marg),
                   verdict=docks_ok and (min(wp_marg) >= min(base_marg) - 1e-9))

    # ---- Criterion 2: torso TRACKING (vs the reference being tracked) ----
    b_ori = jget(bm, 'e_ori_torso_deg_SS_tracking', 'rms_SS')
    w_ori = jget(pm, 'e_ori_torso_deg_SS_tracking', 'rms_SS')
    b_pos = jget(bm, 'e_pos_torso_norm_m_SS_tracking', 'peak_SS') * 1000
    w_pos = jget(pm, 'e_pos_torso_norm_m_SS_tracking', 'peak_SS') * 1000
    R['c2'] = dict(base_ori_rms=b_ori, wp_ori_rms=w_ori, base_pos_peak_mm=b_pos,
                   wp_pos_peak_mm=w_pos, verdict=(w_ori <= b_ori) and (w_pos <= b_pos))

    # ---- Criterion 3: envelope (planned ≤5 hard; realized + B15 reported) ----
    hdot_ss = [jget(pm, 'Hdot_s_per_axis_peak_Nm_SS_only', a) for a in 'xyz']
    hdot_ss_base = [jget(bm, 'Hdot_s_per_axis_peak_Nm_SS_only', a) for a in 'xyz']
    # Envelope evidence: the postproc REALIZED ‖Ḣ_s‖∞ SS-peak (AOCS structure
    # momentum rate) ≤ 5 — z binds (=5.0), x/y below. The PLANNED ‖Ḣ_s‖∞ ≤ 5 at
    # the NMPC knots is the unchanged QP constraint (structural; same formulation
    # as the baseline whose realized SS-peak = 5.0/5.0/5.0). We report the realized
    # SS-peak as the envelope evidence; ≤5 ⇒ the constraint held in execution.
    t = np.array(pl['t'])
    dt = float(np.median(np.diff(t)))
    ph = np.array(pl['phase'])
    ss = ph == 'SS'
    realized_hdot_ss_peak = float(max(hdot_ss))
    # B15: 100 Hz τ_w-sat, aggregate + per-step
    t_hf = np.array(pl['t_ss_hifreq'])
    tw_hf = np.abs(np.array(pl['tau_w_ss_hifreq']))
    sat_hf = tw_hf.max(1) >= TAU_W_MAX - 1e-6
    b15_agg = float(100 * np.mean(sat_hf)) if len(tw_hf) else float('nan')
    # map each hifreq sample to a step via nearest SS F3F4-row time
    ss_t = np.array([float(r['t']) for r in ss_rows])
    ss_step = np.array([int(r['step_idx']) for r in ss_rows])
    b15_step = {}
    if len(ss_t):
        idx = np.clip(np.searchsorted(ss_t, t_hf), 0, len(ss_t) - 1)
        smap = ss_step[idx]
        for si in steps:
            sel = smap == si
            b15_step[si] = float(100 * np.mean(sat_hf[sel])) if sel.any() else 0.0
    R['c3'] = dict(hdot_ss_peak=hdot_ss, hdot_ss_peak_base=hdot_ss_base,
                   realized_hdot_ss_peak=realized_hdot_ss_peak,
                   envelope_ok=realized_hdot_ss_peak <= HDOT_MAX + 1e-2,
                   b15_agg=b15_agg, b15_step=b15_step,
                   tau_w_clip_frac_f3f4_SS=jget(pm, 'tau_w_at_clip_fraction_SS_only'),
                   tau_w_clip_frac_f3f4_SS_base=jget(bm, 'tau_w_at_clip_fraction_SS_only'))

    # ---- Criterion 4: attitude ----
    R['c4'] = dict(peak=jget(pm, 'theta_s_deg', 'peak_full_run'),
                   final=jget(pm, 'theta_s_deg', 'final'),
                   peak_base=jget(bm, 'theta_s_deg', 'peak_full_run'),
                   final_base=jget(bm, 'theta_s_deg', 'final'),
                   verdict=jget(pm, 'theta_s_deg', 'peak_full_run') <= 1.9
                   and jget(pm, 'theta_s_deg', 'final') <= 1.65)

    # ---- Criterion 5: stored momentum h_w peak ∞-norm ----
    hw_wp = float(np.max(np.abs(np.array(pl['hw']))))
    hw_base = float(np.max(np.abs(np.array(bl['hw']))))
    R['c5'] = dict(hw_wp=hw_wp, hw_base=hw_base,
                   verdict=hw_wp <= hw_base + 0.20)   # baseline + small margin

    # ---- Criterion 6: bit-identical OFF + test (from off run) ----
    off_path = os.path.join(RES, 'phase3_off', 'OFF_RESULT.txt')
    R['c6'] = open(off_path).read().strip() if os.path.exists(off_path) else 'PENDING'

    # ---- Two-regime torso arrival (characterisation, NOT gate) ----
    qd = json.load(open(os.path.join(RES, WP, 'step_q_end.json')))
    pt = np.array(pl['p_torso'])
    # align p_torso ticks to F3F4 rows by time
    regime = []
    for si in steps:
        sr = [r for r in ss_rows if int(r['step_idx']) == si]
        tw = np.array([[abs(float(r[f'tau_w_{a}'])) for a in 'xyz'] for r in sr])
        hd = np.array([[abs(float(r[f'Hdot_s_{a}'])) for a in 'xyz'] for r in sr])
        bind_frac = float(np.mean((tw.max(1) >= TAU_W_MAX - 1e-6)
                                  | (hd.max(1) >= 0.9 * HDOT_MAX)) * 100)
        t_end = float(sr[-1]['t'])
        i_end = int(np.argmin(np.abs(t - t_end)))
        qe = next(e for e in qd if e['step_idx'] == si)
        p_t1 = fk_torso(qe['q_end'])
        p_t0 = fk_torso(qe['q_start'])
        travel = float(np.linalg.norm(p_t1 - p_t0) * 1000)
        arrival = float(np.linalg.norm(pt[i_end] - p_t1) * 1000)
        regime.append(dict(step=si, bind_frac=bind_frac,
                           binding=bind_frac >= 20.0, travel_mm=travel,
                           arrival_mm=arrival,
                           arrival_pct=100 * arrival / travel if travel else float('nan'),
                           tau_w_sat_100hz=b15_step.get(si, 0.0)))
    R['two_regime'] = regime

    # ---------------- print + persist ----------------
    def pf(b):
        return 'PASS' if b else 'FAIL'
    print('================ PHASE 3 GATE ================')
    print(f"C1 docking : {pf(R['c1']['verdict'])}  wp_d={[round(x,2) for x in wp_d]} "
          f"base_d={[round(x,2) for x in base_d]}")
    print(f"           per-step margin≥base: {R['c1']['per_step_marg_ok']} ; "
          f"worst margin wp={R['c1']['worst_marg_wp']:.2f} base={R['c1']['worst_marg_base']:.2f}")
    print(f"C2 torso-track: {pf(R['c2']['verdict'])}  ori_rms wp={w_ori:.3f}≤base={b_ori:.3f}? "
          f"pos_peak wp={w_pos:.1f}≤base={b_pos:.1f}mm?")
    print(f"C3 envelope: {pf(R['c3']['envelope_ok'])}  realized‖Ḣ_s‖∞_SS={realized_hdot_ss_peak:.2f}≤5 "
          f"(per-axis {[round(x,2) for x in hdot_ss]}; planned constraint unchanged)")
    print(f"           B15 τ_w-sat@100Hz: agg={b15_agg:.2f}%  per-step={ {k: round(v,1) for k,v in b15_step.items()} }")
    print(f"C4 attitude: {pf(R['c4']['verdict'])}  peak={R['c4']['peak']:.2f}≤1.9 final={R['c4']['final']:.2f}≤1.65")
    print(f"C5 h_w∞    : {pf(R['c5']['verdict'])}  wp={hw_wp:.3f} base={hw_base:.3f}")
    print(f"C6 OFF/test: {R['c6']}")
    print('---- two-regime torso arrival ----')
    for r in regime:
        print(f"  step {r['step']}: {'BIND' if r['binding'] else 'free'}  "
              f"bind_frac={r['bind_frac']:.0f}%  τ_w-sat100={r['tau_w_sat_100hz']:.0f}%  "
              f"travel={r['travel_mm']:.1f}mm  arrival={r['arrival_mm']:.1f}mm "
              f"({r['arrival_pct']:.1f}%)")
    nb = [r for r in regime if not r['binding']]
    bd = [r for r in regime if r['binding']]
    if nb:
        print(f"  NON-binding mean arrival = {np.mean([r['arrival_pct'] for r in nb]):.1f}%")
    if bd:
        print(f"  BINDING     mean arrival = {np.mean([r['arrival_pct'] for r in bd]):.1f}%")
    json.dump(R, open(os.path.join(OUT, 'gate_phase3_results.json'), 'w'), indent=2)
    print(f'\nresults -> {OUT}/gate_phase3_results.json')


if __name__ == '__main__':
    main()
