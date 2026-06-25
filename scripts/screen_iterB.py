#!/usr/bin/env python3
"""Iteration-B single-step screen analysis (legacy_pid_numerical).
Tabulates per grid point: C2 proxy (torso pos peak SS), C2b (ori RMS), C5 proxy
(h_w peak ∞-norm), C4 proxy (θ_s peak), docking, torso arrival. Screen NARROWS
to 1-2 candidates for the full 5-step gate (the screen does not decide).
"""
import json
import os

import numpy as np
from crawlbot.core.robot_interface import RobotInterface

ROOT = os.path.join(os.path.dirname(__file__), '..')
RES = os.path.join(ROOT, 'results')
# (weight, Kp) grid -> dir
GRID = [(16000, 3), (20000, 3), (24000, 3), (16000, 4), (20000, 4), (24000, 4)]
CUR = (20000, 3)   # the current Phase-3 working point
C5_LIMIT = 4.5     # corrected C5 (≤90% of ±5 budget)

_robot = RobotInterface(os.path.join(ROOT, 'models', 'VISPA_crawling_fixed.urdf'),
                        gravity='zero')
_zv = np.zeros(_robot.model.nv)


def fk_torso(q):
    return _robot.update(np.array(q, float), _zv).oMf_torso.translation.copy()


def jget(m, *ks):
    for k in ks:
        m = m[k]
    return m


def screen_one(W, KP):
    d = f'p3b_w{W}_kp{KP}'
    p = os.path.join(RES, d)
    if not os.path.exists(os.path.join(p, 'sim_log.json')):
        return None
    sl = json.load(open(os.path.join(p, 'sim_log.json')))
    m = json.load(open(os.path.join(p, 'postproc_metrics.json')))
    hw = np.array(sl['hw'])
    de = m.get('d_ee_at_docks_mm', [])
    qd = json.load(open(os.path.join(p, 'step_q_end.json')))
    ph = np.array(sl['phase']); ss = ph == 'SS'
    pt = np.array(sl['p_torso'])
    p_t1 = fk_torso(qd[0]['q_end']); p_t0 = fk_torso(qd[0]['q_start'])
    travel = np.linalg.norm(p_t1 - p_t0) * 1000
    arrival = np.linalg.norm(pt[ss][-1] - p_t1) * 1000 if ss.any() else float('nan')
    return dict(
        W=W, KP=KP, dir=d,
        c2_pos_peak_mm=jget(m, 'e_pos_torso_norm_m_SS_tracking', 'peak_SS') * 1000,
        c2_ori_rms=jget(m, 'e_ori_torso_deg_SS_tracking', 'rms_SS'),
        c5_hw_peak=float(np.max(np.abs(hw))),
        c4_theta_peak=jget(m, 'theta_s_deg', 'peak_full_run'),
        dock_mm=(de[0].get('d_mm_sim_log') if de else float('nan')),
        docked=bool(de),
        arrival_mm=arrival, arrival_pct=100 * arrival / travel if travel else float('nan'),
    )


def main():
    rows = [r for r in (screen_one(W, KP) for W, KP in GRID) if r]
    print('## Iteration-B single-step screen (step 0, legacy_pid_numerical, K_omega=50)')
    print('## proxies: C2=torso pos peak (lower=tighter), C5=h_w peak (≤4.5 budget), C4=θ_s peak')
    print(f'{"tp_weight":>9} {"Kp":>3} {"C2_pospk_mm":>11} {"C2_oriRMS°":>10} '
          f'{"C5_hw_pk":>8} {"C4_θs_pk°":>9} {"dock_mm":>8} {"arrival%":>9}')
    for r in sorted(rows, key=lambda x: (x['KP'], x['W'])):
        cur = '  <- current' if (r['W'], r['KP']) == CUR else ''
        print(f'{r["W"]:>9} {r["KP"]:>3} {r["c2_pos_peak_mm"]:>11.2f} {r["c2_ori_rms"]:>10.3f} '
              f'{r["c5_hw_peak"]:>8.2f} {r["c4_theta_peak"]:>9.3f} {r["dock_mm"]:>8.2f} '
              f'{r["arrival_pct"]:>9.2f}{cur}')

    cur_row = next((r for r in rows if (r['W'], r['KP']) == CUR), None)
    print('\n## screen reading (relative — step 0 is a MILD binding step; 5-step decides)')
    if cur_row:
        print(f'  current {CUR}: C2 pos peak {cur_row["c2_pos_peak_mm"]:.2f}mm, '
              f'h_w {cur_row["c5_hw_peak"]:.2f}, arrival {cur_row["arrival_pct"]:.1f}%')
    # candidate logic: tighten C2 (lower pos peak) while h_w-ranking stays low + docked.
    docked = [r for r in rows if r['docked']]
    by_c2 = sorted(docked, key=lambda r: r['c2_pos_peak_mm'])
    print('  tightest-C2 (docked), ranked:')
    for r in by_c2:
        print(f'    ({r["W"]},Kp{r["KP"]}): C2={r["c2_pos_peak_mm"]:.2f}mm  h_w={r["c5_hw_peak"]:.2f}  '
              f'ΔC2_vs_cur={r["c2_pos_peak_mm"]-cur_row["c2_pos_peak_mm"]:+.2f}  '
              f'ΔhW_vs_cur={r["c5_hw_peak"]-cur_row["c5_hw_peak"]:+.2f}')
    json.dump(rows, open(os.path.join(RES, 'iterB_screen.json'), 'w'), indent=2)
    print(f'\n  -> {RES}/iterB_screen.json')


if __name__ == '__main__':
    main()
