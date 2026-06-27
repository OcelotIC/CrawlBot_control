#!/usr/bin/env python3
"""Phase 2.1 two-task weight-ratio sweep analysis: cursor-vs-wall + two-regime.

Arrival measured vs the GEOMETRIC torso target p_t1 (FK torso @ q_end). Per-task
closed-loop residual proxies (momentum=CoM track, torso-pose=torso-vs-quintic,
EE=dock dist) distinguish a tunable cursor from a structural wall. h_w in ∞-norm;
τ_w-sat at 100 Hz. metrics.py untouched.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from crawlbot.core.robot_interface import RobotInterface

ROOT = os.path.join(os.path.dirname(__file__), '..')
RES = os.path.join(ROOT, 'results')
OUT = os.path.join(RES, 'phase2_1_twotask_report')
TAU_W_MAX = 5.0

# (label, dir, mom_weight, torso_weight)
RUNS = [
    ('baseline', 'phase2_1_baseline_1step', None, None),
    ('TMOM-only', 'phase2_1_vA_amom5000', 5000, None),   # Phase-2.1 v1 témoin
    ('M-dom 30k:0.5k', 'phase2_1_tt_Mdom', 30000, 500),
    ('bal 5k:5k', 'phase2_1_tt_bal', 5000, 5000),
    ('TP-dom 0.5k:30k', 'phase2_1_tt_TPdom', 500, 30000),
]

_robot = RobotInterface(os.path.join(ROOT, 'models', 'VISPA_crawling_fixed.urdf'),
                        gravity='zero')


def geom_p_t1(d):
    qe = json.load(open(os.path.join(RES, d, 'step_q_end.json')))[0]
    q_end = np.array(qe['q_end'], float)
    return _robot.update(q_end, np.zeros(_robot.model.nv)).oMf_torso.translation.copy()


def load(d):
    p = os.path.join(RES, d, 'sim_log.json')
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def metrics(log, p_t1):
    t = np.array(log['t']); ph = np.array(log['phase']); ss = ph == 'SS'
    pt = np.array(log['p_torso']); ptr = np.array(log['p_torso_ref'])
    r = np.array(log['r_com']); rr = np.array(log['r_com_ref'])
    de = log.get('dock_events', []); ab = log.get('aborted_steps', [])
    m = {}
    m['docked'] = bool(de)
    m['d_final_mm'] = (de[-1].get('d_mm', de[-1].get('d', 0) * 1000) if de
                       else (ab[-1].get('d_mm', float('nan')) if ab else float('nan')))
    # ARRIVAL vs geometric p_t1
    m['arrival_mm'] = float(np.linalg.norm(pt[ss][-1] - p_t1) * 1000)
    # per-task closed-loop residual proxies (SS)
    m['mom_com_track_mm'] = float(np.max(np.linalg.norm((r - rr)[ss], axis=1)) * 1000)
    m['torso_pos_track_mm'] = float(np.max(np.linalg.norm((pt - ptr)[ss], axis=1)) * 1000)
    m['torso_ori_track_deg'] = float(np.max(np.array(log['e_torso_ori'])[ss]))
    dg = np.array(log['d_grip_swing'])
    m['ee_dock_min_mm'] = float(dg.min() * 1000)
    # h_w ∞-norm (100Hz) + τ_w-sat @100Hz
    hw = np.array(log.get('hw_ss_hifreq') or log['hw'])
    m['hw_inf'] = float(np.max(np.abs(hw)))
    tw = np.abs(np.array(log.get('tau_w_ss_hifreq') or log['tau_w']))
    m['tau_w_sat_100hz_pct'] = float(100 * np.mean((tw >= TAU_W_MAX - 1e-6).any(1))) if len(tw) else float('nan')
    # envelope: realized Ḣ_s (L_dot) vs planned (d L_com_ref/dt); binding fraction
    Ld = np.array(log['L_dot']); Lcr = np.array(log['L_com_ref'])
    dt = np.median(np.diff(t)); Ld_plan = np.gradient(Lcr, dt, axis=0)
    m['Hdot_real_pk'] = float(np.max(np.abs(Ld[ss])))
    m['Hdot_plan_pk'] = float(np.max(np.abs(Ld_plan[ss])))
    m['envelope_binding_pct'] = float(100 * np.mean(
        (np.abs(Ld_plan[ss]) >= 0.9 * TAU_W_MAX).any(1)))
    tau = np.array(log['tau']); m['tau_peak_Nm'] = float(np.max(np.abs(tau)))
    qpt = np.array(log['qp_time_ms'])[ss]
    m['qp_p50'] = float(np.percentile(qpt, 50)); m['qp_p99'] = float(np.percentile(qpt, 99))
    return m


def main():
    os.makedirs(OUT, exist_ok=True)
    p_t1 = geom_p_t1('phase2_1_baseline_1step')
    print(f'geometric p_t1 (FK torso @ q_end) = {np.round(p_t1,4)}')
    logs = {lab: load(d) for lab, d, _, _ in RUNS}
    M = {lab: metrics(logs[lab], p_t1) for lab, d, _, _ in RUNS if logs[lab]}

    cols = ['docked', 'd_final_mm', 'arrival_mm', 'mom_com_track_mm',
            'torso_pos_track_mm', 'torso_ori_track_deg', 'ee_dock_min_mm',
            'hw_inf', 'tau_w_sat_100hz_pct', 'Hdot_real_pk', 'Hdot_plan_pk',
            'envelope_binding_pct', 'tau_peak_Nm', 'qp_p50', 'qp_p99']
    labs = [l for l, d, a, b in RUNS if l in M]
    lines = ['| metric | ' + ' | '.join(labs) + ' |', '|---|' + '---|' * len(labs)]
    for k in cols:
        row = [k]
        for l in labs:
            v = M[l][k]
            row.append(str(v) if isinstance(v, bool) else
                       (f'{v:.2f}' if isinstance(v, float) else str(v)))
        lines.append('| ' + ' | '.join(row) + ' |')
    table = '\n'.join(lines)
    print('\n' + table)
    open(os.path.join(OUT, 'twotask_metrics.md'), 'w').write(table + '\n')

    # ---- cursor vs wall ----
    sweep = [(b / a if a else 0, l, M[l]) for l, d, a, b in RUNS
             if l in M and a and b]   # ratio = torso/mom
    sweep.sort()
    print('\n## CURSOR vs WALL (sweep, ratio = torso_pose/momentum weight)')
    print(f'{"torso/mom":>10} {"point":>16} {"arrival_mm":>11} {"CoM_trk_mm":>11} '
          f'{"EE_dock_mm":>11} {"docked":>7}')
    for ratio, l, mm in sweep:
        print(f'{ratio:>10.3f} {l:>16} {mm["arrival_mm"]:>11.2f} '
              f'{mm["mom_com_track_mm"]:>11.1f} {mm["ee_dock_min_mm"]:>11.3f} '
              f'{str(mm["docked"]):>7}')
    arrivals = [mm['arrival_mm'] for _, _, mm in sweep]
    verdict = ('CURSOR (arrival falls as torso-pose weight rises)'
               if len(arrivals) >= 2 and arrivals[-1] < 0.6 * arrivals[0]
               else 'WALL (arrival pinned; check residual transfer)')
    print(f'\nVERDICT: {verdict}')
    open(os.path.join(OUT, 'cursor_vs_wall.txt'), 'w').write(
        table + f'\n\nVERDICT: {verdict}\n')

    # ---- sweep summary plot ----
    if sweep:
        rr = [s[0] for s in sweep]
        fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
        ax1.plot(rr, [m['arrival_mm'] for _, _, m in sweep], 'o-b', label='torso arrival vs p_t1 [mm]')
        ax1.plot(rr, [m['torso_pos_track_mm'] for _, _, m in sweep], 's-c', label='torso-pose track [mm]')
        ax1.set_xscale('log'); ax1.set_xlabel('weight ratio  torso_pose / momentum')
        ax1.set_ylabel('torso error [mm]', color='b')
        ax2 = ax1.twinx()
        ax2.plot(rr, [m['mom_com_track_mm'] for _, _, m in sweep], '^--r', label='CoM(mom) track [mm]')
        ax2.plot(rr, [m['ee_dock_min_mm'] for _, _, m in sweep], 'x--m', label='EE dock min [mm]')
        ax2.set_ylabel('momentum/EE error [mm]', color='r')
        ax1.axhline(125.5, color='0.7', ls=':', lw=0.8)
        ax1.set_title('Phase 2.1 two-task — cursor vs wall (arrival & per-task vs ratio)')
        h1, _ = ax1.get_legend_handles_labels(); h2, _ = ax2.get_legend_handles_labels()
        ax1.legend(handles=h1 + h2, fontsize=7, loc='best')
        fig.tight_layout(); fig.savefig(os.path.join(OUT, 'cursor_vs_wall.png'), dpi=120)
        plt.close(fig)

    # ---- per-run torso realized-vs-quintic (balanced) ----
    for lab in ('bal 5k:5k',):
        log = logs.get(lab)
        if not log:
            continue
        t = np.array(log['t']); ph = np.array(log['phase']); ss = ph == 'SS'
        pt = np.array(log['p_torso'])[ss]; ptr = np.array(log['p_torso_ref'])[ss]
        tb = np.array(logs['baseline']['t']); phb = np.array(logs['baseline']['phase']) == 'SS'
        ptb = np.array(logs['baseline']['p_torso'])[phb]
        fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        for j, c in enumerate('xyz'):
            ax[j].plot(t[ss], ptr[:, j] * 1000, 'r:', lw=1.3, label='torso ref (raw quintic)')
            ax[j].plot(t[ss], pt[:, j] * 1000, 'b-', lw=1.2, label='two-task realized')
            ax[j].plot(tb[phb], ptb[:, j] * 1000, color='0.6', ls='--', lw=1.1, label='baseline realized')
            ax[j].axhline(p_t1[j] * 1000, color='g', ls='-.', lw=1.0, label='geometric p_t1')
            ax[j].set_ylabel(f'torso {c} [mm]')
            if j == 0:
                ax[j].legend(fontsize=7); ax[j].set_title('two-task (balanced) torso tracking vs raw quintic + geometric target')
        ax[2].set_xlabel('t [s]')
        fig.tight_layout(); fig.savefig(os.path.join(OUT, 'torso_track_balanced.png'), dpi=120)
        plt.close(fig)
    print(f'\nplots+tables -> {OUT}/')


if __name__ == '__main__':
    main()
