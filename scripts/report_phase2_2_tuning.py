#!/usr/bin/env python3
"""Phase 2.2 SS two-task weight×gain tuning analysis (canonical single step).

Two levers explored: torso-pose WEIGHT (priority, alpha_torso_pose) and torso
GAINS Kp/Kd (aggressiveness, ss_Kp_torso/ss_Kd_torso). Question: is there a
(weight × gain) point reaching ~2-3% torso arrival WITHOUT pushing τ_w-sat to the
TP-dom level (~24%)? Arrival measured vs the GEOMETRIC torso target p_t1 (FK torso
@ q_end), as % of the 125.5 mm geometric travel. Metric defs reused verbatim from
report_phase2_1_twotask.py; smoothness/jerk + arrival% + 2-D trade-off added here.
No metrics.py / crawlbot edit.
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
OUT = os.path.join(RES, 'phase2_2_tuning_report')
TAU_W_MAX = 5.0
TRAVEL_MM = 125.5            # |p_t1 - p_t0|, geometric torso displacement over SS

# (label, dir, mom_weight, torso_pose_weight, Kp_torso)  — Kd scaled = 5*Kp/6
RUNS = [
    ('baseline',       'phase2_1_baseline_1step', None,  None,  None),
    ('bal 5k:5k',      'phase2_1_tt_bal',         5000,  5000,  6),
    ('5k:8k',          'p22_w8k_kp6',             5000,  8000,  6),
    ('5k:12k',         'p22_w12k_kp6',            5000,  12000, 6),
    ('5k:20k',         'p22_w20k_kp6',            5000,  20000, 6),
    ('5k:12k Kp4',     'p22_w12k_kp4',            5000,  12000, 4),
    ('5k:20k Kp4',     'p22_w20k_kp4',            5000,  20000, 4),
    ('5k:12k Kp3',     'p22_w12k_kp3',            5000,  12000, 3),
    ('5k:20k Kp3',     'p22_w20k_kp3',            5000,  20000, 3),
    ('TP-dom .5k:30k', 'phase2_1_tt_TPdom',       500,   30000, 6),
    ('TPdom Kp3',      'p22_TPdom_kp3',           500,   30000, 3),
]

_robot = RobotInterface(os.path.join(ROOT, 'models', 'VISPA_crawling_fixed.urdf'),
                        gravity='zero')


def geom_p_t1(d):
    qe = json.load(open(os.path.join(RES, d, 'step_q_end.json')))[0]
    q_end = np.array(qe['q_end'], float)
    return _robot.update(q_end, np.zeros(_robot.model.nv)).oMf_torso.translation.copy()


def load(d):
    p = os.path.join(RES, d, 'sim_log.json')
    return json.load(open(p)) if os.path.exists(p) else None


def metrics(log, p_t1):
    t = np.array(log['t']); ph = np.array(log['phase']); ss = ph == 'SS'
    pt = np.array(log['p_torso']); ptr = np.array(log['p_torso_ref'])
    r = np.array(log['r_com']); rr = np.array(log['r_com_ref'])
    de = log.get('dock_events', []); ab = log.get('aborted_steps', [])
    m = {}
    m['docked'] = bool(de)
    m['d_final_mm'] = (de[-1].get('d_mm', de[-1].get('d', 0) * 1000) if de
                       else (ab[-1].get('d_mm', float('nan')) if ab else float('nan')))
    # ARRIVAL vs geometric p_t1 (THE objective), absolute + % of travel
    m['arrival_mm'] = float(np.linalg.norm(pt[ss][-1] - p_t1) * 1000)
    m['arrival_pct'] = 100 * m['arrival_mm'] / TRAVEL_MM
    # per-task closed-loop residual proxies (SS)
    m['mom_com_track_mm'] = float(np.max(np.linalg.norm((r - rr)[ss], axis=1)) * 1000)
    m['torso_pos_track_mm'] = float(np.max(np.linalg.norm((pt - ptr)[ss], axis=1)) * 1000)
    m['torso_ori_track_deg'] = float(np.max(np.array(log['e_torso_ori'])[ss]))
    dg = np.array(log['d_grip_swing'])
    m['ee_dock_min_mm'] = float(dg.min() * 1000)
    # h_w ∞-norm (100Hz) + τ_w-sat @100Hz (THE cost) + τ_w RMS
    hw = np.array(log.get('hw_ss_hifreq') or log['hw'])
    m['hw_inf'] = float(np.max(np.abs(hw)))
    tw = np.abs(np.array(log.get('tau_w_ss_hifreq') or log['tau_w']))
    m['tau_w_sat_pct'] = float(100 * np.mean((tw >= TAU_W_MAX - 1e-6).any(1))) if len(tw) else float('nan')
    m['tau_w_rms'] = float(np.sqrt(np.mean(tw ** 2))) if len(tw) else float('nan')
    # SMOOTHNESS / aggressiveness on the torso linear trajectory (SS)
    dt = float(np.median(np.diff(t)))
    vt = np.array(log['v_torso'])[ss]           # torso linear velocity
    acc = np.gradient(vt, dt, axis=0)
    jerk = np.gradient(acc, dt, axis=0)
    m['torso_acc_pk'] = float(np.max(np.linalg.norm(acc, axis=1)))      # m/s^2
    m['torso_jerk_rms'] = float(np.sqrt(np.mean(np.sum(jerk ** 2, axis=1))))  # m/s^3
    # envelope fidelity: realized Ḣ_s vs planned (d L_com_ref/dt)
    Ld = np.array(log['L_dot']); Lcr = np.array(log['L_com_ref'])
    Ld_plan = np.gradient(Lcr, dt, axis=0)
    m['Hdot_real_pk'] = float(np.max(np.abs(Ld[ss])))
    m['Hdot_plan_pk'] = float(np.max(np.abs(Ld_plan[ss])))
    tau = np.array(log['tau']); m['tau_peak_Nm'] = float(np.max(np.abs(tau)))
    qpt = np.array(log['qp_time_ms'])[ss]
    m['qp_p50'] = float(np.percentile(qpt, 50)); m['qp_p99'] = float(np.percentile(qpt, 99))
    return m


def fmt(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f'{v:.2f}'
    return str(v)


def main():
    os.makedirs(OUT, exist_ok=True)
    p_t1 = geom_p_t1('phase2_1_baseline_1step')
    print(f'geometric p_t1 (FK torso @ q_end) = {np.round(p_t1, 4)}  travel={TRAVEL_MM} mm')

    logs, M = {}, {}
    for lab, d, mom, tp, kp in RUNS:
        lg = load(d)
        if lg is None:
            print(f'  [skip missing] {lab} ({d})')
            continue
        logs[lab] = lg
        M[lab] = metrics(lg, p_t1)

    # ---- master metrics table ----
    cols = ['docked', 'd_final_mm', 'arrival_mm', 'arrival_pct', 'tau_w_sat_pct',
            'tau_w_rms', 'torso_acc_pk', 'torso_jerk_rms', 'torso_pos_track_mm',
            'mom_com_track_mm', 'ee_dock_min_mm', 'hw_inf', 'Hdot_real_pk',
            'Hdot_plan_pk', 'tau_peak_Nm', 'qp_p50', 'qp_p99']
    labs = [l for l, *_ in RUNS if l in M]
    lines = ['| metric | ' + ' | '.join(labs) + ' |', '|---|' + '---|' * len(labs)]
    for k in cols:
        lines.append('| ' + k + ' | ' + ' | '.join(fmt(M[l][k]) for l in labs) + ' |')
    table = '\n'.join(lines)
    print('\n' + table)
    open(os.path.join(OUT, 'tuning_metrics.md'), 'w').write(table + '\n')

    # ---- Pareto / trade-off table (two-task points only) ----
    pts = [(lab, dict(mom=mom, tp=tp, kp=kp), M[lab])
           for lab, d, mom, tp, kp in RUNS if lab in M and mom is not None]
    print('\n## PARETO: arrival% vs τ_w-sat% (the trade-off)')
    print(f'{"point":>16} {"mom":>6} {"tp":>6} {"Kp":>3} {"arr_mm":>7} '
          f'{"arr%":>6} {"sat%":>6} {"jerk_rms":>9} {"acc_pk":>7} {"dock_mm":>8}')
    for lab, cfg, mm in sorted(pts, key=lambda x: x[2]['arrival_pct']):
        print(f'{lab:>16} {cfg["mom"]:>6} {cfg["tp"]:>6} {str(cfg["kp"]):>3} '
              f'{mm["arrival_mm"]:>7.2f} {mm["arrival_pct"]:>6.2f} '
              f'{mm["tau_w_sat_pct"]:>6.2f} {mm["torso_jerk_rms"]:>9.4f} '
              f'{mm["torso_acc_pk"]:>7.4f} {mm["ee_dock_min_mm"]:>8.3f}')

    # ---- best compromise: tightest arrival with sat under a "low" bar ----
    LOW_SAT = 10.0
    cand = [(lab, mm) for lab, _, mm in pts if mm['tau_w_sat_pct'] < LOW_SAT
            and mm['docked']]
    best = min(cand, key=lambda x: x[1]['arrival_pct']) if cand else None
    reach_3 = [lab for lab, _, mm in pts if mm['arrival_pct'] <= 3.5
               and mm['tau_w_sat_pct'] < LOW_SAT and mm['docked']]
    verdict = []
    if best:
        verdict.append(f'best compromise (sat<{LOW_SAT:.0f}%, dock<5mm): {best[0]} '
                       f'→ arrival {best[1]["arrival_pct"]:.1f}% '
                       f'({best[1]["arrival_mm"]:.1f}mm), sat {best[1]["tau_w_sat_pct"]:.1f}%')
    verdict.append('2-3% arrival reachable at low sat: '
                   + ('YES via ' + ', '.join(reach_3) if reach_3
                      else 'NO — tight arrival (<3.5%) only at high τ_w-sat'))
    vtxt = '\n'.join(verdict)
    print('\n' + vtxt)
    open(os.path.join(OUT, 'verdict.txt'), 'w').write(table + '\n\n' + vtxt + '\n')

    # ================= PLOT 1 — Pareto scatter (arrival% vs τ_w-sat%) =========
    fig, ax = plt.subplots(figsize=(8, 5.5))
    kp_color = {6: 'tab:blue', 4: 'tab:orange', 3: 'tab:red'}
    for lab, cfg, mm in pts:
        c = kp_color.get(cfg['kp'], 'k')
        mk = 'D' if cfg['mom'] != 5000 else 'o'    # diamond = off-grid mom (TP-dom)
        ax.scatter(mm['tau_w_sat_pct'], mm['arrival_pct'], s=90, c=c, marker=mk,
                   edgecolors='k', linewidths=0.6, zorder=3)
        ax.annotate(lab, (mm['tau_w_sat_pct'], mm['arrival_pct']),
                    fontsize=7, xytext=(4, 3), textcoords='offset points')
    ax.axhspan(2.0, 3.0, color='g', alpha=0.10, label='target arrival 2–3%')
    ax.axvline(23.8, color='0.6', ls=':', lw=1.0, label='TP-dom sat 23.8%')
    ax.set_xlabel('τ_w-sat @100 Hz  [%]   (envelope cost →)')
    ax.set_ylabel('torso arrival vs p_t1  [% of 125.5 mm travel]   (← objective)')
    ax.set_title('Phase 2.2 — arrival vs saturation trade-off (lower-left = better)')
    from matplotlib.lines import Line2D
    leg = [Line2D([], [], marker='o', ls='', mec='k', mfc=kp_color[k], label=f'Kp={k}')
           for k in (6, 4, 3)]
    leg += [Line2D([], [], marker='D', ls='', mec='k', mfc='0.7', label='mom≠5k (TP-dom)')]
    ax.legend(handles=leg, fontsize=7, loc='upper right')
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'pareto_arrival_vs_sat.png'), dpi=120)
    plt.close(fig)

    # ============ PLOT 2 — two levers: arrival% & sat% vs weight, per Kp ======
    grid = [(lab, cfg, mm) for lab, cfg, mm in pts if cfg['mom'] == 5000]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    for kp in (6, 4, 3):
        row = sorted([(cfg['tp'], mm) for _, cfg, mm in grid if cfg['kp'] == kp])
        if not row:
            continue
        xs = [r[0] for r in row]
        a1.plot(xs, [r[1]['arrival_pct'] for r in row], 'o-',
                color=kp_color[kp], label=f'Kp={kp}')
        a2.plot(xs, [r[1]['tau_w_sat_pct'] for r in row], 'o-',
                color=kp_color[kp], label=f'Kp={kp}')
    for a in (a1, a2):
        a.set_xlabel('torso-pose weight  (mom fixed 5k)'); a.grid(alpha=0.3); a.legend(fontsize=8)
    a1.axhspan(2.0, 3.0, color='g', alpha=0.10); a1.set_ylabel('torso arrival [% of travel]')
    a1.set_title('arrival vs torso-pose weight, per Kp')
    a2.axhline(23.8, color='0.6', ls=':', lw=1.0); a2.set_ylabel('τ_w-sat @100Hz [%]')
    a2.set_title('saturation vs torso-pose weight, per Kp')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'two_levers.png'), dpi=120)
    plt.close(fig)

    # ====== PLOT 3 — best-candidate torso trajectory + saturation smoothness ==
    bestlab = best[0] if best else 'bal 5k:5k'
    log = logs[bestlab]; bl = logs['bal 5k:5k']
    t = np.array(log['t']); ss = np.array(log['phase']) == 'SS'
    pt = np.array(log['p_torso'])[ss]; ptr = np.array(log['p_torso_ref'])[ss]
    fig, ax = plt.subplots(4, 1, figsize=(9, 10), sharex=False)
    for j, c in enumerate('xyz'):
        ax[j].plot(t[ss], ptr[:, j] * 1000, 'r:', lw=1.3, label='torso ref (quintic)')
        ax[j].plot(t[ss], pt[:, j] * 1000, 'b-', lw=1.2, label=f'{bestlab} realized')
        ax[j].axhline(p_t1[j] * 1000, color='g', ls='-.', lw=1.0, label='geometric p_t1')
        ax[j].set_ylabel(f'torso {c} [mm]')
        if j == 0:
            ax[j].legend(fontsize=7)
            ax[j].set_title(f'best candidate "{bestlab}" — torso tracking + saturation')
    # saturation smoothness: |τ_w|∞ over wheels at 100Hz, best vs balanced
    def tw_inf(lg):
        tw = np.abs(np.array(lg.get('tau_w_ss_hifreq') or lg['tau_w']))
        return tw.max(axis=1)
    twb = tw_inf(bl); twc = tw_inf(log)
    ax[3].plot(np.linspace(0, 1, len(twb)), twb, color='0.6', lw=1.0, label='bal 5k:5k')
    ax[3].plot(np.linspace(0, 1, len(twc)), twc, 'b-', lw=1.0, label=bestlab)
    ax[3].axhline(TAU_W_MAX, color='r', ls='--', lw=1.0, label='τ_w max (5 N·m)')
    ax[3].set_ylabel('|τ_w|∞ [N·m] @100Hz'); ax[3].set_xlabel('SS fraction')
    ax[3].legend(fontsize=7); ax[3].set_title('wheel-torque smoothness (flat-top = saturating)')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'best_candidate_trajectory.png'), dpi=120)
    plt.close(fig)

    print(f'\nplots+tables -> {OUT}/')


if __name__ == '__main__':
    main()
