#!/usr/bin/env python3
"""Phase 2.1 reporting: metrics table + §6 plots + ratio read for the
single-step swing sweep (Variant A/B × ss_alpha_mom) vs the flag-OFF baseline.

h_w reported in per-axis ∞-norm (brief); τ_w-saturation computed at the 100 Hz
QP rate from the log_hifreq_ss buffers. metrics.py is NOT used/modified.

Usage: PYTHONPATH=. MUJOCO_GL=disabled python3 Misc/scripts/report_phase2_1.py
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
OUT = os.path.join(_ROOT, 'Misc', 'runs', 'phase2_1_report')
PROG_AXIS = 0  # swing progression ≈ structure +x (anchor_dx along x)

# (label, dir, alpha_mom, variant)  — baseline first
RUNS = [
    ('baseline', 'phase2_1_baseline_1step', None, '-'),
    ('A@500', 'phase2_1_vA_amom500', 500, 'A'),
    ('A@5000', 'phase2_1_vA_amom5000', 5000, 'A'),
    ('A@30000', 'phase2_1_vA_amom30000', 30000, 'A'),
    ('B@500', 'phase2_1_vB_amom500', 500, 'B'),
    ('B@5000', 'phase2_1_vB_amom5000', 5000, 'B'),
    ('B@30000', 'phase2_1_vB_amom30000', 30000, 'B'),
]


def _load(d):
    p = os.path.join(_ROOT, 'results', d, 'sim_log.json')
    return json.load(open(p)) if os.path.exists(p) else None


def _fsat(d):
    p = os.path.join(_ROOT, 'results', d, 'sat_stats.txt')
    if not os.path.exists(p):
        return None
    return open(p).read().strip()


def metrics(log):
    t = np.array(log['t']); ph = np.array(log['phase']); ss = ph == 'SS'
    r = np.array(log['r_com']); rr = np.array(log['r_com_ref'])
    pt = np.array(log['p_torso'])
    de = log.get('dock_events', []); ab = log.get('aborted_steps', [])
    m = {}
    m['docked'] = bool(de)
    if de:
        m['d_final_mm'] = float(de[-1].get('d_mm', de[-1].get('d', 0) * 1000))
        m['t_event'] = float(de[-1].get('t', t.max()))
    elif ab:
        m['d_final_mm'] = float(ab[-1].get('d_mm', 0))
        m['t_event'] = float(ab[-1].get('t', t.max()))
    else:
        m['d_final_mm'] = float('nan'); m['t_event'] = float(t.max())
    # CoM tracking (SS)
    e_ax = np.abs(r - rr)
    m['com_err_peak_mm'] = float(np.max(np.linalg.norm(r - rr, axis=1)[ss]) * 1000)
    m['com_err_rms_mm'] = float(np.sqrt(np.mean(
        np.linalg.norm(r - rr, axis=1)[ss] ** 2)) * 1000)
    m['com_err_ax_peak_mm'] = (np.max(e_ax[ss], axis=0) * 1000).round(3).tolist()
    # Torso jitter (pk-pk on progression axis, SS) + torso-ori peak
    m['torso_jitter_pkpk_mm'] = float(
        (pt[ss, PROG_AXIS].max() - pt[ss, PROG_AXIS].min()) * 1000)
    m['torso_ori_peak_deg'] = float(np.max(np.array(log['e_torso_ori'])[ss]))
    # EE capture
    dg = np.array(log['d_grip_swing'])
    i_close = int(np.argmin(dg))
    m['ee_closest_mm'] = float(dg[i_close] * 1000)
    m['ee_ori_at_closest_deg'] = float(np.array(log['e_ee_ori'])[i_close])
    # h_w ∞-norm (100 Hz if available, else 10 Hz) + 2-norm
    hw = np.array(log.get('hw_ss_hifreq') or log['hw'])
    m['hw_peak_inf'] = float(np.max(np.abs(hw)))
    m['hw_peak_2'] = float(np.max(np.linalg.norm(hw, axis=1)))
    # τ_w-sat at 100 Hz (any-axis at ±5 bound)
    tw = np.abs(np.array(log.get('tau_w_ss_hifreq') or log['tau_w']))
    n = len(tw)
    m['tau_w_sat_100hz_pct'] = float(100 * np.mean((tw >= 5 - 1e-6).any(1))) if n else float('nan')
    m['tau_w_peak'] = float(tw.max()) if n else float('nan')
    m['hifreq_n'] = int(len(log.get('t_ss_hifreq', [])))
    # Joint torque margin
    tau = np.array(log['tau'])
    m['tau_peak_Nm'] = float(np.max(np.abs(tau)))
    # QP solve time
    qpt = np.array(log['qp_time_ms'])[ss]
    m['qp_p50_ms'] = float(np.percentile(qpt, 50))
    m['qp_p99_ms'] = float(np.percentile(qpt, 99))
    # realized Ḣ_s (robot L_dot) vs NMPC-planned (d L_com_ref/dt), peak ∞-norm
    Ld = np.array(log['L_dot']); Lcr = np.array(log['L_com_ref'])
    dt = np.median(np.diff(t))
    Ld_plan = np.gradient(Lcr, dt, axis=0)
    m['Hdot_realized_peak'] = float(np.max(np.abs(Ld[ss])))
    m['Hdot_planned_peak'] = float(np.max(np.abs(Ld_plan[ss])))
    return m


def main():
    os.makedirs(OUT, exist_ok=True)
    logs = {lab: _load(d) for lab, d, _, _v in RUNS}
    M = {lab: metrics(logs[lab]) for lab, d, a, _v in RUNS if logs[lab]}
    fs = {lab: _fsat(d) for lab, d, _, _v in RUNS}

    # ---- metrics table (markdown) ----
    cols = [('docked', ''), ('d_final_mm', 'mm'), ('com_err_peak_mm', 'mm'),
            ('com_err_rms_mm', 'mm'), ('torso_jitter_pkpk_mm', 'mm'),
            ('torso_ori_peak_deg', 'deg'), ('ee_closest_mm', 'mm'),
            ('ee_ori_at_closest_deg', 'deg'), ('hw_peak_inf', 'N·m·s'),
            ('tau_w_sat_100hz_pct', '%'), ('tau_peak_Nm', 'Nm'),
            ('qp_p50_ms', 'ms'), ('qp_p99_ms', 'ms'),
            ('Hdot_realized_peak', 'N·m'), ('Hdot_planned_peak', 'N·m')]
    labs = [lab for lab, d, a, _v in RUNS if lab in M]
    lines = ['| metric | ' + ' | '.join(labs) + ' |',
             '|---|' + '---|' * len(labs)]
    for key, unit in cols:
        row = [f'{key} [{unit}]' if unit else key]
        for lab in labs:
            v = M[lab][key]
            row.append(str(v) if isinstance(v, bool) else
                       (f'{v:.3f}' if isinstance(v, float) else str(v)))
        lines.append('| ' + ' | '.join(row) + ' |')
    table = '\n'.join(lines)
    print(table)
    open(os.path.join(OUT, 'metrics_table.md'), 'w').write(table + '\n')

    # ---- ratio read ----
    rr_lines = ['', '## Ratio read (CoM authority vs torso-angular cost)']
    for var in ('A', 'B'):
        sv = [(a, M[lab]) for lab, d, a, v in RUNS
              if v == var and lab in M and a is not None]
        if not sv:
            continue
        rr_lines.append(f'-- Variant {var} --')
        rr_lines.append(f'{"alpha_mom":>10} {"docked":>7} {"d_final":>9} '
                        f'{"CoM_pk_mm":>10} {"torso_ori":>10}')
        for a, mm in sorted(sv):
            rr_lines.append(
                f'{a:>10} {str(mm["docked"]):>7} {mm["d_final_mm"]:>9.2f} '
                f'{mm["com_err_peak_mm"]:>10.3f} {mm["torso_ori_peak_deg"]:>10.3f}')
    sweep = [(a, M[lab]) for lab, d, a, v in RUNS
             if v == 'A' and lab in M and a is not None]
    ratio = '\n'.join(rr_lines)
    print(ratio)
    open(os.path.join(OUT, 'ratio_read.txt'), 'w').write(ratio + '\n')

    # ---- plots ----
    base = logs['baseline']
    # (1) sweep summary: CoM peak err + torso-ori peak vs alpha_mom
    if sweep:
        a_arr = [a for a, _ in sweep]
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(a_arr, [m['com_err_peak_mm'] for _, m in sweep], 'o-b',
                 label='CoM peak err [mm]')
        ax1.set_xscale('log'); ax1.set_xlabel('ss_alpha_mom')
        ax1.set_ylabel('CoM peak err [mm]', color='b')
        ax2 = ax1.twinx()
        ax2.plot(a_arr, [m['torso_ori_peak_deg'] for _, m in sweep], 's--r',
                 label='torso-ori peak [deg]')
        ax2.set_ylabel('torso-ori peak [deg]', color='r')
        ax1.set_title('Phase 2.1 ratio read — Variant A authority sweep')
        fig.tight_layout(); fig.savefig(os.path.join(OUT, 'sweep_ratio.png'), dpi=120)
        plt.close(fig)
    # (2) per-run CoM tracking + torso jitter overlay (baseline grey/dashed)
    for lab, d, a, _v in RUNS:
        log = logs.get(lab)
        if not log or lab == 'baseline':
            continue
        t = np.array(log['t']); r = np.array(log['r_com']); rr_ = np.array(log['r_com_ref'])
        tb = np.array(base['t']); ptb = np.array(base['p_torso'])
        pt = np.array(log['p_torso'])
        fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        for j, c in zip(range(3), 'xyz'):
            ax[0].plot(t, rr_[:, j] * 1000, ':', lw=1, label=f'r*_{c}')
            ax[0].plot(t, r[:, j] * 1000, '-', lw=1, label=f'r̂_{c}')
        ax[0].set_ylabel('CoM [mm]'); ax[0].legend(fontsize=7, ncol=3)
        ax[0].set_title(f'Phase 2.1 {lab} — CoM tracking + torso jitter (baseline grey)')
        ax[1].plot(tb, ptb[:, PROG_AXIS] * 1000, color='0.6', ls='--', label='baseline torso_x')
        ax[1].plot(t, pt[:, PROG_AXIS] * 1000, 'k', label=f'{lab} torso_x')
        ax[1].set_ylabel('torso x [mm]'); ax[1].set_xlabel('t [s]'); ax[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f'track_{lab.replace("@","_")}.png'), dpi=120)
        plt.close(fig)

    # F-SAT confirmation
    print('\n## F-SAT telemetry per run (active vs computed-but-unused)')
    for lab, d, a, _v in RUNS:
        if fs.get(lab):
            print(f'  [{lab}] {fs[lab].splitlines()[-1] if fs[lab] else ""}')
    print(f'\nPlots + tables -> {OUT}/')


if __name__ == '__main__':
    main()
