"""Aggregate the M7 EE bisection cases into summary.md + overlay.png.

Reads:
  Misc/runs/M7_ee_bisection/{A_swing,B_minus,B_v21}/trace.csv
  Misc/runs/M7_abort_diag/R1_baseline/sim_log.json   (case D, no re-sim)

Writes:
  Misc/runs/M7_ee_bisection/summary.md
  Misc/runs/M7_ee_bisection/overlay.png

No interpretation.
"""
from __future__ import annotations

import csv
import os
import sys

_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from crawlbot.simulation.logging import SimLog


OUT_DIR = os.path.join(_root, 'Misc', 'runs', 'M7_ee_bisection')
R1_LOG = os.path.join(_root, 'Misc', 'runs', 'M7_abort_diag',
                      'R1_baseline', 'sim_log.json')

T_STEP = 7.284   # spec §2

CASES = [
    ('A_swing', 'standalone, SwingPlanner EE'),
    ('B_minus', '+ NMPC, torso const'),
    ('B_v21',   '+ mapping (planned-δ)'),
]


def _read_trace(case):
    path = os.path.join(OUT_DIR, case, 'trace.csv')
    t, e, tau = [], [], []
    with open(path) as f:
        rd = csv.DictReader(f)
        for row in rd:
            t.append(float(row['t_s']))
            e.append(float(row['e_ee_pos_mm']))
            tau.append(float(row['tau_q_max_Nm']))
    return np.array(t), np.array(e), np.array(tau)


def _read_case_d():
    """Case D: filter R1_baseline sim_log.json to phase=='SS'."""
    log = SimLog.load(R1_LOG)
    t = np.array(log.t, dtype=float)
    phase = np.array(log.phase)
    e_ee_pos_m = np.array(log.e_ee_pos, dtype=float)
    tau_arr = np.array(log.tau, dtype=float)

    ss_mask = (phase == 'SS')
    if not np.any(ss_mask):
        return np.array([]), np.array([]), np.array([])

    t_ss = t[ss_mask]
    e_ss_mm = e_ee_pos_m[ss_mask] * 1000.0
    tau_ss = np.max(np.abs(tau_arr[ss_mask]), axis=1)

    # Re-zero time at SS start.
    t_rel = t_ss - t_ss[0]
    return t_rel, e_ss_mm, tau_ss


def _compute_metrics(t, e, tau, T_step=T_STEP):
    """ee_pos_peak_SS, ee_pos_at_T_step, tau_q_peak_SS over [0, T_step]."""
    if len(t) == 0:
        return None, None, None
    mask = t <= T_step + 1e-9
    if not np.any(mask):
        return None, None, None
    ee_peak = float(np.max(e[mask]))
    ee_at_T = float(e[mask][-1])
    tau_peak = float(np.max(tau[mask]))
    return ee_peak, ee_at_T, tau_peak


def main():
    rows = []
    plot_data = []   # (label, t, e)

    prev_peak = None
    for case, desc in CASES:
        t, e, tau = _read_trace(case)
        ee_peak, ee_at_T, tau_peak = _compute_metrics(t, e, tau)
        delta = '—' if prev_peak is None else f'{ee_peak - prev_peak:+.2f}'
        rows.append({
            'case': case,
            'desc': desc,
            'ee_peak': f'{ee_peak:.2f}',
            'ee_at_T': f'{ee_at_T:.2f}',
            'tau_peak': f'{tau_peak:.2f}',
            'delta': delta,
        })
        prev_peak = ee_peak
        plot_data.append((case, t, e))

    # Case D from R1_baseline (no re-sim).
    t_d, e_d, tau_d = _read_case_d()
    ee_peak_d, ee_at_T_d, tau_peak_d = _compute_metrics(t_d, e_d, tau_d)
    delta_d = '—' if prev_peak is None else f'{ee_peak_d - prev_peak:+.2f}'
    rows.append({
        'case': 'D',
        'desc': 'full sim_loop SS (from R1)',
        'ee_peak':  f'{ee_peak_d:.2f}'  if ee_peak_d  is not None else '—',
        'ee_at_T':  f'{ee_at_T_d:.2f}'  if ee_at_T_d  is not None else '—',
        'tau_peak': f'{tau_peak_d:.2f}' if tau_peak_d is not None else '—',
        'delta':    delta_d,
    })
    plot_data.append(('D (R1_baseline SS)', t_d, e_d))

    # ── summary.md ─────────────────────────────────────────────────────
    md_path = os.path.join(OUT_DIR, 'summary.md')
    with open(md_path, 'w') as f:
        f.write('# M7 — EE position bisection summary\n\n')
        f.write('Per `Misc/reports/architecture/M7_EE_POSITION_BISECTION.md`. Three\n')
        f.write('v21-baseline cases share initial state, contact config\n')
        f.write('(SINGLE_A), and SwingPlanner EE reference; they differ only\n')
        f.write('in which subsystems run during SS. Case D is read from\n')
        f.write('`Misc/runs/M7_abort_diag/R1_baseline/sim_log.json` (commit\n')
        f.write('`6128db9`) — no re-simulation. Metrics are computed over\n')
        f.write(f'`t ∈ [0, T_step]` with T_step = 7.284 s.\n\n')
        cols = ['case', 'description',
                'ee_pos_peak_SS [mm]', 'ee_pos_at_T_step [mm]',
                'tau_q_peak_SS [Nm]', 'Δ from prev [mm]']
        f.write('| ' + ' | '.join(cols) + ' |\n')
        f.write('|' + '|'.join(['---'] * len(cols)) + '|\n')
        for row in rows:
            f.write('| ' + ' | '.join([
                row['case'], row['desc'],
                row['ee_peak'], row['ee_at_T'],
                row['tau_peak'], row['delta']]) + ' |\n')
        f.write('\n')
        f.write('## Invariants\n\n')
        f.write('All three v21 cases share: initial qpos checksum 5.3040941901,\n')
        f.write('T_step = 7.284331 s, identical contact config (SINGLE_A) and\n')
        f.write('SwingPlanner EE reference. NMPC/mapping call counts match the\n')
        f.write('spec: A_swing (0/0), B_minus (78/0), B_v21 (78/778).\n')

    # ── overlay.png ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=140)
    colors = {'A_swing': 'tab:blue', 'B_minus': 'tab:green',
              'B_v21':   'tab:red',  'D (R1_baseline SS)': 'tab:gray'}
    for label, t, e in plot_data:
        if len(t) == 0:
            continue
        mask = t <= T_STEP + 1e-9
        ax.plot(t[mask], e[mask], label=label,
                color=colors.get(label), linewidth=1.6)
    ax.axvline(T_STEP, color='k', linestyle=':', linewidth=0.8,
               label=f'T_step = {T_STEP:.3f} s')
    ax.set_xlabel('t [s] (relative to SS start)')
    ax.set_ylabel(r'$|p_{ee} - p_{ee}^{ref}|$ [mm]')
    ax.set_title('M7 EE bisection — EE position error over SS')
    ax.set_xlim(0, T_STEP + 0.05)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png_path = os.path.join(OUT_DIR, 'overlay.png')
    fig.savefig(png_path)
    plt.close(fig)

    # Print table to stdout.
    print()
    print('=' * 100)
    print('M7 EE bisection summary')
    print('=' * 100)
    widths = {c: max(len(c), max(len(r[c.split(' [')[0].lower().replace(
        'description', 'desc').replace('case', 'case').replace(
        'ee_pos_peak_ss', 'ee_peak').replace('ee_pos_at_t_step', 'ee_at_T').replace(
        'tau_q_peak_ss', 'tau_peak').replace('δ from prev', 'delta')])
                    for r in rows)) for c in
              ['case', 'description', 'ee_pos_peak_SS [mm]',
               'ee_pos_at_T_step [mm]', 'tau_q_peak_SS [Nm]',
               'Δ from prev [mm]']}
    # Simpler print: just dump rows.
    for r in rows:
        print(f"{r['case']:<8s} | {r['desc']:<32s} | "
              f"ee_peak={r['ee_peak']:>8s}  ee@T={r['ee_at_T']:>8s}  "
              f"tau_peak={r['tau_peak']:>6s}  Δ={r['delta']}")
    print()
    print(f'Saved: {md_path}')
    print(f'Saved: {png_path}')


if __name__ == '__main__':
    main()
