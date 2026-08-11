"""h_w carry across crawl steps, and what the terminal settle hides.

Two panels:
  1. h_w,z over the run with each SS entry marked and the 20 s DS_terminal
     shaded — the carry walks the entry point down, the settle undoes it.
  2. extrapolated peak vs step count against the two candidate bounds.

Run:
    MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/audit_nmpc_plot_per_step.py
"""
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(ROOT,
                   'results/j2_adjconv/nmpc_sweep/F2off_ctl_N20/fulldiag_fulldiag.csv')
JSN = os.path.join(ROOT, 'results/j2_adjconv/nmpc_sweep/hw_per_step.json')
OUT = os.path.join(ROOT, 'results/j2_adjconv/nmpc_sweep/nmpc_hw_per_step.png')
H_MAX, H_HONEST = 5.0, 3.1


def col(rows, n):
    o = []
    for r in rows:
        try:
            o.append(float(r[n]))
        except (TypeError, ValueError, KeyError):
            o.append(np.nan)
    return np.asarray(o)


def main():
    rows = list(csv.DictReader(open(RUN)))
    t = col(rows, 't_s')
    ph = np.array([r['phase'] for r in rows])
    si = np.array([r['step_index'] for r in rows])
    hz = col(rows, 'hw_z_Nms')
    d = json.load(open(JSN))
    o = d['per_axis']['z']

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    fig.suptitle('The crawl DOES carry momentum on z — the 20 s terminal '
                 'settle is what hides it', fontsize=13, fontweight='bold')

    # --- 1. the trace -------------------------------------------------
    a = ax[0]
    a.plot(t, hz, lw=1.1, color='#d62728', label='$h_{w,z}$')
    term = ph == 'DS_terminal'
    a.axvspan(t[term][0], t[term][-1], color='0.85', zorder=0,
              label='DS_terminal (20 s settle)')
    ent_t, ent_h = [], []
    for s in sorted({x for x in si if x not in ('', 'None') and float(x) >= 0},
                    key=lambda x: int(float(x))):
        m = (si == s) & (ph == 'SS')
        if m.sum() > 2:
            ent_t.append(t[m][0]); ent_h.append(hz[m][0])
    a.plot(ent_t, ent_h, 'o--', color='k', ms=5, lw=1.0, label='SS entry (carry)')
    fit = np.polyfit(range(len(ent_h)), ent_h, 1)
    a.annotate(f'carry = {o["carry_mean"]:+.3f} Nms/step\n'
               f'(|mean|/std = {o["systematic_ratio"]:.2f}, n=5)\n'
               f'settle returns {o["terminal_settle_delta"]:+.3f}',
               xy=(.02, .06), xycoords='axes fraction', fontsize=8,
               bbox=dict(fc='w', ec='0.7', alpha=.9))
    a.axhline(-H_MAX, color='k', ls='--', lw=1.1, label=f'box −{H_MAX}')
    a.axhline(-H_HONEST, color='#ff7f0e', ls=':', lw=1.4,
              label=f'honest bound −{H_HONEST}')
    a.set(title='1. entry point walks down, then the settle unwinds it',
          xlabel='t [s]', ylabel='$h_{w,z}$ [Nms]')

    # --- 2. extrapolation ---------------------------------------------
    a = ax[1]
    N = np.arange(1, 16)
    for k, c in (('z', '#d62728'), ('y', '#2ca02c'), ('x', '#1f77b4')):
        p = d['per_axis'][k]
        a.plot(N, abs(p['carry_mean']) * N + p['excursion_max'], 'o-', ms=4,
               color=c, label=f'{k}: |{p["carry_mean"]:+.3f}|·N + '
                              f'{p["excursion_max"]:.2f}')
    a.axhline(H_MAX, color='k', ls='--', lw=1.2, label=f'box {H_MAX}')
    a.axhline(H_HONEST, color='#ff7f0e', ls=':', lw=1.4,
              label=f'honest bound {H_HONEST}')
    a.axvline(6, color='0.5', lw=1.0)
    a.annotate('today\n(6 steps)', xy=(6, 0.3), fontsize=8, ha='center')
    nz = o['steps_to_5p0']
    a.annotate(f'z reaches the box at\n~{nz:.0f} steps ≈ '
               f'{o["metres_to_5p0"]:.1f} m of CoM travel',
               xy=(.40, .70), xycoords='axes fraction', fontsize=8,
               bbox=dict(fc='w', ec='0.7', alpha=.9))
    a.set(title='2. peak(N) with no terminal settle — 20 m is 8× more than needed',
          xlabel='crawl steps N', ylabel='predicted $|h_w|$ peak [Nms]',
          ylim=(0, 7))

    for x in ax:
        x.grid(alpha=.3)
        x.legend(fontsize=7, loc='best')
    fig.tight_layout()
    fig.savefig(OUT, dpi=130, bbox_inches='tight')
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
