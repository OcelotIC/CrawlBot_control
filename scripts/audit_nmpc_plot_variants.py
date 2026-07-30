"""Plot the NMPC horizon/discretization variants against the frozen canonical.

Reads each variant's exported fulldiag CSV plus its nmpc_step_log.json and
emits a multi-panel comparison figure:

  1. CoM tracking error |e_com| over time
  2. Structure attitude excursion theta_s
  3. Wheel momentum norm ||h_w|| against the +/-5 Nms envelope
  4. Realized momentum-rate ||Hdot_s||_inf against the tau_w_max cap
  5. NMPC solve time against the control period
  6. Per-solve status mix

Usage:
    MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/audit_nmpc_plot_variants.py
"""
import collections
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWEEP = os.path.join(ROOT, 'results/j2_adjconv/nmpc_sweep')
BASE = os.path.join(ROOT, 'results/j2_adjconv/c25_fulldiag.csv')
OUT = os.path.join(ROOT, 'results/j2_adjconv/nmpc_sweep/nmpc_variants.png')

# label -> (fulldiag csv, nmpc step log or None, control period, colour)
def variant_paths():
    v = [('frozen N=8 dt=0.10 (10 Hz)', BASE, None, 0.10, '#888888')]
    for tag, colour in (('N20_dt05_p10', '#d62728'), ('N20_dt05_p05', '#1f77b4')):
        d = os.path.join(SWEEP, tag)
        csv_p = os.path.join(d, 'fulldiag_fulldiag.csv')
        log_p = os.path.join(d, 'nmpc_step_log.json')
        if not os.path.exists(csv_p):
            print(f'  (skipping {tag}: no fulldiag)')
            continue
        period = 0.10 if tag.endswith('p10') else 0.05
        nice = (f'N=20 dt=0.05 ({1/period:.0f} Hz)'
                + ('  [ref dilated 2x]' if period == 0.10 else '  [consistent]'))
        v.append((nice, csv_p, log_p, period, colour))
    return v


def load(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def col(rows, name):
    out = []
    for r in rows:
        try:
            out.append(float(r.get(name, 'nan')))
        except (TypeError, ValueError):
            out.append(np.nan)
    return np.asarray(out)


def main():
    variants = variant_paths()
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('NMPC horizon / discretization variants — 6-step managed traversal',
                 fontsize=14, fontweight='bold')

    summary = []
    for label, csv_p, log_p, period, colour in variants:
        rows = load(csv_p)
        t = col(rows, 't_s')
        if not np.isfinite(t).any():
            t = np.arange(len(rows), dtype=float) * 0.01
        phase = np.array([r.get('phase', '') for r in rows])
        ss = phase == 'SS'

        # 1. CoM tracking error
        e = col(rows, 'e_com_m') * 1e3
        ax = axes[0, 0]
        ax.plot(t[ss], e[ss], '.', ms=2, color=colour, label=label, alpha=.7)

        # 2. theta_s
        th = np.linalg.norm(
            np.stack([col(rows, f'theta_s_{k}_deg') for k in 'xyz'], axis=1), axis=1)
        axes[0, 1].plot(t, th, lw=1, color=colour, label=label)

        # 3. ||h_w||
        hw = np.stack([col(rows, f'hw_{k}_Nms') for k in 'xyz'], axis=1)
        axes[1, 0].plot(t, np.linalg.norm(hw, axis=1), lw=1, color=colour, label=label)

        # 4. realized ||Hdot_s||_inf
        hd = np.max(np.abs(np.stack(
            [col(rows, f'Hdot_s_realized_cont_{k}_Nm') for k in 'xyz'], axis=1)), axis=1)
        axes[1, 1].plot(t, hd, lw=.8, color=colour, label=label, alpha=.8)

        # 5 + 6. solver
        if log_p and os.path.exists(log_p):
            d = json.load(open(log_p))
            ms = np.array([r['time_ms'] for r in d], float)
            axes[2, 0].plot(np.arange(len(ms)), ms, lw=.7, color=colour,
                            label=f'{label}  max={ms.max():.0f} ms')
            axes[2, 0].axhline(period * 1e3, color=colour, ls=':', lw=1.2)
            cnt = collections.Counter(str(r['status']) for r in d)
            summary.append((label, len(d), float(np.median(ms)), float(ms.max()),
                            int(sum(1 for x in ms if x > period * 1e3)),
                            cnt.get('Solved_To_Acceptable_Level', 0)))

    axes[0, 0].set(title='CoM tracking error (SS ticks only)',
                   xlabel='t [s]', ylabel='|e_com| [mm]')
    axes[0, 1].set(title='Structure attitude excursion',
                   xlabel='t [s]', ylabel=r'$\theta_s$ [deg]')
    axes[1, 0].set(title='Wheel momentum norm', xlabel='t [s]',
                   ylabel=r'$\|h_w\|$ [Nms]')
    axes[1, 0].axhline(5.0, color='k', ls='--', lw=1, label='envelope 5 Nms')
    axes[1, 1].set(title=r'Realized $\|\dot{H}_s\|_\infty$ vs cap',
                   xlabel='t [s]', ylabel=r'$\|\dot{H}_s\|_\infty$ [Nm]')
    axes[1, 1].axhline(2.5, color='k', ls='--', lw=1, label=r'$\tau_{w,max}=2.5$')
    axes[2, 0].set(title='NMPC solve time (dotted = that variant\'s control period)',
                   xlabel='solve index', ylabel='ms')

    # Panel 6: table
    ax = axes[2, 1]
    ax.axis('off')
    if summary:
        cells = [[l[:26], f'{n}', f'{med:.1f}', f'{mx:.1f}', f'{ov}', f'{acc}']
                 for l, n, med, mx, ov, acc in summary]
        tbl = ax.table(cellText=cells,
                       colLabels=['variant', 'solves', 'med ms', 'max ms',
                                  'over period', 'acceptable-tol'],
                       loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.6)
    ax.set_title('NMPC solver summary', fontweight='bold')

    for a in axes.flat[:5]:
        a.grid(alpha=.3)
        a.legend(fontsize=7, loc='best')

    fig.tight_layout()
    fig.savefig(OUT, dpi=130, bbox_inches='tight')
    print(f'wrote {OUT}')
    for s in summary:
        print('  ', s)


if __name__ == '__main__':
    main()
