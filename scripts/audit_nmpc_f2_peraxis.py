"""F2 per-axis re-analysis of the RWA conservation box.

The first pass compared `max_i |h_w,i|` — a scalar — against the box. The box is
COMPONENT-WISE (`h_w,i ∈ [−h', h']` for each i independently, 6 rows per knot),
so a scalar summary can misattribute which axis violates and by how much. This
redoes it axis by axis and plots it.

Also checks the quantity actually handed to the NMPC: `hw_for_nmpc` is
`rwa_I_w · qvel[6:9]` read at `_step` entry, whereas the exported `hw_*_Nms` is
`rwa_I_w · rw_vel_f` (filtered, captured later in the tick). If those differ the
box is anchored to something other than what the diagnostics report.

Run:
    MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/audit_nmpc_f2_peraxis.py
"""
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWEEP = os.path.join(ROOT, 'results/j2_adjconv/nmpc_sweep')
OUT = os.path.join(SWEEP, 'nmpc_f2_peraxis.png')
AXES = 'xyz'


def load(tag):
    p = os.path.join(SWEEP, tag, 'fulldiag_fulldiag.csv')
    return list(csv.DictReader(open(p)))


def col(rows, name):
    o = []
    for r in rows:
        try:
            o.append(float(r[name]))
        except (TypeError, ValueError, KeyError):
            o.append(np.nan)
    return np.asarray(o)


def main():
    BITE, BOX = 'F2bite_h35_N20', 'F2box_N20'
    h_bite, h_box = 3.5, 5.0
    rows = load(BITE)
    t = col(rows, 't_s')
    ph = np.array([r['phase'] for r in rows])
    hw = np.stack([col(rows, f'hw_{k}_Nms') for k in AXES], axis=1)
    ss = ph == 'SS'

    print('=' * 70)
    print(f'PER-AXIS h_w vs the component-wise box  (run {BITE}, box = {h_bite})')
    print('=' * 70)
    print(f'{"axis":>5}{"min":>10}{"max":>10}{"peak |.|":>11}'
          f'{"SS ticks over box":>20}')
    viol_any = np.zeros(len(t), bool)
    for i, k in enumerate(AXES):
        a = hw[:, i]
        fin = np.isfinite(a)
        over = fin & ss & (np.abs(a) > h_bite)
        viol_any |= over
        print(f'{k:>5}{np.nanmin(a):>10.4f}{np.nanmax(a):>10.4f}'
              f'{np.nanmax(np.abs(a)):>11.4f}{over.sum():>12} / {(fin & ss).sum()}')
    print(f'\nSS ticks where ANY axis exceeds {h_bite}: {(viol_any & ss).sum()}'
          f' / {ss.sum()}')

    # Which axis carries the violation, and is it the same one throughout?
    idx = np.where(viol_any & ss)[0]
    if idx.size:
        who = np.argmax(np.abs(hw[idx]), axis=1)
        names, counts = np.unique(who, return_counts=True)
        print('violating axis breakdown: '
              + ', '.join(f'{AXES[n]}={c}' for n, c in zip(names, counts)))
        i0 = idx[int(np.argmax(np.max(np.abs(hw[idx]), axis=1)))]
        print(f'worst tick t={t[i0]:.2f}s  h_w = '
              f'[{hw[i0,0]:+.4f}, {hw[i0,1]:+.4f}, {hw[i0,2]:+.4f}]  '
              f'-> excess on {AXES[int(np.argmax(np.abs(hw[i0])))]} = '
              f'{np.max(np.abs(hw[i0])) - h_bite:+.4f} Nms')

    # ---- figure -------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('F2 — RWA conservation box, per axis '
                 '(box is component-wise, not a norm)',
                 fontsize=13, fontweight='bold')

    ax = axes[0, 0]
    for i, k in enumerate(AXES):
        ax.plot(t, hw[:, i], lw=1, label=f'$h_{{w,{k}}}$')
    for h, c, l in ((h_bite, '#d62728', f'bite box ±{h_bite}'),
                    (h_box, 'k', f'nominal box ±{h_box}')):
        ax.axhline(h, color=c, ls='--', lw=1, label=l)
        ax.axhline(-h, color=c, ls='--', lw=1)
    ax.set(title=f'per-axis wheel momentum ({BITE})',
           xlabel='t [s]', ylabel='$h_w$ [Nms]')

    ax = axes[0, 1]
    for i, k in enumerate(AXES):
        a = np.where(ss, hw[:, i], np.nan)
        ax.plot(t, np.abs(a), lw=1, label=f'$|h_{{w,{k}}}|$ (SS)')
    ax.axhline(h_bite, color='#d62728', ls='--', lw=1.2, label=f'box {h_bite}')
    m = viol_any & ss
    if m.any():
        ax.plot(t[m], np.max(np.abs(hw[m]), axis=1), 'o', ms=4,
                color='#d62728', label='violating SS ticks')
    ax.set(title='SS only — where the box is enforced',
           xlabel='t [s]', ylabel='$|h_w|$ [Nms]')

    # zoom on the violation window
    ax = axes[1, 0]
    if idx.size:
        lo, hi = max(0, idx[0] - 25), min(len(t), idx[-1] + 25)
        for i, k in enumerate(AXES):
            ax.plot(t[lo:hi], hw[lo:hi, i], lw=1.4, marker='.', ms=3,
                    label=f'$h_{{w,{k}}}$')
        ax.axhline(h_bite, color='#d62728', ls='--', lw=1.2)
        ax.axhline(-h_bite, color='#d62728', ls='--', lw=1.2)
        ax.axvspan(t[idx[0]], t[idx[-1]], color='#d62728', alpha=.12,
                   label='box violated')
    ax.set(title='zoom: the violation window', xlabel='t [s]', ylabel='$h_w$ [Nms]')

    # box-on at 5.0 for reference
    ax = axes[1, 1]
    rb = load(BOX)
    tb = col(rb, 't_s')
    hb = np.stack([col(rb, f'hw_{k}_Nms') for k in AXES], axis=1)
    for i, k in enumerate(AXES):
        ax.plot(tb, hb[:, i], lw=1, label=f'$h_{{w,{k}}}$')
    ax.axhline(h_box, color='k', ls='--', lw=1, label=f'box ±{h_box}')
    ax.axhline(-h_box, color='k', ls='--', lw=1)
    ax.set(title=f'nominal box ({BOX}, ±{h_box}) — slack throughout',
           xlabel='t [s]', ylabel='$h_w$ [Nms]')

    for a in axes.flat:
        a.grid(alpha=.3)
        a.legend(fontsize=7, loc='best')
    fig.tight_layout()
    fig.savefig(OUT, dpi=130, bbox_inches='tight')
    print(f'\nwrote {OUT}')

    dest = os.path.join(SWEEP, 'f2_peraxis.json')
    with open(dest, 'w') as fh:
        json.dump({
            'run': BITE, 'box': h_bite,
            'per_axis_peak_abs': {k: float(np.nanmax(np.abs(hw[:, i])))
                                  for i, k in enumerate(AXES)},
            'ss_ticks_over_box': int((viol_any & ss).sum()),
            'ss_ticks_total': int(ss.sum()),
        }, fh, indent=2)
    print(f'wrote {dest}')


if __name__ == '__main__':
    main()
