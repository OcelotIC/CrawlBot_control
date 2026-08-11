"""Curves for the momentum-budget argument: transient vs secular, rate vs level.

Four panels:
  1. h_w per axis with the +/-5 box — showing the peak is a within-step
     EXCURSION that returns to ~0, not a traversal-scale wind-up.
  2. realized |Hdot_s| per axis against the tau_w_max cap — showing z is
     already clipped, so a longer lever cannot raise the delivered rate.
  3. the box margin expressed in SECONDS of saturated tau_w, which is the
     operational unit.
  4. I_s * d(omega_s) over one horizon against that margin — the
     reconstruction error, on the same axes.

Run:
    MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/audit_nmpc_plot_momentum.py
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(ROOT,
                   'results/j2_adjconv/nmpc_sweep/F2off_ctl_N20/fulldiag_fulldiag.csv')
MJCF = os.path.join(ROOT, 'models/VISPA_crawling_rwa3.xml')
OUT = os.path.join(ROOT, 'results/j2_adjconv/nmpc_sweep/nmpc_momentum_budget.png')
AXES = 'xyz'
COL = {'x': '#1f77b4', 'y': '#2ca02c', 'z': '#d62728'}
TAU_W_MAX, H_MAX, T = 2.5, 5.0, 2.0


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
    hw = np.stack([col(rows, f'hw_{k}_Nms') for k in AXES], axis=1)
    hd = np.stack([col(rows, f'Hdot_s_realized_cont_{k}_Nm') for k in AXES], axis=1)
    om = np.stack([col(rows, f'omega_s_{k}_radps') for k in AXES], axis=1)

    fig, ax = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('NMPC momentum budget — the box defends a within-step excursion, '
                 'not an accumulation', fontsize=13, fontweight='bold')

    # 1. h_w with the box
    a = ax[0, 0]
    for i, k in enumerate(AXES):
        a.plot(t, hw[:, i], lw=1.2, color=COL[k], label=f'$h_{{w,{k}}}$')
    a.axhline(H_MAX, color='k', ls='--', lw=1.2, label=f'box ±{H_MAX}')
    a.axhline(-H_MAX, color='k', ls='--', lw=1.2)
    a.axhline(0, color='0.6', lw=.7)
    fin = np.isfinite(hw).all(1)
    a.annotate(f'net drift after 6 steps:\n'
               f'x {hw[fin][-1,0]-hw[fin][0,0]:+.3f}   '
               f'y {hw[fin][-1,1]-hw[fin][0,1]:+.3f}   '
               f'z {hw[fin][-1,2]-hw[fin][0,2]:+.3f}  Nms',
               xy=(.98, .04), xycoords='axes fraction', ha='right', fontsize=8,
               bbox=dict(fc='w', ec='0.7', alpha=.9))
    a.set(title='1. wheel momentum — peaks, then returns to ~0',
          xlabel='t [s]', ylabel='$h_w$ [Nms]')

    # 2. realized rate vs the cap
    a = ax[0, 1]
    for i, k in enumerate(AXES):
        a.plot(t, np.abs(hd[:, i]), lw=.8, color=COL[k], alpha=.85,
               label=f'$|\\dot H_{{s,{k}}}|$')
    a.axhline(TAU_W_MAX, color='k', ls='--', lw=1.2,
              label=f'$\\tau_{{w,max}}$ = {TAU_W_MAX}')
    hdm = hd[np.isfinite(hd).all(1)]
    txt = '  '.join(f'{k}: {int((np.abs(hdm[:,i])>TAU_W_MAX-0.1).sum())}'
                    for i, k in enumerate(AXES))
    a.annotate(f'ticks at the cap — {txt}  (of {len(hdm)})',
               xy=(.98, .04), xycoords='axes fraction', ha='right', fontsize=8,
               bbox=dict(fc='w', ec='0.7', alpha=.9))
    a.set(title='2. realized rate is ALREADY clipped on z',
          xlabel='t [s]', ylabel='$|\\dot H_s|$ [Nm]')

    # 3. margin in saturated seconds
    a = ax[1, 0]
    pk = [float(np.nanmax(np.abs(hw[:, i]))) for i in range(3)]
    room = [H_MAX - p for p in pk]
    secs = [r / TAU_W_MAX for r in room]
    b = a.bar(list(AXES), secs, color=[COL[k] for k in AXES], alpha=.85)
    a.axhline(T, color='k', ls='--', lw=1.2, label=f'horizon = {T} s')
    for r, s, p in zip(b, secs, pk):
        a.text(r.get_x() + r.get_width() / 2, s + .04,
               f'{s:.2f} s\n(peak {p:.2f})', ha='center', fontsize=8)
    a.set(title='3. box margin, in seconds of SATURATED $\\tau_w$',
          ylabel='seconds', ylim=(0, max(secs) * 1.35))

    # 4. reconstruction error vs margin
    a = ax[1, 1]
    try:
        import mujoco
        I = np.asarray(mujoco.MjModel.from_xml_path(MJCF).body_inertia[1], float)
        f = np.isfinite(om).all(1) & np.isfinite(t)
        tt, ww = t[f], om[f]
        worst = np.zeros(3)
        for i in range(len(tt)):
            j = np.searchsorted(tt, tt[i] + T)
            if j < len(tt):
                worst = np.maximum(worst, np.abs(ww[j] - ww[i]) * I)
        xx = np.arange(3)
        a.bar(xx - .2, room, .4, label='box margin', color='0.7')
        a.bar(xx + .2, worst, .4, label='$|I_s\\,\\Delta\\omega_s|$ over one horizon',
              color=[COL[k] for k in AXES], alpha=.9)
        for i, k in enumerate(AXES):
            a.text(i + .2, worst[i] + .05, f'{100*worst[i]/room[i]:.0f}%',
                   ha='center', fontsize=9, fontweight='bold')
        a.set_xticks(xx); a.set_xticklabels(list(AXES))
        a.set(title='4. reconstruction error vs the margin it must fit inside',
              ylabel='Nms')
    except Exception as exc:
        a.text(.5, .5, f'inertia unavailable: {exc}', ha='center')

    for x in ax.flat:
        x.grid(alpha=.3)
        x.legend(fontsize=7, loc='best')
    fig.tight_layout()
    fig.savefig(OUT, dpi=130, bbox_inches='tight')
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
