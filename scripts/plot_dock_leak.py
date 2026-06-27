#!/usr/bin/env python3
"""Dock-leak Part 1 figure: H_sys staircase + per-dock sub-event attribution.

Top:    |H_sys|(t) with SS/DS (n_weld) shading and dock markers — the leak is a
        staircase: flat (conserved) within SS swings AND within DS dwell, with a
        single-tick step at each dock engagement.
Bottom: per-dock attribution bars — weld-activation vs inelastic impact
        projection vs gap-closing stabilization. The impact projection dominates.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), '..')
INSTR = os.path.join(ROOT, 'results', 'dockleak_instrumented')
OUT = os.path.join(INSTR, 'dockleak_plots')
os.makedirs(OUT, exist_ok=True)


def main():
    sl = json.load(open(os.path.join(INSTR, 'sim_log.json')))
    t = np.array(sl['t']); H = np.array(sl['H_sys']); nw = np.array(sl['n_weld'])
    Hn = np.linalg.norm(H, axis=1)
    pr = sl['dock_probe']
    docks = [i for i in range(1, len(nw)) if nw[i] == 2 and nw[i - 1] == 1]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 8),
                                   gridspec_kw={'height_ratios': [2, 1]})

    # ── top: H_sys staircase ──
    ax0.plot(t, Hn, '-', lw=1.4, color='C3', label=r'$\|H_{sys}\|$ (total ang. mom.)')
    ax0.plot(t, H[:, 0], '-', lw=0.7, alpha=0.5, color='C0', label='$H_x$')
    ax0.plot(t, H[:, 1], '-', lw=0.7, alpha=0.5, color='C1', label='$H_y$')
    ax0.plot(t, H[:, 2], '-', lw=0.7, alpha=0.5, color='C2', label='$H_z$')
    # shade DS (n_weld==2) windows
    i = 0
    lab = True
    while i < len(nw):
        j = i
        while j < len(nw) and nw[j] == nw[i]:
            j += 1
        if nw[i] == 2:
            ax0.axvspan(t[i], t[min(j, len(t) - 1) - 1], color='0.85',
                        label='DS (2 welds)' if lab else None)
            lab = False
        i = j
    for k, di in enumerate(docks):
        ax0.axvline(t[di], color='k', ls=':', lw=0.8)
        ax0.annotate(f'dock {pr[k]["step"]}', (t[di], Hn[di]),
                     textcoords='offset points', xytext=(3, 6), fontsize=8)
    ax0.axhline(0.0, color='k', lw=0.5)
    ax0.set_ylabel(r'angular momentum [N·m·s]')
    ax0.set_title(r'Total system angular momentum $H_{sys}=$subtree_angmom[0] '
                  r'(world) — free system, must be 0; leaks to 0.203 N·m·s')
    ax0.legend(loc='upper left', fontsize=8, ncol=2)
    ax0.grid(alpha=0.3)

    # ── bottom: per-dock attribution ──
    steps = [p['step'] for p in pr]
    d01 = [np.linalg.norm(np.array(p['H1']) - np.array(p['H0'])) for p in pr]
    d12 = [np.linalg.norm(np.array(p['H2']) - np.array(p['H1'])) for p in pr]
    d23 = [np.linalg.norm(np.array(H[di]) - np.array(p['H2']))
           for p, di in zip(pr, docks)]
    x = np.arange(len(steps)); w = 0.26
    ax1.bar(x - w, d01, w, label='weld activation (H0→H1)', color='C7')
    ax1.bar(x, d12, w, label='inelastic impact projection (H1→H2)', color='C3')
    ax1.bar(x + w, d23, w, label='gap-closing stabilization (H2→H3)', color='C0')
    ax1.set_xticks(x); ax1.set_xticklabels([f'dock {s}' for s in steps])
    ax1.set_ylabel(r'$|\Delta H|$ [N·m·s]')
    ax1.set_title(r'Per-dock sub-event attribution — the impact velocity '
                  r'projection injects essentially all of it')
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3, axis='y')
    for xi, v in zip(x, d12):
        ax1.annotate(f'{v:.3f}', (xi, v), textcoords='offset points',
                     xytext=(0, 2), ha='center', fontsize=7)

    fig.tight_layout()
    p = os.path.join(OUT, 'dock_leak_attribution.png')
    fig.savefig(p, dpi=130)
    print('wrote', p)


if __name__ == '__main__':
    main()
