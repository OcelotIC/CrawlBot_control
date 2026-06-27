#!/usr/bin/env python3
"""Dock-leak Part 2 figure: full-DOF vs current-partial impact map + velocity scaling."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from audit_dock_leak_part2 import (
    build_loop, partial_map, full_dof_map, H_sys, INSTR)


def main():
    sl = json.load(open(os.path.join(INSTR, 'sim_log.json')))
    snaps = {lbl: (np.array(q), np.array(v))
             for (t, q, v, lbl) in sl['snapshots']}
    sim = build_loop()
    m, d = sim.mj_model, sim.mj_data

    steps = list(range(5))
    a2, a1 = [], []
    alphas = [1.0, 0.75, 0.5, 0.25, 0.0]
    scale = {n: [] for n in steps}
    for n in steps:
        q, v = snaps[f'dock_step{n}']
        H0 = H_sys(m, d, q, v)
        qv2, _ = partial_map(sim, q, v, apply_impulse=True)
        a2.append(np.linalg.norm(H_sys(m, d, q, qv2) - H0))
        qv1, _, _, _ = full_dof_map(sim, q, v)
        a1.append(np.linalg.norm(H_sys(m, d, q, qv1) - H0))
        for al in alphas:
            va = al * v
            H0a = H_sys(m, d, q, va)
            qva, _ = partial_map(sim, q, va, apply_impulse=True)
            scale[n].append(np.linalg.norm(H_sys(m, d, q, qva) - H0a))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(len(steps)); w = 0.38
    ax0.bar(x - w / 2, a2, w, label='current partial map (robot-only, partial write-back)',
            color='C3')
    ax0.bar(x + w / 2, a1, w, label='full-DOF momentum-consistent map (Fix A)', color='C0')
    ax0.set_yscale('log')
    ax0.set_xticks(x); ax0.set_xticklabels([f'dock {n}' for n in steps])
    ax0.set_ylabel(r'$\|\Delta H_{sys}\|$ [N·m·s]  (log)')
    ax0.set_title('Test A — full-DOF map collapses the leak (110×–2275×)\n'
                  'down to the O(gap×f) residual')
    ax0.legend(fontsize=8, loc='lower left')
    ax0.grid(alpha=0.3, axis='y', which='both')
    for xi, (v2, v1) in enumerate(zip(a2, a1)):
        ax0.annotate(f'{v2/v1:.0f}×', (xi, max(v2, v1)),
                     textcoords='offset points', xytext=(0, 3),
                     ha='center', fontsize=8)

    for n in steps:
        ax1.plot(alphas, scale[n], 'o-', label=f'dock {n}', lw=1.3)
    ax1.set_xlabel(r'pre-impact velocity scale $\alpha$  (gap held fixed)')
    ax1.set_ylabel(r'$\|\Delta H_{sys}\|$ [N·m·s]')
    ax1.set_title('Test B.2 — the leak is VELOCITY-driven, not gap-driven\n'
                  r'$\Delta H \propto \alpha$, $\to 0$ at zero approach velocity')
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ax1.set_xlim(-0.05, 1.05)

    fig.tight_layout()
    out = os.path.join(INSTR, 'dockleak_plots', 'dock_leak_part2.png')
    fig.savefig(out, dpi=130)
    print('wrote', out)


if __name__ == '__main__':
    main()
