#!/usr/bin/env python3
"""Dock-leak Part 3 figure: H_sys staircase under Fix A vs frozen canonical."""
import json
import os
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), '..')
MJCF = os.path.join(ROOT, 'models', 'VISPA_crawling_rwa3.xml')
FIXA = os.path.join(ROOT, 'results', 'fixA_gate')
CANON = os.path.join(ROOT, 'results', 'p3b_gate_w24000_kp3')
OUT = os.path.join(FIXA, 'fixA_plots')
os.makedirs(OUT, exist_ok=True)


def series(path):
    sl = json.load(open(os.path.join(path, 'sim_log.json')))
    m = mujoco.MjModel.from_xml_path(MJCF)
    d = mujoco.MjData(m)
    ts, Hs = [], []
    for (t, q, v, lbl) in sl['snapshots']:
        d.qpos[:] = np.array(q); d.qvel[:] = np.array(v)
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        mujoco.mj_comVel(m, d); mujoco.mj_subtreeVel(m, d)
        ts.append(t); Hs.append(np.linalg.norm(d.subtree_angmom[0]))
    order = np.argsort(ts)
    return np.array(ts)[order], np.array(Hs)[order]


def main():
    tf, Hf = series(FIXA)
    tc, Hc = series(CANON)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(tc, Hc, 'o-', color='C3', lw=1.6, label='canonical (current impact map)')
    ax.plot(tf, Hf, 's-', color='C0', lw=1.6, label='Fix A (full-DOF impact map)')
    ax.axhline(0.203, color='C3', ls=':', lw=0.8)
    ax.axhline(0.004, color='C0', ls=':', lw=0.8)
    ax.set_xlabel('time [s]'); ax.set_ylabel(r'$\|H_{sys}\|$  [N·m·s]')
    ax.set_title('Total system angular-momentum leak across the 5-step traversal\n'
                 'Fix A (full-DOF impact map): 0.203 → 0.004 N·m·s (51×); '
                 'residual = the un-touched DS weld gap-couple')
    ax.legend(loc='center left'); ax.grid(alpha=0.3)
    ax.annotate('canonical final 0.203', (tc[-1], Hc[-1]),
                textcoords='offset points', xytext=(-120, -4), fontsize=9, color='C3')
    ax.annotate('Fix A final 0.004', (tf[-1], Hf[-1]),
                textcoords='offset points', xytext=(-100, 12), fontsize=9, color='C0')
    fig.tight_layout()
    p = os.path.join(OUT, 'dock_leak_fixA.png')
    fig.savefig(p, dpi=130)
    print('wrote', p)


if __name__ == '__main__':
    main()
