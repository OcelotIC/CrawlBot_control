#!/usr/bin/env python3
"""Dock-leak Part 3 (3.2) — full-traversal leak under Fix A vs frozen canonical.

Computes subtree_angmom[0] at each logged snapshot (initial / release_stepN /
dock_stepN / final) for the Fix-A run and the frozen canonical run, side by side.
READ-ONLY static eval (set qpos/qvel -> mj_subtreeVel). Metric = subtree_angmom[0].
"""
import json
import os
import sys
import numpy as np
import mujoco

ROOT = os.path.join(os.path.dirname(__file__), '..')
MJCF = os.path.join(ROOT, 'models', 'VISPA_crawling_rwa3.xml')
FIXA = os.path.join(ROOT, 'results', 'fixA_gate')
CANON = os.path.join(ROOT, 'results', 'p3b_gate_w24000_kp3')


def H_at_snapshots(path):
    sl = json.load(open(os.path.join(path, 'sim_log.json')))
    m = mujoco.MjModel.from_xml_path(MJCF)
    d = mujoco.MjData(m)
    out = []
    for (t, q, v, lbl) in sl['snapshots']:
        d.qpos[:] = np.array(q); d.qvel[:] = np.array(v)
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        mujoco.mj_comVel(m, d); mujoco.mj_subtreeVel(m, d)
        out.append((lbl, float(t), d.subtree_angmom[0].copy()))
    return out


def main():
    fixa = H_at_snapshots(FIXA)
    canon = H_at_snapshots(CANON)
    cmap = {lbl: H for (lbl, t, H) in canon}

    print('Dock-leak Part 3 (3.2) — full-traversal leak: Fix A vs frozen canonical\n')
    print(f'{"snapshot":>16} {"t[s]":>6} {"|H_sys| Fix A":>13} {"|H_sys| canon":>14} '
          f'{"H_z Fix A":>10} {"H_z canon":>10}')
    for (lbl, t, H) in fixa:
        Hc = cmap.get(lbl, np.full(3, np.nan))
        print(f'{lbl:>16} {t:>6.2f} {np.linalg.norm(H):>13.4f} '
              f'{np.linalg.norm(Hc):>14.4f} {H[2]:>10.4f} {Hc[2]:>10.4f}')

    fin_fixa = np.linalg.norm(fixa[-1][2])
    fin_canon = np.linalg.norm(cmap.get('final', np.full(3, np.nan)))
    print(f'\n  FINAL |H_sys|:  Fix A = {fin_fixa:.4f}   canonical = {fin_canon:.4f}   '
          f'reduction = {fin_canon/fin_fixa:.0f}x' if fin_fixa > 1e-9 else '')
    print(f'  Expected: Fix A ~0.001 (A.1 residual), >=110x below canonical 0.203')

    # per-dock injection: H at dock_stepN -> release_step(N+1) (H conserved in
    # DS dwell + SS swing, Part 1), i.e. the step-N impact contribution.
    print('\n  per-dock injection (|H(release_{N+1}) - H(dock_N)|), Fix A vs canon:')
    fmap = {lbl: H for (lbl, t, H) in fixa}
    print(f'  {"dock N":>7} {"Fix A ΔH":>10} {"canon ΔH":>10}  (canon Part-1: '
          f'.041/.048/.049/.146/.073)')
    for n in range(5):
        dock = f'dock_step{n}'; rel = f'release_step{n+1}'
        if dock in fmap and rel in fmap:
            dfa = np.linalg.norm(fmap[rel] - fmap[dock])
            dca = np.linalg.norm(cmap[rel] - cmap[dock]) if (rel in cmap and dock in cmap) else float('nan')
            print(f'  {n:>7} {dfa:>10.4f} {dca:>10.4f}')
        else:
            print(f'  {n:>7}   (snapshots missing — traversal may have aborted)')


if __name__ == '__main__':
    main()
