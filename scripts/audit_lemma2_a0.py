#!/usr/bin/env python3
"""J1 closing — Lemma 2 (A.0): data availability + model structure.

Checks what the ae0673e gate run (results/fixA_gate) logs per-tick, and maps the
MuJoCo body tree / robot subtree / masses / SS windows needed for the inertial
reduction test. READ-ONLY.
"""
import json
import os
import numpy as np
import mujoco

ROOT = os.path.join(os.path.dirname(__file__), '..')
MJCF = os.path.join(ROOT, 'models', 'VISPA_crawling_rwa3.xml')
GATE = os.path.join(ROOT, 'results', 'fixA_gate')


def main():
    sl = json.load(open(os.path.join(GATE, 'sim_log.json')))
    print('=== A.0 — per-tick log availability (results/fixA_gate) ===')
    n = len(sl['t'])
    print(f'  per-tick rows: {n}')
    need = {
        'inertial robot-subtree angmom about O_s': None,   # not a single key
        'per-tick qpos/qvel (to recompute subtree)': 'qpos_full' in sl or 'qvel_full' in sl,
        'plant weld wrench efc_force': 'efc_force' in sl,
        'qfrc_constraint (full nv)': 'qfrc_constraint' in sl,
        'qfrc_constraint_torque (struct[3:6] only)': 'qfrc_constraint_torque' in sl,
        'lambda_qp (QP command wrench)': 'lambda_qp' in sl,
        'L_com (STRUCTURE-RELATIVE — wrong frame for Lemma2)': 'L_com' in sl,
    }
    for k, v in need.items():
        print(f'    {k:<52} : {v}')
    print('  snapshot labels:', [s[3] for s in sl['snapshots']])
    print('  => per-tick qpos/qvel NOT logged (snapshots only); efc_force/qfrc_constraint(full) NOT logged.')
    print('     => instrument per-tick (Part-1 pattern) required for both LHS and RHS.\n')

    # ---- model structure ----
    m = mujoco.MjModel.from_xml_path(MJCF)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'structure')
    tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'torso')
    print('=== body tree (id: name [parent] mass) ===')
    for b in range(m.nbody):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
        par = m.body_parentid[b]
        pnm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, par)
        mark = '  <-- structure' if b == sid else ('  <-- torso(robot root)' if b == tid else '')
        print(f'  {b:>2}: {str(nm):<16} [{str(pnm):<12}] m={m.body_mass[b]:7.3f}{mark}')

    # robot subtree = bodies whose ancestor chain hits torso (incl torso)
    def is_desc(b, anc):
        while b != 0:
            if b == anc:
                return True
            b = m.body_parentid[b]
        return False
    robot_bodies = [b for b in range(m.nbody) if is_desc(b, tid)]
    struct_bodies = [b for b in range(m.nbody) if is_desc(b, sid)]
    m_R = float(sum(m.body_mass[b] for b in robot_bodies))
    m_S = float(sum(m.body_mass[b] for b in struct_bodies))
    print(f'\n  robot subtree (root=torso {tid}): {len(robot_bodies)} bodies, m_R = {m_R:.3f} kg')
    print(f'  structure subtree (root={sid}): {len(struct_bodies)} bodies, m_S = {m_S:.3f} kg')
    print(f'  total system mass = {m_R + m_S:.3f} kg (A\'.1 used M_ROBOT=72.556)')

    # subtree-momentum readout sanity (rest state)
    mujoco.mj_comPos(m, d); mujoco.mj_comVel(m, d); mujoco.mj_subtreeVel(m, d)
    print(f'\n  subtree_angmom[torso] at rest = {np.round(d.subtree_angmom[tid],6)} (≈0 expected)')
    print(f'  O_s = xipos[structure] (world CoM) = {np.round(d.xipos[sid],4)}')
    print(f'  robot subtree CoM subtree_com[torso] = {np.round(d.subtree_com[tid],4)}')

    # ---- SS windows + rates (from the logged phase + struct-rel L_com norm as a proxy) ----
    ph = np.array(sl['phase']); t = np.array(sl['t'])
    def contig(mask):
        segs, i = [], 0
        while i < len(mask):
            if mask[i]:
                j = i
                while j < len(mask) and mask[j]:
                    j += 1
                segs.append((i, j-1)); i = j
            else:
                i += 1
        return segs
    ss = contig(ph == 'SS')
    print(f'\n=== SS windows (phase==SS), {len(ss)} total ===')
    Ln = np.array(sl['L_com_norm'])
    print(f'  {"win":>3} {"ticks":>11} {"t[s]":>14} {"npts":>5} {"dt[s]":>7} {"mean|L_com|(struct-rel proxy)":>30}')
    for k, (a, b) in enumerate(ss):
        seg_t = t[a:b+1]
        dt = float(np.median(np.diff(seg_t))) if b > a else float('nan')
        print(f'  {k:>3} {f"{a}-{b}":>11} {f"{seg_t[0]:.2f}-{seg_t[-1]:.2f}":>14} {b-a+1:>5} '
              f'{dt:>7.3f} {np.mean(Ln[a:b+1]):>30.4f}')
    print('  (highest mean|L_com| swings = highest-rate; pick >=2 for the test)')


if __name__ == '__main__':
    main()
