#!/usr/bin/env python3
"""DS-Active J1 — Part A′: conservation frame resolution. READ-ONLY, STATIC.
A′.1 MuJoCo-direct total angular momentum (subtree_angmom[0], must be ≈0 for the
free system); A′.2 inertial decomposition (confirm robot co-rotation + I_s=1777);
A′.3 controller structure-relative invariant c = hw + L_com + r_com×m·v_com (logged).
No stepping, no controller, no sim re-run — set qpos/qvel on snapshots and read out.
"""
import csv
import json
import os
import numpy as np
import pinocchio as pin
import mujoco

ROOT = os.path.join(os.path.dirname(__file__), '..')
RES = os.path.join(ROOT, 'results', 'p3b_gate_w24000_kp3')
M_ROBOT = 72.556
RWA_I_W = 0.01


def contiguous(mask):
    segs, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            segs.append((i, j - 1)); i = j
        else:
            i += 1
    return segs


def main():
    sl = json.load(open(os.path.join(RES, 'sim_log.json')))
    snaps = sl['snapshots']
    m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, 'models', 'VISPA_crawling_rwa3.xml'))
    d = mujoco.MjData(m)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'structure')
    Ipr = m.body_inertia[sid].copy(); iq = m.body_iquat[sid]
    R_i = pin.Quaternion(iq[0], iq[1], iq[2], iq[3]).toRotationMatrix()
    I_s_body = R_i @ np.diag(Ipr) @ R_i.T

    print('## J1 Part A′ — conservation frame resolution (static, read-only)\n')
    print('=== A′.1 — MuJoCo-direct total system angular momentum (must be ≈0; free system) ===')
    print('  route: set qpos/qvel → mj_kinematics, mj_comPos, mj_comVel, mj_subtreeVel → subtree_angmom[0]')
    print(f'{"snapshot":>16} {"t":>6} {"‖H_sys‖":>9} {"H_sys[xyz]":>26} {"‖v_comSys‖":>10}')
    rows_A1 = {}
    for (t, qpos, qvel, label) in snaps:
        d.qpos[:] = np.array(qpos); d.qvel[:] = np.array(qvel)
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        mujoco.mj_comVel(m, d); mujoco.mj_subtreeVel(m, d)
        H = d.subtree_angmom[0].copy(); vcom = d.subtree_linvel[0].copy()
        rows_A1[label] = (t, H, vcom, np.array(qvel))
        print(f'{label:>16} {t:>6.2f} {np.linalg.norm(H):>9.4f} '
              f'[{H[0]:+.3f},{H[1]:+.3f},{H[2]:+.3f}] {np.linalg.norm(vcom):>10.5f}')
    Hmax = max(np.linalg.norm(v[1]) for v in rows_A1.values())
    print(f'  -> max ‖H_sys‖ over all snapshots = {Hmax:.4f} N·m·s '
          f'({"≈0 — CONSERVATION HOLDS in plant" if Hmax < 0.05 else "NON-NEGLIGIBLE — FLAG"})')

    # ---- A′.2 — inertial decomposition at settled ticks ----
    print('\n=== A′.2 — inertial budget at settled ticks: H_sys = H_robot/Os + I_s·ω_s + h_w ===')
    # log lookup by time for structure-relative robot terms
    tlog = np.array(sl['t']); Lc = np.array(sl['L_com']); rc = np.array(sl['r_com'])
    vc = np.array(sl['v_com']); hwl = np.array(sl['hw'])
    for label in ('release_step1', 'release_step3', 'final'):
        if label not in rows_A1:
            continue
        t, H, vcom, qvel = rows_A1[label]
        # structure pose at this snapshot (qpos free-joint struct: pos[0:3], quat[3:7] wxyz)
        sq = [s for s in snaps if s[3] == label][0][1][3:7]
        R_s = pin.Quaternion(sq[0], sq[1], sq[2], sq[3]).toRotationMatrix()
        om_s = qvel[3:6]                          # structure ang vel (local body frame)
        hw_b = RWA_I_W * qvel[6:9]                # wheel momentum (local)
        H_struct_w = R_s @ (I_s_body @ om_s)      # → world
        hw_w = R_s @ hw_b
        H_robot_Os = H - H_struct_w - hw_w        # inertial robot momentum about ~O_s (H_sys≈0)
        # structure-relative logged robot momentum (should be ≈0 at settle)
        k = int(np.argmin(np.abs(tlog - t)))
        H_robot_srel = Lc[k] + np.cross(rc[k], M_ROBOT * vc[k])
        I_rob = np.linalg.norm(H_robot_Os) / np.linalg.norm(om_s) if np.linalg.norm(om_s) > 1e-9 else float('nan')
        print(f'  [{label} t={t:.2f}]  ‖ω_s‖={np.linalg.norm(om_s)*1e3:.3f} mrad/s')
        print(f'    I_s·ω_s (world)        = [{H_struct_w[0]:+.3f},{H_struct_w[1]:+.3f},{H_struct_w[2]:+.3f}] |{np.linalg.norm(H_struct_w):.3f}')
        print(f'    h_w (world)            = [{hw_w[0]:+.3f},{hw_w[1]:+.3f},{hw_w[2]:+.3f}] |{np.linalg.norm(hw_w):.3f}')
        print(f'    H_robot/Os (inertial)  = [{H_robot_Os[0]:+.3f},{H_robot_Os[1]:+.3f},{H_robot_Os[2]:+.3f}] |{np.linalg.norm(H_robot_Os):.3f}')
        print(f'    H_robot (struct-rel,log)= [{H_robot_srel[0]:+.3f},{H_robot_srel[1]:+.3f},{H_robot_srel[2]:+.3f}] |{np.linalg.norm(H_robot_srel):.3f}  (≈0 expected)')
        print(f'    co-rotation difference  = {np.linalg.norm(H_robot_Os - H_robot_srel):.3f} ; back-out I_robot/Os = {I_rob:.0f} kg·m²')

    # ---- A′.3 — controller structure-relative invariant c ----
    print('\n=== A′.3 — controller invariant c = h_w + L_com + r_com×m·v_com (struct-rel, logged) ===')
    ph = np.array(sl['phase'])
    ds_segs = contiguous(ph == 'DS')
    roles = ['initial'] + [f'post-step{i}' for i in range(len(ds_segs) - 2)] + ['TERMINAL']
    c_all = hwl + Lc + np.cross(rc, M_ROBOT * vc)
    print(f'{"window":>12} {"n":>4} {"‖c_ref‖":>9} {"max‖Δc‖":>9} {"mean‖Δc‖":>9} {"max‖Δc‖/‖c‖":>12}')
    for (a, b), role in zip(ds_segs, roles):
        idx = np.arange(a, b + 1)
        c = c_all[idx]; cref = c[0]
        dc = np.linalg.norm(c - cref, axis=1); cn = np.linalg.norm(cref)
        rel = dc.max() / cn if cn > 1e-9 else float('nan')
        print(f'{role:>12} {len(idx):>4} {cn:>9.4f} {dc.max():>9.4f} {dc.mean():>9.4f} {rel:>12.4f}')

    # ---- secondary: structure mass/inertia ----
    print('\n=== secondary — structure mass/inertia (model hygiene, report only) ===')
    print(f'  structure body_mass = {m.body_mass[sid]:.1f} kg ; principal inertia = {np.round(Ipr,1)} kg·m² ; iquat={np.round(iq,4)}')
    print(f'  radius of gyration ≈ {np.sqrt(Ipr.mean()/m.body_mass[sid]):.3f} m (sanity)')


if __name__ == '__main__':
    main()
