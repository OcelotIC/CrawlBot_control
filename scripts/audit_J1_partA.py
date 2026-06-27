#!/usr/bin/env python3
"""DS-Active J1 — Part A: conservation residual (A.1-A.3) + centroidal reduction
identity / Lemma 2 (A.4). READ-ONLY on the canonical sim_log.

Conventions established from source:
- L_com,r_com,v_com: Pinocchio fixed-base (base=structure) ⇒ structure frame R_s,
  r_com relative to O_s (struct origin, ipos=0). (transport term ω_s×H at
  sim_loop:2917 confirms R_s rotating frame.)
- omega_s,hw: structure body frame R_s (qvel[3:6]; wheel momentum).
- I_s: principal [1777,1493,597] in the iquat-rotated frame → tensor in R_s body
  frame = R_i diag R_iᵀ.   m_robot=72.556 kg.
- lambda (12,) = [f_A, m_A, f_B, m_B], contact wrench ON the robot, R_s frame
  (compute_momentum_map: L̇_com = Σ_j (r_Cj−r_com)×f_j + τ_j = M_λ λ).
- L_dot/transport/H_dot_est are 0 in inter-step DS (sim_loop:923/992/956) → L̇_com,
  v̇_com finite-differenced with a boundary guard there.
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
GUARD = 2   # ticks dropped at each window edge for finite-difference rates


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


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def main():
    sl = json.load(open(os.path.join(RES, 'sim_log.json')))
    rows = list(csv.DictReader(open(os.path.join(RES, 'postproc_F3F4.csv'))))
    t = np.array(sl['t']); ph = np.array(sl['phase'])
    Lc = np.array(sl['L_com']); rc = np.array(sl['r_com']); vc = np.array(sl['v_com'])
    om = np.array(sl['omega_s']); hw = np.array(sl['hw'])
    lam_qp = np.array(sl['lambda_qp']); lam_ref = np.array(sl['lambda_ref'])
    Lcr = np.array(sl['L_com_ref'])
    sq = np.array(sl['struct_quat'])
    sa = np.array([r['stance_a_idx'] for r in rows]); sb = np.array([r['stance_b_idx'] for r in rows])

    # structure inertia tensor in R_s body frame (iquat-rotated principal)
    m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, 'models', 'VISPA_crawling_rwa3.xml'))
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'structure')
    Iprinc = m.body_inertia[sid].copy()
    iq = m.body_iquat[sid]  # wxyz
    R_i = pin.Quaternion(iq[0], iq[1], iq[2], iq[3]).toRotationMatrix()
    I_s = R_i @ np.diag(Iprinc) @ R_i.T

    # anchors in R_s (replicate postproc: R_s0ᵀ(p_anchor_world − p_s0))
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    p_s0 = d.xpos[sid].copy(); R_s0 = d.xmat[sid].reshape(3, 3).copy()
    anchors = {}
    for i in range(1, 20):
        for arm in ('a', 'b'):
            site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f'anchor_{i}{arm}')
            if site >= 0:
                anchors[(i, arm)] = R_s0.T @ (d.site_xpos[site].copy() - p_s0)

    # R_struct(t) for inertial-frame check
    def Rstruct(k):
        return pin.Quaternion(sq[k, 0], sq[k, 1], sq[k, 2], sq[k, 3]).toRotationMatrix()

    ds_segs = contiguous(ph == 'DS')
    roles = ['initial'] + [f'post-step{i}' for i in range(len(ds_segs) - 2)] + ['TERMINAL']

    print('## J1 Part A — conservation (A.1-A.3) + reduction Lemma 2 (A.4). DS windows.\n')
    print('=== A.1-A.3: H_tot/Os = L_com + r_com×m·v_com + I_s·ω_s + h_w ; R=H−c_ref (R_s frame) ===')
    print(f'{"window":>12} {"n":>4} {"‖c_ref‖":>9} {"max|R|/‖c‖":>11} {"mean|R|/‖c‖":>12} '
          f'{"max|R|_inert/‖c‖":>16} {"corr(|R|,|ω|)":>13}')
    cons = []
    for (a, b), role in zip(ds_segs, roles):
        idx = np.arange(a, b + 1)
        H = (Lc[idx] + np.cross(rc[idx], M_ROBOT * vc[idx])
             + (I_s @ om[idx].T).T + hw[idx])
        cref = H[0]
        R = H - cref
        Rn = np.linalg.norm(R, axis=1); cn = np.linalg.norm(cref)
        # inertial frame: rotate each H by R_struct(t), residual vs start
        Hin = np.array([Rstruct(k) @ H[i] for i, k in enumerate(idx)])
        Rin = Hin - Hin[0]
        omn = np.linalg.norm(om[idx], axis=1)
        corr = float(np.corrcoef(Rn, omn)[0, 1]) if np.std(Rn) > 1e-12 and np.std(omn) > 1e-12 else float('nan')
        print(f'{role:>12} {len(idx):>4} {cn:>9.3f} {Rn.max()/cn:>11.4f} {Rn.mean()/cn:>12.4f} '
              f'{np.linalg.norm(Rin,axis=1).max()/cn:>16.4f} {corr:>13.2f}')
        # term magnitudes (at window mid) for attribution
        mid = len(idx) // 2; k = idx[mid]
        terms = dict(L_com=np.linalg.norm(Lc[k]),
                     orbital=np.linalg.norm(np.cross(rc[k], M_ROBOT * vc[k])),
                     I_s_om=np.linalg.norm(I_s @ om[k]),
                     h_w=np.linalg.norm(hw[k]))
        cons.append((role, terms, cn, Rn.max() / cn))
    print('  term magnitudes at mid-window (Nms): ' )
    for role, terms, cn, _ in cons:
        print(f'    {role:>12}: ' + ' '.join(f'{kk}={vv:.3f}' for kk, vv in terms.items()))

    # ---- A.4: reduction identity ----
    dt = float(np.median(np.diff(t)))
    print('\n=== A.4: centroidal reduction — wrench-implied rate vs realized/planned rate ===')
    print('  angular: M_λ λ  vs  L̇_com   |   linear: Σf  vs  m·v̇_com   (per DS window)')
    print(f'{"window":>12} {"side":>5} {"ang_resid_med":>13} {"ang_resid_pk":>12} {"‖L̇‖_ref":>9} '
          f'{"lin_resid_med":>13} {"‖Σf‖":>8}')
    for (a, b), role in zip(ds_segs, roles):
        idx = np.arange(a, b + 1)
        if len(idx) < 2 * GUARD + 3:
            print(f'{role:>12}  (too short, n={len(idx)})'); continue
        # stance anchors for this window (from F3F4, mid tick)
        kmid = idx[len(idx) // 2]
        try:
            ia = int(sa[kmid]); ib = int(sb[kmid])
        except Exception:
            ia = ib = -1
        # which arm is at which anchor: F3F4 stance_a_idx/stance_b_idx
        rA = anchors.get((ia, 'a')) if (ia, 'a') in anchors else anchors.get((ia, 'b'))
        rB = anchors.get((ib, 'b')) if (ib, 'b') in anchors else anchors.get((ib, 'a'))
        have_anch = rA is not None and rB is not None
        for side, lam, Lref, useFD in (('plan', lam_ref, Lcr, True), ('exec', lam_qp, None, True)):
            # rate (realized for exec via FD of Lc/vc; planned for plan via FD of Lcr)
            g = slice(idx[0] + GUARD, idx[-1] - GUARD + 1)
            Lsrc = Lcr if side == 'plan' else Lc
            Ldot = np.gradient(Lsrc[idx], dt, axis=0)[GUARD:len(idx) - GUARD]
            vdot = np.gradient(vc[idx], dt, axis=0)[GUARD:len(idx) - GUARD]
            lamw = lam[idx][GUARD:len(idx) - GUARD]
            rcw = rc[idx][GUARD:len(idx) - GUARD]
            # linear: Σf vs m v̇_com
            fnet = lamw[:, 0:3] + lamw[:, 6:9]
            lin_res = np.linalg.norm(fnet - M_ROBOT * vdot, axis=1)
            # angular: M_λ λ vs L̇_com   (M_λ λ = Σ_j (r_Cj−r_com)×f_j + τ_j)
            if have_anch:
                mlam = np.empty((len(rcw), 3))
                for i in range(len(rcw)):
                    mlam[i] = (np.cross(rA - rcw[i], lamw[i, 0:3]) + lamw[i, 3:6]
                               + np.cross(rB - rcw[i], lamw[i, 6:9]) + lamw[i, 9:12])
                ang_res = np.linalg.norm(mlam - Ldot, axis=1)
                angpk = ang_res.max(); angmed = np.median(ang_res)
            else:
                angmed = angpk = float('nan')
            print(f'{role:>12} {side:>5} {angmed:>13.4f} {angpk:>12.4f} '
                  f'{np.linalg.norm(Ldot,axis=1).mean():>9.4f} {np.median(lin_res):>13.4f} '
                  f'{np.linalg.norm(fnet,axis=1).mean():>8.3f}')
        if not have_anch:
            print(f'             [{role}: anchors a={ia} b={ib} not both reconstructable]')


if __name__ == '__main__':
    main()
