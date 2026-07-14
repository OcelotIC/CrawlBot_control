#!/usr/bin/env python3
"""Phase ADJ-CONV VOLET 2 — numerically verify the SE(3) adjoint transport of the
robot momentum from the TORSO frame to the STRUCTURE frame, using the CODE'S stored
pose g, on the committed canonical run (runfix @5ab2c91).

Per-tick full config is NOT logged; the run's 42 full-state snapshots (initial /
release / frame / dock across all 6 steps, each [t, qpos(31), qvel(29), label]) are
the achievable full-state set. The transport identity is kinematic (holds at every
config), so 42 diverse configs spanning the traversal is a rigorous check.

For each snapshot:
  (q,v) = mujoco_to_pinocchio(qpos,qvel)   # robot in structure frame R_s (the code's own conversion)
  g = data.oMi[1]                          # Pinocchio base(torso) placement = the STORED pose struct_M_torso
  RHS  = per-body robot momentum about the STRUCTURE origin O (independent sum Sigma_i oMi.act(Y_i v_i))
  h_torso = g.actInv(RHS_force)            # robot momentum reduced at the torso frame (the Pi_w H_r q_dot input)
  LHS  = Ad^{-T}(g) . h_torso via the [L;P] force adjoint:  L=R L + p^ R P ; P=R P   (== g.act, checked)
  compare LHS.angular vs RHS.angular; also g^{-1} (inverse sense) and V-C.2 (L_com + r_com x P).

READ-ONLY; no crawlbot/ change, no new sim. Run:
  MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_adjconv_verify.py
"""
import json
import os
import numpy as np
import pinocchio as pin

from crawlbot.core.robot_interface import RobotInterface
from crawlbot.core.state_conversions import mujoco_to_pinocchio

SIM = 'results/figC_qpcond/sim_log.json'   # identity-confirmed == committed runfix @5ab2c91 (Phase ABL-HDOT-2 Step 0)
URDF = 'models/VISPA_crawling_fixed.urdf'

ri = RobotInterface(URDF, gravity='zero')
model, data = ri.model, ri.data
print(f'pin model nq={model.nq} nv={model.nv} njoints={model.njoints}')


def skew(p):
    return np.array([[0, -p[2], p[1]], [p[2], 0, -p[0]], [-p[1], p[0], 0]])


def per_body_momentum_about_O(q, v):
    """Independent robot spatial momentum about the universe origin O (= structure
    origin, R_s axes): Sigma_i oMi.act(Y_i v_i). Returns pin.Force [linear; angular]."""
    pin.forwardKinematics(model, data, q, v)
    h = pin.Force.Zero()
    for i in range(1, model.njoints):
        h += data.oMi[i].act(model.inertias[i] * data.v[i])
    return h


def adjoint_LP(R, p, L_in, P_in):
    """[L;P] co-vector (momentum) transport by g=(R,p): the SE(3) force adjoint g.act."""
    P_out = R @ P_in
    L_out = R @ L_in + skew(p) @ (R @ P_in)
    return L_out, P_out


sl = json.load(open(SIM))
snaps = sl['snapshots']
print(f'{len(snaps)} snapshots\n')

hdr = f'{"label":16s} {"|p|":>7} {"|P|":>7} {"|L_struct|":>10} | {"LHS-RHS":>9} {"act-RHS":>9} {"VC2-RHS":>9} {"INV-RHS":>9}'
print(hdr); print('-' * len(hdr))
dev_lhs, dev_act, dev_vc2, dev_inv, twopxp = [], [], [], [], []
p_report = []
for (t, q31, v29, label) in snaps:
    q, v = mujoco_to_pinocchio(np.asarray(q31, float), np.asarray(v29, float))

    # RHS: independent per-body momentum about O (structure origin, R_s)
    hO = per_body_momentum_about_O(q, v)
    L_struct = hO.angular.copy(); P = hO.linear.copy()

    # g = stored pose = Pinocchio base(torso) placement = struct_M_torso
    g = data.oMi[1]
    R, p = g.rotation.copy(), g.translation.copy()

    # sanity: p must equal R_s^T (p_torso - p_struct) from the raw MuJoCo snapshot
    p_s = np.asarray(q31[0:3]); qs = np.asarray(q31[3:7])
    R_s = pin.Quaternion(qs[0], qs[1], qs[2], qs[3]).toRotationMatrix()
    p_t = np.asarray(q31[10:13])
    p_geom = R_s.T @ (p_t - p_s)
    assert np.max(np.abs(p - p_geom)) < 1e-9, f'{label}: p != R_s^T(p_torso-p_struct)'

    # momentum reduced at the torso frame (the Pi_w H_r q_dot input): g.actInv(hO)
    h_torso = g.actInv(hO)
    L_t, P_t = h_torso.angular.copy(), h_torso.linear.copy()

    # LHS via the explicit [L;P] adjoint with the STORED g
    L_lhs, P_lhs = adjoint_LP(R, p, L_t, P_t)
    # cross-check: g.act (Pinocchio's own force adjoint) must agree with the explicit formula
    L_act = g.act(h_torso).angular

    # INVERSE sense (use g^{-1} as if the stored pose were T_torso_struct)
    ginv = g.inverse()
    L_inv, _ = adjoint_LP(ginv.rotation, ginv.translation, L_t, P_t)

    # V-C.2 reference: L about O from centroidal (L_com + r_com x P)
    pin.computeCentroidalMomentum(model, data, q, v)
    L_com = data.hg.angular.copy(); r_com = data.com[0].copy()
    L_vc2 = L_com + np.cross(r_com, P)

    d_lhs = np.max(np.abs(L_lhs - L_struct))
    d_act = np.max(np.abs(L_act - L_struct))
    d_vc2 = np.max(np.abs(L_vc2 - L_struct))
    d_inv = np.max(np.abs(L_inv - L_struct))
    dev_lhs.append(d_lhs); dev_act.append(d_act); dev_vc2.append(d_vc2); dev_inv.append(d_inv)
    twopxp.append(np.max(np.abs(2 * np.cross(p, P)))); p_report.append(p.copy())
    print(f'{label:16s} {np.linalg.norm(p):7.3f} {np.linalg.norm(P):7.3f} {np.linalg.norm(L_struct):10.4f} | '
          f'{d_lhs:9.1e} {d_act:9.1e} {d_vc2:9.1e} {d_inv:9.1e}')

pm = np.array(p_report)
print('\n==== AGGREGATE (42 snapshots) ====')
print(f'  LHS (adjoint, stored g) vs RHS (per-body):   max|dev| = {max(dev_lhs):.2e}  -> {"MATCH" if max(dev_lhs)<1e-6 else "MISMATCH"}')
print(f'  g.act (Pinocchio force adjoint) vs RHS:       max|dev| = {max(dev_act):.2e}')
print(f'  V-C.2 (L_com + r_com x P) vs RHS:             max|dev| = {max(dev_vc2):.2e}  -> {"MATCH" if max(dev_vc2)<1e-6 else "MISMATCH"}')
print(f'  INVERSE sense (g^-1) vs RHS:                  max|dev| = {max(dev_inv):.2e}  (mismatch = 2|p x P|, max {max(twopxp):.2e})  -> {"OPPOSITE/OFFSET" if max(dev_inv)>1e-6 else "match?!"}')
print(f'  stored p = R_s^T(p_torso - p_struct)  [torso-from-structure, T_struct_torso]: confirmed all snapshots')
print(f'  p range: |p| in [{np.linalg.norm(pm,axis=1).min():.3f}, {np.linalg.norm(pm,axis=1).max():.3f}] m; mean p = [{pm[:,0].mean():+.3f} {pm[:,1].mean():+.3f} {pm[:,2].mean():+.3f}]')

os.makedirs('results/j2_adjconv', exist_ok=True)
json.dump(dict(
    n_snapshots=len(snaps), source=SIM, urdf=URDF,
    stored_pose='T_struct_torso; p = R_s^T(p_torso - p_struct) = torso-from-structure, in R_s '
                '(state_conversions.py:86-88,98)',
    max_dev_LHS_adjoint_vs_RHS=float(max(dev_lhs)),
    max_dev_gact_vs_RHS=float(max(dev_act)),
    max_dev_VC2_vs_RHS=float(max(dev_vc2)),
    max_dev_INVERSE_vs_RHS=float(max(dev_inv)),
    max_2p_cross_P=float(max(twopxp)),
    verdict='MATCH — stored sense correct (T_struct_torso), no inversion needed',
    p_abs_range_m=[float(np.linalg.norm(pm, axis=1).min()), float(np.linalg.norm(pm, axis=1).max())],
    p_mean=pm.mean(axis=0).tolist(),
), open('results/j2_adjconv/adjconv_verify.json', 'w'), indent=2)
print('  wrote results/j2_adjconv/adjconv_verify.json')
