#!/usr/bin/env python3
"""C1.1 / C1.2 / C1.3 structural probe on the controller model (read-only).

Loads the canonical CONTROLLER model (models/VISPA_crawling_fixed.urdf, the one
gate/run_gate.py:46 names) and measures, at a non-trivial configuration:

  C1.1  Is the QP momentum task the LINEAR CMM rows A_{G,v}?
        -> compare J_com (what _com_task_rows assembles) against ccrba's
           linear block A_G[0:3,:] / m.
        Is L_com tracked by any Stage-2 COST task?
        -> static check on wholebody_qp.py.

  C1.3  Block structure of the task Jacobians in the column order
        [q_t(6) | arm A(7) | arm B(7)]:
            J_torso  expected [Jt 0_{6x7} 0_{6x7}]
            J_ee     expected [Jt 0 (stance) | J_q (swing)]

  C1.2  Frame consistency of the orientation error against the Jacobian
        convention: the tasks use pin.log3(R^T R_ref) (LOCAL / body frame)
        while J_torso and J_ee are LOCAL_WORLD_ALIGNED. Quantify the induced
        angular-error discrepancy ||(R - I) log3(R^T R_ref)|| at the torso
        attitudes the canonical run actually visits.

Writes results/review_closure/c1/c1_jacobian_probe.json
"""
import json
import os

import numpy as np
import pinocchio as pin

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
URDF = os.path.join(ROOT, 'models', 'VISPA_crawling_fixed.urdf')
OUT = os.path.join(ROOT, 'results', 'review_closure', 'c1')

from crawlbot.core.robot_interface import RobotInterface   # noqa: E402


def blocks(J, na, nb):
    """Split a (6, 6+na+nb) Jacobian into [base | armA | armB] and report norms."""
    return {
        'base': float(np.abs(J[:, :6]).max()),
        'armA': float(np.abs(J[:, 6:6 + na]).max()),
        'armB': float(np.abs(J[:, 6 + na:6 + na + nb]).max()),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    ri = RobotInterface(URDF, tau_max=20.0, gravity='zero')
    model, data = ri.model, ri.data
    nv = model.nv
    nj = ri.n_joints
    na = nb = nj // 2

    rng = np.random.default_rng(0)
    q = pin.neutral(model)
    qs = ri.joints_q_slice
    q[qs] = 0.35 * rng.standard_normal(len(q[qs]))
    # a non-identity floating-base attitude, ~8 deg, so R != I in the frame test
    ang = np.deg2rad(8.0)
    axis = np.array([0.3, -0.5, 0.81]); axis /= np.linalg.norm(axis)
    quat = pin.Quaternion(np.cos(ang / 2), *(np.sin(ang / 2) * axis))
    quat.normalize()
    q[3:7] = np.array([quat.x, quat.y, quat.z, quat.w])   # Pinocchio xyzw
    q[0:3] = np.array([0.12, -0.05, -0.31])
    v = 0.2 * rng.standard_normal(nv)

    ri.update(q, v)
    st = ri.state

    out = {'model': os.path.relpath(URDF, ROOT),
           'nq': int(model.nq), 'nv': int(nv), 'n_joints': int(nj),
           'total_mass_kg': float(st.total_mass)}

    # ---- C1.1 : is J_com the linear CMM rows / m ? -----------------------
    A_G = pin.computeCentroidalMap(model, data, q)       # (6, nv) [lin; ang]
    m = st.total_mass
    lin_from_cmm = A_G[0:3, :] / m
    out['C1_1'] = {
        'J_com_shape': list(st.J_com.shape),
        'A_G_shape': list(A_G.shape),
        'max_abs_diff_Jcom_vs_AGlin_over_m': float(
            np.abs(st.J_com - lin_from_cmm).max()),
        'max_abs_Jcom': float(np.abs(st.J_com).max()),
        'note': ('A_G[0:3,:] = m * J_com exactly => the QP momentum task rows '
                 'ARE the linear CMM rows A_{G,v}, scaled by 1/m'),
        # the angular CMM rows, for contrast: not assembled anywhere in the QP
        'A_G_angular_rows_max_abs': float(np.abs(A_G[3:6, :]).max()),
        'L_com_current': [float(x) for x in st.L_com],
    }

    # ---- C1.3 : task-Jacobian block structure ---------------------------
    J_t = st.J_torso
    J_ea, J_eb = st.J_tool_a, st.J_tool_b
    out['C1_3'] = {
        'column_order': 'q_t(6) | arm A(%d) | arm B(%d)' % (na, nb),
        'J_torso_block_maxabs': blocks(J_t, na, nb),
        'J_tool_a_block_maxabs': blocks(J_ea, na, nb),
        'J_tool_b_block_maxabs': blocks(J_eb, na, nb),
        'J_torso_arm_block_exactly_zero': bool(
            np.all(J_t[:, 6:] == 0.0)),
        'J_tool_a_armB_block_exactly_zero': bool(
            np.all(J_ea[:, 6 + na:] == 0.0)),
        'J_tool_b_armA_block_exactly_zero': bool(
            np.all(J_eb[:, 6:6 + na] == 0.0)),
    }

    # ---- C1.2 : orientation-error frame consistency ----------------------
    # The task uses e_ang = log3(R^T R_ref)  (LOCAL / body frame).
    # J (LOCAL_WORLD_ALIGNED) maps qdd -> accel in WORLD-ALIGNED axes.
    # The frame-consistent error would be  log3(R_ref R^T) = R log3(R^T R_ref).
    R = st.oMf_torso.rotation
    res = []
    for deg in (0.0, 0.5, 1.0, 2.0, 5.0, 8.0):
        a = np.deg2rad(deg)
        ax = np.array([0.2, 0.9, -0.39]); ax /= np.linalg.norm(ax)
        Rr = R @ pin.exp3(a * ax)                     # ref = current * delta
        e_local = pin.log3(R.T @ Rr)                  # what the code uses
        e_world = pin.log3(Rr @ R.T)                  # frame-consistent form
        res.append({
            'attitude_error_deg': deg,
            'e_local_norm_rad': float(np.linalg.norm(e_local)),
            'e_world_norm_rad': float(np.linalg.norm(e_world)),
            'norms_equal': bool(abs(np.linalg.norm(e_local)
                                    - np.linalg.norm(e_world)) < 1e-12),
            'direction_gap_rad': float(np.linalg.norm(e_world - e_local)),
            'direction_gap_deg': float(np.rad2deg(
                np.linalg.norm(e_world - e_local))),
            'relative_direction_gap': (
                float(np.linalg.norm(e_world - e_local)
                      / max(np.linalg.norm(e_local), 1e-15))
                if deg > 0 else 0.0),
        })
    out['C1_2_frame'] = {
        'torso_R_deviation_from_identity_deg': float(np.rad2deg(
            np.linalg.norm(pin.log3(R)))),
        'code_uses': 'pin.log3(R.T @ R_ref)  -> LOCAL (body) frame',
        'jacobian_convention': 'pin.LOCAL_WORLD_ALIGNED -> world-aligned axes',
        'frame_consistent_form': 'log3(R_ref @ R.T) = R @ log3(R.T @ R_ref)',
        'note': ('the two differ by the rotation R itself: same magnitude, '
                 'different direction. The gap scales with the ATTITUDE of '
                 'the frame (how far R is from I), not with the tracking '
                 'error.'),
        'samples': res,
    }

    with open(os.path.join(OUT, 'c1_jacobian_probe.json'), 'w') as fh:
        json.dump(out, fh, indent=2)

    print('=' * 74)
    print(f"model {out['model']}   nq={out['nq']} nv={out['nv']} "
          f"n_joints={out['n_joints']}  m={out['total_mass_kg']:.3f} kg")
    print()
    print('C1.1 — momentum task rows')
    c = out['C1_1']
    print(f"   J_com {c['J_com_shape']}   A_G {c['A_G_shape']}")
    print(f"   max| J_com - A_G[0:3,:]/m | = "
          f"{c['max_abs_diff_Jcom_vs_AGlin_over_m']:.3e}   "
          f"(scale max|J_com| = {c['max_abs_Jcom']:.3f})")
    print()
    print('C1.3 — task-Jacobian block structure (max |.| per block)')
    b = out['C1_3']
    for k in ('J_torso_block_maxabs', 'J_tool_a_block_maxabs',
              'J_tool_b_block_maxabs'):
        v = b[k]
        print(f"   {k:26s} base={v['base']:9.4f}  "
              f"armA={v['armA']:9.4f}  armB={v['armB']:9.4f}")
    print(f"   J_torso arm block exactly zero : "
          f"{b['J_torso_arm_block_exactly_zero']}")
    print(f"   J_tool_a  armB block exactly zero : "
          f"{b['J_tool_a_armB_block_exactly_zero']}")
    print(f"   J_tool_b  armA block exactly zero : "
          f"{b['J_tool_b_armA_block_exactly_zero']}")
    print()
    print('C1.2 — orientation-error frame check')
    fr = out['C1_2_frame']
    print(f"   torso attitude |log3(R)| = "
          f"{fr['torso_R_deviation_from_identity_deg']:.3f} deg")
    print(f"   {'err[deg]':>9} {'|e_local|':>11} {'|e_world|':>11} "
          f"{'gap[rad]':>11} {'gap/|e|':>9}")
    for s in fr['samples']:
        print(f"   {s['attitude_error_deg']:9.2f} {s['e_local_norm_rad']:11.6f} "
              f"{s['e_world_norm_rad']:11.6f} {s['direction_gap_rad']:11.6f} "
              f"{s['relative_direction_gap']:9.4f}")
    print()
    print(f"wrote {os.path.join(OUT, 'c1_jacobian_probe.json')}")


if __name__ == '__main__':
    main()
