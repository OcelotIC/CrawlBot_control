"""Step-2 dock-stall reachability decomposition.

Question: step 2's swing EE parks ~17mm short of the anchor with
tiny torque (3 Nm of 20). Is that residual:
  (A) kinematically un-closeable while the P1-strict torso-ANGULAR
      task holds orientation fixed (i.e. reaching needs torso yaw),
      OR
  (B) closeable in principle, but the QP declines (task-weight /
      conditioning between EE α=3000 and torso-linear α=500)?

Method (held config = last step-2 snapshot, t≈30.43s):
  - Load the same Pinocchio model the QP uses.
  - J_torso (6×nv, LWA), J_ee = J_tool_b (6×nv). Angular rows = [3:6]
    (matches wholebody_qp A_torso_full[3:] strict-P1 task).
  - N_ang = I − pinv(J_torso_ang) J_torso_ang  (null-space of torso
    orientation — the projector the EE task lives in, cooperative mode).
  - Reachable EE-linear subspace under orientation lock =
    range(J_ee_lin @ N_ang). Project the EE error onto it.
      reachable_frac ≈ 1  → mechanism (B), weight/conditioning.
      reachable_frac < 1  → mechanism (A), orientation lock blocks reach.
  - Cross-checks: arm-only reach (base frozen), directional
    manipulability of J_ee_lin@N_ang along the error direction, and
    the same for steps 0/1 held configs for contrast.

Usage:
  PYTHONPATH=. MUJOCO_GL=osmesa python3 scripts/diag_dock_reach.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pinocchio as pin

from crawlbot.core.robot_interface import RobotInterface
from crawlbot.core.state_conversions import mujoco_to_pinocchio

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF = os.path.join(_root, 'models', 'VISPA_crawling_fixed.urdf')
LOG = os.path.join(_root, 'results', 'diag_cooperative_arms', 'sim_log.json')

# Per-step swing arm + anchor (dock target = p_ee_ref endpoint).
STEP_INFO = {0: ('b', None), 1: ('a', None), 2: ('b', None)}


def dir_manip(J3xn, direction):
    """Smallest task-space gain of J3xn along `direction` (unit).
    Returns ||direction|| achievable per unit ||qdd|| in that dir =
    1 / ||pinv(J)^T direction|| (Yoshikawa directional). Larger = easier."""
    Jp = np.linalg.pinv(J3xn, rcond=1e-8)
    u = direction / (np.linalg.norm(direction) + 1e-12)
    # qdd needed for unit cartesian motion u: qdd = Jp @ u ; gain = 1/||qdd||
    qdd = Jp @ u
    return 1.0 / (np.linalg.norm(qdd) + 1e-12)


def analyze(robot, snap, swing_arm, p_ee_ref):
    t, qpos, qvel = snap[0], np.array(snap[1]), np.array(snap[2])
    pq, pv = mujoco_to_pinocchio(qpos, qvel)
    rs = robot.update(pq, pv)
    if swing_arm == 'b':
        J_ee = rs.J_tool_b
        oMf_ee = rs.oMf_tool_b
        arm_slice = robot.arm_b_v_slice
    else:
        J_ee = rs.J_tool_a
        oMf_ee = rs.oMf_tool_a
        arm_slice = robot.arm_a_v_slice
    p_ee = oMf_ee.translation
    if p_ee_ref is None:
        p_ee_ref = p_ee  # docked steps: no residual of interest
    e = p_ee_ref - p_ee                      # error to close (toward anchor)

    J_torso = rs.J_torso                     # (6, nv) LWA
    nv = J_torso.shape[1]
    J_torso_ang = J_torso[3:6, :]            # strict-P1 task (cooperative)
    N_ang = np.eye(nv) - np.linalg.pinv(J_torso_ang, rcond=1e-8) @ J_torso_ang

    J_ee_lin = J_ee[0:3, :]                  # linear rows

    # (1) full-body EE-linear reach under orientation lock
    M = J_ee_lin @ N_ang                     # (3, nv) achievable cart accel map
    P_reach = M @ np.linalg.pinv(M, rcond=1e-8)   # (3,3) projector onto range
    e_reach = P_reach @ e
    reach_frac = np.linalg.norm(e_reach) / (np.linalg.norm(e) + 1e-12)

    # (2) arm-only reach (base frozen) — what the arm can do alone
    J_ee_lin_arm = J_ee_lin[:, arm_slice]    # (3, n_arm)
    P_arm = J_ee_lin_arm @ np.linalg.pinv(J_ee_lin_arm, rcond=1e-8)
    arm_frac = np.linalg.norm(P_arm @ e) / (np.linalg.norm(e) + 1e-12)

    # (3) directional gains along the error direction
    g_full = dir_manip(M, e) if np.linalg.norm(e) > 1e-9 else float('nan')
    g_arm = dir_manip(J_ee_lin_arm, e) if np.linalg.norm(e) > 1e-9 else float('nan')

    # singular values of the orientation-locked reach map (overall cond)
    sv = np.linalg.svd(M, compute_uv=False)
    return dict(
        t=t, e_mm=e * 1000, e_norm_mm=np.linalg.norm(e) * 1000,
        reach_frac=reach_frac, arm_frac=arm_frac,
        g_full=g_full, g_arm=g_arm,
        sv_min=sv.min(), sv_max=sv.max(),
        p_ee=p_ee, p_ee_ref=p_ee_ref,
    )


def main():
    sl = json.load(open(LOG))
    snaps = sl['snapshots']
    # Map: find the LAST snapshot whose t falls in each step's SS/hold.
    t_arr = np.array(sl['t']); sidx = np.array(sl['step_idx'])
    ph = np.array(sl['phase'], dtype=object)

    # dock targets from p_ee_ref endpoint per step (the anchor).
    pref = np.array(sl['p_ee_ref'])
    step_target = {}
    for s in (0, 1, 2):
        m = (sidx == s) & (ph == 'SS')
        if m.any():
            step_target[s] = pref[m][-1].copy()

    robot = RobotInterface(URDF, gravity='zero')

    # pick representative snapshots: closest snapshot time to each step's
    # SS end.
    def snap_near(t_target):
        best = min(snaps, key=lambda s: abs(s[0] - t_target))
        return best

    step_end_t = {}
    for s in (0, 1, 2):
        m = (sidx == s) & (ph == 'SS')
        if m.any():
            step_end_t[s] = t_arr[m][-1]

    print(f"{'step':>4} {'t':>7} {'|e|mm':>7} {'reach%':>7} {'arm%':>7} "
          f"{'g_full':>8} {'g_arm':>8} {'sv_min':>8} {'sv_max':>8}")
    print('-' * 76)
    for s in (0, 1, 2):
        if s not in step_end_t:
            continue
        snap = snap_near(step_end_t[s])
        arm = STEP_INFO[s][0]
        # use the step's anchor as target (so steps 0/1 show their
        # residual at dock too, for contrast).
        tgt = step_target.get(s)
        r = analyze(robot, snap, arm, tgt)
        print(f"{s:>4} {r['t']:>7.2f} {r['e_norm_mm']:>7.2f} "
              f"{r['reach_frac']*100:>6.1f}% {r['arm_frac']*100:>6.1f}% "
              f"{r['g_full']:>8.4f} {r['g_arm']:>8.4f} "
              f"{r['sv_min']:>8.4f} {r['sv_max']:>8.4f}")
        if s == 2:
            print(f"       e vec [mm] = {r['e_mm'].round(2).tolist()}")
            print(f"       p_ee={r['p_ee'].round(4).tolist()}  "
                  f"p_ee_ref={r['p_ee_ref'].round(4).tolist()}")

    print()
    print("Legend: reach% = fraction of EE error closeable WITHOUT torso")
    print("        rotation (range of J_ee_lin @ N_torso_ang).")
    print("        arm% = fraction closeable by the arm alone (base frozen).")
    print("        g_* = directional task-space gain along the error dir")
    print("        (cart motion per unit joint accel; smaller = stiffer/")
    print("        closer to singular in that direction).")


if __name__ == '__main__':
    main()
