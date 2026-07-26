"""Feasibility probe for dock-posture redundancy resolution (fix option 1).

Question: step 2's swing arm parks with approach-direction manipulability
gain 0.34 (vs ~0.80 for docked steps 0/1). The 7-DOF arm at a 6D dock
pose has a 1-DOF self-motion manifold. Does that single redundant DOF
have enough freedom to raise the approach-direction gain toward ~0.8?

If yes  -> redundancy resolution at the dock IK is a viable fix.
If no   -> 1 DOF is insufficient; need a different lever (torso
           placement / re-plan).

Method:
  - Load the held config (last step-2 snapshot). Fix base + stance arm.
  - IK the swing arm (arm b, 7-DOF) to the dock pose (anchor position
    + held orientation).
  - Trace the 1-DOF null-space manifold (null vector of the 6x7 arm
    Jacobian) in both directions, re-projecting onto the pose each
    step, respecting joint limits.
  - Report approach-direction gain g(s) along the manifold: baseline,
    min, max, and the joint-limit extent reached.

Usage:
  PYTHONPATH=. MUJOCO_GL=osmesa python3 scripts/diag_dock_redundancy.py
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
LOG = os.path.join(_root, 'Misc', 'runs', 'diag_cooperative_arms', 'sim_log.json')

P_STAR = np.array([1.2, -0.3, 0.025])           # step-2 anchor (arm b)
APPROACH = np.array([15.68, 7.62, 1.58])         # stalled error dir (world)


def main():
    robot = RobotInterface(URDF, gravity='zero')
    model, data = robot.model, robot.data
    fid = robot.frame_tool_b
    vsl = robot.arm_b_v_slice
    qsl = robot.arm_b_q_slice
    a_hat = APPROACH / np.linalg.norm(APPROACH)

    sl = json.load(open(LOG))
    snap = min(sl['snapshots'], key=lambda s: abs(s[0] - 30.43))
    pq, pv = mujoco_to_pinocchio(np.array(snap[1]), np.array(snap[2]))

    qlo = model.lowerPositionLimit.copy()
    qhi = model.upperPositionLimit.copy()
    arm_q_idx = np.arange(model.nq)[qsl]
    has_limits = np.all(np.isfinite(qlo[arm_q_idx])) and np.all(
        np.isfinite(qhi[arm_q_idx]))

    def fk(q):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        return data.oMf[fid]

    def J_local(q):
        pin.computeJointJacobians(model, data, q)
        pin.updateFramePlacements(model, data)
        return pin.getFrameJacobian(model, data, fid, pin.LOCAL)[:, vsl]

    def J_lwa_lin(q):
        pin.computeJointJacobians(model, data, q)
        pin.updateFramePlacements(model, data)
        J = pin.getFrameJacobian(model, data, fid, pin.LOCAL_WORLD_ALIGNED)
        return J[0:3, vsl]

    def gain(q):
        J = J_lwa_lin(q)
        Jp = np.linalg.pinv(J, rcond=1e-8)
        return 1.0 / (np.linalg.norm(Jp @ a_hat) + 1e-12)

    R_star = fk(pq).rotation.copy()
    oMdes = pin.SE3(R_star, P_STAR)

    def arm_ik(q_in, iters=2000, tol=1e-9, damp=1e-4):
        # Damped least-squares CLIK with adaptive step (stable near
        # singularities, unlike raw pinv).
        q = q_in.copy()
        best = q.copy()
        best_res = np.inf
        for _ in range(iters):
            oM = fk(q)
            err = pin.log6(oM.actInv(oMdes)).vector
            res = np.linalg.norm(err)
            if res < best_res:
                best_res, best = res, q.copy()
            if res < tol:
                break
            J = J_local(q)
            JJt = J @ J.T + damp * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, err)
            step = min(1.0, 0.2 / (np.linalg.norm(dq) + 1e-9))
            q[qsl] = q[qsl] + step * dq
        return best, best_res

    # 1. project held config onto the exact dock pose
    q0, res0 = arm_ik(pq.copy())
    g0 = gain(q0)
    print(f"baseline (held->dock pose): IK residual={res0:.2e}  "
          f"approach gain={g0:.4f}")
    if res0 > 1e-6:
        print("  WARN: arm IK to dock pose did not fully converge.")

    def limits_ok(q):
        if not has_limits:
            return True
        qa = q[arm_q_idx]
        return np.all(qa >= qlo[arm_q_idx] - 1e-9) and np.all(
            qa <= qhi[arm_q_idx] + 1e-9)

    # 2. trace the 1-DOF self-motion manifold both directions
    def trace(direction, step=0.02, n=200):
        q = q0.copy()
        gains = [gain(q)]
        ss = [0.0]
        for k in range(1, n + 1):
            J = J_local(q)
            # null-space basis: smallest singular direction of 6x7
            U, S, Vt = np.linalg.svd(J)
            z = Vt[-1]                      # (7,) null vector
            q_try = q.copy()
            q_try[qsl] = q_try[qsl] + direction * step * z
            q_try, res = arm_ik(q_try, iters=60)
            if res > 1e-6 or not limits_ok(q_try):
                break
            q = q_try
            gains.append(gain(q))
            ss.append(direction * step * k)
        return np.array(ss), np.array(gains)

    s_pos, g_pos = trace(+1.0)
    s_neg, g_neg = trace(-1.0)
    all_g = np.concatenate([g_neg[::-1], g_pos[1:]])
    all_s = np.concatenate([s_neg[::-1], s_pos[1:]])

    print()
    print(f"manifold traced: s in [{all_s.min():+.2f}, {all_s.max():+.2f}] "
          f"rad of null-DOF travel ({len(all_g)} samples)")
    print(f"  approach gain: baseline={g0:.4f}  min={all_g.min():.4f}  "
          f"max={all_g.max():.4f}")
    imax = all_g.argmax()
    print(f"  max gain {all_g.max():.4f} at s={all_s[imax]:+.3f} rad "
          f"(baseline 0 -> {((all_g.max()/g0)-1)*100:+.0f}%)")
    print()
    print(f"  docked steps 0/1 sit at gain ~0.79-0.81.")
    if all_g.max() >= 0.70:
        print(f"  => VIABLE: 1-DOF redundancy reaches "
              f"{all_g.max():.2f} >= 0.70. Dock-posture resolution can fix it.")
    elif all_g.max() >= 0.50:
        print(f"  => PARTIAL: redundancy lifts gain to {all_g.max():.2f}; "
              f"may help but not to 0/1 levels.")
    else:
        print(f"  => INSUFFICIENT: gain stays < 0.50 across the manifold; "
              f"1 DOF is not enough — need torso placement / re-plan.")

    # profile dump for the record
    print()
    print("  s[rad]   gain")
    for s, g in zip(all_s, all_g):
        if abs(s * 100 % 4) < 2:   # ~every 0.04 rad
            print(f"  {s:+6.3f}  {g:.4f}")


if __name__ == '__main__':
    main()
