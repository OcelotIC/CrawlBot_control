"""Feasibility prototype: constant CoM-z standoff in the dock IK.

The traversal currently lets CoM-z float (range 408mm), so dock configs
drift toward the beam and the torso grazes/penetrates it (steps 2-4).
This probes whether pinning CoM-z to a constant standoff is feasible for
EVERY anchor pair the gait visits, while keeping the torso clear of the
beam and the arms well-conditioned.

For each step we take the existing dock targets (both tool poses, read
by FK at the logged q_end), then re-solve a damped-LS whole-body IK with
tasks:
    tool_a 6D = anchor_a pose      (6)
    tool_b 6D = anchor_b pose      (6)
    torso orientation = leveled    (3, from the seed's leveled R)
    CoM_z = z_target               (1)
seeded from q_end. We report convergence, achieved CoM-z, torso-z and
clearance to the beam (z in [-0.025, 0.025]), and per-arm σ_min (worst-
direction manipulability), compared to the unconstrained baseline.

Sweeps a few z_target values so we can pick the standoff that keeps the
torso clear with the least manipulability cost.

Usage:
  PYTHONPATH=. MUJOCO_GL=osmesa python3 scripts/diag_standoff_feasibility.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pinocchio as pin

from crawlbot.core.robot_interface import RobotInterface

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF = os.path.join(_root, 'models', 'VISPA_crawling_fixed.urdf')
QEND = os.path.join(_root, 'results', 'diag_cooperative_arms', 'step_q_end.json')

BEAM_HALF_Z = 0.025          # beam spans z in [-0.025, 0.025] (structure frame)
Z_TARGETS = [-0.20, -0.25, -0.274, -0.32, -0.35, -0.40, -0.45]


def main():
    robot = RobotInterface(URDF, gravity='zero')
    model, data = robot.model, robot.data
    fa, fb, ft = robot.frame_tool_a, robot.frame_tool_b, robot.frame_torso
    va, vb = robot.arm_a_v_slice, robot.arm_b_v_slice

    qends = json.load(open(QEND))

    def fk(q):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

    def pose(q, fid):
        fk(q)
        return data.oMf[fid].copy()

    def Jlocal(q, fid):
        pin.computeJointJacobians(model, data, q)
        pin.updateFramePlacements(model, data)
        return pin.getFrameJacobian(model, data, fid, pin.LOCAL)

    def Jcom(q):
        return pin.jacobianCenterOfMass(model, data, q)   # (3, nv)

    def com(q):
        return pin.centerOfMass(model, data, q).copy()

    def arm_sigmin(q, fid, vsl):
        pin.computeJointJacobians(model, data, q)
        pin.updateFramePlacements(model, data)
        J = pin.getFrameJacobian(model, data, fid, pin.LOCAL_WORLD_ALIGNED)[:, vsl]
        return float(np.linalg.svd(J, compute_uv=False).min())

    def solve(q_seed, oMa, oMb, R_torso_level, z_tgt,
              iters=4000, damp=1e-3, tol=1e-7):
        q = q_seed.copy()
        best_q, best_res = q.copy(), np.inf
        for _ in range(iters):
            fk(q)
            ea = pin.log6(data.oMf[fa].actInv(oMa)).vector       # 6
            eb = pin.log6(data.oMf[fb].actInv(oMb)).vector       # 6
            Rt = data.oMf[ft].rotation
            eo = pin.log3(Rt.T @ R_torso_level)                  # 3 torso ori
            ecz = np.array([z_tgt - com(q)[2]])                  # 1
            err = np.concatenate([ea, eb, eo, ecz])
            res = np.linalg.norm(err)
            if res < best_res:
                best_res, best_q = res, q.copy()
            if res < tol:
                break
            Ja = Jlocal(q, fa)
            Jb = Jlocal(q, fb)
            Jt = Jlocal(q, ft)[3:6, :]                           # torso angular
            Jcz = Jcom(q)[2:3, :]                                # CoM-z row
            J = np.vstack([Ja, Jb, Jt, Jcz])                     # (16, nv)
            JJt = J @ J.T + damp * np.eye(J.shape[0])
            dq = J.T @ np.linalg.solve(JJt, err)
            step = min(1.0, 0.3 / (np.linalg.norm(dq) + 1e-9))
            q = pin.integrate(model, q, step * dq)
        return best_q, best_res

    # Baseline (unconstrained) per step
    print("Baseline dock configs (unconstrained CoM-z):")
    print(f"{'step':>4} {'CoM-z':>8} {'torso-z':>8} {'clear':>8} "
          f"{'a_smin':>7} {'b_smin':>7}")
    base = {}
    for e in qends:
        s = e['step_idx']
        q = np.asarray(e['q_end'], float)
        cz = com(q)[2]
        tz = pose(q, ft).translation[2]
        clr = (abs(tz) - BEAM_HALF_Z) * 1000   # torso-center clearance to box (z only)
        sa, sb = arm_sigmin(q, fa, va), arm_sigmin(q, fb, vb)
        base[s] = dict(q=q, cz=cz, tz=tz)
        print(f"{s:>4} {cz:>+8.3f} {tz:>+8.3f} {clr:>6.0f}mm "
              f"{sa:>7.3f} {sb:>7.3f}")

    # Pinned-CoM-z sweep
    for z_tgt in Z_TARGETS:
        print(f"\n=== pin CoM-z = {z_tgt:+.3f} m ===")
        print(f"{'step':>4} {'resid':>9} {'CoM-z':>8} {'torso-z':>8} "
              f"{'clear':>8} {'a_smin':>7} {'b_smin':>7}")
        feasible = True
        for e in qends:
            s = e['step_idx']
            q_seed = base[s]['q']
            oMa = pose(q_seed, fa)
            oMb = pose(q_seed, fb)
            R_lvl = pose(q_seed, ft).rotation.copy()
            q_sol, res = solve(q_seed, oMa, oMb, R_lvl, z_tgt)
            cz = com(q_sol)[2]
            tz = pose(q_sol, ft).translation[2]
            clr = (abs(tz) - BEAM_HALF_Z) * 1000
            sa, sb = arm_sigmin(q_sol, fa, va), arm_sigmin(q_sol, fb, vb)
            ok = res < 1e-4
            feasible &= ok
            flag = '' if ok else '  <-- IK FAIL'
            print(f"{s:>4} {res:>9.2e} {cz:>+8.3f} {tz:>+8.3f} "
                  f"{clr:>6.0f}mm {sa:>7.3f} {sb:>7.3f}{flag}")
        print(f"  feasible for all steps: {feasible}")

    print("\nLegend: clear = torso-center z-distance outside the beam box "
          "(neg = inside). a_smin/b_smin = worst-direction arm manipulability "
          "(higher=better; baseline docked steps ran ~0.28-0.41).")


if __name__ == '__main__':
    main()
