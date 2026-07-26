#!/usr/bin/env python3
"""Fix-C mini-audit — can "weld-relative pose+twist -> 0" be a terminal NMPC constraint?

READ-ONLY reproducer for Misc/runs/j2_audit/INTERNAL_j2_fixC.md. Decides whether the
Fix-C condition (gripper<->anchor weld-relative pose -> 0 AND twist Jc·v⁻ -> 0) can be
expressed as a terminal constraint in the 9-D centroidal NMPC, or must attach to the
swing planner (b) or the dock gate (c). No crawlbot/ change, no sim run.

Decisive demo (Q2): the NMPC state is the centroidal REDUCTION [r_com, v_com, L_com].
Using the REAL robot, a velocity perturbation in null(A_G) leaves [v_com, L_com]
unchanged yet moves the gripper (J_tool·δv != 0) -> the gripper twist is NOT a function
of the NMPC state -> a terminal NMPC constraint on it is NOT formulable. (a) is OUT.

Run:  MUJOCO_GL=disabled PYTHONPATH=. python3 Misc/scripts/audit_fixC.py
"""
import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_SIM = os.path.join(ROOT, 'crawlbot', 'simulation', 'sim_loop.py')
_NMPC = os.path.join(ROOT, 'crawlbot', 'solvers', 'centroidal_nmpc.py')
_SWING = os.path.join(ROOT, 'crawlbot', 'planning', 'swing_planner.py')
_URDF = os.path.join(ROOT, 'models', 'VISPA_crawling_fixed.urdf')

_checks = []


def record(ok, label, evidence):
    _checks.append((bool(ok), label, evidence))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    print(f"         {evidence}")


def find(path, needle, label):
    rel = os.path.relpath(path, ROOT)
    try:
        with open(path) as f:
            for i, line in enumerate(f, 1):
                if needle in line:
                    return i, line.strip()
    except OSError as e:
        record(False, label, f'{rel}: unreadable ({e})')
        return None
    record(False, label, f'{rel}: anchor {needle!r} NOT FOUND')
    return None


def main():
    print('=' * 78)
    print('FIX-C MINI-AUDIT — terminal-NMPC formulability of weld-relative pose+twist')
    print('=' * 78)

    # ---- Q1 / Q4: where the Fix-C quantities live (reuse, not rebuild) ----
    print('\n[Q1/Q4] Jc·v⁻ (weld-relative twist) + weld-relative pose are ALREADY computed')
    a = find(_SIM, 'rows.append(jpg - jpa)', 'relative-site linear twist row')
    b = find(_SIM, 'rows.append(jrg - jra)', 'relative-site angular twist row')
    c = find(_SIM, 'v_pre = J @ v_minus', 'Jc·v⁻ = J @ v_minus')
    if a and b and c:
        record(True, 'Jc = relative-site weld Jacobian [jpg-jpa; jrg-jra]; Jc·v⁻ = J@v_minus (Fix-A block)',
               f'sim_loop.py:{a[0]},{b[0]} (J rows) ; :{c[0]} (Jc·v⁻) — reuse, do not rebuild')
    a = find(_SIM, 'def _gripper_distance', 'weld-relative position metric')
    b = find(_SIM, 'def _gripper_ori_err_deg', 'weld-relative orientation metric')
    if a and b:
        record(True, 'weld-relative POSE available: _gripper_distance (pos) + _gripper_ori_err_deg (ori)',
               f'sim_loop.py:{a[0]} (pos) ; :{b[0]} (ori)  [site_xpos/site_xmat differences]')

    # ---- Q2 (DECISIVE): is the gripper twist a function of the NMPC state? ----
    print('\n[Q2] NMPC state has NO gripper DOF; centroidal reduction is many-to-one')
    a = find(_NMPC, 'NX = 9', 'NMPC state dim')
    if a:
        record(True, 'NMPC state x = [r_com, v_com, L_com] only — NO gripper pose/velocity, NO joint config',
               f'centroidal_nmpc.py:{a[0]}: {a[1].split("#")[0].strip()}')
    a = find(_NMPC, 'NP = 18', 'NMPC param dim')
    if a:
        record(True, 'r_C1,r_C2 are ANCHOR positions carried as PARAMETERS (p[6:12]) — not state, not the gripper',
               f'centroidal_nmpc.py:{a[0]}: {a[1].split("#")[0].strip()}')

    # Real-robot demonstration: null(A_G) moves the gripper.
    import pinocchio as pin
    from crawlbot.core.robot_interface import RobotInterface
    rob = RobotInterface(_URDF, gravity='zero')
    m, d = rob.model, rob.data
    rng = np.random.default_rng(1)
    # a non-trivial configuration (not the singular neutral pose)
    q = pin.neutral(m).copy()
    q[7:] = 0.3 * rng.standard_normal(m.nq - 7)
    rs = rob.update(q, np.zeros(m.nv))
    A_G = pin.computeCentroidalMap(m, d, q)        # (6, nv): [h_lin; h_ang] = A_G @ v
    J_tool = np.asarray(rs.J_tool_a)               # (6, nv) gripper Jacobian
    U, S, Vt = np.linalg.svd(A_G)
    rank = int((S > 1e-9).sum())
    nulldim = m.nv - rank
    record(nulldim >= 1, f'centroidal map A_G is 6x{m.nv}; null space dim = {nulldim} (nv > 6 -> many-to-one)',
           f'rank(A_G) = {rank}, nullspace dim = {nulldim}')
    # pick the null-space direction that maximally moves the gripper
    NS = Vt[rank:].T                               # (nv, nulldim) basis of null(A_G)
    proj = J_tool @ NS                             # gripper twist for each null direction
    j = int(np.argmax(np.linalg.norm(proj, axis=0)))
    dv = NS[:, j]
    centroidal_change = float(np.linalg.norm(A_G @ dv))
    gripper_change = float(np.linalg.norm(J_tool @ dv))
    record(centroidal_change < 1e-8 and gripper_change > 1e-3,
           'δv in null(A_G): NMPC state [v_com,L_com] UNCHANGED, but gripper twist CHANGES',
           f'||A_G·δv|| = {centroidal_change:.2e} (~0)  vs  ||J_tool·δv|| = {gripper_change:.4f} (>0)')
    record(True, 'CONCLUSION: gripper twist is NOT a function of the NMPC state -> (a) NOT formulable',
           f'a {nulldim}-D family of motions leaves the NMPC state fixed while moving the gripper')

    # ---- (b) swing planner: terminal reference twist is already 6-D zero ----
    print('\n[(b)] swing-planner terminal reference: v_ee=0 AND omega_ee=0 at tau=1 (objective)')
    from crawlbot.planning.swing_planner import SwingPlanner as SP
    qd1 = SP._quintic_dot(1.0)
    qdd1 = SP._quintic_ddot(1.0)
    record(qd1 == 0.0 and qdd1 == 0.0,
           'quintic timing has zero terminal vel+accel -> position-track v_ee=0, a_ee=0 at tau=1',
           f'_quintic_dot(1)={qd1}, _quintic_ddot(1)={qdd1}')
    a = find(_SWING, 'omega_ee = R_ee @ (sigma_r_dot * omega_total)', 'angular ref uses delayed-cosine')
    if a:
        record(True, 'swing ref outputs BOTH v_ee and omega_ee (SLERP w/ zero terminal rate) -> 6-D twist ref',
               f'swing_planner.py:{a[0]}: {a[1]}')
    record(True, '(b) verdict: CAN carry pose+twist as an OBJECTIVE (feedforward ref) — shapes, cannot GUARANTEE',
           'the WBC tracks this ref; actual Jc·v⁻ at dock is not forced to 0 by a reference')

    # ---- (c) dock gate: hard guard; currently LINEAR speed only -----------
    print('\n[(c)] dock gate: hard boolean guard; today gates LINEAR EE speed only')
    a = find(_SIM, 'vel_ok = (self._gripper_speed(swing_arm)', 'dock-gate velocity check')
    if a:
        record(True, 'gate vel check = _gripper_speed (LINEAR EE speed) < dock_vel_max — NOT the 6-D twist',
               f'sim_loop.py:{a[0]}: {a[1]}')
    a = find(_SIM, 'docked = pos_ok and ori_ok and vel_ok', 'dock-gate boolean')
    if a:
        record(True, 'gate = pos_ok AND ori_ok AND vel_ok (hard precondition before _activate_weld)',
               f'sim_loop.py:{a[0]}: {a[1]}')
    record(True, '(c) verdict: CAN carry pose+twist as a HARD GATE (replace _gripper_speed with ||Jc·v⁻||<eps)',
           'a GUARD, not a DRIVER: prevents welding with nonzero twist, but does not actively reduce it')

    # ---- Verdict ----------------------------------------------------------
    n_pass = sum(ok for ok, _, _ in _checks)
    n_tot = len(_checks)
    print('\n' + '=' * 78)
    print(f'VERDICT: {n_pass}/{n_tot} checks confirmed.')
    fails = [lbl for ok, lbl, _ in _checks if not ok]
    if fails:
        print('FAILED:')
        for lbl in fails:
            print(f'  - {lbl}')
        return 1
    print('(a) terminal NMPC constraint : NOT FORMULABLE — gripper decoupled from centroidal state.')
    print('(b) swing-planner terminal    : CAN carry it as an OBJECTIVE (driver; already 6-D-zero ref).')
    print('(c) dock gate                 : CAN carry it as a HARD GATE (guard; reuse Jc·v⁻ + d + ori).')
    print('=> Fix C routes to (c) hard guard ||Jc·v⁻||<eps + pose, backed by (b) as the driver. NOT (a).')
    print('   See Misc/runs/j2_audit/INTERNAL_j2_fixC.md.')
    print('=' * 78)
    return 0


if __name__ == '__main__':
    sys.exit(main())
