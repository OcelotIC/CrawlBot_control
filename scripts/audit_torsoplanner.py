#!/usr/bin/env python3
"""TorsoPlanner audit (READ-ONLY) — map the planner mechanics to ground a
future "DS-mobile" mode (moving torso-ori + arm-posture in the reorientation
null space, gait-triggered, NOT CoM transport). Maps; does not design.

Source anchors for Q1–Q3 (planner output / set_hold / gait trigger / QP
consumption) + a numeric Q4: the reorientation null-space dimension at a
docked 2-contact config (null of the stacked contact Jacobian, ± CoM).

Run: MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/audit_torsoplanner.py
"""
import json
import os
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_TP = os.path.join(ROOT, 'crawlbot', 'planning', 'torso_planner.py')
_QP = os.path.join(ROOT, 'crawlbot', 'solvers', 'wholebody_qp.py')
_SIM = os.path.join(ROOT, 'crawlbot', 'simulation', 'sim_loop.py')
_URDF = os.path.join(ROOT, 'models', 'VISPA_crawling_fixed.urdf')
_DOCK = os.path.join(ROOT, 'Misc', 'runs', 'fixA_gate', 'sim_log.json')

_checks = []


def rec(ok, label, ev):
    _checks.append(bool(ok))
    print(f"  [{'OK ' if ok else 'XX '}] {label}\n         {ev}")


def find(path, needle, label):
    rel = os.path.relpath(path, ROOT)
    try:
        for i, ln in enumerate(open(path), 1):
            if needle in ln:
                return i, ln.strip()
    except OSError as e:
        rec(False, label, f'{rel}: {e}'); return None
    rec(False, label, f'{rel}: {needle!r} NOT FOUND'); return None


def main():
    print('=' * 78)
    print('TORSOPLANNER AUDIT — DS-mobile grounding  (ae0673e)')
    print('=' * 78)

    # ---- Q1: planner output + set_hold ---------------------------------
    print('\n[Q1] TorsoPlanner output + set_hold mechanics')
    a = find(_TP, 'class TorsoReference', 'TorsoReference')
    if a: rec(True, 'output = TorsoReference: p(3)+R(3x3)+v(6 twist)+a(6 accel) — SE(3)+derivs, struct frame',
              f'torso_planner.py:{a[0]}')
    a = find(_TP, 'class ComReference', 'ComReference')
    if a: rec(True, 'plus ComReference: r_com(3)+v_com(3); plus l_com_reference_at -> NMPC L_com_ref(3)',
              f'torso_planner.py:{a[0]}')
    a = find(_TP, 'def set_hold', 'set_hold')
    b = find(_TP, 'p=self._hold_p.copy(), R=self._hold_R.copy()', 'hold returns v=0,a=0')
    if a and b: rec(True, 'set_hold = STATIC setpoint (_hold_p/_hold_R/_hold_com); _hold_reference v=0,a=0',
                    f'torso_planner.py:{a[0]} (set) ; :{b[0]} (static, zero twist/accel)')
    a = find(_TP, 'def add_phase', 'add_phase')
    b = find(_TP, 'def reference_at', 'reference_at')
    if a and b: rec(True, 'TIME-VARYING machinery ALREADY EXISTS: add_phase/set_from_waypoints + '
                    'reference_at interpolates p,R,v,a (quintic/trapezoid/FK)',
                    f'torso_planner.py:{a[0]} (add_phase) ; :{b[0]} (reference_at: phase else hold)')
    a = find(_TP, 'q_seq: Optional[list] = None', 'FK q_seq')
    if a: rec(True, 'add_phase can carry a JOINT trajectory q_seq (FK mode) — but planner has NO arm-posture output',
              f'torso_planner.py:{a[0]} (q_seq is for FK CoM/L_com, not a posture task feed)')

    # ---- Q2: gait trigger ----------------------------------------------
    print('\n[Q2] Gait trigger: how the phase drives the planner')
    # SS builds a time-varying torso trajectory (add_phase directly).
    a = None
    for i, ln in enumerate(open(_SIM), 1):
        if 'self.torso_planner.add_phase' in ln or \
           'self.torso_planner.set_from_waypoints' in ln:
            a = (i, ln.strip()); break
    rec(a is not None, 'SS swing builds a TIME-VARYING torso trajectory (add_phase/set_from_waypoints)',
        f'sim_loop.py:{a[0]}' if a else 'NOT FOUND')
    a = find(_SIM, 'self.torso_planner.set_hold(', 'DS uses set_hold')
    if a: rec(True, 'DS (trailing/DWELL) drives the planner via set_hold(welded-equilibrium) — STATIC',
              f'sim_loop.py:{a[0]} (the DS-mobile hook: replace with a moving phase here)')
    a = find(_SIM, 'tr = self.torso_planner.reference_at(tq_planner)', '_step queries reference_at per tick')
    if a: rec(True, '_step calls reference_at(t) EACH tick -> qp.solve (so a moving phase flows through unchanged)',
              f'sim_loop.py:{a[0]}')

    # ---- Q3: QP consumption --------------------------------------------
    print('\n[Q3] QP consumption of the reference (centroidal-DS tasks)')
    a = find(_SIM, 'R_torso_ref=R_torso_ref_used', 'R_torso_ref into qp.solve')
    if a: rec(True, '_step passes p_torso_ref/R_torso_ref into qp.solve per tick (time-varying-CAPABLE interface)',
              f'sim_loop.py:{a[0]}')
    a = find(_QP, 'if settle_mode and cfg.ds_centroidal_mode and ds_centroidal_active:', 'DS task gate')
    if a: rec(True, 'DS tracking tasks: CoM-3D (ds_alpha_com) + torso-ANG-3D (ds_alpha_torso_ori) — track R_torso_ref',
              f'wholebody_qp.py:{a[0]}')
    a = find(_QP, '_posture_in_ds = (settle_mode and cfg.ds_centroidal_mode', 'DS posture task')
    b = find(_QP, 'cfg.Kp_posture * (self._q_nominal - q)', 'posture tracks q_nominal')
    if a and b: rec(True, 'arm-posture HALF: a posture task IS active in DS (constrains arm-null DOFs) BUT tracks '
                    'FIXED self._q_nominal',
                    f'wholebody_qp.py:{a[0]} (gate) ; :{b[0]} (fixed target)')
    a = find(_SIM, 'q_nominal=getattr(cfg', 'q_nominal set once')
    if a: rec(True, 'self._q_nominal set ONCE at construction; qp.solve has NO per-tick posture-target arg',
              f'sim_loop.py:{a[0]} (=> moving arm-posture needs NEW plumbing)')

    # ---- Q4: reorientation null space (numeric) ------------------------
    print('\n[Q4] Reorientation null space at a docked 2-contact config')
    a = find(_QP, 'P_int = np.eye(12) - G_pinv @ G', 'P_int wrench null space')
    if a: rec(True, 'P_int (=I-G+G) is the 12-D CONTACT-WRENCH internal-stress null space — a FORCE object, '
              'NOT the motion/reorientation freedom',
              f'wholebody_qp.py:{a[0]}')

    import pinocchio as pin
    from crawlbot.core.robot_interface import RobotInterface
    from crawlbot.core.state_conversions import mujoco_to_pinocchio
    rob = RobotInterface(_URDF, gravity='zero')
    sl = json.load(open(_DOCK))
    snap = [s for s in sl['snapshots'] if s[3] == 'final'][-1]
    pq, pv = mujoco_to_pinocchio(np.array(snap[1]), np.array(snap[2]))
    rs = rob.update(pq, pv)
    Ja = np.asarray(rs.J_tool_a)          # (6, nv)
    Jb = np.asarray(rs.J_tool_b)          # (6, nv)
    Jc = np.vstack([Ja, Jb])              # (12, nv) both welds
    Jcom = np.asarray(rs.J_com)           # (3, nv)
    nv = rob.model.nv

    def ndim(M, tol=1e-9):
        s = np.linalg.svd(M, compute_uv=False)
        r = int((s > tol * max(1.0, s[0])).sum())
        return r, M.shape[1] - r

    rJc, nJc = ndim(Jc)
    rJcg, nJcg = ndim(np.vstack([Jc, Jcom]))
    rec(True, f'nv(robot)={nv}; rank(Jc[12xnv])={rJc} -> dim null(Jc) = {nJc} '
        '(internal motions with BOTH welds active)',
        f'= the genuine reorientation+posture freedom at the dock')
    rec(True, f'rank([Jc;J_com])={rJcg} -> dim null([Jc;J_com]) = {nJcg} '
        '(reorient+posture at FIXED CoM — the DS-mobile space)',
        f'CoM-transport removed: {nJc}->{nJcg} (so ~{nJc-nJcg} CoM dof, ~3 torso-ori, rest arm-posture)')
    rec(True, 'commanded today: torso-ori-3D + CoM-3D tasks use part of null(Jc); posture pins the REST to q_nominal',
        f'DS-mobile = make the torso-ori target (and the posture target) MOVE within this {nJcg}-dim space')

    n_ok = sum(_checks)
    print('\n' + '=' * 78)
    print(f'{n_ok}/{len(_checks)} checks OK.')
    print('=' * 78)
    return 0 if n_ok == len(_checks) else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
