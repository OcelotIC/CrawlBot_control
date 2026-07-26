"""Offline IK feasibility check for the step 2 reference trajectory.

Pinocchio-only. No MuJoCo, no WBC, no dynamics. At 10 uniformly
sampled instants along the step 2 swing (t=0, T/9, ..., T with
T=T_step from the preplanner), solve IK to find q such that:

    FK(q).tool_b  = anchor_5b  (position only, tol 5 mm)
    FK(q).torso   = c_ref(t)   (position only, tol 50 mm)
    q_min ≤ q[7:] ≤ q_max

Floating base (q[0:7]) is free. Stance arm A is NOT constrained
(more permissive than the closed-chain reality; if THIS check fails
the reference is definitely infeasible).

References pulled from Misc/runs/diag_alpha_torso/alpha_500/sim_log.json:
    - r_com_ref[t]  (used as torso target with 50 mm tol)
    - preplanner_T_steps[2]  = T_step for step 2
    - dock_events / aborted_steps  to anchor the step 2 SS window
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pinocchio as pin

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF = os.path.join(_root, 'models', 'VISPA_crawling_fixed.urdf')

# anchor_5b position in structure frame (MJCF: <site name="anchor_5b"
# pos=" 1.2 -0.3 0.025"/>). Step 2 swing target.
ANCHOR_5B = np.array([1.2, -0.3, 0.025])

TOL_EE_M = 5e-3       # 5 mm
TOL_TORSO_M = 50e-3   # 50 mm
MAX_ITERS = 200
DAMP = 1e-2
STEP_CLIP = 0.5       # rad/step on the integration


def load_step2_reference(sim_log_path: str, n_instants: int = 10):
    """Return (target_times_t, c_ref_samples, T_step, q_init_guess)."""
    sl = json.load(open(sim_log_path))
    t = np.array(sl['t'])
    phase = np.array(sl['phase'], dtype=object)
    sidx = np.array(sl['step_idx'])
    r_com_ref = np.array(sl['r_com_ref'])
    p_torso = np.array(sl['p_torso']) if sl.get('p_torso') else None
    q_torso_arr = np.array(sl['q_torso']) if sl.get('q_torso') else None
    qva = np.array(sl['qvel_joints_a']) if sl.get('qvel_joints_a') else None
    qjA = sl.get('q_joints_a', None)
    qjB = sl.get('q_joints_b', None)

    mask = (sidx == 2) & (phase == 'SS')
    if not mask.any():
        raise RuntimeError('No step 2 SS samples in log')
    t_step2 = t[mask]
    t_start = float(t_step2[0])
    T_step = float(sl['preplanner_T_steps'][2])

    # Sample at t_start + k * T_step / (n_instants - 1) for k=0..n-1.
    target_times = t_start + np.linspace(0, T_step, n_instants)
    indices_in_log = []
    for tt in target_times:
        # Find the closest sample index in the FULL log (not the mask).
        idx = int(np.argmin(np.abs(t - tt)))
        indices_in_log.append(idx)
    c_ref_samples = r_com_ref[indices_in_log]

    # Initial q for warm-starting IK: use the log's actual torso state
    # (p_torso, q_torso) at step 2 SS entry; arm joints aren't logged,
    # so fall back to zero (a valid starting guess for the iterative
    # damped-LS solver below).
    idx0 = indices_in_log[0]
    model_tmp = pin.buildModelFromUrdf(URDF, pin.JointModelFreeFlyer())
    q_init = pin.neutral(model_tmp)
    if p_torso is not None and q_torso_arr is not None:
        q_init[0:3] = p_torso[idx0]
        # Log stores quaternion as wxyz; Pinocchio wants xyzw.
        q_wxyz = q_torso_arr[idx0]
        q_init[3:7] = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    return target_times, c_ref_samples, T_step, q_init, t_start


def ik_solve(model, data, frame_tool_b, frame_torso,
             p_ee_target, p_torso_target, q0):
    """Damped-LS Newton iteration. Both targets are position-only (3D)."""
    q = q0.copy()
    nv = model.nv
    q_lo = model.lowerPositionLimit
    q_hi = model.upperPositionLimit

    for it in range(MAX_ITERS):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        p_ee = data.oMf[frame_tool_b].translation
        p_to = data.oMf[frame_torso].translation
        e_ee = p_ee_target - p_ee
        e_to = p_torso_target - p_to
        err_ee = float(np.linalg.norm(e_ee))
        err_to = float(np.linalg.norm(e_to))
        if err_ee < TOL_EE_M and err_to < TOL_TORSO_M:
            return q, True, err_ee, err_to, it + 1

        # Stacked 6-row Jacobian: [J_ee_lin; J_torso_lin]
        Jee_full = pin.computeFrameJacobian(
            model, data, q, frame_tool_b, pin.LOCAL_WORLD_ALIGNED)
        Jto_full = pin.computeFrameJacobian(
            model, data, q, frame_torso, pin.LOCAL_WORLD_ALIGNED)
        J = np.vstack([Jee_full[:3], Jto_full[:3]])  # (6, nv)
        e = np.concatenate([e_ee, e_to])              # (6,)

        JJt = J @ J.T + DAMP ** 2 * np.eye(6)
        v = J.T @ np.linalg.solve(JJt, e)             # (nv,)
        # Clip step size for stability
        nv_step = float(np.linalg.norm(v))
        if nv_step > STEP_CLIP:
            v = v * (STEP_CLIP / nv_step)
        q = pin.integrate(model, q, v)
        # Joint-limit clipping on arm joints (indices 7:21 in nq).
        for i in range(7, model.nq):
            q[i] = float(np.clip(q[i], q_lo[i], q_hi[i]))

    # Did not converge in MAX_ITERS
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    err_ee = float(np.linalg.norm(p_ee_target - data.oMf[frame_tool_b].translation))
    err_to = float(np.linalg.norm(p_torso_target - data.oMf[frame_torso].translation))
    return q, False, err_ee, err_to, MAX_ITERS


def main():
    sim_log_path = os.path.join(
        _root, 'Misc', 'runs', 'diag_alpha_torso', 'alpha_500', 'sim_log.json')
    out_dir = os.path.join(_root, 'Misc', 'runs', 'diag_step2_ik_feasibility')
    os.makedirs(out_dir, exist_ok=True)

    target_times, c_ref_samples, T_step, q_init, t_start = (
        load_step2_reference(sim_log_path, n_instants=10))

    model = pin.buildModelFromUrdf(URDF, pin.JointModelFreeFlyer())
    data = model.createData()
    frame_tool_b = model.getFrameId('tool_b')
    frame_torso = model.getFrameId('Link_0')

    print('=' * 84)
    print('  Offline IK feasibility check — step 2 reference trajectory')
    print('  Tool: tool_b, target: anchor_5b = [1.2, -0.3, 0.025]')
    print(f'  T_step = {T_step:.3f} s, t_ss_start = {t_start:.3f} s')
    print(f'  Tol: EE {TOL_EE_M*1000:.0f} mm, Torso {TOL_TORSO_M*1000:.0f} mm')
    print(f'  IK warm-start: actual q at step 2 SS entry from sim log')
    print(f'  Stance arm A: UNCONSTRAINED (more permissive than reality)')
    print('=' * 84)

    rows = []
    q_prev = q_init
    for k, (tt, c_ref) in enumerate(zip(target_times, c_ref_samples)):
        t_off = tt - t_start
        q_sol, ok, err_ee, err_to, n_it = ik_solve(
            model, data, frame_tool_b, frame_torso,
            ANCHOR_5B, c_ref, q_prev)
        if ok:
            q_prev = q_sol   # warm-start next instant
        # Decide which constraint dominates if failed
        worst = None
        if not ok:
            if err_ee > TOL_EE_M and err_to <= TOL_TORSO_M:
                worst = 'EE'
            elif err_to > TOL_TORSO_M and err_ee <= TOL_EE_M:
                worst = 'TORSO'
            elif err_ee > TOL_EE_M and err_to > TOL_TORSO_M:
                worst = 'BOTH'
            else:
                worst = '?'
        row = {
            'k': k, 't_offset_s': float(t_off), 't_sim_s': float(tt),
            'c_ref': c_ref.tolist(),
            'converged': bool(ok),
            'ee_residual_mm': err_ee * 1000.0,
            'torso_residual_mm': err_to * 1000.0,
            'iters': int(n_it),
            'violated': worst,
        }
        rows.append(row)
        flag = 'OK ' if ok else 'FAIL'
        print(f"  k={k}  t_off={t_off:6.2f}s  c_ref=[{c_ref[0]:+.3f},"
              f"{c_ref[1]:+.3f},{c_ref[2]:+.3f}]  "
              f"{flag}  ee={err_ee*1000:7.2f}mm  torso={err_to*1000:7.2f}mm"
              f"  iters={n_it:3d}"
              + (f"  worst={worst}" if not ok else ''))

    n_ok = sum(1 for r in rows if r['converged'])
    print('=' * 84)
    print(f'  IK converged at {n_ok}/{len(rows)} instants')
    if n_ok < len(rows):
        # Find earliest failure
        for r in rows:
            if not r['converged']:
                print(f'  EARLIEST FAILURE: k={r["k"]}, t_off={r["t_offset_s"]:.2f}s,'
                      f' ee={r["ee_residual_mm"]:.1f}mm,'
                      f' torso={r["torso_residual_mm"]:.1f}mm,'
                      f' violated={r["violated"]}')
                break
    print('=' * 84)
    print('  VERDICT:')
    if n_ok == len(rows):
        print('  → Reference is KINEMATICALLY FEASIBLE.')
        print('  → WBC architecture is preventing a reachable solution.')
        print('  → Do NOT implement active-DS yet.')
    else:
        print('  → Reference is KINEMATICALLY INFEASIBLE.')
        print('  → active-DS split is necessary.')

    # Save report + JSON
    out_json = os.path.join(out_dir, 'ik_feasibility.json')
    with open(out_json, 'w') as f:
        json.dump({'T_step': T_step, 't_ss_start': t_start,
                   'anchor_5b': ANCHOR_5B.tolist(),
                   'tol_ee_mm': TOL_EE_M * 1000.0,
                   'tol_torso_mm': TOL_TORSO_M * 1000.0,
                   'instants': rows}, f, indent=2)
    out_txt = os.path.join(out_dir, 'ik_feasibility.txt')
    with open(out_txt, 'w') as f:
        f.write('Step 2 IK feasibility — Pinocchio-only, position constraints\n')
        f.write(f'T_step={T_step:.3f}s, t_ss_start={t_start:.3f}s\n')
        f.write(f'Target: tool_b at anchor_5b={ANCHOR_5B.tolist()}\n')
        f.write(f'Tol: ee={TOL_EE_M*1000:.0f}mm, torso={TOL_TORSO_M*1000:.0f}mm\n\n')
        f.write(f"{'k':>3s} {'t_off':>7s} "
                f"{'c_ref_x':>9s} {'c_ref_y':>9s} {'c_ref_z':>9s} "
                f"{'OK':>4s} {'ee_mm':>9s} {'torso_mm':>10s} "
                f"{'iter':>5s} {'worst':>6s}\n")
        f.write('-' * 84 + '\n')
        for r in rows:
            ok = 'YES' if r['converged'] else 'NO'
            worst = r['violated'] if r['violated'] else ''
            c = r['c_ref']
            f.write(
                f"{r['k']:>3d} {r['t_offset_s']:>7.2f} "
                f"{c[0]:>+9.3f} {c[1]:>+9.3f} {c[2]:>+9.3f} "
                f"{ok:>4s} {r['ee_residual_mm']:>9.2f} "
                f"{r['torso_residual_mm']:>10.2f} "
                f"{r['iters']:>5d} {worst:>6s}\n")
        f.write(f'\nConverged: {n_ok}/{len(rows)}\n')
    print(f'  Outputs: {out_dir}')


if __name__ == '__main__':
    main()
