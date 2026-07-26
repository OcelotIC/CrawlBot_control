"""T15 step-2 path-geometry diagnostic (read-only).

Distinguishes three hypotheses for the step-2 closed-loop dock failure
in the IK-fix run, given that the IK output is well-conditioned
(w_end = 5.10e-2 at step 2):

  H1 (time budget): closed-loop sigma_min stays ok; EE just doesn't
                    reach in T_step.
  H2 (reference-path singular): sigma_min drops in the idealised
                    reference path. Both ideal and actual go singular
                    together. Fix: reference shape.
  H3 (QP-induced detour): ideal path stays ok; actual diverges into
                    a singular region. Fix: QP / mapping layer.

Method:
  - At 21 tau samples per SS window, evaluate:
      (a) idealised q via 3-task IK (torso + swing-EE + stance-EE
          targets sourced from sim_log refs at the same sim-time).
      (b) actual q from physics_trace.pkl at the same sim-time
          (interpolated to the tau grid).
  - Compute sigma_min(J_a) * sigma_min(J_b) at both.
  - Also report tracking errors p_torso, q_torso, p_ee_swing, q_ee_swing.

No code change to ik.py / sim_loop / planners / QP / NMPC / MJCF.

Outputs:
  Misc/runs/diagnostic/step2_path_geometry/
    step{0,1,2}_data.json      — tau-sampled raw data
    step{0,1,2}_w.png          — ideal vs actual sigma_min product per step
    step{0,1,2}_tracking.png   — torso / swing pos & rot tracking errors per step
    all_steps_w.png            — multi-panel sigma_min comparison (§5)
"""
from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt
import pinocchio as pin

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.environ.setdefault('MUJOCO_GL', 'disabled')

URDF = os.path.join(_ROOT, 'models', 'VISPA_crawling_fixed.urdf')
SIM_LOG = os.path.join(_ROOT, 'Misc', 'runs', 'M7_1pct_3step_v22_t15_ik_fix', 'sim_log.json')
PHYS_TRACE = os.path.join(_ROOT, 'Misc', 'runs', 'M7_1pct_3step_v22_t15_ik_fix', 'physics_trace.pkl')
IK_TRACE = os.path.join(_ROOT, 'Misc', 'runs', 'M7_1pct_3step_v22_t15_ik_fix', 'ik_trace.json')
OUTDIR = os.path.join(_ROOT, 'Misc', 'runs', 'diagnostic', 'step2_path_geometry')


def _load_robot():
    from crawlbot.core.robot_interface import RobotInterface
    from crawlbot.core.ik import _arm_v_slice, _get_tool_frames
    robot = RobotInterface(URDF, tau_max=20.0, gravity='zero')
    fid_a, fid_b = _get_tool_frames(robot.model)
    sl_a = _arm_v_slice(robot.model, fid_a)
    sl_b = _arm_v_slice(robot.model, fid_b)
    fid_torso = robot.frame_torso
    return robot.model, fid_a, fid_b, sl_a, sl_b, fid_torso


def _quat_xyzw_to_R(q):
    """Pinocchio quaternion convention is xyzw at q[3:7]. Sim_log
    stores q_torso / q_ee as wxyz (MuJoCo convention)."""
    # Detect: sim_log stores wxyz (first comp ~1.0 at neutral).
    # All tests below assume input is wxyz.
    w, x, y, z = q[0], q[1], q[2], q[3]
    return pin.Quaternion(w, x, y, z).toRotationMatrix()


def _ik_three_tasks(model, fid_torso, fid_swing, fid_stance,
                    se3_torso, se3_swing, se3_stance,
                    q_seed, max_iter=2000, tol=1e-6):
    """IK satisfying torso pose + swing EE pose + stance EE pose.

    Damped least-squares Newton iteration with backtracking line
    search on the 18-D task error. Returns (q, err_total, converged).
    """
    nv = model.nv
    data = model.createData()
    q = q_seed.copy()
    targets = [(fid_torso, se3_torso), (fid_swing, se3_swing),
               (fid_stance, se3_stance)]

    def _err_at(qq):
        pin.forwardKinematics(model, data, qq)
        pin.updateFramePlacements(model, data)
        ev = []
        for fid, tgt in targets:
            ev.append(pin.log6(data.oMf[fid].actInv(tgt)).vector)
        return np.concatenate(ev)

    err_total = _err_at(q)
    err_norm_prev = float(np.linalg.norm(err_total))
    for it in range(max_iter):
        pin.computeJointJacobians(model, data, q)
        J_stack = []
        for fid, _ in targets:
            J_stack.append(pin.getFrameJacobian(model, data, fid, pin.LOCAL))
        J_total = np.vstack(J_stack)
        # Damped LS with adaptive damping
        lam = max(1e-6, min(1e-2, err_norm_prev * 1e-3))
        H = J_total.T @ J_total + lam * np.eye(nv)
        dq = np.linalg.solve(H, J_total.T @ err_total)
        # Backtracking line search
        step_size = 1.0
        for _ in range(10):
            q_try = pin.integrate(model, q, step_size * dq)
            err_try = _err_at(q_try)
            err_norm_try = float(np.linalg.norm(err_try))
            if err_norm_try < err_norm_prev:
                q = q_try
                err_total = err_try
                err_norm_prev = err_norm_try
                break
            step_size *= 0.5
        else:
            # No line search step accepted: gradient stuck; bail
            break
        if err_norm_prev < tol:
            return q, err_norm_prev, True
    return q, err_norm_prev, err_norm_prev < 1e-3


def _sigma_min_product(model, data, q, fid_a, fid_b, sl_a, sl_b):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    pin.computeJointJacobians(model, data, q)
    Ja = pin.getFrameJacobian(model, data, fid_a, pin.LOCAL)[:, sl_a]
    Jb = pin.getFrameJacobian(model, data, fid_b, pin.LOCAL)[:, sl_b]
    sa = float(np.linalg.svd(Ja, compute_uv=False)[-1])
    sb = float(np.linalg.svd(Jb, compute_uv=False)[-1])
    return sa, sb, sa * sb


def _interp_q_at_t(trace, t_target):
    """Linear interpolation of q in the physics_trace at time t_target.
    Pinocchio configurations need pin.interpolate, not naive linear blend.
    """
    ts = np.array([e['t'] for e in trace])
    qs = [e['q'] for e in trace]
    # Find bracket
    if t_target <= ts[0]:
        return qs[0]
    if t_target >= ts[-1]:
        return qs[-1]
    idx = int(np.searchsorted(ts, t_target))
    t0, t1 = ts[idx - 1], ts[idx]
    if t1 == t0:
        return qs[idx]
    return None  # caller handles via Pinocchio interpolate using model


def _interp_sim_log(sim_log, key, t_target, dim=None):
    """Sim-log signals are at (near) regular dt; linear interpolate."""
    ts = np.array(sim_log['t'])
    arr = np.asarray(sim_log[key])
    idx = int(np.searchsorted(ts, t_target))
    idx = max(1, min(len(ts) - 1, idx))
    t0, t1 = ts[idx - 1], ts[idx]
    if t1 == t0:
        return arr[idx]
    f = (t_target - t0) / (t1 - t0)
    return (1 - f) * arr[idx - 1] + f * arr[idx]


def _quat_slerp_simlog(sim_log, key, t_target):
    """SLERP for quaternion signals (q_torso, q_ee). sim_log uses wxyz."""
    ts = np.array(sim_log['t'])
    arr = np.asarray(sim_log[key])
    idx = int(np.searchsorted(ts, t_target))
    idx = max(1, min(len(ts) - 1, idx))
    t0, t1 = ts[idx - 1], ts[idx]
    if t1 == t0:
        return arr[idx]
    f = (t_target - t0) / (t1 - t0)
    q0, q1 = arr[idx - 1], arr[idx]
    # Pinocchio Quaternion expects (w, x, y, z); normalize sign for shortest arc.
    if np.dot(q0, q1) < 0:
        q1 = -q1
    qa = pin.Quaternion(q0[0], q0[1], q0[2], q0[3])
    qb = pin.Quaternion(q1[0], q1[1], q1[2], q1[3])
    qs = qa.slerp(f, qb)
    return np.array([qs.w, qs.x, qs.y, qs.z])


def _segment_ss_blocks(trace, gap_threshold_s=0.5):
    ts = np.array([e['t'] for e in trace])
    gaps = np.where(np.diff(ts) > gap_threshold_s)[0].tolist()
    blocks, start = [], 0
    for g in gaps:
        blocks.append((start, g))
        start = g + 1
    blocks.append((start, len(ts) - 1))
    return blocks


def _get_q_at_time(model, trace, t_target):
    """Pinocchio-correct interpolation in q at t_target (free-flyer-aware)."""
    ts = np.array([e['t'] for e in trace])
    if t_target <= ts[0]:
        return np.asarray(trace[0]['q'], dtype=float)
    if t_target >= ts[-1]:
        return np.asarray(trace[-1]['q'], dtype=float)
    idx = int(np.searchsorted(ts, t_target))
    t0, t1 = ts[idx - 1], ts[idx]
    f = float((t_target - t0) / (t1 - t0))
    q0 = np.asarray(trace[idx - 1]['q'], dtype=float)
    q1 = np.asarray(trace[idx]['q'], dtype=float)
    return pin.interpolate(model, q0, q1, f)


def analyse_step(step_idx, swing_arm,
                 t_ss_start, t_ss_end,
                 model, fid_a, fid_b, sl_a, sl_b, fid_torso,
                 sim_log, phys_trace, ik_trace_step,
                 n_tau=21):
    """Per-step §1+§2+§3 analysis.

    Stance pose is read from FK on q_actual at t_ss_start (the moment
    the stance constraint engages). This places the stance target in
    the same Pinocchio frame as sim_log's references.
    """
    data = model.createData()
    print(f"\n=== Step {step_idx}: SS [{t_ss_start:.2f}, {t_ss_end:.2f}] s, "
          f"swing={swing_arm}")
    if swing_arm == 'b':
        fid_swing = fid_b
        fid_stance = fid_a
    else:
        fid_swing = fid_a
        fid_stance = fid_b

    # Stance pose from FK at SS start.
    q_start = _get_q_at_time(model, phys_trace, t_ss_start)
    pin.forwardKinematics(model, data, q_start)
    pin.updateFramePlacements(model, data)
    se3_stance = data.oMf[fid_stance].copy()
    print(f"  stance ({'a' if swing_arm=='b' else 'b'}) "
          f"pos at SS start (Pinocchio frame): {se3_stance.translation}")

    taus = np.linspace(0.0, 1.0, n_tau)
    rows = []
    for k, tau in enumerate(taus):
        t = t_ss_start + tau * (t_ss_end - t_ss_start)
        # ── Idealised path
        p_torso_ref = _interp_sim_log(sim_log, 'p_torso_ref', t)
        q_torso_ref_wxyz = _quat_slerp_simlog(sim_log, 'q_torso_ref', t)
        p_ee_ref = _interp_sim_log(sim_log, 'p_ee_ref', t)
        q_ee_ref_wxyz = _quat_slerp_simlog(sim_log, 'q_ee_ref', t)

        R_torso_ref = _quat_xyzw_to_R(q_torso_ref_wxyz)
        R_ee_ref = _quat_xyzw_to_R(q_ee_ref_wxyz)
        se3_torso_ref = pin.SE3(R_torso_ref, p_torso_ref)
        se3_swing_ref = pin.SE3(R_ee_ref, p_ee_ref)

        # Seed from the actual q at this time — already a good
        # approximation to the ref-satisfying q (closed-loop tracking
        # is at most a few cm / few deg off the references).
        q_seed = _get_q_at_time(model, phys_trace, t)
        q_ideal, err_ideal, conv = _ik_three_tasks(
            model, fid_torso, fid_swing, fid_stance,
            se3_torso_ref, se3_swing_ref, se3_stance,
            q_seed.copy(), max_iter=2000)
        sa_i, sb_i, w_i = _sigma_min_product(model, data, q_ideal,
                                              fid_a, fid_b, sl_a, sl_b)

        # ── Actual path
        q_actual = _get_q_at_time(model, phys_trace, t)
        sa_a, sb_a, w_a = _sigma_min_product(model, data, q_actual,
                                              fid_a, fid_b, sl_a, sl_b)

        # ── Tracking errors at this tau (sim_log, actual vs ref)
        p_torso_actual = _interp_sim_log(sim_log, 'p_torso', t)
        q_torso_actual_wxyz = _quat_slerp_simlog(sim_log, 'q_torso', t)
        p_ee_actual = _interp_sim_log(sim_log, 'p_ee', t)
        q_ee_actual_wxyz = _quat_slerp_simlog(sim_log, 'q_ee', t)
        R_torso_actual = _quat_xyzw_to_R(q_torso_actual_wxyz)
        R_ee_actual = _quat_xyzw_to_R(q_ee_actual_wxyz)
        e_torso_pos = float(np.linalg.norm(p_torso_actual - p_torso_ref))
        e_torso_rot = float(np.linalg.norm(
            pin.log3(R_torso_ref.T @ R_torso_actual)))
        e_swing_pos = float(np.linalg.norm(p_ee_actual - p_ee_ref))
        e_swing_rot = float(np.linalg.norm(
            pin.log3(R_ee_ref.T @ R_ee_actual)))

        # joint-space deltas (arm joints only, q[7:])
        dq_arm = float(np.linalg.norm(
            np.asarray(q_actual)[7:] - np.asarray(q_ideal)[7:]))

        rows.append({
            'tau': float(tau), 't': float(t),
            'ik_ideal_err': float(err_ideal),
            'ik_ideal_converged': bool(conv),
            'sigma_min_a_ideal': sa_i, 'sigma_min_b_ideal': sb_i,
            'w_ideal': w_i,
            'sigma_min_a_actual': sa_a, 'sigma_min_b_actual': sb_a,
            'w_actual': w_a,
            'e_torso_pos_m': e_torso_pos,
            'e_torso_rot_rad': e_torso_rot,
            'e_swing_pos_m': e_swing_pos,
            'e_swing_rot_rad': e_swing_rot,
            'dq_arm_norm_rad': dq_arm,
        })
        if k < 3 or k == n_tau - 1 or not conv:
            print(f"  tau={tau:.2f} t={t:.2f}s  "
                  f"w_ideal={w_i:.3e}  w_actual={w_a:.3e}  "
                  f"e_torso_pos={e_torso_pos*1000:.1f}mm  "
                  f"e_swing_pos={e_swing_pos*1000:.1f}mm  "
                  f"ik_conv={conv}  ik_err={err_ideal:.2e}")

    # Save raw
    out_json = os.path.join(OUTDIR, f'step{step_idx}_data.json')
    with open(out_json, 'w') as f:
        json.dump({'rows': rows,
                   't_ss_start': t_ss_start, 't_ss_end': t_ss_end,
                   'swing_arm': swing_arm,
                   'stance_pos_pinocchio': se3_stance.translation.tolist(),
                   }, f, indent=2)
    print(f"  -> {out_json}")
    return rows


def _plot_step_w(step_idx, rows, out_path):
    taus = [r['tau'] for r in rows]
    w_i = [r['w_ideal'] for r in rows]
    w_a = [r['w_actual'] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(taus, w_i, 'o-', color='#1f77b4', label='ideal (planner refs + 3-task IK)')
    ax.semilogy(taus, w_a, 's--', color='#d62728', label='actual (closed-loop q)')
    ax.set_xlabel(r'$\tau$ (fraction of step-{} SS window)'.format(step_idx))
    ax.set_ylabel(r'$\sigma_{\min}(J_a)\,\sigma_{\min}(J_b)$')
    ax.set_title(f'Step {step_idx} — manipulability along ideal vs actual path')
    ax.grid(True, which='both', alpha=0.3)
    ax.axhline(1e-3, color='gray', linestyle=':', linewidth=0.8,
               label='safety threshold 1e-3')
    ax.legend(loc='best', fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_step_tracking(step_idx, rows, out_path):
    taus = [r['tau'] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    axes[0, 0].plot(taus, [r['e_torso_pos_m'] * 1000 for r in rows], 'o-')
    axes[0, 0].set_ylabel('|e_torso_pos| [mm]')
    axes[0, 0].set_title('Torso position tracking error')
    axes[0, 1].plot(taus, [np.degrees(r['e_torso_rot_rad']) for r in rows], 'o-')
    axes[0, 1].set_ylabel('|e_torso_rot| [°]')
    axes[0, 1].set_title('Torso rotation tracking error')
    axes[1, 0].plot(taus, [r['e_swing_pos_m'] * 1000 for r in rows], 'o-', color='#d62728')
    axes[1, 0].set_ylabel('|e_swing_pos| [mm]')
    axes[1, 0].set_title('Swing-EE position tracking error')
    axes[1, 0].set_xlabel(r'$\tau$')
    axes[1, 1].plot(taus, [np.degrees(r['e_swing_rot_rad']) for r in rows], 'o-', color='#d62728')
    axes[1, 1].set_ylabel('|e_swing_rot| [°]')
    axes[1, 1].set_title('Swing-EE rotation tracking error')
    axes[1, 1].set_xlabel(r'$\tau$')
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    fig.suptitle(f'Step {step_idx} — closed-loop tracking errors (actual − reference)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_all_steps_w(rows_per_step, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for s, ax in enumerate(axes):
        rows = rows_per_step[s]
        taus = [r['tau'] for r in rows]
        ax.semilogy(taus, [r['w_ideal'] for r in rows], 'o-', color='#1f77b4', label='ideal')
        ax.semilogy(taus, [r['w_actual'] for r in rows], 's--', color='#d62728', label='actual')
        ax.set_xlabel(r'$\tau$')
        ax.set_title(f'Step {s}')
        ax.grid(True, which='both', alpha=0.3)
        ax.axhline(1e-3, color='gray', linestyle=':', linewidth=0.8)
        ax.set_ylim(1e-5, 1e0)
    axes[0].set_ylabel(r'$\sigma_{\min}(J_a)\,\sigma_{\min}(J_b)$')
    axes[0].legend(loc='lower left', fontsize=9)
    fig.suptitle('All steps — ideal (planner refs) vs actual (closed-loop) manipulability')
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # Load data
    sim_log = json.load(open(SIM_LOG))
    phys_trace = pickle.load(open(PHYS_TRACE, 'rb'))
    ik_trace = json.load(open(IK_TRACE))

    model, fid_a, fid_b, sl_a, sl_b, fid_torso = _load_robot()
    print(f"Model: nq={model.nq}, nv={model.nv}, "
          f"fid_a={fid_a}, fid_b={fid_b}, fid_torso={fid_torso}")

    # Anchor positions from MJCF
    import mujoco
    MJCF = os.path.join(_ROOT, 'models', 'VISPA_crawling_rwa3.xml')
    mj_model = mujoco.MjModel.from_xml_path(MJCF)
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)
    from crawlbot.planning.contact_scheduler import read_anchors_from_mujoco
    anchors_a, anchors_b = read_anchors_from_mujoco(mj_model, mj_data)

    # Identify SS windows from physics_trace gaps
    blocks = _segment_ss_blocks(phys_trace)
    print(f"SS blocks: {len(blocks)} -> {blocks}")
    assert len(blocks) == 3
    ss_windows = []
    for s, e in blocks:
        ss_windows.append((float(phys_trace[s]['t']), float(phys_trace[e]['t'])))
    print(f"SS windows: {ss_windows}")

    # Anchor pair per step from ik_trace
    pairs = [(int(e['step_ab_end'][0]), int(e['step_ab_end'][1]))
             for e in ik_trace]
    swing_arms = []
    for ev in sim_log['dock_events']:
        swing_arms.append((int(ev['step']), ev['arm']))
    swing_arms_dict = dict(swing_arms)
    swing_arms_dict[2] = 'b'  # step 2 aborts but we know it's b's turn (anchor pair 3->4)
    print(f"Anchor pairs (a_idx, b_idx) per step: {pairs}")
    print(f"Swing arm per step: {swing_arms_dict}")

    # Per-step analysis. Stance pose is recovered from FK on q_actual
    # at SS start (the moment the stance constraint engages), keeping
    # everything in Pinocchio frame consistent with sim_log refs.
    all_rows = []
    for step in range(3):
        swing = swing_arms_dict[step]
        rows = analyse_step(step, swing,
                            ss_windows[step][0], ss_windows[step][1],
                            model, fid_a, fid_b, sl_a, sl_b, fid_torso,
                            sim_log, phys_trace, ik_trace[step])
        all_rows.append(rows)
        _plot_step_w(step, rows, os.path.join(OUTDIR, f'step{step}_w.png'))
        _plot_step_tracking(step, rows, os.path.join(OUTDIR, f'step{step}_tracking.png'))

    _plot_all_steps_w(all_rows, os.path.join(OUTDIR, 'all_steps_w.png'))
    print(f"\nDone. Outputs in {OUTDIR}")


if __name__ == '__main__':
    main()
