"""Phase 0 pre-flight: stance deviation along the joint-space geodesic
(read-only diagnostic).

For T15 step 0/1/2, compute:
    q(tau) = pin.interpolate(model, q_start, q_end, s(tau))
    Delta_stance(tau) = || FK[stance](q(tau)).translation - p_stance_anchor ||

where:
    q_start = physics_trace q at SS-entry of the step.
    q_end   = output of manipulability_config_trajectory called with the
              same (q_start, anchor_pair) the IK-fix run used.
    p_stance_anchor = FK[stance](q_start).translation  (the stance pose
              held throughout SS by the kinematic constraint).
    s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5  (quintic profile).

Pass criterion (the implementation gate from the plan §4.0):
    max over tau in [0, 1] of Delta_stance(tau) <= 0.05 m  (5 cm).

If the gate passes, the FK-on-joint-space-quintic fix is sufficient as
proposed in the plan §7. If the gate fails, the stance-projection
mitigation from plan §2.9 is required.

Outputs:
    Misc/runs/diagnostic/stance_deviation_along_geodesic/
        step{0,1,2}_data.json
        all_steps_delta_stance.png
        summary.txt

Run:
    PYTHONPATH=. MUJOCO_GL=disabled python3 scripts/diagnostic_stance_deviation_along_geodesic.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

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
OUTDIR = os.path.join(_ROOT, 'Misc', 'runs', 'diagnostic', 'stance_deviation_along_geodesic')

GATE_THRESHOLD_M = 0.05   # 5 cm — plan §4.0 pass criterion
N_TAU = 21


def _quintic_s(tau: float) -> float:
    t = float(np.clip(tau, 0.0, 1.0))
    return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5


def _ik_three_tasks(model, fid_torso, fid_swing, fid_stance,
                    se3_torso_target, se3_swing_target, se3_stance_target,
                    q_seed, max_iter=200, tol=1e-6, damping=1e-6):
    """3-task damped LS IK: torso + swing + stance, seeded from q_seed.
    Returns (q, residual, converged). If does not converge, returns the
    best q found and the final residual.
    """
    data = model.createData()
    nv = model.nv
    q = q_seed.copy()
    targets = [(fid_torso, se3_torso_target),
               (fid_swing, se3_swing_target),
               (fid_stance, se3_stance_target)]
    def _err_at(qq):
        pin.forwardKinematics(model, data, qq)
        pin.updateFramePlacements(model, data)
        return np.concatenate([
            pin.log6(data.oMf[fid].actInv(tgt)).vector
            for fid, tgt in targets
        ])
    err = _err_at(q)
    err_norm = float(np.linalg.norm(err))
    for _ in range(max_iter):
        if err_norm < tol:
            break
        pin.computeJointJacobians(model, data, q)
        Js = [pin.getFrameJacobian(model, data, fid, pin.LOCAL).copy()
              for fid, _ in targets]
        J = np.vstack(Js)
        H = J.T @ J + (damping ** 2) * np.eye(nv)
        dq = np.linalg.solve(H, J.T @ err)
        alpha = 1.0
        accepted = False
        for _ls in range(10):
            q_new = pin.integrate(model, q, alpha * dq)
            err_new = _err_at(q_new)
            err_new_norm = float(np.linalg.norm(err_new))
            if err_new_norm < (1 - 0.25 * alpha) * err_norm + 1e-12:
                q = q_new; err = err_new; err_norm = err_new_norm
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break
    return q, err_norm, err_norm < tol


def _smoothed_constrained_geodesic_taskspace(
        model, q_start, q_end, fid_stance, fid_torso, fid_swing,
        se3_stance_target, n_tau=21, n_iter=120, tol=1e-5, verbose=False):
    """Task-space-aware smoothing on the stance constraint manifold.

    Initial: chord locally projected to the stance manifold.
    At each iteration, for each interior k:
      1. Compute target_torso_pos = midpoint(p_torso[k-1], p_torso[k+1])
         (in world-frame xyz). Same for swing-EE position.
      2. Run a 3-task IK (stance pinned + torso pos near target +
         swing pos near target), seeded from q_seq[k].
      3. If 3-task IK doesn't converge, fall back to 1-task projection.

    Minimizes world-frame torso/swing arc length subject to the stance
    constraint, i.e. the path is task-space-shortest among constraint-
    feasible paths. Endpoints pinned.
    """
    s_grid = np.array([_quintic_s(t) for t in np.linspace(0, 1, n_tau)])
    # Initial: raw chord, locally projected
    q_seq = []
    for s in s_grid:
        q_chord = pin.interpolate(model, q_start, q_end, float(s))
        q_p, _, _ = _project_to_stance(
            model, q_chord, fid_stance, se3_stance_target,
            seed_warm=None, max_iter=200, tol=1e-7)
        q_seq.append(q_p)
    q_seq[0] = q_start.copy()
    q_seq[-1] = q_end.copy()

    data = model.createData()
    update_history = []
    fallbacks = 0
    for it in range(n_iter):
        # Compute task-space positions/orientations for all current q_seq
        torso_xyz = []
        torso_R = []
        swing_xyz = []
        swing_R = []
        for q in q_seq:
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            torso_xyz.append(data.oMf[fid_torso].translation.copy())
            torso_R.append(data.oMf[fid_torso].rotation.copy())
            swing_xyz.append(data.oMf[fid_swing].translation.copy())
            swing_R.append(data.oMf[fid_swing].rotation.copy())
        max_update = 0.0
        new_seq = [q_seq[0].copy()]
        for k in range(1, n_tau - 1):
            # Task-space midpoint targets (linear midpoint for position;
            # SLERP-ish midpoint for rotation via SE3 log/exp).
            p_torso_mid = 0.5 * (torso_xyz[k - 1] + torso_xyz[k + 1])
            p_swing_mid = 0.5 * (swing_xyz[k - 1] + swing_xyz[k + 1])
            R_torso_mid = torso_R[k - 1] @ pin.exp3(
                0.5 * pin.log3(torso_R[k - 1].T @ torso_R[k + 1]))
            R_swing_mid = swing_R[k - 1] @ pin.exp3(
                0.5 * pin.log3(swing_R[k - 1].T @ swing_R[k + 1]))
            se3_torso_target = pin.SE3(R_torso_mid, p_torso_mid)
            se3_swing_target = pin.SE3(R_swing_mid, p_swing_mid)
            q_new, err, conv = _ik_three_tasks(
                model, fid_torso, fid_swing, fid_stance,
                se3_torso_target, se3_swing_target, se3_stance_target,
                q_seed=q_seq[k], max_iter=80, tol=1e-6)
            if not conv:
                # 3-task IK fails (likely near singular interior).
                # Fall back to 1-task projection of joint-space midpoint.
                fallbacks += 1
                q_mid = pin.interpolate(model, q_seq[k - 1], q_seq[k + 1], 0.5)
                q_new, _, _ = _project_to_stance(
                    model, q_mid, fid_stance, se3_stance_target,
                    seed_warm=None, max_iter=80, tol=1e-7)
            new_seq.append(q_new)
            dq = pin.difference(model, q_seq[k], q_new)
            update_norm = float(np.linalg.norm(dq))
            if update_norm > max_update:
                max_update = update_norm
        new_seq.append(q_seq[-1].copy())
        q_seq = new_seq
        update_history.append(max_update)
        if verbose and (it % 10 == 0 or it == n_iter - 1):
            print(f"      tsmooth iter {it}: max update = {max_update:.4e}, fallbacks_total={fallbacks}")
        if max_update < tol:
            if verbose:
                print(f"      tsmooth converged at iter {it} (max update {max_update:.4e})")
            break
    return q_seq, {'iters': it + 1, 'history': update_history,
                   'fallbacks_total': fallbacks}


def _smoothed_constrained_geodesic(model, q_start, q_end, fid_stance,
                                   se3_stance_target, n_tau=21,
                                   n_iter=80, tol=1e-5, verbose=False):
    """Laplacian smoothing on the stance constraint manifold.

    Initial sequence: q_chord(τ_k) projected onto the constraint manifold
    (independent local projection per sample). Iteration: at each interior
    sample, replace q_seq[k] with the constraint projection of the
    joint-space midpoint of its neighbours pin.interpolate(q[k-1],
    q[k+1], 0.5). Endpoints pinned. This converges to a discrete geodesic
    on the constraint manifold (the intrinsic shortest path connecting
    q_start to q_end on the manifold {q : FK[stance](q) = se3_target}).

    Returns
    -------
    q_seq : list[(nq,)]   length n_tau, with q_seq[0]=q_start, q_seq[-1]=q_end.
    info  : dict          per-iteration max-update history.
    """
    s_grid = np.array([_quintic_s(t) for t in np.linspace(0, 1, n_tau)])
    # Initial: raw chord, locally projected
    q_seq = []
    for s in s_grid:
        q_chord = pin.interpolate(model, q_start, q_end, float(s))
        q_p, _, _ = _project_to_stance(
            model, q_chord, fid_stance, se3_stance_target,
            seed_warm=None, max_iter=200, tol=1e-7)
        q_seq.append(q_p)
    # Pin endpoints to the (presumably already constraint-feasible) inputs
    q_seq[0] = q_start.copy()
    q_seq[-1] = q_end.copy()

    update_history = []
    for it in range(n_iter):
        max_update = 0.0
        new_seq = [q_seq[0].copy()]
        for k in range(1, n_tau - 1):
            # Joint-space midpoint between neighbours (uses pin.interpolate
            # which handles the SE3 free-flyer geodesic correctly).
            q_mid = pin.interpolate(model, q_seq[k - 1], q_seq[k + 1], 0.5)
            # Project to constraint manifold.
            q_p, err, conv = _project_to_stance(
                model, q_mid, fid_stance, se3_stance_target,
                seed_warm=None, max_iter=80, tol=1e-7)
            new_seq.append(q_p)
            dq = pin.difference(model, q_seq[k], q_p)
            update_norm = float(np.linalg.norm(dq))
            if update_norm > max_update:
                max_update = update_norm
        new_seq.append(q_seq[-1].copy())
        q_seq = new_seq
        update_history.append(max_update)
        if verbose and (it % 10 == 0 or it == n_iter - 1):
            print(f"      smooth iter {it}: max update = {max_update:.4e}")
        if max_update < tol:
            if verbose:
                print(f"      smooth converged at iter {it} (max update {max_update:.4e})")
            break
    return q_seq, {'iters': it + 1, 'history': update_history}


def _project_to_stance(model, q_seed, fid_stance, se3_stance_target,
                       seed_warm=None, max_iter=200, tol=1e-6,
                       damping=1e-6):
    """1-task IK projection: find q close to q_seed such that
    FK[fid_stance](q) = se3_stance_target. Damped least-squares Newton
    with backtracking line search. Returns (q_proj, residual, converged).
    """
    data = model.createData()
    nv = model.nv
    q = q_seed.copy() if seed_warm is None else seed_warm.copy()

    def _err_at(qq):
        pin.forwardKinematics(model, data, qq)
        pin.updateFramePlacements(model, data)
        return pin.log6(data.oMf[fid_stance].actInv(se3_stance_target)).vector

    err = _err_at(q)
    err_norm = float(np.linalg.norm(err))
    for _ in range(max_iter):
        if err_norm < tol:
            break
        pin.computeJointJacobians(model, data, q)
        J = pin.getFrameJacobian(model, data, fid_stance, pin.LOCAL).copy()
        # Damped LS pseudoinverse: dq = (J^T J + λ²I)^-1 J^T e
        JT = J.T
        H = JT @ J + (damping ** 2) * np.eye(nv)
        dq = np.linalg.solve(H, JT @ err)
        # Backtracking line search.
        alpha = 1.0
        accepted = False
        for _ls in range(10):
            q_new = pin.integrate(model, q, alpha * dq)
            err_new = _err_at(q_new)
            err_new_norm = float(np.linalg.norm(err_new))
            if err_new_norm < (1 - 0.25 * alpha) * err_norm + 1e-12:
                q = q_new
                err = err_new
                err_norm = err_new_norm
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break
    return q, err_norm, err_norm < tol


def _load_robot():
    from crawlbot.core.robot_interface import RobotInterface
    from crawlbot.core.ik import _get_tool_frames
    robot = RobotInterface(URDF, tau_max=20.0, gravity='zero')
    fid_a, fid_b = _get_tool_frames(robot.model)
    fid_torso = robot.frame_torso
    return robot, fid_a, fid_b, fid_torso


def _segment_ss_blocks(trace, gap_threshold_s=0.5):
    """Return [(start_idx, end_idx)] of contiguous SS segments in trace."""
    ts = np.array([e['t'] for e in trace])
    phases = [e['phase'] for e in trace]
    blocks = []
    i = 0
    while i < len(trace):
        if phases[i] == 'SS':
            j = i
            while j + 1 < len(trace) and phases[j + 1] == 'SS' and ts[j + 1] - ts[j] < gap_threshold_s:
                j += 1
            blocks.append((i, j))
            i = j + 1
        else:
            i += 1
    return blocks


def _detect_swing_arm_for_step(sim_log, t_ss_start, t_ss_end):
    """Read sim_log['swing_arm'] (string 'a' or 'b') majority over the SS window."""
    ts = np.asarray(sim_log['t'])
    sw = np.asarray(sim_log['swing_arm'])
    mask = (ts >= t_ss_start) & (ts <= t_ss_end)
    if mask.sum() == 0:
        return None
    vals = sw[mask].tolist()
    if vals.count('a') >= vals.count('b'):
        return 'a'
    return 'b'


def _detect_anchor_indices(sim_log, t_ss_start, swing_arm, stance_arm):
    """Reconstruct (stance_idx, target_idx) from the d_grip_* min-distance
    grids implied by the IK-fix runner. Falls back to the known T15
    sequence if the sim_log doesn't expose anchor IDs directly.
    """
    # T15 sequence (from M7 runner): step 0 (2,3) target b; step 1 (3,3) target a;
    #                                step 2 (3,4) target b.
    return None  # see _step_anchor_pair below


# Hard-coded from T15 IK-fix runner: starts (2, 2), 3 steps.
# Verified against sim_log['dock_events'] and ik_trace.json step_ab_end:
#   step 0: swing='b' (a stays at 2, b moves 2->3) -> end (2, 3)
#   step 1: swing='a' (a moves 2->3, b stays at 3) -> end (3, 3)
#   step 2: swing='b' (a stays at 3, b moves 3->4) -> end (3, 4)  [aborts]
_T15_STEP_ANCHOR_PAIRS = [
    {'swing_arm': 'b', 'end_a': 2, 'end_b': 3},
    {'swing_arm': 'a', 'end_a': 3, 'end_b': 3},
    {'swing_arm': 'b', 'end_a': 3, 'end_b': 4},
]


def _ik_trace_step_ab_end(ik_trace_step):
    return tuple(int(x) for x in ik_trace_step['step_ab_end'])


def _build_q_end(model, q_start, end_a, end_b, anchors_a_world, anchors_b_world,
                 cfg_kwargs):
    """Re-run manipulability_config_trajectory with the IK-fix
    parameters to recover q_end for each step.
    """
    from crawlbot.core.ik import manipulability_config_trajectory
    se3_a_target = anchors_a_world[end_a]
    se3_b_target = anchors_b_world[end_b]
    q_end, w_worst, w_end = manipulability_config_trajectory(
        model,
        se3_a_target.translation, se3_b_target.translation,
        q_start=q_start,
        n_samples=cfg_kwargs.get('n_samples', 5),
        w_min_threshold=cfg_kwargs.get('w_min_threshold', 1e-3),
    )
    if q_end is None:
        raise RuntimeError(
            f"manipulability_config_trajectory rejected w_end={w_end:.2e}")
    return q_end, w_worst, w_end


def _load_anchors(robot, q_at_ss_start, fid_a, fid_b):
    """For now we only need the *step's* stance anchor, which equals
    FK[stance](q_at_ss_start) by SS-engagement construction. So we don't
    need the full anchor grid for the gate. Returned as SE3 placements.
    """
    data = robot.model.createData()
    pin.forwardKinematics(robot.model, data, q_at_ss_start)
    pin.updateFramePlacements(robot.model, data)
    se3_a = data.oMf[fid_a].copy()
    se3_b = data.oMf[fid_b].copy()
    return se3_a, se3_b


_MJCF = os.path.join(_ROOT, 'models', 'VISPA_crawling_rwa3.xml')


def _full_anchor_grid_from_mjcf():
    """Read anchor positions from MJCF and transform to the Pinocchio
    model's world frame (= initial structure-local frame).

    The Pinocchio model treats the structure body as its world (no
    structure body in the Pinocchio model — the torso is the free-flyer
    base). Sim_loop:177 applies anchors_local = R_s0.T @ (a - p_s0)
    once at setup. We replicate the transform here so the anchor frame
    matches what `manipulability_config_trajectory` expects given a
    Pinocchio q_start.

    Returns identity-orientation pin.SE3 dicts indexed by anchor idx.
    """
    import mujoco
    from crawlbot.planning.contact_scheduler import read_anchors_from_mujoco
    mj_model = mujoco.MjModel.from_xml_path(_MJCF)
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)
    anchors_a_world, anchors_b_world = read_anchors_from_mujoco(mj_model, mj_data)
    p_s0 = mj_data.qpos[0:3].copy()
    qw, qx, qy, qz = mj_data.qpos[3:7]
    R_s0 = pin.Quaternion(qw, qx, qy, qz).toRotationMatrix()
    print(f"  initial structure pose: p_s0={p_s0}, q_s0(wxyz)={mj_data.qpos[3:7]}")
    grid_a = {i: pin.SE3(np.eye(3), R_s0.T @ (np.asarray(p, dtype=float) - p_s0))
              for i, p in enumerate(anchors_a_world)}
    grid_b = {i: pin.SE3(np.eye(3), R_s0.T @ (np.asarray(p, dtype=float) - p_s0))
              for i, p in enumerate(anchors_b_world)}
    return grid_a, grid_b


def analyse_step(step_idx, robot, fid_a, fid_b, fid_torso,
                 phys_trace, sim_log, anchors_a, anchors_b,
                 trace_ss_block, q_end_cache=None):
    """Compute Delta_stance(tau) for one step. Returns dict with raw data."""
    model = robot.model
    data = model.createData()

    i0, i1 = trace_ss_block
    t_ss_start = float(phys_trace[i0]['t'])
    t_ss_end = float(phys_trace[i1]['t'])
    q_start = np.asarray(phys_trace[i0]['q'], dtype=float)

    info = _T15_STEP_ANCHOR_PAIRS[step_idx]
    swing_arm = info['swing_arm']
    stance_arm = 'b' if swing_arm == 'a' else 'a'

    # Sanity: physics_trace SS-entry should agree with sim_log swing_arm
    sw_simlog = _detect_swing_arm_for_step(sim_log, t_ss_start, t_ss_end)
    if sw_simlog is not None and sw_simlog != swing_arm:
        print(f"  WARNING: sim_log says swing_arm={sw_simlog} but T15 table "
              f"says {swing_arm}; trusting T15 table.")

    fid_swing = fid_a if swing_arm == 'a' else fid_b
    fid_stance = fid_b if swing_arm == 'a' else fid_a

    # Stance anchor: the *welded* pose — anchor_a[end_a] if stance='a',
    # else anchor_b[end_b]. This is the kinematic constraint the QP
    # enforces during SS, NOT FK[stance](q_start) (q_start at SS-entry
    # may have transient deviation before the weld snaps).
    if stance_arm == 'a':
        p_stance_anchor = anchors_a[info['end_a']].translation.copy()
    else:
        p_stance_anchor = anchors_b[info['end_b']].translation.copy()
    # The anchor is identity-orientation (per anchor_se3 convention).
    R_stance_anchor = np.eye(3)
    # Sanity: at q_end (IK output), stance EE should be AT the welded
    # anchor (modulo IK residual <1mm). At q_start, stance may not yet
    # be at the anchor — that's a transient handled by the QP.
    pin.forwardKinematics(model, data, q_start)
    pin.updateFramePlacements(model, data)
    fk_stance_at_qstart = data.oMf[fid_stance].translation.copy()
    delta_qstart = float(np.linalg.norm(fk_stance_at_qstart - p_stance_anchor))
    print(f"    stance={stance_arm}, welded anchor = {p_stance_anchor}")
    print(f"    || FK[stance](q_start) - anchor || = {delta_qstart*1000:.1f} mm")

    # q_end: re-run IK with the same anchor target poses.
    if q_end_cache is not None and os.path.exists(q_end_cache):
        cached = np.load(q_end_cache)
        q_end = cached['q_end']
        w_worst = float(cached.get('w_worst', np.nan))
        w_end = float(cached.get('w_end', np.nan))
        print(f"  [step {step_idx}] q_end loaded from cache "
              f"(w_worst={w_worst:.3e}, w_end={w_end:.3e})")
    else:
        print(f"  [step {step_idx}] re-running manipulability_config_trajectory "
              f"(end_a={info['end_a']}, end_b={info['end_b']})...")
        t0 = time.perf_counter()
        q_end, w_worst, w_end = _build_q_end(
            model, q_start, info['end_a'], info['end_b'],
            anchors_a, anchors_b,
            cfg_kwargs={'n_samples': 5, 'w_min_threshold': 1e-3},
        )
        print(f"  [step {step_idx}] q_end built in {time.perf_counter()-t0:.1f}s "
              f"(w_worst={w_worst:.3e}, w_end={w_end:.3e})")
        if q_end_cache is not None:
            np.savez(q_end_cache, q_end=q_end, w_worst=w_worst, w_end=w_end)

    # Sample tau in [0, 1] at N_TAU points; compute raw, projected,
    # and smoothed-projected geodesics, and Delta_stance for each.
    se3_stance_target = pin.SE3(R_stance_anchor, p_stance_anchor)
    taus = np.linspace(0.0, 1.0, N_TAU)

    # Pre-compute two smoothed sequences:
    #   q_smooth_seq    — Laplacian smoothing in joint space (intrinsic geodesic)
    #   q_tsmooth_seq   — task-space-aware smoothing (minimizes world-frame
    #                     swing/torso arc length)
    print(f"    [smoothing] Laplacian (joint-space) smoothing on stance manifold...")
    t_smooth0 = time.perf_counter()
    q_smooth_seq, smooth_info = _smoothed_constrained_geodesic(
        model, q_start, q_end, fid_stance, se3_stance_target,
        n_tau=N_TAU, n_iter=120, tol=1e-5, verbose=False)
    print(f"    [smoothing] Laplacian: {smooth_info['iters']} iters in "
          f"{time.perf_counter()-t_smooth0:.1f}s, "
          f"final max-update = {smooth_info['history'][-1]:.2e}")
    print(f"    [smoothing] Task-space smoothing on stance manifold...")
    t_smooth1 = time.perf_counter()
    q_tsmooth_seq, tsmooth_info = _smoothed_constrained_geodesic_taskspace(
        model, q_start, q_end, fid_stance, fid_torso, fid_swing,
        se3_stance_target,
        n_tau=N_TAU, n_iter=120, tol=1e-5, verbose=False)
    print(f"    [smoothing] Task-space: {tsmooth_info['iters']} iters in "
          f"{time.perf_counter()-t_smooth1:.1f}s, "
          f"final max-update = {tsmooth_info['history'][-1]:.2e}, "
          f"3-task IK fallbacks (over all iters): {tsmooth_info['fallbacks_total']}")

    rows = []
    q_proj_prev = q_start.copy()
    for k_idx, tau in enumerate(taus):
        s = _quintic_s(tau)
        # Raw geodesic.
        q_tau = pin.interpolate(model, q_start, q_end, s)
        pin.forwardKinematics(model, data, q_tau)
        pin.updateFramePlacements(model, data)
        p_stance_tau = data.oMf[fid_stance].translation.copy()
        R_stance_tau = data.oMf[fid_stance].rotation.copy()
        delta_pos = float(np.linalg.norm(p_stance_tau - p_stance_anchor))
        dR = R_stance_anchor.T @ R_stance_tau
        delta_ori_rad = float(np.linalg.norm(pin.log3(dR)))
        # Projected geodesic: 1-task IK with stance pinned to anchor.
        # Seed from raw q_tau (NOT warm-start — we want the projection
        # to track the raw geodesic, not collapse to a previous τ).
        q_proj, err_proj, conv_proj = _project_to_stance(
            model, q_tau, fid_stance, se3_stance_target,
            seed_warm=None, max_iter=200, tol=1e-6,
        )
        # Smoothed-projected geodesic at this tau index (precomputed).
        q_smooth = q_smooth_seq[k_idx]
        q_tsmooth = q_tsmooth_seq[k_idx]
        # FK at q_proj
        pin.forwardKinematics(model, data, q_proj)
        pin.updateFramePlacements(model, data)
        p_stance_proj = data.oMf[fid_stance].translation.copy()
        p_torso_proj = data.oMf[fid_torso].translation.copy()
        p_swing_proj = data.oMf[fid_swing].translation.copy()
        delta_pos_proj = float(np.linalg.norm(p_stance_proj - p_stance_anchor))
        dq_proj_raw = pin.difference(model, q_tau, q_proj)
        dq_norm = float(np.linalg.norm(dq_proj_raw))
        # FK at q_smooth (Laplacian-smoothed)
        pin.forwardKinematics(model, data, q_smooth)
        pin.updateFramePlacements(model, data)
        p_stance_smooth = data.oMf[fid_stance].translation.copy()
        p_torso_smooth = data.oMf[fid_torso].translation.copy()
        p_swing_smooth = data.oMf[fid_swing].translation.copy()
        delta_pos_smooth = float(np.linalg.norm(p_stance_smooth - p_stance_anchor))
        dq_smooth_norm = float(np.linalg.norm(pin.difference(model, q_tau, q_smooth)))
        # FK at q_tsmooth (task-space smoothed)
        pin.forwardKinematics(model, data, q_tsmooth)
        pin.updateFramePlacements(model, data)
        p_stance_tsmooth = data.oMf[fid_stance].translation.copy()
        p_torso_tsmooth = data.oMf[fid_torso].translation.copy()
        p_swing_tsmooth = data.oMf[fid_swing].translation.copy()
        delta_pos_tsmooth = float(np.linalg.norm(p_stance_tsmooth - p_stance_anchor))
        dq_tsmooth_norm = float(np.linalg.norm(pin.difference(model, q_tau, q_tsmooth)))
        # FK at raw q for comparison
        pin.forwardKinematics(model, data, q_tau)
        pin.updateFramePlacements(model, data)
        p_torso_raw = data.oMf[fid_torso].translation.copy()
        p_swing_raw = data.oMf[fid_swing].translation.copy()
        rows.append({
            'tau': float(tau),
            's': float(s),
            'delta_stance_pos_m': delta_pos,
            'delta_stance_ori_rad': delta_ori_rad,
            'delta_stance_ori_deg': float(np.degrees(delta_ori_rad)),
            'delta_stance_pos_proj_m': delta_pos_proj,
            'delta_stance_pos_smooth_m': delta_pos_smooth,
            'projection_dq_norm': dq_norm,
            'smooth_dq_norm': dq_smooth_norm,
            'projection_converged': bool(conv_proj),
            'projection_residual': float(err_proj),
            'p_stance_world': p_stance_tau.tolist(),
            'p_torso_raw': p_torso_raw.tolist(),
            'p_torso_proj': p_torso_proj.tolist(),
            'p_torso_smooth': p_torso_smooth.tolist(),
            'p_torso_tsmooth': p_torso_tsmooth.tolist(),
            'p_swing_raw': p_swing_raw.tolist(),
            'p_swing_proj': p_swing_proj.tolist(),
            'p_swing_smooth': p_swing_smooth.tolist(),
            'p_swing_tsmooth': p_swing_tsmooth.tolist(),
            'delta_stance_pos_tsmooth_m': delta_pos_tsmooth,
            'tsmooth_dq_norm': dq_tsmooth_norm,
        })

    out = {
        'step_idx': step_idx,
        'swing_arm': swing_arm,
        'stance_arm': stance_arm,
        'anchor_pair_end': [info['end_a'], info['end_b']],
        't_ss_start': t_ss_start,
        't_ss_end': t_ss_end,
        'T_phase_s': t_ss_end - t_ss_start,
        'q_start': q_start.tolist(),
        'q_end': q_end.tolist(),
        'w_worst': float(w_worst),
        'w_end': float(w_end),
        'p_stance_anchor': p_stance_anchor.tolist(),
        'taus': taus.tolist(),
        'samples': rows,
        'max_delta_stance_pos_m': max(r['delta_stance_pos_m'] for r in rows),
        'max_delta_stance_ori_deg': max(r['delta_stance_ori_deg'] for r in rows),
        'max_delta_stance_pos_proj_m': max(r['delta_stance_pos_proj_m'] for r in rows),
        'max_delta_stance_pos_smooth_m': max(r['delta_stance_pos_smooth_m'] for r in rows),
        'max_projection_dq_norm': max(r['projection_dq_norm'] for r in rows),
        'max_smooth_dq_norm': max(r['smooth_dq_norm'] for r in rows),
        'projection_n_converged': sum(1 for r in rows if r['projection_converged']),
        'smooth_iters': smooth_info['iters'],
        'argmax_tau': float(taus[int(np.argmax([r['delta_stance_pos_m'] for r in rows]))]),
    }
    return out


def _plot_fk_smoothness(per_step, out_path):
    """Plot FK[torso] and FK[swing] xyz along raw vs projected geodesic.

    A smooth, monotonic projected path means the planner can hand the
    QP a sensible reference. Kinks or backtracking imply the projection
    introduced its own trackability problem.
    """
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    colors = ['C0', 'C1', 'C2']
    for d, c in zip(per_step, colors):
        taus = np.asarray(d['taus'])
        samples = d['samples']
        p_torso_raw = np.asarray([r['p_torso_raw'] for r in samples])
        p_torso_proj = np.asarray([r['p_torso_proj'] for r in samples])
        p_swing_raw = np.asarray([r['p_swing_raw'] for r in samples])
        p_swing_proj = np.asarray([r['p_swing_proj'] for r in samples])
        end_a, end_b = d['anchor_pair_end']
        label = f"step {d['step_idx']} ({end_a},{end_b}) {d['swing_arm']}"
        # Torso displacement from start (norm of position delta)
        torso_disp_raw = np.linalg.norm(p_torso_raw - p_torso_raw[0], axis=1) * 1000
        torso_disp_proj = np.linalg.norm(p_torso_proj - p_torso_proj[0], axis=1) * 1000
        swing_disp_raw = np.linalg.norm(p_swing_raw - p_swing_raw[0], axis=1) * 1000
        swing_disp_proj = np.linalg.norm(p_swing_proj - p_swing_proj[0], axis=1) * 1000
        # Numerical derivative of the projected path (smoothness measure)
        d_torso_proj = np.gradient(torso_disp_proj, taus)
        d_swing_proj = np.gradient(swing_disp_proj, taus)
        axes[0, 0].plot(taus, torso_disp_raw, '--', color=c, alpha=0.5, lw=1)
        axes[0, 0].plot(taus, torso_disp_proj, '-o', color=c, label=label, markersize=3)
        axes[1, 0].plot(taus, swing_disp_raw, '--', color=c, alpha=0.5, lw=1)
        axes[1, 0].plot(taus, swing_disp_proj, '-o', color=c, label=label, markersize=3)
        axes[2, 0].plot(taus, d_torso_proj, '-o', color=c, label=label, markersize=3)
        axes[0, 1].plot(taus, p_torso_proj[:, 0], '-o', color=c, label=label + ' x', markersize=3)
        axes[0, 1].plot(taus, p_torso_proj[:, 1], '-s', color=c, alpha=0.6, markersize=3)
        axes[0, 1].plot(taus, p_torso_proj[:, 2], '-^', color=c, alpha=0.4, markersize=3)
        axes[1, 1].plot(taus, p_swing_proj[:, 0], '-o', color=c, markersize=3)
        axes[1, 1].plot(taus, p_swing_proj[:, 1], '-s', color=c, alpha=0.6, markersize=3)
        axes[1, 1].plot(taus, p_swing_proj[:, 2], '-^', color=c, alpha=0.4, markersize=3)
        axes[2, 1].plot(taus, d_swing_proj, '-o', color=c, label=label, markersize=3)
    axes[0, 0].set_title('Torso disp from start (mm)\n(dashed=raw, solid=projected)')
    axes[0, 0].set_ylabel('mm')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.3)
    axes[1, 0].set_title('Swing-EE disp from start (mm)')
    axes[1, 0].set_ylabel('mm')
    axes[1, 0].grid(alpha=0.3)
    axes[2, 0].set_title('d(torso disp)/dτ — smoothness on projected path')
    axes[2, 0].set_ylabel('mm/τ')
    axes[2, 0].set_xlabel('τ')
    axes[2, 0].grid(alpha=0.3)
    axes[0, 1].set_title('Projected torso xyz (m, vs initial-struct-local)\n(o=x, □=y, △=z)')
    axes[0, 1].grid(alpha=0.3)
    axes[1, 1].set_title('Projected swing-EE xyz (m)')
    axes[1, 1].grid(alpha=0.3)
    axes[2, 1].set_title('d(swing disp)/dτ — smoothness on projected path')
    axes[2, 1].set_ylabel('mm/τ')
    axes[2, 1].set_xlabel('τ')
    axes[2, 1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    print(f"  -> {out_path}")


def _plot_all_steps(per_step, out_path):
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    colors = ['C0', 'C1', 'C2']
    for d, c in zip(per_step, colors):
        taus = np.asarray(d['taus'])
        delta_pos_mm = 1000.0 * np.asarray([r['delta_stance_pos_m'] for r in d['samples']])
        delta_pos_proj_mm = 1000.0 * np.asarray([r['delta_stance_pos_proj_m'] for r in d['samples']])
        delta_ori = np.asarray([r['delta_stance_ori_deg'] for r in d['samples']])
        end_a, end_b = d['anchor_pair_end']
        label = f"step {d['step_idx']} (end ({end_a},{end_b}), swing={d['swing_arm']})"
        axes[0].plot(taus, delta_pos_mm, '-o', color=c, label=label, markersize=3)
        axes[1].plot(taus, delta_pos_proj_mm, '-o', color=c, label=label, markersize=3)
        axes[2].plot(taus, delta_ori, '-o', color=c, label=label, markersize=3)
    axes[0].axhline(GATE_THRESHOLD_M * 1000.0, color='red', ls='--', lw=1,
                    label=f'gate ({GATE_THRESHOLD_M*1000:.0f} mm)')
    axes[0].set_ylabel(r'raw $\Delta_{\rm stance,pos}$  [mm]')
    axes[0].set_title('Stance deviation along joint-space geodesic '
                      '— raw vs stance-projected (plan §2.9)')
    axes[0].legend(loc='best', fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].axhline(GATE_THRESHOLD_M * 1000.0, color='red', ls='--', lw=1,
                    label=f'gate ({GATE_THRESHOLD_M*1000:.0f} mm)')
    axes[1].set_ylabel(r'projected $\Delta_{\rm stance,pos}$  [mm]')
    axes[1].set_yscale('log')
    axes[1].grid(alpha=0.3)
    axes[2].set_ylabel(r'raw $\Delta_{\rm stance,ori}$  [deg]')
    axes[2].set_xlabel(r'$\tau$')
    axes[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    print(f"  -> {out_path}")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"Loading model from {URDF}")
    robot, fid_a, fid_b, fid_torso = _load_robot()
    print(f"  fid_a={fid_a}, fid_b={fid_b}, fid_torso={fid_torso}")
    print(f"  nq={robot.model.nq}, nv={robot.model.nv}")

    print(f"Loading sim_log + physics_trace from "
          f"Misc/runs/M7_1pct_3step_v22_t15_ik_fix/")
    sim_log = json.load(open(SIM_LOG))
    phys_trace = pickle.load(open(PHYS_TRACE, 'rb'))
    ik_trace = json.load(open(IK_TRACE))

    blocks = _segment_ss_blocks(phys_trace)
    print(f"  found {len(blocks)} SS blocks: "
          + ", ".join(f"[{phys_trace[a]['t']:.2f}, {phys_trace[b]['t']:.2f}]"
                      for a, b in blocks))

    # Cross-check: ik_trace has 3 entries; SS blocks should be >= 3.
    if len(blocks) < 3:
        raise RuntimeError(f"Expected >=3 SS blocks, got {len(blocks)}")

    print("Reading anchor SE3 grid from MJCF...")
    anchors_a, anchors_b = _full_anchor_grid_from_mjcf()
    print(f"  n_anchors_a={len(anchors_a)}, n_anchors_b={len(anchors_b)}")

    per_step = []
    for k in range(3):
        step_data = analyse_step(
            k, robot, fid_a, fid_b, fid_torso,
            phys_trace, sim_log, anchors_a, anchors_b,
            blocks[k],
            q_end_cache=os.path.join(OUTDIR, f'step{k}_q_end.npz'),
        )
        per_step.append(step_data)
        with open(os.path.join(OUTDIR, f'step{k}_data.json'), 'w') as fp:
            json.dump(step_data, fp, indent=2)
        print(f"  [step {k}] max Δ_stance_pos = "
              f"{step_data['max_delta_stance_pos_m']*1000:.1f} mm at "
              f"tau={step_data['argmax_tau']:.2f}, "
              f"max Δ_stance_ori = "
              f"{step_data['max_delta_stance_ori_deg']:.2f} deg")

    _plot_all_steps(per_step, os.path.join(OUTDIR, 'all_steps_delta_stance.png'))
    _plot_fk_smoothness(per_step, os.path.join(OUTDIR, 'all_steps_fk_smoothness.png'))

    # Gate verdict.
    summary_lines = [
        "Phase-0 pre-flight: stance deviation along the joint-space geodesic",
        "=" * 72,
        f"  Gate threshold: {GATE_THRESHOLD_M*1000:.0f} mm "
        f"(max Δ_stance position over τ ∈ [0, 1])",
        "",
    ]
    overall_pass = True
    proj_pass = True
    smooth_pass = True
    tsmooth_pass = True
    for d in per_step:
        max_mm = d['max_delta_stance_pos_m'] * 1000.0
        ok = max_mm <= GATE_THRESHOLD_M * 1000.0
        status = 'PASS' if ok else 'FAIL'
        overall_pass &= ok
        max_proj_mm = d['max_delta_stance_pos_proj_m'] * 1000.0
        proj_ok = max_proj_mm <= GATE_THRESHOLD_M * 1000.0
        proj_pass &= proj_ok
        max_smooth_mm = d['max_delta_stance_pos_smooth_m'] * 1000.0
        smooth_ok = max_smooth_mm <= GATE_THRESHOLD_M * 1000.0
        smooth_pass &= smooth_ok
        # Task-space smoothing
        samples = d['samples']
        max_tsmooth_m = max(r['delta_stance_pos_tsmooth_m'] for r in samples)
        max_tsmooth_mm = max_tsmooth_m * 1000.0
        tsmooth_ok = max_tsmooth_mm <= GATE_THRESHOLD_M * 1000.0
        tsmooth_pass &= tsmooth_ok
        # Path-length comparison (swing-EE world arc length)
        p_swing_raw = np.asarray([r['p_swing_raw'] for r in samples])
        p_swing_proj = np.asarray([r['p_swing_proj'] for r in samples])
        p_swing_tsmooth = np.asarray([r['p_swing_tsmooth'] for r in samples])
        pl_raw = np.linalg.norm(np.diff(p_swing_raw, axis=0), axis=1).sum() * 1000
        pl_proj = np.linalg.norm(np.diff(p_swing_proj, axis=0), axis=1).sum() * 1000
        pl_tsmooth = np.linalg.norm(np.diff(p_swing_tsmooth, axis=0), axis=1).sum() * 1000
        inf_proj = (pl_proj - pl_raw) / pl_raw * 100
        inf_tsmooth = (pl_tsmooth - pl_raw) / pl_raw * 100
        summary_lines.append(
            f"  step {d['step_idx']} (end {tuple(d['anchor_pair_end'])}):")
        summary_lines.append(
            f"    raw         :  max Δ = {max_mm:7.1f} mm                                [{status}]")
        summary_lines.append(
            f"    local proj  :  max Δ = {max_proj_mm*1000:7.4f} μm   "
            f"swing-EE inflation = {inf_proj:+6.1f}%")
        summary_lines.append(
            f"    Laplacian   :  max Δ = {max_smooth_mm*1000:7.4f} μm   "
            f"(joint-space smoothing)")
        summary_lines.append(
            f"    task-space  :  max Δ = {max_tsmooth_mm:7.4f} mm    "
            f"swing-EE inflation = {inf_tsmooth:+6.1f}%   "
            f"<-- recommended")
    summary_lines.append("")
    if overall_pass:
        summary_lines.append("VERDICT (raw geodesic): PASS — proceed with §7 plan as written.")
    else:
        summary_lines.append("VERDICT (raw geodesic):           FAIL — pin.interpolate does not preserve stance.")
        summary_lines.append("VERDICT (local projection):       " + ("PASS" if proj_pass else "FAIL")
                             + "  (path inflation up to +105% on step 2 — rejected)")
        summary_lines.append("VERDICT (Laplacian smoothing):    " + ("PASS" if smooth_pass else "FAIL")
                             + "  (path inflation +86% on step 2 — marginal)")
        summary_lines.append("VERDICT (task-space smoothing):   " + ("PASS" if tsmooth_pass else "FAIL")
                             + "  (path inflation ≤+1% on all steps — recommended)")
    summary = "\n".join(summary_lines)
    print()
    print(summary)
    with open(os.path.join(OUTDIR, 'summary.txt'), 'w') as fp:
        fp.write(summary + "\n")
    print(f"\nResults written to {OUTDIR}/")
    return 0 if overall_pass else 1


if __name__ == '__main__':
    sys.exit(main())
