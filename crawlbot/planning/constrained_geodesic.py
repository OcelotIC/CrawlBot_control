"""Task-space-smoothed constrained geodesic on the stance manifold.

This module computes a sequence of robot configurations
``q_seq[0..N-1]`` connecting ``q_start`` to ``q_end`` such that:

1. ``q_seq[0] = q_start`` and ``q_seq[N-1] = q_end`` (endpoints pinned).
2. ``FK[fid_stance](q_seq[k]).translation = p_anchor`` for all k
   (stance-EE constraint satisfied to numerical tolerance,
   typically <40 μm).
3. The world-frame torso and swing-EE arc lengths are minimised
   over the choice of interior ``q_seq[k]``.

The output is a discrete reference path on the stance constraint
manifold ``M_stance := {q : FK[fid_stance](q).translation =
p_anchor}``. Used by the SS planners (TorsoPlanner, SwingPlanner)
under ``cfg.reference_source = 'joint_space_fk'`` to produce
kinematically-consistent torso + swing references via FK.

Algorithm per
``docs/architecture/T15_step2_diagnosis_and_resolution.md`` §7
and ``/root/.claude/plans/magical-munching-book.md`` §2.2:

    1. Initial sequence: q_chord(τ_k) projected to M_stance
       per-sample (1-task IK).
    2. Iterate: for each interior k, solve a 3-task IK with stance
       pinned + torso/swing targets at the task-space midpoints
       of FK(q_seq[k-1]) and FK(q_seq[k+1]).
    3. If 3-task IK diverges (typically near singular interior),
       fall back to 1-task projection of the joint-space midpoint.
    4. Stop when max τ-update < tol.

Phase-0 measurement on T15 (commit eb38c72):
- All 3 steps converge in ≤120 iterations (~0.3 s wall-clock).
- Stance-EE deviation: ≤40 μm (gate 50 mm).
- Step-2 swing-EE world-frame inflation vs raw chord: +0.2 %.
- Zero 3-task IK fallbacks across all interior τ × all iterations.

See `results/diagnostic/stance_deviation_along_geodesic/` for
full data and `PHASE0_FINDINGS.md` for the analysis.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pinocchio as pin


def _quintic_s(tau: float) -> float:
    """Quintic time-scaling s(τ) = 10τ³ − 15τ⁴ + 6τ⁵.

    Boundary values: s(0)=0, s(1)=1, s'(0)=s'(1)=0,
    s''(0)=s''(1)=0. Used here only to seed the initial chord
    projection; not used at runtime by the planners.
    """
    t = float(np.clip(tau, 0.0, 1.0))
    return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5


def project_to_stance(
    model: pin.Model,
    q_seed: np.ndarray,
    fid_stance: int,
    se3_stance_target: pin.SE3,
    max_iter: int = 200,
    tol: float = 1e-6,
    damping: float = 1e-6,
) -> tuple[np.ndarray, float, bool]:
    """1-task damped-LS IK projecting q_seed onto the stance manifold.

    Find q close to q_seed (under pseudo-inverse local distance)
    such that ``FK[fid_stance](q) = se3_stance_target``.
    Backtracking line search; pin.LOCAL frame for the task error.

    Returns
    -------
    q : (nq,) ndarray
        Projected configuration. If IK fails, returns the best q
        found so far.
    residual : float
        Final ‖log6(M.actInv(target))‖.
    converged : bool
        ``residual < tol``.
    """
    data = model.createData()
    nv = model.nv
    q = q_seed.copy()

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
        JT = J.T
        H = JT @ J + (damping ** 2) * np.eye(nv)
        dq = np.linalg.solve(H, JT @ err)
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


def ik_three_tasks(
    model: pin.Model,
    fid_torso: int,
    fid_swing: int,
    fid_stance: int,
    se3_torso_target: pin.SE3,
    se3_swing_target: pin.SE3,
    se3_stance_target: pin.SE3,
    q_seed: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-6,
    damping: float = 1e-6,
) -> tuple[np.ndarray, float, bool]:
    """3-task damped-LS IK: torso + swing + stance, seeded from q_seed.

    Stacks the three frame Jacobians (each LOCAL frame, 6×nv)
    vertically and solves the regularised normal equations with
    backtracking line search.

    Returns ``(q, residual, converged)``. If does not converge,
    returns the best q found and the final stacked task error norm.
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
        J = np.vstack([
            pin.getFrameJacobian(model, data, fid, pin.LOCAL).copy()
            for fid, _ in targets
        ])
        H = J.T @ J + (damping ** 2) * np.eye(nv)
        dq = np.linalg.solve(H, J.T @ err)
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


def smoothed_constrained_geodesic(
    model: pin.Model,
    q_start: np.ndarray,
    q_end: np.ndarray,
    fid_stance: int,
    fid_torso: int,
    fid_swing: int,
    n_tau: int = 21,
    n_iter: int = 120,
    tol: float = 1e-5,
    verbose: bool = False,
    se3_stance_target: Optional[pin.SE3] = None,
) -> tuple[list[np.ndarray], dict]:
    """Compute a stance-feasible, task-space-shortest q-sequence.

    Parameters
    ----------
    model : pin.Model
    q_start, q_end : (nq,) ndarray
        Endpoint configurations, both expected to satisfy the
        stance constraint (the IK that produced q_end places the
        stance EE at the anchor; q_start is the SS-entry state
        which is at the anchor by SS-engagement convention).
    fid_stance, fid_torso, fid_swing : int
        Frame IDs of the stance EE, torso, and swing EE.
    n_tau : int
        Number of samples in the τ-grid (uniform in τ ∈ [0, 1]).
    n_iter : int
        Maximum smoothing iterations.
    tol : float
        Convergence tolerance on max joint-space update per
        iteration (in tangent-vector norm, rad).
    verbose : bool
        Log per-10-iteration progress.
    se3_stance_target : pin.SE3, optional
        Stance-EE target pose. Defaults to ``FK[fid_stance](q_start)``,
        which is the welded anchor pose by SS-engagement convention.

    Returns
    -------
    q_seq : list[(nq,) ndarray] of length n_tau
        Smoothed sequence with q_seq[0]=q_start, q_seq[-1]=q_end.
    info : dict
        Diagnostics: ``'iters'`` (iterations consumed),
        ``'history'`` (per-iteration max-update list),
        ``'fallbacks_total'`` (total 3-task IK fallbacks across all
        iterations × interior samples), ``'final_max_update'``,
        ``'converged'``.
    """
    if se3_stance_target is None:
        data0 = model.createData()
        pin.forwardKinematics(model, data0, q_start)
        pin.updateFramePlacements(model, data0)
        se3_stance_target = data0.oMf[fid_stance].copy()

    s_grid = np.array([_quintic_s(t) for t in np.linspace(0.0, 1.0, n_tau)])
    # Initial sequence: chord locally projected.
    q_seq: list[np.ndarray] = []
    for s in s_grid:
        q_chord = pin.interpolate(model, q_start, q_end, float(s))
        q_p, _, _ = project_to_stance(
            model, q_chord, fid_stance, se3_stance_target,
            max_iter=200, tol=1e-7,
        )
        q_seq.append(q_p)
    q_seq[0] = q_start.copy()
    q_seq[-1] = q_end.copy()

    data = model.createData()
    update_history: list[float] = []
    fallbacks_total = 0
    final_iter = 0

    for it in range(n_iter):
        final_iter = it
        # Cache task-space data for all current q_seq.
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
            # Task-space midpoint targets.
            p_torso_mid = 0.5 * (torso_xyz[k - 1] + torso_xyz[k + 1])
            p_swing_mid = 0.5 * (swing_xyz[k - 1] + swing_xyz[k + 1])
            R_torso_mid = torso_R[k - 1] @ pin.exp3(
                0.5 * pin.log3(torso_R[k - 1].T @ torso_R[k + 1]))
            R_swing_mid = swing_R[k - 1] @ pin.exp3(
                0.5 * pin.log3(swing_R[k - 1].T @ swing_R[k + 1]))
            se3_torso_target = pin.SE3(R_torso_mid, p_torso_mid)
            se3_swing_target = pin.SE3(R_swing_mid, p_swing_mid)
            q_new, err_new, conv = ik_three_tasks(
                model, fid_torso, fid_swing, fid_stance,
                se3_torso_target, se3_swing_target, se3_stance_target,
                q_seed=q_seq[k], max_iter=80, tol=1e-6,
            )
            if not conv:
                fallbacks_total += 1
                q_mid = pin.interpolate(
                    model, q_seq[k - 1], q_seq[k + 1], 0.5)
                q_new, _, _ = project_to_stance(
                    model, q_mid, fid_stance, se3_stance_target,
                    max_iter=80, tol=1e-7,
                )
            new_seq.append(q_new)
            dq = pin.difference(model, q_seq[k], q_new)
            update_norm = float(np.linalg.norm(dq))
            if update_norm > max_update:
                max_update = update_norm
        new_seq.append(q_seq[-1].copy())
        q_seq = new_seq
        update_history.append(max_update)

        if verbose and (it % 10 == 0 or it == n_iter - 1):
            print(
                f"      smoothed_constrained_geodesic iter {it}: "
                f"max update = {max_update:.4e}, "
                f"fallbacks_total = {fallbacks_total}"
            )
        if max_update < tol:
            if verbose:
                print(
                    f"      smoothed_constrained_geodesic converged at "
                    f"iter {it} (max update {max_update:.4e})"
                )
            break

    return q_seq, {
        'iters': final_iter + 1,
        'history': update_history,
        'fallbacks_total': fallbacks_total,
        'final_max_update': update_history[-1] if update_history else 0.0,
        'converged': bool(update_history and update_history[-1] < tol),
    }


# ----------------------------------------------------------------------
# Runtime-side helpers: extract per-τ frame references via FK on a
# pre-computed smoothed q-sequence.
# ----------------------------------------------------------------------


def precompute_segment_tangents(
    model: pin.Model,
    q_seq: list[np.ndarray],
) -> list[np.ndarray]:
    """Per-segment tangent vectors ``dq_seg[k] = pin.difference(q_seq[k],
    q_seq[k+1])`` cached at planner.add_phase time.

    Used by ``frame_reference_at_tau`` to compute v_full and a_full
    via finite-difference on the smoothed sequence (plan §2.4 eqs.
    3, 4, 7).
    """
    n_tau = len(q_seq)
    return [
        pin.difference(model, q_seq[k], q_seq[k + 1])
        for k in range(n_tau - 1)
    ]


def frame_reference_at_tau(
    model: pin.Model,
    data: pin.Data,
    q_seq: list[np.ndarray],
    dq_seg: list[np.ndarray],
    fid_target: int,
    tau: float,
    T_phase: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract frame placement, twist, and acceleration in
    ``LOCAL_WORLD_ALIGNED`` for ``fid_target`` at parameter τ along
    a pre-smoothed q-sequence.

    Implements plan §2.3, §2.4, §2.6, §2.7, §2.8:
        q(τ)    = piecewise pin.interpolate between adjacent samples
        v_full  = dq_seg[k_lo] / Δτ                            (eq. 4)
        a_full  = (dq_seg[k_lo] − dq_seg[k_lo−1]) / Δτ²        (eq. 7)
                 (zero on boundary segments)
        v_real  = v_full / T_phase
        a_real  = a_full / T_phase²
        v_LWA   = pin.getFrameVelocity(...).vector             (eq. 10)
        a_LWA   = pin.getFrameAcceleration(...).vector         (eq. 11)

    Parameters
    ----------
    model : pin.Model
    data : pin.Data
        Fresh data buffer for FK calls (caller-owned).
    q_seq : list[(nq,) ndarray]
        Smoothed sequence from ``smoothed_constrained_geodesic``.
    dq_seg : list[(nv,) ndarray] of length len(q_seq)-1
        Per-segment tangents from ``precompute_segment_tangents``.
    fid_target : int
        Frame ID (torso, swing-EE, stance-EE, etc.).
    tau : float
        Parameter in [0, 1].
    T_phase : float
        Real-time duration of the SS phase in seconds.

    Returns
    -------
    p : (3,) ndarray
        Frame translation in world frame.
    R : (3, 3) ndarray
        Frame rotation matrix.
    v6 : (6,) ndarray
        LOCAL_WORLD_ALIGNED twist [v_lin_world(3), ω_world(3)].
    a6 : (6,) ndarray
        LOCAL_WORLD_ALIGNED acceleration [a_lin(3), α(3)].
    """
    n_tau = len(q_seq)
    if n_tau < 2:
        raise ValueError("q_seq must have at least 2 samples")
    if len(dq_seg) != n_tau - 1:
        raise ValueError(
            f"dq_seg length {len(dq_seg)} must equal n_tau - 1 = {n_tau - 1}")
    delta_tau = 1.0 / (n_tau - 1)
    tau_c = float(np.clip(tau, 0.0, 1.0))
    rt = tau_c / delta_tau
    k_lo = min(int(np.floor(rt)), n_tau - 2)
    alpha = float(rt - k_lo)
    q_lo = q_seq[k_lo]
    q_hi = q_seq[k_lo + 1]
    q_tau = pin.interpolate(model, q_lo, q_hi, alpha)

    dq_seg_k = dq_seg[k_lo]
    v_full = dq_seg_k / delta_tau

    # Acceleration: zero on boundary segments (k_lo == 0 or k_lo == n_tau-2).
    if 0 < k_lo < n_tau - 2:
        a_full = (dq_seg_k - dq_seg[k_lo - 1]) / (delta_tau * delta_tau)
    else:
        a_full = np.zeros_like(dq_seg_k)

    v_real = v_full / T_phase
    a_real = a_full / (T_phase * T_phase)

    pin.forwardKinematics(model, data, q_tau, v_real, a_real)
    pin.updateFramePlacements(model, data)

    M = data.oMf[fid_target]
    p = M.translation.copy()
    R = M.rotation.copy()
    v6 = pin.getFrameVelocity(
        model, data, fid_target, pin.LOCAL_WORLD_ALIGNED).vector.copy()
    a6 = pin.getFrameAcceleration(
        model, data, fid_target, pin.LOCAL_WORLD_ALIGNED).vector.copy()
    return p, R, v6, a6


def q_v_real_at_tau(
    model: pin.Model,
    q_seq: list[np.ndarray],
    dq_seg: list[np.ndarray],
    tau: float,
    T_phase: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (q(τ), v_real(τ)) for downstream callers that need the
    raw configuration + tangent without going through frame FK.

    Used by:
      - TorsoPlanner.l_com_reference_at(t) → pin.computeCentroidalMomentum
      - sim_loop._planned_arm_config(t) → q_planned for the M5 mapping
    """
    n_tau = len(q_seq)
    delta_tau = 1.0 / (n_tau - 1)
    tau_c = float(np.clip(tau, 0.0, 1.0))
    rt = tau_c / delta_tau
    k_lo = min(int(np.floor(rt)), n_tau - 2)
    alpha = float(rt - k_lo)
    q_tau = pin.interpolate(model, q_seq[k_lo], q_seq[k_lo + 1], alpha)
    v_real = dq_seg[k_lo] / delta_tau / T_phase
    return q_tau, v_real
