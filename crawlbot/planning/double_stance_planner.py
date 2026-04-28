"""Double-stance planning primitives for the active-DS architecture.

This module supports the inter-step DS phase upgrade described in
``docs/architecture/active_ds_torso_advance.md``. During inter-step
DS, both arms remain welded at their CURRENT anchors; the torso
advances toward a "ready pose" for the upcoming SS step. This
module provides the geometric primitives:

  - ``compute_q_target_DS``: pick the DS-end body pose by
    interpolating between q_DS_start and q_SS_end at fraction β,
    then projecting onto the double-stance constraint manifold via
    a 2-task IK.

  - ``smoothed_constrained_geodesic_double_stance``: build a
    21-sample q-sequence connecting q_DS_start to q_target_DS_proj
    that minimises world-frame torso arc length subject to BOTH
    arms pinned at their current anchors throughout. Mirrors
    ``crawlbot/planning/constrained_geodesic.py``'s SS smoother
    with the swing-arm task replaced by the second stance task.

  - ``reachability_gate``: at runtime during DS active-advance,
    determine whether the upcoming SS's swing arm can reach its
    target anchor from the current body pose. Two-stage check:
    cheap workspace-sphere screen + expensive 1-task IK
    feasibility.

See ``active_ds_torso_advance.md`` §3 for the math derivation and
§4.1 for the public API contract.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pinocchio as pin

from crawlbot.core.ik import solve_ik
from crawlbot.planning.constrained_geodesic import (
    ik_three_tasks,
    project_to_stance,
)


def _quintic_s(tau: float) -> float:
    """Quintic time-scaling s(τ) = 10τ³ − 15τ⁴ + 6τ⁵."""
    t = float(np.clip(tau, 0.0, 1.0))
    return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5


# ----------------------------------------------------------------------
# §4.1: compute_q_target_DS — DS-end body pose
# ----------------------------------------------------------------------


def compute_q_target_DS(
    model: pin.Model,
    q_DS_start: np.ndarray,
    q_SS_end: np.ndarray,
    beta: float,
    fid_a: int,
    fid_b: int,
    anchor_a_now: np.ndarray,
    anchor_b_now: np.ndarray,
    max_iter: int = 500,
    tol: float = 1e-7,
) -> tuple[np.ndarray, float, bool]:
    """Pick the DS-end body pose at fraction β between q_DS_start and
    q_SS_end, projected onto the double-stance manifold.

    Eq. §3.2 of active_ds_torso_advance.md:

        q_target_DS_unconstrained = pin.interpolate(q_DS_start,
                                                     q_SS_end, β)
        q_target_DS_proj = solve_ik(q_target_DS_unconstrained,
                                    targets={fid_a: anchor_a_now,
                                             fid_b: anchor_b_now})

    The interpolation lives in the 21-D configuration manifold
    (free-flyer + arms); the projection enforces both stance arms
    at their CURRENT anchors (not the next-step's anchors —
    transitions to the new pair happen at SS-undock, not during
    DS).

    Parameters
    ----------
    model : pin.Model
    q_DS_start, q_SS_end : (nq,) ndarray
        Endpoints in joint-space configuration.
    beta : float
        Interpolation fraction in [0, 1]. Recommended β = 0.7 per
        Phase-0 evidence (advances body 70 % of SS-needed
        translation under double-stance constraint).
    fid_a, fid_b : int
        Frame IDs of the two stance EEs.
    anchor_a_now, anchor_b_now : (3,) ndarray
        Stance anchor positions in Pinocchio (initial-structure-
        local) frame. Both arms must be at these anchors at
        q_DS_start AND q_target_DS_proj.

    Returns
    -------
    q_target_DS_proj : (nq,) ndarray
    residual : float
        Final ‖task error‖ from the 2-task IK.
    converged : bool
    """
    q_unconstrained = pin.interpolate(model, q_DS_start, q_SS_end,
                                      float(np.clip(beta, 0.0, 1.0)))
    targets = {
        fid_a: pin.SE3(np.eye(3), np.asarray(anchor_a_now, dtype=float)),
        fid_b: pin.SE3(np.eye(3), np.asarray(anchor_b_now, dtype=float)),
    }
    q_proj, err = solve_ik(model, q_unconstrained, targets,
                           max_iter=max_iter, tol=tol)
    return q_proj, float(err), float(err) < tol


# ----------------------------------------------------------------------
# §4.1: smoothed_constrained_geodesic_double_stance — DS path
# ----------------------------------------------------------------------


def project_to_double_stance(
    model: pin.Model,
    q_seed: np.ndarray,
    fid_a: int,
    fid_b: int,
    se3_a_target: pin.SE3,
    se3_b_target: pin.SE3,
    max_iter: int = 200,
    tol: float = 1e-7,
) -> tuple[np.ndarray, float, bool]:
    """2-task projection: stance_a + stance_b pinned. Wraps solve_ik
    for a consistent return shape with the SS module's
    ``project_to_stance``.
    """
    targets = {fid_a: se3_a_target, fid_b: se3_b_target}
    q_proj, err = solve_ik(model, q_seed, targets,
                           max_iter=max_iter, tol=tol)
    return q_proj, float(err), float(err) < tol


def smoothed_constrained_geodesic_double_stance(
    model: pin.Model,
    q_DS_start: np.ndarray,
    q_target_DS_proj: np.ndarray,
    fid_a: int,
    fid_b: int,
    fid_torso: int,
    se3_stance_a_target: Optional[pin.SE3] = None,
    se3_stance_b_target: Optional[pin.SE3] = None,
    n_tau: int = 11,
    n_iter: int = 80,
    tol: float = 1e-5,
    verbose: bool = False,
) -> tuple[list[np.ndarray], dict]:
    """Build a stance-feasible q-sequence on M_double from
    q_DS_start to q_target_DS_proj minimising world-frame torso
    arc length.

    Mirrors
    ``crawlbot.planning.constrained_geodesic.smoothed_constrained_geodesic``
    with the SWING-arm task replaced by the SECOND stance task —
    both arms pinned, only torso moves.

    Algorithm (per active_ds_torso_advance.md §3.3):

      1. Initial sequence: chord locally projected to M_double
         (independent 2-task IK at each τ_k seed = chord).
      2. Iterate: for each interior k, solve a 3-task IK with both
         stance pinned + torso target at the SE3 midpoint of
         neighbours' FK[fid_torso].
      3. Fall back to 2-task projection of joint-space midpoint
         if 3-task IK diverges (rare on the double-stance manifold
         given its 14 free DoF).
      4. Stop when worst joint-space update < tol.

    Returns
    -------
    q_seq : list of (nq,) ndarray, length n_tau
        Smoothed sequence with q_seq[0] = q_DS_start,
        q_seq[-1] = q_target_DS_proj.
    info : dict
        ``iters``, ``history`` (per-iteration max-update),
        ``fallbacks_total``, ``final_max_update``, ``converged``.
    """
    if se3_stance_a_target is None or se3_stance_b_target is None:
        # Default: take both stance targets from FK at q_DS_start.
        data0 = model.createData()
        pin.forwardKinematics(model, data0, q_DS_start)
        pin.updateFramePlacements(model, data0)
        if se3_stance_a_target is None:
            se3_stance_a_target = data0.oMf[fid_a].copy()
        if se3_stance_b_target is None:
            se3_stance_b_target = data0.oMf[fid_b].copy()

    s_grid = np.array([_quintic_s(t) for t in np.linspace(0.0, 1.0, n_tau)])
    # Initial sequence: chord locally projected.
    q_seq: list[np.ndarray] = []
    for s in s_grid:
        q_chord = pin.interpolate(model, q_DS_start,
                                  q_target_DS_proj, float(s))
        q_p, _, _ = project_to_double_stance(
            model, q_chord, fid_a, fid_b,
            se3_stance_a_target, se3_stance_b_target,
            max_iter=200, tol=1e-7,
        )
        q_seq.append(q_p)
    q_seq[0] = q_DS_start.copy()
    q_seq[-1] = q_target_DS_proj.copy()

    data = model.createData()
    update_history: list[float] = []
    fallbacks_total = 0
    final_iter = 0

    for it in range(n_iter):
        final_iter = it
        # Cache torso pose for all current samples.
        torso_xyz = []
        torso_R = []
        for q in q_seq:
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            torso_xyz.append(data.oMf[fid_torso].translation.copy())
            torso_R.append(data.oMf[fid_torso].rotation.copy())

        max_update = 0.0
        new_seq = [q_seq[0].copy()]
        for k in range(1, n_tau - 1):
            # Task-space midpoint for torso only.
            p_torso_mid = 0.5 * (torso_xyz[k - 1] + torso_xyz[k + 1])
            R_torso_mid = torso_R[k - 1] @ pin.exp3(
                0.5 * pin.log3(torso_R[k - 1].T @ torso_R[k + 1]))
            se3_torso_target = pin.SE3(R_torso_mid, p_torso_mid)
            # 3-task IK: torso + stance_a + stance_b.
            # The constrained_geodesic.ik_three_tasks signature is
            # (fid_torso, fid_swing, fid_stance, se3_torso, se3_swing,
            #  se3_stance, ...). Here we reuse it by mapping
            # 'swing' → stance_b (it just calls solve_ik with three
            # targets, the role labels are documentation only).
            q_new, err_new, conv = ik_three_tasks(
                model, fid_torso, fid_b, fid_a,
                se3_torso_target,
                se3_stance_b_target,
                se3_stance_a_target,
                q_seed=q_seq[k], max_iter=80, tol=1e-6,
            )
            if not conv:
                fallbacks_total += 1
                q_mid = pin.interpolate(model, q_seq[k - 1],
                                        q_seq[k + 1], 0.5)
                q_new, _, _ = project_to_double_stance(
                    model, q_mid, fid_a, fid_b,
                    se3_stance_a_target, se3_stance_b_target,
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
            print(f"      DS smoother iter {it}: "
                  f"max update = {max_update:.4e}, "
                  f"fallbacks_total = {fallbacks_total}")
        if max_update < tol:
            if verbose:
                print(f"      DS smoother converged at iter {it} "
                      f"(max update {max_update:.4e})")
            break

    return q_seq, {
        'iters': final_iter + 1,
        'history': update_history,
        'fallbacks_total': fallbacks_total,
        'final_max_update': update_history[-1] if update_history else 0.0,
        'converged': bool(update_history and update_history[-1] < tol),
    }


# ----------------------------------------------------------------------
# §4.1: reachability_gate — DS-to-SS transition trigger
# ----------------------------------------------------------------------


def reachability_gate(
    model: pin.Model,
    q_now: np.ndarray,
    fid_swing: int,
    anchor_swing_target: np.ndarray,
    arm_max_reach: float = 1.7,
    safety_margin: float = 0.1,
    w_min_threshold: float = 0.02,
    fid_stance_a: Optional[int] = None,
    fid_stance_b: Optional[int] = None,
    se3_stance_a_target: Optional[pin.SE3] = None,
    se3_stance_b_target: Optional[pin.SE3] = None,
    fid_torso: Optional[int] = None,
    ik_max_iter: int = 200,
    ik_tol: float = 1e-4,
    # Body-advance gate (§3.6 v2 — added after Phase 4 found the
    # kinematic gate fires too early when the body hasn't moved).
    p_torso_target_DS: Optional[np.ndarray] = None,
    p_torso_DS_start: Optional[np.ndarray] = None,
    advance_fraction_threshold: float = 0.85,
) -> tuple[bool, dict]:
    """Determine whether the upcoming SS's swing arm can reach
    ``anchor_swing_target`` from the current body pose ``q_now``,
    AND whether the body has advanced enough during DS for the SS
    phase to be dynamically tractable.

    Three-stage check (§3.6 v2):

      1. **Cheap workspace sphere.** Compute
         ``d_reach = ‖FK[fid_torso](q_now) − anchor_swing_target‖``.
         If ``d_reach ≥ arm_max_reach − safety_margin``, the body
         is geometrically too far; gate FAILS.

      2. **Body advance progress** (NEW). If
         ``p_torso_target_DS`` and ``p_torso_DS_start`` are
         supplied, require
         ``‖FK[fid_torso](q_now) − p_torso_DS_start‖ ≥
           advance_fraction_threshold · ‖p_torso_target_DS − p_torso_DS_start‖``.
         The body must have travelled at least
         ``advance_fraction_threshold`` of the planned DS advance.
         Without this check the kinematic reachability fires at k=0
         (the body is "in principle" reachable from the un-advanced
         pose), bypassing the active-advance entirely.

      3. **Expensive 1-task IK feasibility.** Run a 1-task IK
         pinning the swing arm at anchor_swing_target (stance arms
         left free during the IK — we only test "can the swing
         REACH"). If the IK converges and σ_min(J_swing) ≥
         w_min_threshold, gate PASSES.

    Parameters
    ----------
    model : pin.Model
    q_now : (nq,) ndarray
        Current configuration.
    fid_swing : int
        Frame ID of the upcoming-SS swing arm.
    anchor_swing_target : (3,) ndarray
        World/Pinocchio-frame target.
    arm_max_reach, safety_margin : float
        Cheap-check thresholds (default 1.7 m and 0.1 m).
    w_min_threshold : float
        Manipulability lower bound at the IK solution.
    fid_torso, fid_stance_a, fid_stance_b, se3_stance_a/b_target :
        Optional. If all four stance-related kwargs are provided,
        the IK is run as 3-task (swing target + both stance pinned)
        which is closer to the actual SS feasibility check. If
        omitted, run as 1-task swing only (looser).

    Returns
    -------
    passed : bool
    info : dict
        ``d_reach_m``, ``w_swing``, ``ik_residual``, ``stage_failed``
        (one of ``'workspace'``, ``'ik'``, or ``None`` on PASS).
    """
    data = model.createData()
    pin.forwardKinematics(model, data, q_now)
    pin.updateFramePlacements(model, data)
    p_torso = data.oMf[fid_torso].translation if fid_torso is not None \
              else q_now[:3]
    d_reach = float(np.linalg.norm(
        np.asarray(anchor_swing_target, dtype=float) - p_torso))

    # Stage 1: workspace sphere.
    if d_reach >= (arm_max_reach - safety_margin):
        return False, {
            'd_reach_m': d_reach,
            'w_swing': 0.0,
            'ik_residual': float('inf'),
            'advance_fraction': 0.0,
            'stage_failed': 'workspace',
        }

    # Stage 2: body-advance progress (only when supplied).
    advance_fraction = 1.0    # default to "no constraint" if not supplied
    if (p_torso_target_DS is not None and p_torso_DS_start is not None):
        p_target = np.asarray(p_torso_target_DS, dtype=float)
        p_start = np.asarray(p_torso_DS_start, dtype=float)
        full_advance = float(np.linalg.norm(p_target - p_start))
        if full_advance > 1e-6:
            done_advance = float(np.linalg.norm(p_torso - p_start))
            advance_fraction = done_advance / full_advance
            if advance_fraction < advance_fraction_threshold:
                return False, {
                    'd_reach_m': d_reach,
                    'w_swing': 0.0,
                    'ik_residual': float('inf'),
                    'advance_fraction': advance_fraction,
                    'stage_failed': 'body_advance',
                }

    # Stage 3: feasibility IK.
    targets = {
        fid_swing: pin.SE3(np.eye(3),
                           np.asarray(anchor_swing_target, dtype=float)),
    }
    if (fid_stance_a is not None and fid_stance_b is not None
            and se3_stance_a_target is not None
            and se3_stance_b_target is not None):
        targets[fid_stance_a] = se3_stance_a_target
        targets[fid_stance_b] = se3_stance_b_target

    q_test, err = solve_ik(model, q_now, targets,
                           max_iter=ik_max_iter, tol=ik_tol)

    # σ_min(J_swing) at the IK solution.
    pin.computeJointJacobians(model, data, q_test)
    J = pin.getFrameJacobian(model, data, fid_swing,
                             pin.LOCAL_WORLD_ALIGNED)
    # Only linear part for position-only reachability.
    J_lin = J[:3, :]
    sv = np.linalg.svd(J_lin, compute_uv=False)
    w_swing = float(sv[-1])

    ik_passed = (err < ik_tol)
    w_passed = (w_swing >= w_min_threshold)
    if not ik_passed or not w_passed:
        return False, {
            'd_reach_m': d_reach,
            'w_swing': w_swing,
            'ik_residual': float(err),
            'advance_fraction': advance_fraction,
            'stage_failed': 'ik',
        }

    return True, {
        'd_reach_m': d_reach,
        'w_swing': w_swing,
        'ik_residual': float(err),
        'advance_fraction': advance_fraction,
        'stage_failed': None,
    }
