"""Inverse kinematics for VISPA docking configurations.

Provides:
    solve_ik              — iterative 6D IK for multiple frame targets
    dock_configuration    — convenience: both tools at anchor poses
    manipulability_config — optimal-manipulability configuration for a dock pair
    manipulability_config_trajectory — trajectory-aware variant (M7 / Candidate 1)
    precompute_torso_map  — offline map of optimal configs for all anchor pairs
    solve_ik_waypoints    — chain of IK solutions along a swing arc
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import pinocchio as pin
from scipy.optimize import minimize as scipy_minimize

from crawlbot.core.robot_interface import FRAME_TOOL_A, FRAME_TOOL_B, _detect_arm_slices


def _get_tool_frames(model: pin.Model):
    """Get tool frame IDs from model by name lookup."""
    return model.getFrameId("tool_a"), model.getFrameId("tool_b")

# Cache arm slices per model (by id) to avoid recomputing
_arm_slice_cache: Dict[int, dict] = {}

def _get_arm_slices(model: pin.Model) -> dict:
    """Get arm velocity-space slices for a model, cached by (id, nv)."""
    key = (id(model), model.nv)
    if key not in _arm_slice_cache:
        _arm_slice_cache[key] = _detect_arm_slices(model)
    return _arm_slice_cache[key]


def _arm_v_slice(model: pin.Model, frame_id: int) -> slice:
    """Return the velocity-space slice for the arm owning frame_id."""
    slices = _get_arm_slices(model)
    # Determine which arm owns this frame by checking if the frame's
    # parent joint falls within arm A or arm B's joint range
    pj = model.frames[frame_id].parentJoint
    a_start = slices['arm_a_v'].start
    a_stop = slices['arm_a_v'].stop
    b_start = slices['arm_b_v'].start
    b_stop = slices['arm_b_v'].stop
    # Parent joint's velocity index
    jv = model.idx_vs[pj]
    if a_start <= jv < a_stop:
        return slices['arm_a_v']
    elif b_start <= jv < b_stop:
        return slices['arm_b_v']
    else:
        raise ValueError(f"Frame {frame_id} parent joint {pj} not in arm A or B")


def _level_torso(q: np.ndarray, level_axis: np.ndarray) -> np.ndarray:
    """Project the free-flyer torso rotation to zero pitch/roll about
    ``level_axis`` (a unit vector in the structure/Pinocchio-world
    frame) while preserving yaw about it.

    Forces the torso body z-axis parallel to ±``level_axis`` (whichever
    side it is currently closest to, so a torso that naturally hangs
    "below" the structure is not flipped 180°). The minimal rotation
    that maps the current torso-z onto the target is left-multiplied
    onto the torso rotation, which removes tilt without touching the
    yaw component about ``level_axis``.

    q[3:7] is the Pinocchio free-flyer quaternion in (x, y, z, w) order.
    """
    qx, qy, qz, qw = q[3], q[4], q[5], q[6]
    R = pin.Quaternion(qw, qx, qy, qz).toRotationMatrix()
    u = R[:, 2]                                  # torso body-z in struct
    n = np.asarray(level_axis, dtype=float)
    n = n / max(np.linalg.norm(n), 1e-12)
    if float(u @ n) < 0.0:
        n = -n                                   # align to nearest side
    c = float(np.clip(u @ n, -1.0, 1.0))
    if c > 1.0 - 1e-12:
        return q                                 # already leveled
    axis = np.cross(u, n)
    s = np.linalg.norm(axis)
    if s < 1e-12:
        # u antiparallel to n after sign pick — should not happen, but
        # rotate 180° about any axis ⟂ u.
        axis = np.array([1.0, 0.0, 0.0])
        if abs(u[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = axis - (axis @ u) * u
        axis /= np.linalg.norm(axis)
        angle = np.pi
    else:
        axis = axis / s
        angle = np.arccos(c)
    R_corr = pin.AngleAxis(angle, axis).toRotationMatrix()
    R_new = R_corr @ R
    quat = pin.Quaternion(R_new)
    q = q.copy()
    q[3] = quat.x
    q[4] = quat.y
    q[5] = quat.z
    q[6] = quat.w
    return q


def solve_ik(
    model: pin.Model,
    q0: np.ndarray,
    targets: Dict[int, pin.SE3],
    max_iter: int = 500,
    tol: float = 1e-8,
    base_gain: float = 0.3,
    q_nominal: Optional[np.ndarray] = None,
    w_posture: float = 0.0,
    level_axis: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """
    Iterative IK placing tool frames at target SE3 poses.

    Uses damped least-squares on the full Jacobian (base + arm joints)
    for each target. When multiple targets are specified, contributions
    are summed — no priority ordering.

    Parameters
    ----------
    model : pin.Model
        Pinocchio model (free-flyer).
    q0 : (nq,) initial configuration guess.
    targets : {frame_id: SE3}
        e.g. {FRAME_TOOL_A: anchor_3a, FRAME_TOOL_B: anchor_3b}
    max_iter : int
    tol : float
        Convergence on sum of ||log6(err)||.
    q_nominal : (nq-7,), optional
        Arm-joint reference pose. When given together with
        ``w_posture > 0`` the posture gradient ``q_nominal − q_arms``
        is projected into the WHOLE-SYSTEM task null space (both tool
        Jacobians stacked) and added to the velocity step each
        iteration, so the base repositions to let the arms extend
        toward ``q_nominal`` without disturbing either tool target.
        Default None ⇒ behaviour bit-identical to the legacy solver.
    w_posture : float
        Weight on the posture regularizer. 0 disables.
    level_axis : (3,), optional
        Unit vector in the structure (Pinocchio-world) frame. When
        given, the torso rotation is projected after every iteration
        so its body-z stays parallel to ±``level_axis`` (pitch/roll
        leveled, yaw free). Default None ⇒ legacy free rotation.

    Returns
    -------
    q : (nq,) converged configuration
    err : float, final error norm
    """
    q = q0.copy()
    if level_axis is not None:
        q = _level_torso(q, level_axis)
    data = model.createData()
    nv = model.nv

    for it in range(max_iter):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        pin.computeJointJacobians(model, data, q)
        posture_active = q_nominal is not None and w_posture > 0.0
        dq = np.zeros(nv)
        err_tot = 0.0
        J_list = []
        err_list = []

        for fid, tgt in targets.items():
            err = pin.log6(data.oMf[fid].actInv(tgt)).vector
            J = pin.getFrameJacobian(model, data, fid, pin.LOCAL)
            err_tot += np.linalg.norm(err)

            if posture_active:
                # Defer to the stacked resolved-rate solve below — the
                # whole-system null-space projector is only consistent
                # with a stacked primary step, not the decoupled one.
                J_list.append(J)
                err_list.append(err)
                continue

            # ── Legacy decoupled path (flags off OR leveling-only) ──
            # Bit-identical to the original solver.
            # Arm-specific joint slice (DOF-generic)
            idx = _arm_v_slice(model, fid)
            n_arm = idx.stop - idx.start

            # Arm joints: primary contribution (low regularization)
            Ja = J[:, idx]
            dq[idx] += np.linalg.solve(
                Ja.T @ Ja + 1e-4 * np.eye(n_arm), Ja.T @ err)

            # Base (free-flyer): scaled by base_gain to control how much
            # the base moves vs the arm. Default 0.3 is conservative (good
            # from neutral). Use 1.0 for chain-seeded IK where the base
            # needs to advance naturally.
            Jb = J[:, :6]
            dq[:6] += np.linalg.solve(
                Jb.T @ Jb + 1e-3 * np.eye(6), Jb.T @ err) * base_gain

        if posture_active:
            # ── Stacked resolved-rate primary + whole-system
            #    null-space posture (consistent projector) ──
            # The per-arm posture form (project into one arm's 6-DOF
            # task null space) has only 1 redundant DOF/arm and
            # empirically drives that arm to a near-singularity (σ_min
            # collapses 100–1000×) without reducing ‖q_arm‖. Bolting a
            # whole-system projector onto the *decoupled* primary step
            # is mathematically inconsistent (the projector assumes the
            # primary step is J⁺·err) and breaks task convergence
            # (err ~0.1–2). So when posture is requested the primary
            # step itself becomes the stacked damped least-squares
            # dq = J⁺·err over BOTH tool tasks (12×nv), and the posture
            # gradient is projected into its true null space
            # N = I − J⁺J (≈8 DOF: base xyz + yaw + arm self-motion).
            # A pull toward q_nominal then recruits BASE motion to let
            # the arms extend — the redundancy the per-arm form cannot
            # reach — while keeping both tools on target. Damped pinv
            # J⁺=Jᵀ(JJᵀ+λ²I)⁻¹ avoids amplifying near-singular task
            # directions. q_nominal[k] ↔ q[7+k] ↔ v-index 6+k, so an
            # arm v-slice s maps to q_nominal[s.start-6:s.stop-6] and
            # q[s.start+1:s.stop+1]. Reached only when q_nominal is
            # not None and w_posture > 0 ⇒ legacy/leveling-only paths
            # are untouched.
            J_full = np.vstack(J_list)               # (m_t, nv)
            err_stack = np.concatenate(err_list)
            lam = 1e-3
            m_t = J_full.shape[0]

            # Constraint-manifold reduction. When leveling, the base
            # angular velocity is restricted to pure yaw about
            # ``n_body``. Doing that *post hoc* (zeroing dq[3:6]'s
            # off-yaw part after a full-space minimum-norm J⁺ solve)
            # makes J⁺ keep spending task correction on the forbidden
            # base-tilt DOFs every iteration → limit cycle, err never
            # drops below ~0.1–1 (verified: err 0.4 leveled vs 1e-7
            # unleveled). Folding the constraint into the column space
            # — solve the stacked task in reduced coordinates
            # [base_lin(3); yaw(1); arm] that *cannot express* base
            # tilt — removes the conflict so both the task converges
            # and the null space is consistent.
            if level_axis is not None:
                qx, qy, qz, qw = q[3], q[4], q[5], q[6]
                R_b = pin.Quaternion(qw, qx, qy, qz).toRotationMatrix()
                n = np.asarray(level_axis, dtype=float)
                n = n / max(np.linalg.norm(n), 1e-12)
                n_body = R_b.T @ n
                n_body = n_body / max(np.linalg.norm(n_body), 1e-12)
                E = np.zeros((nv, nv - 2))           # reduced → full
                E[0:3, 0:3] = np.eye(3)              # base linear
                E[3:6, 3] = n_body                   # base yaw only
                E[6:nv, 4:nv - 2] = np.eye(nv - 6)   # arm joints
            else:
                E = np.eye(nv)

            J_r = J_full @ E                          # (m_t, nr)
            nr = E.shape[1]
            J_r_pinv = J_r.T @ np.linalg.solve(
                J_r @ J_r.T + lam ** 2 * np.eye(m_t), np.eye(m_t))
            dq_r = J_r_pinv @ err_stack               # primary task
            N_r = np.eye(nr) - J_r_pinv @ J_r
            qn = np.asarray(q_nominal)
            dq_post = np.zeros(nv)
            slices = _get_arm_slices(model)
            for s in (slices['arm_a_v'], slices['arm_b_v']):
                dq_post[s] = (qn[s.start - 6: s.stop - 6]
                              - q[s.start + 1: s.stop + 1])
            # E has orthonormal columns ⇒ Eᵀ maps the arm posture
            # gradient into reduced coords (base-lin & yaw entries
            # become 0: no base posture target).
            dq_r = dq_r + w_posture * (N_r @ (E.T @ dq_post))
            dq = E @ dq_r

        # Pitch/roll leveling — constraint-manifold descent. Restrict
        # the base angular velocity to pure yaw about ``level_axis``
        # (structure frame) so the solver only ever moves along the
        # leveled manifold, instead of fighting a post-hoc projection
        # (which causes a limit cycle and false "infeasible"). Mirrors
        # dock_configuration_fixed_rotation, but keeps 1 rotational DOF
        # (yaw) free instead of zeroing all 3. Pinocchio free-flyer nv
        # layout is [linear(3); angular(3)] in the body frame, so the
        # leveled yaw axis must be expressed in the body frame.
        if level_axis is not None:
            qx, qy, qz, qw = q[3], q[4], q[5], q[6]
            R_b = pin.Quaternion(qw, qx, qy, qz).toRotationMatrix()
            n = np.asarray(level_axis, dtype=float)
            n = n / max(np.linalg.norm(n), 1e-12)
            n_body = R_b.T @ n
            n_body = n_body / max(np.linalg.norm(n_body), 1e-12)
            omega = dq[3:6]
            dq[3:6] = (omega @ n_body) * n_body   # keep yaw only

        alpha = min(1.0, 0.5 / max(np.max(np.abs(dq)), 1e-10))
        q = pin.integrate(model, q, alpha * dq)

        # Numerical-drift cleanup: re-project the torso to exactly
        # leveled (the dq restriction keeps it on the manifold to
        # first order; this removes accumulated integration error).
        if level_axis is not None:
            q = _level_torso(q, level_axis)

        if err_tot < tol:
            break

    return q, err_tot



def dock_configuration(
    model: pin.Model,
    anchor_a: pin.SE3,
    anchor_b: pin.SE3,
    torso_pos: np.ndarray = None,
    q_init: np.ndarray = None,
    *,
    level_axis: Optional[np.ndarray] = None,
    q_nominal: Optional[np.ndarray] = None,
    w_posture: float = 0.0,
) -> np.ndarray:
    """
    Convenience: compute a valid configuration with both tools at anchors.

    Parameters
    ----------
    model : pin.Model
    anchor_a, anchor_b : SE3 target poses for tool_a, tool_b
    torso_pos : (3,) initial torso position guess (default: midpoint of anchors)
    q_init : (nq,) full configuration to use as seed. If provided, used
             instead of neutral + torso_pos. This ensures the IK converges
             to the same branch as the current robot configuration.
    level_axis, q_nominal, w_posture : forwarded to ``solve_ik``
        (see that docstring). All default off ⇒ legacy behaviour.

    Returns
    -------
    q : (nq,) valid docking configuration
    """
    if q_init is not None:
        q0 = q_init.copy()
    else:
        q0 = pin.neutral(model)
    if torso_pos is None and q_init is None:
        torso_pos = 0.5 * (anchor_a.translation + anchor_b.translation)
    if torso_pos is not None:
        q0[:3] = torso_pos

    fid_a, fid_b = _get_tool_frames(model)
    targets = {fid_a: anchor_a, fid_b: anchor_b}
    q, err = solve_ik(model, q0, targets, max_iter=2000,
                       q_nominal=q_nominal, w_posture=w_posture,
                       level_axis=level_axis)
    if err > 1e-4:
        raise RuntimeError(f"IK failed to converge: err={err:.2e}")
    return q


def dock_configuration_fixed_rotation(
    model: pin.Model,
    anchor_a: pin.SE3,
    anchor_b: pin.SE3,
    R_torso_fixed: np.ndarray,
    torso_pos: np.ndarray = None,
    q_init: np.ndarray = None,
    max_iter: int = 2000,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, float, float]:
    """IK with torso rotation held at R_torso_fixed (M7 / change A).

    Solves for (torso_position, arm joints) such that both tools reach
    their anchor poses while the torso orientation stays at
    `R_torso_fixed`. Per the M7 crawling philosophy: the robot
    translates between steps and only reorients when the geometry
    strictly demands it.

    The torso has 3 free DOFs (position) plus 2*7=14 arm DOFs = 17
    DOFs to satisfy 12 constraints (2 × SE3), leaving a 5-dim null
    space — enough slack for a good solution when one exists.

    Parameters
    ----------
    R_torso_fixed : (3,3)
        Desired torso rotation matrix (structure frame).
    torso_pos : (3,), optional
        Initial guess for torso position. Defaults to the midpoint of
        the two anchors.
    q_init : (nq,), optional
        Full configuration seed. Its torso rotation is overwritten by
        `R_torso_fixed`.

    Returns
    -------
    q : (nq,) converged configuration (torso rotation = R_torso_fixed)
    err : float — final residual norm
    w_product : float
        Yoshikawa manipulability product per IK_FORMULATION.md §4.1:
        ``√det(J_a J_aᵀ) · √det(J_b J_bᵀ)``, equal to the product
        of all 12 task-direction singular values across both arms.
        Primary diagnostic; backwards-compatible interface.
    w_sigma_min : float
        σ_min(J_a) · σ_min(J_b) per IK_FORMULATION.md §4.2 — the
        worst-direction manipulability used by IK 3
        (manipulability_config_trajectory). Secondary diagnostic
        for cross-comparison; resolves the metric-mismatch
        artefact identified in IK_ANOMALY_REPORT.md §4.3.
    """
    # Seed q0 with the fixed torso rotation. Pinocchio free-flyer uses
    # quaternion (x, y, z, w) at q[3:7].
    q0 = q_init.copy() if q_init is not None else pin.neutral(model)
    if torso_pos is None:
        torso_pos = 0.5 * (anchor_a.translation + anchor_b.translation)
    q0[:3] = torso_pos
    quat = pin.Quaternion(R_torso_fixed)
    q0[3] = quat.x
    q0[4] = quat.y
    q0[5] = quat.z
    q0[6] = quat.w

    fid_a, fid_b = _get_tool_frames(model)
    targets = {fid_a: anchor_a, fid_b: anchor_b}

    q = q0.copy()
    data = model.createData()
    nv = model.nv

    # Custom IK loop: same as solve_ik but zero out base angular dq
    # at every iteration to hold the rotation fixed.
    for it in range(max_iter):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        pin.computeJointJacobians(model, data, q)
        dq = np.zeros(nv)
        err_tot = 0.0

        for fid, tgt in targets.items():
            err = pin.log6(data.oMf[fid].actInv(tgt)).vector
            J = pin.getFrameJacobian(model, data, fid, pin.LOCAL)

            # Arm joints: primary contribution
            idx = _arm_v_slice(model, fid)
            n_arm = idx.stop - idx.start
            Ja = J[:, idx]
            dq[idx] += np.linalg.solve(
                Ja.T @ Ja + 1e-4 * np.eye(n_arm), Ja.T @ err)

            # Base: only the *linear* 3 DOFs are free (drop angular cols).
            Jb_lin = J[:, :3]
            dq[:3] += np.linalg.solve(
                Jb_lin.T @ Jb_lin + 1e-3 * np.eye(3),
                Jb_lin.T @ err) * 0.3

            err_tot += np.linalg.norm(err)

        # Integrate; base angular dq stays zero so rotation is held.
        alpha = min(1.0, 0.5 / max(np.max(np.abs(dq)), 1e-10))
        q = pin.integrate(model, q, alpha * dq)

        if err_tot < tol:
            break

    # Manipulability metrics at the solution (IK_FORMULATION §4):
    #   w_product   = Yoshikawa, √det(JJᵀ) per arm, product across arms.
    #   w_sigma_min = σ_min(Ja) · σ_min(Jb) (worst-direction product).
    # Both reported so downstream callers can cross-compare with IK 3.
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    pin.computeJointJacobians(model, data, q)
    sl_a = _arm_v_slice(model, fid_a)
    sl_b = _arm_v_slice(model, fid_b)
    Ja = pin.getFrameJacobian(model, data, fid_a, pin.LOCAL)[:, sl_a]
    Jb = pin.getFrameJacobian(model, data, fid_b, pin.LOCAL)[:, sl_b]
    w_a = float(np.sqrt(max(np.linalg.det(Ja @ Ja.T), 0.0)))
    w_b = float(np.sqrt(max(np.linalg.det(Jb @ Jb.T), 0.0)))
    sa = float(np.linalg.svd(Ja, compute_uv=False)[-1])
    sb = float(np.linalg.svd(Jb, compute_uv=False)[-1])
    return q, float(err_tot), w_a * w_b, sa * sb


def manipulability_config(
    model: pin.Model,
    anchor_a: pin.SE3,
    anchor_b: pin.SE3,
    *,
    level_axis: Optional[np.ndarray] = None,
    q_nominal: Optional[np.ndarray] = None,
    w_posture: float = 0.0,
) -> Tuple[np.ndarray, float]:
    """Find the configuration maximizing combined arm manipulability.

    Optimizes the torso (base) position to maximize the product of
    Yoshikawa manipulability indices for both arms, subject to both
    tools reaching their respective anchors via IK.

    Optional ``level_axis`` / ``q_nominal`` / ``w_posture`` are passed
    straight through to the inner ``solve_ik`` calls (see that
    function's docstring). When ``level_axis`` is given the inner IK
    keeps the torso pitch/roll leveled to the structure (yaw free);
    when ``q_nominal`` + ``w_posture`` are given the inner IK is biased
    away from contorted/entangled branches. All default off ⇒
    behaviour bit-identical to the legacy call. The outer Nelder-Mead
    still optimises only torso xyz and the σ_min objective is
    unchanged.

    Parameters
    ----------
    model : pin.Model
    anchor_a, anchor_b : SE3
        Target poses for tool_a and tool_b.

    Returns
    -------
    q_opt : ndarray (nq,)
        Optimal configuration.
    w_product : float
        Manipulability product w_a * w_b at the optimum.
    """
    data = model.createData()
    fid_a, fid_b = _get_tool_frames(model)
    targets = {fid_a: anchor_a, fid_b: anchor_b}
    midpoint = 0.5 * (anchor_a.translation + anchor_b.translation)

    # Cache for IK solutions at each torso position.
    # Reuse previous solution as seed to avoid singularity traps.
    _cache = {'q_prev': None}

    def cost(torso_xyz):
        q0 = pin.neutral(model)
        q0[:3] = torso_xyz
        # Also try seeding from previous successful solve
        candidates = [q0]
        if _cache['q_prev'] is not None:
            q_seed = _cache['q_prev'].copy()
            q_seed[:3] = torso_xyz
            candidates.insert(0, q_seed)

        best_score = -1.0
        best_q = None
        for q_try in candidates:
            q, err = solve_ik(model, q_try, targets, max_iter=500,
                               q_nominal=q_nominal, w_posture=w_posture,
                               level_axis=level_axis)
            if err > 1e-3:
                continue
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            pin.computeJointJacobians(model, data, q)

            sl_a = _arm_v_slice(model, fid_a)
            sl_b = _arm_v_slice(model, fid_b)
            Ja = pin.getFrameJacobian(
                model, data, fid_a, pin.LOCAL)[:, sl_a]
            Jb = pin.getFrameJacobian(
                model, data, fid_b, pin.LOCAL)[:, sl_b]
            # Minimum singular value: distance from singularity
            sigma_a = np.linalg.svd(Ja, compute_uv=False)[-1]
            sigma_b = np.linalg.svd(Jb, compute_uv=False)[-1]
            score = sigma_a * sigma_b
            if score > best_score:
                best_score = score
                best_q = q

        if best_q is not None:
            _cache['q_prev'] = best_q
        return -best_score if best_score > 0 else 1e6

    # Run Nelder-Mead from multiple starting points to escape singularities
    best_result = None
    best_cost = 1e6
    for dz in [0.0, -0.3, -0.6]:
        x0 = midpoint.copy()
        x0[2] += dz
        result = scipy_minimize(cost, x0, method='Nelder-Mead',
                                options={'xatol': 1e-3, 'fatol': 1e-8,
                                         'maxiter': 200, 'adaptive': True})
        if result.fun < best_cost:
            best_cost = result.fun
            best_result = result
        _cache['q_prev'] = None  # reset cache between starts

    if best_result is None:
        raise RuntimeError(
            "manipulability_config: all multi-starts produced IK failure "
            "(infeasible anchor pair)."
        )

    # Recover the optimal configuration
    q0 = pin.neutral(model)
    q0[:3] = best_result.x
    q_opt, err = solve_ik(model, q0, targets, max_iter=2000,
                          q_nominal=q_nominal, w_posture=w_posture,
                          level_axis=level_axis)
    if err > 1e-4:
        # Fallback: use midpoint
        q_opt = dock_configuration(model, anchor_a, anchor_b,
                                   level_axis=level_axis,
                                   q_nominal=q_nominal,
                                   w_posture=w_posture)

    # Compute final manipulability
    pin.forwardKinematics(model, data, q_opt)
    pin.updateFramePlacements(model, data)
    pin.computeJointJacobians(model, data, q_opt)
    sl_a = _arm_v_slice(model, fid_a)
    sl_b = _arm_v_slice(model, fid_b)
    Ja = pin.getFrameJacobian(model, data, fid_a, pin.LOCAL)[:, sl_a]
    Jb = pin.getFrameJacobian(model, data, fid_b, pin.LOCAL)[:, sl_b]
    w_a = np.sqrt(max(np.linalg.det(Ja @ Ja.T), 0.0))
    w_b = np.sqrt(max(np.linalg.det(Jb @ Jb.T), 0.0))

    return q_opt, w_a * w_b


def precompute_torso_map(
    model: pin.Model,
    anchors_a: np.ndarray,
    anchors_b: np.ndarray,
    *,
    anchor_pair_sequence: Optional[List[Tuple[int, int]]] = None,
    q_initial: Optional[np.ndarray] = None,
    n_samples: int = 5,
    use_trajectory_aware: bool = False,
) -> Dict[Tuple[int, int], object]:
    """Precompute optimal configurations for all anchor pairs.

    Two modes:

    * use_trajectory_aware=False (default, backwards-compatible):
        returns {(ai, bi): q_opt: ndarray} for the full grid product.
        This is the endpoint-only behaviour and matches the legacy signature.

    * use_trajectory_aware=True:
        walks ``anchor_pair_sequence`` in order, chaining q_start from
        the previous entry's q_end. Seeds from ``q_initial`` (required).
        Returns {(ai, bi): {'q_end', 'q_start_assumed', 'w_worst', 'w_end'}}.
        For repeated pairs, the last occurrence overwrites earlier ones
        — earlier occurrences will hit the drift-tolerance fallback at
        runtime (which is the intended graceful-degradation behaviour).

    Parameters
    ----------
    model : pin.Model
    anchors_a : (N, 3) array of anchor positions for tool A.
    anchors_b : (M, 3) array of anchor positions for tool B.
    anchor_pair_sequence : ordered list of (ai, bi) pairs (trajectory mode only).
    q_initial : (nq,) seed for the first chained pair (trajectory mode only).
        Typically the endpoint-IK solution for the first pair.
    n_samples : K — number of τ∈(0,1] samples (trajectory mode only).
    use_trajectory_aware : enable chained trajectory-aware IK.

    Returns
    -------
    torso_map : dict
        Endpoint mode: {(a_idx, b_idx): q_opt}.
        Trajectory mode: {(a_idx, b_idx): {q_end, q_start_assumed, w_worst, w_end}}.
    """
    if not use_trajectory_aware:
        torso_map: Dict[Tuple[int, int], object] = {}
        for ai in range(len(anchors_a)):
            for bi in range(len(anchors_b)):
                se3_a = pin.SE3(np.eye(3), anchors_a[ai].copy())
                se3_b = pin.SE3(np.eye(3), anchors_b[bi].copy())
                try:
                    q_opt, w = manipulability_config(model, se3_a, se3_b)
                    torso_map[(ai, bi)] = q_opt
                except RuntimeError:
                    pass  # infeasible pair — skip
        return torso_map

    # Trajectory-aware chained mode.
    if anchor_pair_sequence is None or q_initial is None:
        raise ValueError(
            "precompute_torso_map(use_trajectory_aware=True) requires "
            "both `anchor_pair_sequence` and `q_initial`."
        )

    traj_map: Dict[Tuple[int, int], object] = {}
    q_start = np.asarray(q_initial, dtype=float).copy()
    for (ai, bi) in anchor_pair_sequence:
        a_pos = anchors_a[ai]
        b_pos = anchors_b[bi]
        try:
            q_end, w_worst, w_end = manipulability_config_trajectory(
                model, a_pos, b_pos, q_start, n_samples=n_samples,
            )
        except RuntimeError:
            # Infeasible: skip this entry so runtime falls back to endpoint map.
            continue
        traj_map[(ai, bi)] = {
            'q_end': q_end,
            'q_start_assumed': q_start.copy(),
            'w_worst': float(w_worst),
            'w_end': float(w_end),
        }
        q_start = q_end  # chain
    return traj_map


def _interpolate_q_quintic(
    model: pin.Model,
    q_start: np.ndarray,
    q_end: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Quintic time-scaling on the configuration manifold.

    s(τ) = 10τ³ − 15τ⁴ + 6τ⁵  (C² with s(0)=0, s(1)=1, ṡ(0)=ṡ(1)=s̈(0)=s̈(1)=0)

    The time-scaled parameter is then passed to ``pin.interpolate``, which
    uses the Lie-group structure of the free-flyer base (SE(3) geodesic)
    plus linear interpolation for revolute joints.

    Matches the joint-space quintic used in
    ``sim_loop._planned_arm_config`` for SS swing timing.
    """
    s = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
    return pin.interpolate(model, q_start, q_end, s)


def _trajectory_worst_w(
    model: pin.Model,
    data,
    q_start: np.ndarray,
    q_end: np.ndarray,
    n_samples: int,
    fid_a: int,
    fid_b: int,
    sl_a: slice,
    sl_b: slice,
) -> Tuple[float, float]:
    """Return (w_worst, w_end): worst-case and endpoint σ_min products.

    Samples τ ∈ {1/K, 2/K, …, 1.0} (τ=0 is q_start, which is fixed input
    and common to all candidates, so excluded from the optimised set).
    """
    w_worst = np.inf
    w_end = 0.0
    for k in range(1, n_samples + 1):
        tau = k / n_samples
        q_k = _interpolate_q_quintic(model, q_start, q_end, tau)
        pin.forwardKinematics(model, data, q_k)
        pin.updateFramePlacements(model, data)
        pin.computeJointJacobians(model, data, q_k)
        Ja = pin.getFrameJacobian(model, data, fid_a, pin.LOCAL)[:, sl_a]
        Jb = pin.getFrameJacobian(model, data, fid_b, pin.LOCAL)[:, sl_b]
        sigma_a = float(np.linalg.svd(Ja, compute_uv=False)[-1])
        sigma_b = float(np.linalg.svd(Jb, compute_uv=False)[-1])
        w_k = sigma_a * sigma_b
        if w_k < w_worst:
            w_worst = w_k
        if k == n_samples:
            w_end = w_k
    return float(w_worst), float(w_end)


def manipulability_config_trajectory(
    model: pin.Model,
    anchor_a: np.ndarray,
    anchor_b: np.ndarray,
    q_start: np.ndarray,
    n_samples: int = 5,
    q_guess: Optional[np.ndarray] = None,
    w_min_threshold: Optional[float] = None,
) -> Tuple[Optional[np.ndarray], float, float]:
    """Trajectory-aware manipulability IK (M7 / Manipulability-IK-1).

    Optimise torso xyz to maximise the **worst-case** σ_min(J_a)·σ_min(J_b)
    across K interior samples of the planned quintic interpolation from
    q_start to the docked end configuration. Trades a small amount of
    endpoint optimality for guarantees on the interior trajectory —
    addresses the near-singular interior configurations flagged in the
    T15 post-5 audit §8.4 (Candidate 1).

    Decision variables and multi-start strategy match ``manipulability_config``
    (torso xyz, dz ∈ {0.0, −0.3, −0.6}, Nelder-Mead adaptive). The arm
    joints are solved inside the cost via ``solve_ik``.

    Parameters
    ----------
    model : pin.Model
    anchor_a, anchor_b : (3,) world positions for tool A / tool B anchors.
    q_start : (nq,) fixed τ=0 configuration (chained from the previous step).
    n_samples : K — samples at τ ∈ {1/K, 2/K, …, 1.0}.
    q_guess : optional (nq,) extra seed for the internal IK.
    w_min_threshold : optional safety check per IK_FORMULATION.md §9.3.
        If not None and the converged ``w_end < w_min_threshold``, the
        result is rejected as pathological and the function returns
        ``(None, w_worst, w_end)`` to let the caller fall back to
        fixed_rotation. Default None disables the check.

    Returns
    -------
    q_end : (nq,) or None — optimal end configuration, or None if the
        post-convergence safety check rejected the result.
    w_worst : float — worst-case σ_min product across the K samples (maximised).
    w_end : float — endpoint-only σ_min product at q_end (diagnostic).
    """
    se3_a = pin.SE3(np.eye(3), np.asarray(anchor_a, dtype=float).copy())
    se3_b = pin.SE3(np.eye(3), np.asarray(anchor_b, dtype=float).copy())
    data = model.createData()
    fid_a, fid_b = _get_tool_frames(model)
    targets = {fid_a: se3_a, fid_b: se3_b}
    midpoint = 0.5 * (se3_a.translation + se3_b.translation)
    q_start = np.asarray(q_start, dtype=float)
    sl_a = _arm_v_slice(model, fid_a)
    sl_b = _arm_v_slice(model, fid_b)

    # IK 3 inner-solve seed: q_start with torso xyz overwritten.
    # The seed is deterministic in the decision variable (torso xyz)
    # and identical across all cost evaluations within a single IK
    # invocation. This makes cost(p_t) a deterministic function of
    # p_t and eliminates pathology (C) from IK_ANOMALY_REPORT §3.1,
    # §5.3 (warm-start `_cache['q_prev']` allowed cost(xyz) to vary
    # by 7+ orders of magnitude depending on the prior Nelder-Mead
    # path). Spec: docs/architecture/IK_FORMULATION.md §9.1.
    def cost(torso_xyz):
        q_seed = q_start.copy()
        q_seed[:3] = torso_xyz
        q_end, err = solve_ik(model, q_seed, targets, max_iter=500)
        if err > 1e-3:
            return 1e6
        w_worst, _ = _trajectory_worst_w(
            model, data, q_start, q_end, n_samples, fid_a, fid_b, sl_a, sl_b,
        )
        return -w_worst if w_worst > 0 else 1e6

    # Multi-start seed set per IK_FORMULATION.md §9.2: 7 seeds spanning
    # all three Cartesian axes plus two physically motivated alternatives.
    # The pre-fix 3-seed dz-only set could not reach better basins of
    # the cost landscape (IK_ANOMALY_REPORT §3.2 — grid max at xyz
    # ~0.9 m from any current seed). Falls back to 6 seeds if the
    # fixed-rotation hybrid seed itself fails to converge.
    p_start_b = q_start[:3].copy()
    seeds = [
        ('q_start', p_start_b),
        ('midpoint', midpoint.copy()),
        ('mid+x', midpoint + np.array([+0.3, 0.0, 0.0])),
        ('mid-x', midpoint + np.array([-0.3, 0.0, 0.0])),
        ('mid+y', midpoint + np.array([0.0, +0.3, 0.0])),
        ('mid-y', midpoint + np.array([0.0, -0.3, 0.0])),
    ]
    # 7th seed: fixed-rotation IK output (hybrid). If it fails to
    # converge, omit the seed (see §9.2 "fall back to 6 seeds").
    try:
        qx, qy, qz, qw = q_start[3], q_start[4], q_start[5], q_start[6]
        R_torso_start = pin.Quaternion(qw, qx, qy, qz).toRotationMatrix()
        q_fixed, err_fixed, _, _ = dock_configuration_fixed_rotation(
            model, se3_a, se3_b,
            R_torso_fixed=R_torso_start,
            torso_pos=midpoint.copy(),
            q_init=q_start.copy(),
            max_iter=500, tol=1e-6,
        )
        if err_fixed < 1e-3:
            seeds.append(('p_fixed', q_fixed[:3].copy()))
    except Exception:
        pass  # silent: 6-seed fallback per §9.2

    best_result = None
    best_cost = 1e6
    best_seed_label = None
    for label, x0 in seeds:
        result = scipy_minimize(
            cost, x0.copy(), method='Nelder-Mead',
            options={'xatol': 1e-3, 'fatol': 1e-8,
                     'maxiter': 200, 'adaptive': True},
        )
        if result.fun < best_cost:
            best_cost = result.fun
            best_result = result
            best_seed_label = label

    if best_result is None or best_cost >= 1e6:
        raise RuntimeError(
            "manipulability_config_trajectory: all multi-starts produced "
            f"IK failure across {len(seeds)} seeds (infeasible anchor pair)."
        )

    # Recover the optimal q_end with the same deterministic seed used
    # in the cost (q_start with torso xyz overwritten); no fallback
    # to dock_configuration which would change the kinematic branch.
    q_seed = q_start.copy()
    q_seed[:3] = best_result.x
    q_end, err = solve_ik(model, q_seed, targets, max_iter=2000)
    if err > 1e-4:
        q_end = dock_configuration(model, se3_a, se3_b)

    w_worst, w_end = _trajectory_worst_w(
        model, data, q_start, q_end, n_samples, fid_a, fid_b, sl_a, sl_b,
    )

    # Post-convergence safety check (IK_FORMULATION.md §9.3): reject
    # pathologically singular endpoints. The caller (sim_loop) falls
    # back to fixed_rotation when q_end is None.
    if w_min_threshold is not None and w_end < w_min_threshold:
        return None, float(w_worst), float(w_end)

    return q_end, float(w_worst), float(w_end)


def _sigma_min_pair(model, data, q, fid_a, fid_b, sl_a, sl_b):
    """σ_min(J_a) and σ_min(J_b) at q. Helper for path-evaluation cost."""
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    pin.computeJointJacobians(model, data, q)
    Ja = pin.getFrameJacobian(model, data, fid_a, pin.LOCAL)[:, sl_a]
    Jb = pin.getFrameJacobian(model, data, fid_b, pin.LOCAL)[:, sl_b]
    sa = float(np.linalg.svd(Ja, compute_uv=False)[-1])
    sb = float(np.linalg.svd(Jb, compute_uv=False)[-1])
    return sa, sb


def manipulability_config_mid_waypoint(
    model: pin.Model,
    anchor_a_pose: pin.SE3,
    anchor_b_pose: pin.SE3,
    q_start: np.ndarray,
    q_end: np.ndarray,
    swing_arm: str = 'b',
    n_interior_samples: int = 5,
    w_min_threshold: float = 1e-3,
) -> Tuple[Optional[np.ndarray], float, bool]:
    """Mid-waypoint IK for piecewise-quintic SS reference (Option B).

    Implements T15_step2_path_geometry.md §7.3 Option B: insert a
    manipulability-aware mid-waypoint between ``q_start`` and ``q_end``
    so that the resulting two-segment quintic stays well-conditioned
    throughout. Addresses the H2 finding that the single-quintic
    reference for anchor pair (3,4) at T15 visits a near-singular
    interior region.

    The decision variable is the mid-waypoint torso xyz
    ``p_t_mid ∈ R^3``. The torso orientation and arm joints at the
    mid-waypoint are determined by the inner IK, which enforces only
    the stance-arm target (the swing arm is mid-flight at ``q_mid``).
    The cost samples the σ_min product across both sub-segments and
    returns the worst case.

    Parameters
    ----------
    model : pin.Model
    anchor_a_pose, anchor_b_pose : SE3
        EE targets at ``q_end``. The stance pose (the one held
        throughout the SS) is selected by ``swing_arm``: if
        ``swing_arm == 'b'`` the stance is ``anchor_a_pose``, else
        the stance is ``anchor_b_pose``.
    q_start, q_end : (nq,) endpoint configurations.
    swing_arm : 'a' or 'b' — which arm is swinging during the SS.
    n_interior_samples : K — number of interior τ samples per
        sub-segment. Sample positions are τ ∈ {1/(K+1), 2/(K+1), …,
        K/(K+1)}, i.e. interior of (0, 1) excluding the endpoints
        (which are q_start, q_mid, q_end and evaluated separately).
    w_min_threshold : safety threshold; if the converged worst-case
        ``w_worst < w_min_threshold`` the function returns
        ``(q_mid, w_worst, False)`` so the caller falls back to
        single-quintic.

    Returns
    -------
    q_mid : (nq,) or None — mid-waypoint configuration. ``None`` only
        if the multi-start failed entirely (no seed converged).
    w_worst : float — worst-case σ_min product across both sub-segments
        and the mid-waypoint itself.
    success : bool — True iff ``w_worst >= w_min_threshold``.
    """
    fid_a, fid_b = _get_tool_frames(model)
    sl_a = _arm_v_slice(model, fid_a)
    sl_b = _arm_v_slice(model, fid_b)
    data = model.createData()

    if swing_arm.lower() == 'b':
        stance_fid = fid_a
        stance_pose = anchor_a_pose
    else:
        stance_fid = fid_b
        stance_pose = anchor_b_pose

    q_start = np.asarray(q_start, dtype=float)
    q_end = np.asarray(q_end, dtype=float)

    # Joint-space midpoint as inner-IK seed. Deterministic in the
    # decision variable (torso xyz). Same pattern as
    # manipulability_config_trajectory's q_start seed (IK_FORMULATION
    # §9.1) — avoids the path-dependency pathology that was fixed in
    # IK_ANOMALY_REPORT §3.1.
    q_jmid_base = pin.interpolate(model, q_start, q_end, 0.5)

    # Interior τ samples for cost evaluation (per sub-segment).
    # We sample (n_interior_samples) equally-spaced interior points in
    # (0, 1), excluding the segment endpoints which are evaluated
    # separately as q_start, q_mid, q_end.
    K = max(1, int(n_interior_samples))
    interior_taus = [(k + 1) / (K + 1) for k in range(K)]

    def _solve_mid(p_t_mid):
        """Inner IK at p_t_mid: enforce stance only."""
        q_seed = q_jmid_base.copy()
        q_seed[:3] = p_t_mid
        targets = {stance_fid: stance_pose}
        q_mid, err = solve_ik(model, q_seed, targets, max_iter=500)
        return q_mid, err

    def cost(p_t_mid):
        q_mid, err = _solve_mid(p_t_mid)
        if err > 1e-3:
            return 1e6
        # Sample both sub-segments + the mid-waypoint itself.
        w_worst = np.inf
        for tau in interior_taus:
            # Segment 1: q_start → q_mid
            q_k = _interpolate_q_quintic(model, q_start, q_mid, tau)
            sa, sb = _sigma_min_pair(model, data, q_k,
                                     fid_a, fid_b, sl_a, sl_b)
            w_worst = min(w_worst, sa * sb)
            # Segment 2: q_mid → q_end
            q_k = _interpolate_q_quintic(model, q_mid, q_end, tau)
            sa, sb = _sigma_min_pair(model, data, q_k,
                                     fid_a, fid_b, sl_a, sl_b)
            w_worst = min(w_worst, sa * sb)
        # Mid-waypoint itself
        sa, sb = _sigma_min_pair(model, data, q_mid,
                                 fid_a, fid_b, sl_a, sl_b)
        w_worst = min(w_worst, sa * sb)
        return -w_worst if w_worst > 0 else 1e6

    # Multi-start seed set per IK_FORMULATION.md §9.2 (7 seeds).
    midpoint_anchors = 0.5 * (
        anchor_a_pose.translation + anchor_b_pose.translation)
    p_start_b = q_start[:3].copy()
    seeds = [
        ('q_start', p_start_b),
        ('midpoint', midpoint_anchors.copy()),
        ('mid+x', midpoint_anchors + np.array([+0.3, 0.0, 0.0])),
        ('mid-x', midpoint_anchors + np.array([-0.3, 0.0, 0.0])),
        ('mid+y', midpoint_anchors + np.array([0.0, +0.3, 0.0])),
        ('mid-y', midpoint_anchors + np.array([0.0, -0.3, 0.0])),
    ]
    # 7th seed: fixed-rotation IK output (hybrid).
    try:
        qx, qy, qz, qw = q_start[3], q_start[4], q_start[5], q_start[6]
        R_torso_start = pin.Quaternion(qw, qx, qy, qz).toRotationMatrix()
        q_fixed, err_fixed, _, _ = dock_configuration_fixed_rotation(
            model, anchor_a_pose, anchor_b_pose,
            R_torso_fixed=R_torso_start,
            torso_pos=midpoint_anchors.copy(),
            q_init=q_start.copy(),
            max_iter=500, tol=1e-6,
        )
        if err_fixed < 1e-3:
            seeds.append(('p_fixed', q_fixed[:3].copy()))
    except Exception:
        pass

    best_result = None
    best_cost = 1e6
    for label, x0 in seeds:
        result = scipy_minimize(
            cost, x0.copy(), method='Nelder-Mead',
            options={'xatol': 1e-3, 'fatol': 1e-8,
                     'maxiter': 200, 'adaptive': True},
        )
        if result.fun < best_cost:
            best_cost = result.fun
            best_result = result

    if best_result is None or best_cost >= 1e6:
        # No seed converged — caller falls back to single-quintic.
        return None, 0.0, False

    # Recover q_mid at the optimum.
    q_mid, err = _solve_mid(best_result.x)
    if err > 1e-3:
        # Final inner solve failed despite cost minimum found —
        # treat as unsuccessful. Caller falls back.
        return None, 0.0, False

    w_worst = -best_cost  # cost was negated
    success = bool(w_worst >= w_min_threshold)
    return q_mid, float(w_worst), success


def _ik_three_tasks(model, fid_torso, fid_swing, fid_stance,
                    se3_torso, se3_swing, se3_stance,
                    q_seed, max_iter=2000, tol=1e-6):
    """Damped-LS Newton IK on three SE(3) frame tasks.

    Used by ``check_path_feasibility`` to evaluate whether the
    planner-style reference at a given τ admits a feasible
    whole-body configuration. Returns ``(q, err_norm, converged)``.
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
    for _ in range(max_iter):
        pin.computeJointJacobians(model, data, q)
        J_stack = []
        for fid, _tgt in targets:
            J_stack.append(pin.getFrameJacobian(model, data, fid, pin.LOCAL))
        J_total = np.vstack(J_stack)
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
            break
        if err_norm_prev < tol:
            return q, err_norm_prev, True
    return q, err_norm_prev, err_norm_prev < 1e-3


def check_path_feasibility(
    model: pin.Model,
    q_start: np.ndarray,
    q_end: np.ndarray,
    anchor_a_pose: pin.SE3,
    anchor_b_pose: pin.SE3,
    swing_arm: str,
    fid_torso: int,
    n_samples: int = 11,
    w_min_threshold: float = 1e-3,
    convergence_tol: float = 1e-3,
    clearance: float = 0.08,
    away_normal: np.ndarray = None,
) -> dict:
    """Evaluate whether the planner-style reference path is feasible.

    At ``n_samples`` evenly-spaced τ values across [0, 1], constructs
    the reference triple ``(torso_ref, swing_ref, stance_ref)`` that
    the TorsoPlanner / SwingPlanner / stance constraint would
    produce, and tries to find a whole-body q that satisfies all
    three simultaneously via 3-task damped-LS IK. Reports the
    minimum ``σ_min(J_a) · σ_min(J_b)`` across τ samples (at the
    converged q where the IK converged; using the joint-space q
    seeded from interpolation otherwise).

    This is the same procedure as
    ``scripts/diagnostic_step2_path_geometry.py`` §1, extracted into
    a runtime-callable helper.

    Reference construction (matches planner shapes):
        - torso_ref(τ): quintic SLERP between (p_t_start, R_t_start)
          and (p_t_end, R_t_end).
        - swing_ref(τ): quintic between (p_swing_start, R_swing_start)
          and (p_swing_end, R_swing_end) PLUS a sin²(πτ) clearance
          bump in the +``away_normal`` direction with peak amplitude
          ``clearance``. This matches SwingPlanner's symmetric default
          (M5/M7 ``bump_peak_tau=0.5``). If the SwingPlanner
          instance uses different bump parameters, pass them via the
          function arguments. The bump matters: it pulls the swing
          EE off the structure surface and contributes materially to
          the path's σ_min profile.
        - stance_ref(τ) = stance pose (held throughout).

    Parameters
    ----------
    model : pin.Model
    q_start, q_end : (nq,) endpoint configurations.
    anchor_a_pose, anchor_b_pose : SE3 — EE targets at q_end.
    swing_arm : 'a' or 'b' — which arm is swinging.
    fid_torso : int — Pinocchio frame ID of the torso.
    n_samples : K — total samples at τ ∈ linspace(0, 1, K).
    w_min_threshold : float — samples with σ_min product strictly
        below this are counted as infeasible.
    convergence_tol : float — per-sample IK convergence tolerance
        (residual task error norm). Defaults to 1e-3 m/rad mixed.

    Returns
    -------
    dict with fields:
        - all_samples_feasible : bool — every sample has IK
          convergence and ``w >= w_min_threshold``.
        - n_infeasible_w : int — count of samples with w below threshold.
        - n_ik_failures : int — count of samples where 3-task IK did
          not converge to ``convergence_tol``.
        - w_min : float — minimum w across all samples.
        - w_min_tau : float — τ at which w_min occurred.
        - per_sample : list of dicts ``{tau, t_residual, w, converged}``.
    """
    fid_a, fid_b = _get_tool_frames(model)
    sl_a = _arm_v_slice(model, fid_a)
    sl_b = _arm_v_slice(model, fid_b)
    data = model.createData()

    q_start = np.asarray(q_start, dtype=float)
    q_end = np.asarray(q_end, dtype=float)

    if swing_arm.lower() == 'b':
        fid_swing, fid_stance = fid_b, fid_a
        stance_pose = anchor_a_pose
    else:
        fid_swing, fid_stance = fid_a, fid_b
        stance_pose = anchor_b_pose

    # Endpoint poses in Pinocchio world frame.
    pin.forwardKinematics(model, data, q_start)
    pin.updateFramePlacements(model, data)
    p_t_start = data.oMf[fid_torso].translation.copy()
    R_t_start = data.oMf[fid_torso].rotation.copy()
    p_swing_start = data.oMf[fid_swing].translation.copy()
    R_swing_start = data.oMf[fid_swing].rotation.copy()

    pin.forwardKinematics(model, data, q_end)
    pin.updateFramePlacements(model, data)
    p_t_end = data.oMf[fid_torso].translation.copy()
    R_t_end = data.oMf[fid_torso].rotation.copy()
    # Swing-end pose is the IK's q_end for the swing arm.
    p_swing_end = data.oMf[fid_swing].translation.copy()
    R_swing_end = data.oMf[fid_swing].rotation.copy()

    # Quintic time scaling.
    def _s(tau):
        return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5

    if away_normal is None:
        # Default matches DEFAULT_AWAY_NORMAL = [0, 0, 1] in swing_planner.
        away_normal_n = np.array([0.0, 0.0, 1.0])
    else:
        an = np.asarray(away_normal, dtype=float)
        away_normal_n = an / max(float(np.linalg.norm(an)), 1e-12)

    taus = np.linspace(0.0, 1.0, n_samples)
    per_sample = []
    n_infeasible_w = 0
    n_ik_failures = 0
    w_min = float('inf')
    w_min_tau = 0.0
    for tau in taus:
        s = _s(float(tau))
        # Torso ref at τ
        p_torso = (1 - s) * p_t_start + s * p_t_end
        dR = R_t_start.T @ R_t_end
        R_torso = R_t_start @ pin.exp3(s * pin.log3(dR))
        # Swing ref at τ: linear quintic + sin²(πτ) clearance bump.
        # The bump matches SwingPlanner's default (symmetric peak
        # at τ=0.5). Asymmetric profiles (M7 bump_peak_tau≠0.5) and
        # delayed-cosine SLERP are not modelled here — those would
        # require coupling to a live SwingPlanner instance. The
        # symmetric-bump approximation is sufficient for detecting
        # the singular-interior failure mode (T15_step2_path
        # _geometry.md §2 measured the same drop with the actual
        # SwingPlanner-generated reference).
        bump_amp = float(np.sin(np.pi * float(tau)) ** 2)
        p_swing = ((1 - s) * p_swing_start + s * p_swing_end
                   + clearance * away_normal_n * bump_amp)
        dR_s = R_swing_start.T @ R_swing_end
        R_swing = R_swing_start @ pin.exp3(s * pin.log3(dR_s))
        # IK seed: joint-space interpolation
        q_seed = _interpolate_q_quintic(model, q_start, q_end, float(tau))
        q_ideal, err_norm, conv = _ik_three_tasks(
            model, fid_torso, fid_swing, fid_stance,
            pin.SE3(R_torso, p_torso),
            pin.SE3(R_swing, p_swing),
            stance_pose,
            q_seed,
            max_iter=400, tol=convergence_tol,
        )
        sa, sb = _sigma_min_pair(model, data, q_ideal,
                                 fid_a, fid_b, sl_a, sl_b)
        w = sa * sb
        per_sample.append({
            'tau': float(tau),
            'task_residual': float(err_norm),
            'w': float(w),
            'converged': bool(conv),
        })
        if w < w_min:
            w_min = w
            w_min_tau = float(tau)
        if w < w_min_threshold:
            n_infeasible_w += 1
        if not conv:
            n_ik_failures += 1
    return {
        'all_samples_feasible': bool(
            n_infeasible_w == 0 and n_ik_failures == 0),
        'n_infeasible_w': int(n_infeasible_w),
        'n_ik_failures': int(n_ik_failures),
        'w_min': float(w_min),
        'w_min_tau': float(w_min_tau),
        'per_sample': per_sample,
    }


def solve_ik_waypoints(
    model: pin.Model,
    q_start: np.ndarray,
    stance_frame: int,
    stance_target: pin.SE3,
    swing_frame: int,
    swing_start: np.ndarray,
    swing_end: np.ndarray,
    n_waypoints: int = 10,
    clearance: float = 0.08,
    away_normal: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    """Compute a chain of IK solutions along a swing arc.

    Produces n_waypoints+1 configurations from swing_start to swing_end.
    Each waypoint is seeded from the previous one (chain seeding), ensuring
    smooth configuration transitions with no branch jumps.

    The swing arm follows a straight line with a clearance bump:
        p(s) = (1-s)*start + s*end + clearance * sin²(π*s) * away_normal

    Parameters
    ----------
    model : pin.Model
        Pinocchio model (free-flyer).
    q_start : ndarray (nq,)
        Starting configuration (current robot state).
    stance_frame : int
        Frame ID of the stance (docked) tool (FRAME_TOOL_A or FRAME_TOOL_B).
    stance_target : pin.SE3
        SE3 pose of the stance anchor (stays fixed throughout).
    swing_frame : int
        Frame ID of the swing (free) tool.
    swing_start : ndarray (3,)
        Starting position of the swing tool (departure anchor).
    swing_end : ndarray (3,)
        Target position of the swing tool (arrival anchor).
    n_waypoints : int
        Number of intermediate waypoints (total configs = n_waypoints + 1).
    clearance : float
        Maximum clearance height of the swing arc [m].
    away_normal : ndarray (3,), optional
        Direction of the clearance bump. Default: [0, 0, -1] (away from structure).

    Returns
    -------
    q_chain : list of ndarray (nq,)
        Chain of n_waypoints+1 configurations.
        q_chain[0] ≈ q_start (re-converged at swing_start).
        q_chain[-1] has swing tool at swing_end.
    """
    if away_normal is None:
        away_normal = np.array([0.0, 0.0, -1.0])
    away_normal = np.asarray(away_normal, dtype=float)

    q_chain = []
    q_prev = q_start.copy()
    data = model.createData()

    # Determine which arm slice corresponds to stance/swing (DOF-generic)
    stance_idx = _arm_v_slice(model, stance_frame)
    swing_idx = _arm_v_slice(model, swing_frame)
    n_stance = stance_idx.stop - stance_idx.start
    n_swing = swing_idx.stop - swing_idx.start

    for i in range(n_waypoints + 1):
        s = i / n_waypoints

        # Swing position with clearance bump
        p_swing = ((1.0 - s) * swing_start + s * swing_end
                   + clearance * np.sin(np.pi * s) ** 2 * away_normal)

        # Iterative IK: stance arm 6D (full SE3), swing arm 3D (position only).
        # This is a custom loop rather than calling solve_ik, because we need
        # asymmetric targets (6D stance + 3D swing).
        q = q_prev.copy()
        for it in range(300):
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            pin.computeJointJacobians(model, data, q)
            dq = np.zeros(model.nv)
            err_tot = 0.0

            # Stance arm: full 6D constraint (keep at anchor)
            err_s = pin.log6(data.oMf[stance_frame].actInv(stance_target)).vector
            J_s = pin.getFrameJacobian(model, data, stance_frame, pin.LOCAL)
            Ja_s = J_s[:, stance_idx]
            dq[stance_idx] = np.linalg.solve(
                Ja_s.T @ Ja_s + 1e-4 * np.eye(n_stance), Ja_s.T @ err_s)
            Jb_s = J_s[:, :6]
            dq[:6] += np.linalg.solve(
                Jb_s.T @ Jb_s + 1e-3 * np.eye(6), Jb_s.T @ err_s) * 0.3
            err_tot += np.linalg.norm(err_s)

            # Swing arm: 3D position constraint only
            p_cur = data.oMf[swing_frame].translation
            err_p = p_swing - p_cur
            J_sw = pin.getFrameJacobian(
                model, data, swing_frame, pin.LOCAL_WORLD_ALIGNED)
            J_sw_pos = J_sw[:3, :]  # position rows only
            Ja_p = J_sw_pos[:, swing_idx]
            dq[swing_idx] += np.linalg.solve(
                Ja_p.T @ Ja_p + 1e-4 * np.eye(n_swing), Ja_p.T @ err_p)
            Jb_p = J_sw_pos[:, :6]
            dq[:6] += np.linalg.solve(
                Jb_p.T @ Jb_p + 1e-3 * np.eye(6), Jb_p.T @ err_p) * 0.5
            err_tot += np.linalg.norm(err_p)

            alpha = min(1.0, 0.5 / max(np.max(np.abs(dq)), 1e-10))
            q = pin.integrate(model, q, alpha * dq)

            if err_tot < 1e-4:
                break

        q_chain.append(q)
        q_prev = q

    return q_chain
