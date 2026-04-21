"""Dock-activation velocity audit for the T11 closed run.

Reads results/M7_1pct_1step_v21_armature_fix_damping_off_mapping_off/sim_log.json
and emits results/.../dock_velocity_audit.md.

All velocities are computed as rigid-body formulae on logged fields
only — no new MuJoCo simulation. For a rigid body with world-frame
position p_b and world angular velocity ω_b, the world velocity of a
body-attached site at body-local offset r_off is

    v_site_lin = v_b + ω_b × (R_b · r_off)
    v_site_ang = ω_b

Logged inputs used:
    gripper_b (on arm) : p_ee, q_ee    → finite-diff for v_lin, ω_ang
    anchor_3b (on structure, site pos in body = (-0.4, -0.3, 0.025)):
        v_lin = d(struct_pos)/dt + ω_struct × (R_struct · r_off)
        ω     = ω_struct  (= logged `omega_s`, world frame)
    structure body : struct_pos (finite-diff → v_lin), omega_s (→ ω)

Reference velocity (item 4) is finite-diff of logged p_ee_ref /
q_ee_ref; those come from SwingPlanner.reference_at(t).

No efc_force in the committed log; item 5 reports the available
relative-velocity signal only.
"""
from __future__ import annotations

import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

import numpy as np

from crawlbot.simulation.logging import SimLog


LOG_PATH = os.path.join(_root, 'results',
                        'M7_1pct_1step_v21_armature_fix_damping_off_mapping_off',
                        'sim_log.json')
OUT_PATH = os.path.join(_root, 'results',
                        'M7_1pct_1step_v21_armature_fix_damping_off_mapping_off',
                        'dock_velocity_audit.md')

# The T11 run uses swing arm 'b' stepping from scheduler anchor-index
# 2 to index 3. The scheduler is 0-indexed; dock_events reports
# `anchor=3`. In the MJCF the per-arm anchor sites are named
# anchor_{1..6}b, so index 3 is **anchor_4b** at body-local
# (+0.4, -0.3, 0.025) on body `structure`.
R_OFFSET_TARGET_ANCHOR = np.array([+0.4, -0.3, 0.025])


# ── Pure-numpy helpers ────────────────────────────────────────────────

def _quat_wxyz_to_mat(q):
    """wxyz → 3×3 rotation matrix (no Pinocchio import)."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n = np.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def _omega_from_quat_series(q_wxyz_k, q_wxyz_kp1, dt):
    """Average world-frame angular velocity over [k, k+1] from two
    wxyz quaternions. Uses ΔR = R_k^T · R_kp1, then axis-angle log.
    Returns the angular velocity in WORLD frame (rotated through R_k).
    """
    R_k  = _quat_wxyz_to_mat(q_wxyz_k)
    R_kp = _quat_wxyz_to_mat(q_wxyz_kp1)
    dR   = R_k.T @ R_kp
    # robust axis-angle: cos θ = (tr(dR) - 1)/2, axis from skew-symm part
    c = np.clip((np.trace(dR) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(c))
    if theta < 1e-12:
        return np.zeros(3)
    skew = (dR - dR.T) * 0.5
    axis_local = np.array([skew[2, 1], skew[0, 2], skew[1, 0]]) / np.sin(theta)
    omega_body = axis_local * (theta / dt)
    return R_k @ omega_body       # local body → world axes


def _central_diff(x, i, dt):
    """Central finite difference of a (T, ...) array; falls back to
    one-sided at boundaries."""
    T = len(x)
    if i == 0:
        return (np.asarray(x[1], dtype=float)
                - np.asarray(x[0], dtype=float)) / dt
    if i == T - 1:
        return (np.asarray(x[-1], dtype=float)
                - np.asarray(x[-2], dtype=float)) / dt
    return (np.asarray(x[i+1], dtype=float)
            - np.asarray(x[i-1], dtype=float)) / (2*dt)


def _omega_at(q_list, i, dt):
    """World-frame angular velocity at log index i via adjacent wxyz quats."""
    T = len(q_list)
    if i == 0:
        return _omega_from_quat_series(q_list[0], q_list[1], dt)
    if i == T - 1:
        return _omega_from_quat_series(q_list[-2], q_list[-1], dt)
    # central difference over 2·dt spanning [i-1, i+1]
    return _omega_from_quat_series(q_list[i-1], q_list[i+1], 2*dt)


# ── Rigid-body anchor velocities from structure state ────────────────

def _anchor_state_at(log, i, dt):
    """Return (p_anchor_world, v_anchor_lin_world, v_anchor_ang_world)
    for anchor_3b at log index i."""
    struct_pos = np.asarray(log.struct_pos[i], dtype=float)
    R_struct   = _quat_wxyz_to_mat(log.struct_quat[i])
    omega_str  = np.asarray(log.omega_s[i], dtype=float)      # world frame

    r_off_world = R_struct @ R_OFFSET_TARGET_ANCHOR
    p_anchor_world = struct_pos + r_off_world

    v_struct_lin = _central_diff(log.struct_pos, i, dt)
    v_anchor_lin = v_struct_lin + np.cross(omega_str, r_off_world)
    v_anchor_ang = omega_str
    return p_anchor_world, v_anchor_lin, v_anchor_ang


def _gripper_state_at(log, i, dt):
    """Gripper b world position + velocities from p_ee / q_ee finite diff."""
    p   = np.asarray(log.p_ee[i], dtype=float)
    v_l = _central_diff(log.p_ee, i, dt)
    v_a = _omega_at(log.q_ee, i, dt)
    return p, v_l, v_a


def _ref_vel_at(log, i, dt):
    """Reference linear + angular velocity from p_ee_ref / q_ee_ref."""
    v_l = _central_diff(log.p_ee_ref, i, dt)
    v_a = _omega_at(log.q_ee_ref, i, dt)
    return v_l, v_a


# ── Main audit ───────────────────────────────────────────────────────

def main():
    log = SimLog.load(LOG_PATH)
    t = np.array(log.t)
    dt = float(np.mean(np.diff(t[:10])))   # 0.1 s cadence (NMPC tick)

    # Dock tick
    dock_t = log.dock_events[0]['t']
    k_dock = int(np.argmin(np.abs(t - dock_t)))

    # Last SS tick (immediately before phase transition at dock)
    phase = np.array(log.phase)
    ss_idx = np.where(phase == 'SS')[0]
    k_ss_last = int(ss_idx[-1])

    # ── (1) Dock-activation relative velocity ────────────────────────
    p_g,  v_g_l,  v_g_a  = _gripper_state_at(log, k_dock, dt)
    p_a,  v_a_l,  v_a_a  = _anchor_state_at (log, k_dock, dt)
    v_rel_lin_dock = v_g_l - v_a_l
    v_rel_ang_dock = v_g_a - v_a_a
    approach = p_a - p_g
    approach_n = np.linalg.norm(approach)
    approach_hat = approach / approach_n if approach_n > 1e-12 else np.zeros(3)
    v_rel_lin_along = float(v_rel_lin_dock @ approach_hat)

    # ── (2) Profiles over last 1 s of SS (t ∈ [6.3, 7.3]) ────────────
    k_lo = int(np.argmin(np.abs(t - 6.3)))
    k_hi = k_ss_last
    profile_rows = []
    for k in range(k_lo, k_hi + 1):
        _, vg_l, vg_a = _gripper_state_at(log, k, dt)
        _, va_l, va_a = _anchor_state_at(log, k, dt)
        vr_l = vg_l - va_l
        vr_a = vg_a - va_a
        profile_rows.append({
            'k': k, 't': float(t[k]),
            'vrel_lin_norm': float(np.linalg.norm(vr_l)),
            'vrel_ang_norm': float(np.linalg.norm(vr_a)),
        })

    # ── (3) Structure body velocity at dock tick ─────────────────────
    v_struct_lin = _central_diff(log.struct_pos, k_dock, dt)
    v_struct_ang = np.asarray(log.omega_s[k_dock], dtype=float)

    # ── (4) Reference velocity at T_step and at dock tick ────────────
    T_step = float(log.preplanner_T_steps[0])
    # the log reference buffer p_ee_ref[k] is the sample fed to the QP
    # at time t[k]; find tick at t ≈ T_step (= 7.284)
    # log t grid starts at 0.11, so T_step corresponds to k ≈ 71 or 72.
    k_Tstep = int(np.argmin(np.abs(t - (T_step + t[0]))))  # SS-relative offset
    # Alternatively: simply pick the last SS tick (k_ss_last = 71).
    v_ref_l_Tstep, v_ref_a_Tstep = _ref_vel_at(log, k_ss_last, dt)
    v_ref_l_dock,  v_ref_a_dock  = _ref_vel_at(log, k_dock,     dt)
    # Actual relative velocity at the same ticks (for comparison)
    _, vg_l_Tstep, vg_a_Tstep = _gripper_state_at(log, k_ss_last, dt)
    _, va_l_Tstep, va_a_Tstep = _anchor_state_at (log, k_ss_last, dt)
    vrel_l_Tstep = vg_l_Tstep - va_l_Tstep
    vrel_a_Tstep = vg_a_Tstep - va_a_Tstep

    # ── (5) Post-dock residual over 200 ms ────────────────────────────
    k_post_hi = int(np.argmin(np.abs(t - (dock_t + 0.20))))
    post_rows = []
    for k in range(k_dock, k_post_hi + 1):
        _, vg_l, vg_a = _gripper_state_at(log, k, dt)
        _, va_l, va_a = _anchor_state_at (log, k, dt)
        vr_l = vg_l - va_l
        vr_a = vg_a - va_a
        post_rows.append({
            'k': k, 't': float(t[k]),
            'vrel_lin_norm': float(np.linalg.norm(vr_l)),
            'vrel_ang_norm': float(np.linalg.norm(vr_a)),
        })

    # ── Write report ──────────────────────────────────────────────────
    with open(OUT_PATH, 'w') as f:
        f.write('# M7 dock-activation velocity audit (T11 closed run)\n\n')
        f.write(f'Log: `{os.path.relpath(LOG_PATH, _root)}`. '
                f'Log cadence `dt = {dt:.3f} s`. Dock event at '
                f'`t = {dock_t:.3f} s` (log index `k_dock = {k_dock}`). '
                f'Last SS tick: `k = {k_ss_last}` at `t = {t[k_ss_last]:.3f} s`.\n\n')

        f.write('## Method\n\n')
        f.write('No new simulation. All velocities are finite-differences '
                'of logged fields, then combined via the rigid-body '
                'formula `v_site = v_body_cm + ω × (R_body · r_off_body)` '
                'for anchor_3b (body = `structure`, site offset '
                f'`{R_OFFSET_TARGET_ANCHOR.tolist()}` from MJCF line 101). '
                'Gripper_b velocities are finite-differences of `p_ee` '
                '(linear) and `q_ee` (angular via axis-angle log of '
                '`R_k^T · R_{k+1}`). Structure angular velocity is the '
                'logged `omega_s` (MuJoCo `qvel[3:6]` for the structure '
                'free joint, world frame). Central differences are used '
                'interior-of-trace; one-sided at boundaries.\n\n')

        # ── Item 1 ──
        f.write('## 1. At dock activation (t = 7.31 s, k = 72)\n\n')
        f.write(f'Approach direction `(p_anchor − p_gripper)` (world): '
                f'length `{approach_n*1000:.3f}` mm, unit vector '
                f'`{approach_hat.tolist()}`.\n\n')
        f.write(f'- `‖v_rel_lin‖` = **`{np.linalg.norm(v_rel_lin_dock):.6e}`** m/s\n')
        f.write(f'- `‖v_rel_ang‖` = **`{np.linalg.norm(v_rel_ang_dock):.6e}`** rad/s\n')
        f.write(f'- Component of `v_rel_lin` along approach direction '
                f'(positive = gripper closing on anchor): '
                f'**`{v_rel_lin_along:+.6e}`** m/s\n\n')
        f.write('Vector components (world):\n\n')
        f.write(f'- `v_rel_lin` = `{v_rel_lin_dock.tolist()}` m/s\n')
        f.write(f'- `v_rel_ang` = `{v_rel_ang_dock.tolist()}` rad/s\n\n')

        # ── Item 2 ──
        f.write('## 2. Profile over last 1 s of SS (t ∈ [6.3, 7.3] s)\n\n')
        f.write('| k | t [s] | ‖v_rel_lin‖ [m/s] | ‖v_rel_ang‖ [rad/s] |\n')
        f.write('|---|---|---|---|\n')
        for row in profile_rows:
            f.write(f'| {row["k"]} | {row["t"]:.3f} | '
                    f'{row["vrel_lin_norm"]:.6e} | '
                    f'{row["vrel_ang_norm"]:.6e} |\n')
        f.write('\n')

        # ── Item 3 ──
        f.write('## 3. Structure body velocity at dock tick\n\n')
        f.write(f'- `v_struct_lin` (world, finite-diff of `struct_pos`): '
                f'`{v_struct_lin.tolist()}` m/s, '
                f'`‖·‖ = {np.linalg.norm(v_struct_lin):.6e}` m/s\n')
        f.write(f'- `ω_struct` (world, logged `omega_s`): '
                f'`{v_struct_ang.tolist()}` rad/s, '
                f'`‖·‖ = {np.linalg.norm(v_struct_ang):.6e}` rad/s\n')
        f.write(f'- For context, `struct_pos[k_dock]` = '
                f'`{np.asarray(log.struct_pos[k_dock]).tolist()}`, '
                f'`‖struct_quat‖` ≈ 1 '
                f'(`{np.linalg.norm(log.struct_quat[k_dock]):.6f}`).\n\n')

        # ── Item 4 ──
        f.write('## 4. SwingPlanner reference velocity (from logged '
                '`p_ee_ref` / `q_ee_ref`)\n\n')
        f.write(f'`T_step = {T_step:.3f} s` (from `preplanner_T_steps[0]`). '
                f'Nearest log tick ≤ T_step: `k = {k_ss_last}` '
                f'(t = {t[k_ss_last]:.3f} s). Dock tick: `k = {k_dock}` '
                f'(t = {t[k_dock]:.3f} s).\n\n')
        f.write('| tick | t [s] | ‖v_ref_lin‖ [m/s] | ‖v_ref_ang‖ [rad/s] | '
                '‖v_rel_actual_lin‖ [m/s] | ‖v_rel_actual_ang‖ [rad/s] |\n')
        f.write('|---|---|---|---|---|---|\n')
        f.write(f'| k_ss_last = {k_ss_last} | {t[k_ss_last]:.3f} | '
                f'{np.linalg.norm(v_ref_l_Tstep):.6e} | '
                f'{np.linalg.norm(v_ref_a_Tstep):.6e} | '
                f'{np.linalg.norm(vrel_l_Tstep):.6e} | '
                f'{np.linalg.norm(vrel_a_Tstep):.6e} |\n')
        f.write(f'| k_dock = {k_dock} | {t[k_dock]:.3f} | '
                f'{np.linalg.norm(v_ref_l_dock):.6e} | '
                f'{np.linalg.norm(v_ref_a_dock):.6e} | '
                f'{np.linalg.norm(v_rel_lin_dock):.6e} | '
                f'{np.linalg.norm(v_rel_ang_dock):.6e} |\n\n')

        # ── Item 5 ──
        f.write('## 5. Post-dock residual relative motion (t ∈ [7.31, 7.51] s)\n\n')
        f.write('| k | t [s] | ‖v_rel_lin‖ [m/s] | ‖v_rel_ang‖ [rad/s] |\n')
        f.write('|---|---|---|---|\n')
        for row in post_rows:
            f.write(f'| {row["k"]} | {row["t"]:.3f} | '
                    f'{row["vrel_lin_norm"]:.6e} | '
                    f'{row["vrel_ang_norm"]:.6e} |\n')
        f.write('\n')
        f.write('**`efc_force` on the newly activated weld is not in the '
                'committed log** — `SimLog` does not capture MuJoCo `efc_*` '
                'arrays. Only the relative-velocity signal above is '
                'available from the committed data.\n')

        # ── Section 6 (appended): Post-T_step velocity decay ─────────
        d_grip_swing_mm = np.array(log.d_grip_swing, dtype=float) * 1000.0

        # (6.1) Last SS sample ≤ T_step
        k_le = int(np.where(t <= T_step + 1e-9)[0][-1])
        # (6.2) First sample with t > T_step
        k_gt = int(np.where(t > T_step)[0][0])
        # (6.3) First sample with t ≥ T_step + 0.5 AND d < 5 mm
        mask_delay = (t >= T_step + 0.5) & (d_grip_swing_mm < 5.0)
        k_delay = (int(np.where(mask_delay)[0][0])
                   if np.any(mask_delay) else None)
        # (6.4) First sample with d < 5 mm
        mask_dock = (d_grip_swing_mm < 5.0)
        k_first_d = (int(np.where(mask_dock)[0][0])
                     if np.any(mask_dock) else None)

        def _sample(k):
            p_g,  vg_l,  vg_a  = _gripper_state_at(log, k, dt)
            p_a,  va_l,  va_a  = _anchor_state_at (log, k, dt)
            vr_l = vg_l - va_l
            approach = p_a - p_g
            d_approach = float(np.linalg.norm(approach))
            u = approach / d_approach if d_approach > 1e-12 else np.zeros(3)
            vr_l_along = float(vr_l @ u)
            return {
                'k': k, 't': float(t[k]),
                'vrel_lin_norm': float(np.linalg.norm(vr_l)),
                'vrel_lin_along': vr_l_along,
                'd_mm': float(d_grip_swing_mm[k]),
            }

        s_le    = _sample(k_le)
        s_gt    = _sample(k_gt)
        s_delay = _sample(k_delay)    if k_delay    is not None else None
        s_first = _sample(k_first_d)  if k_first_d is not None else None

        f.write('\n## 6. Post-T_step velocity decay\n\n')
        f.write(f'`T_step = {T_step:.3f} s` (`preplanner_T_steps[0]`). '
                f'`d_grip_swing` pulled from the committed log and '
                f'converted to mm; `v_rel_lin` computed as in sections '
                f'1–5 (finite-diff + rigid-body anchor).\n\n')
        f.write('| label | k | t [s] | d [mm] | ‖v_rel_lin‖ [m/s] | '
                'v_rel_lin along approach [m/s] |\n')
        f.write('|---|---|---|---|---|---|\n')
        def _row(label, s):
            if s is None:
                return f'| {label} | — | — | — | — | — |\n'
            return (f'| {label} | {s["k"]} | {s["t"]:.3f} | '
                    f'{s["d_mm"]:.3f} | {s["vrel_lin_norm"]:.6e} | '
                    f'{s["vrel_lin_along"]:+.6e} |\n')
        f.write(_row(f'(1) last SS, t ≤ T_step',    s_le))
        f.write(_row(f'(2) first post-T_step, t > T_step', s_gt))
        f.write(_row(f'(3) first tick w/ t ≥ T_step+0.5 AND d<5mm', s_delay))
        f.write(_row(f'(4) first tick w/ d<5mm',    s_first))
        f.write('\n')
        f.write('**Note on the `d` column.** `d` above is pulled verbatim '
                'from `log.d_grip_swing`, which `sim_loop` computes as '
                '|gripper − scheduled-target-anchor|. During SS the '
                'scheduled target is anchor index 3 (`anchor_4b`, the '
                'dock target), so row (1) is the real gripper-to-dock '
                'distance. During the trailing DS branch '
                '(`sim_loop.py:1383-1388`), `_step` is called with '
                '`target_anchor = 0`, so `log.d_grip_swing` at k ≥ 72 '
                'reports distance to anchor index 0 (`anchor_1b` at '
                'body-local `(-2.0, -0.3, 0.025)`), which is ~2.4 m from '
                'the welded gripper — this explains the 2399.840 mm '
                'reading at row (2). It does not affect rows (1), (3), '
                '(4)\'s interpretation of dock proximity — but it means '
                'the "d < 5 mm" mask is satisfied only at k=71 (SS with '
                'scheduled target = dock target).\n\n')
        f.write('**Note on item (3).** The dock actually activates at '
                f'`t = {dock_t:.3f} s`, so the post-dock phase enters '
                'DS at that tick. The weld is therefore **active** for '
                'every tick with `t ≥ T_step + 0.5` in the committed log. '
                'No counterfactual (dock gate delayed 500 ms with weld '
                'still inactive) is simulable from the committed log — '
                'it would require a new simulation with the dock gate '
                'logic altered. Row (3) reads "—" because the '
                '`log.d_grip_swing < 5 mm` mask is never True for '
                '`t ≥ T_step + 0.5` (see note above on the `d` column '
                'definition during DS).\n')

    # Console summary
    print()
    print('=' * 78)
    print(f'Dock velocity audit — t_dock = {dock_t:.3f} s, k = {k_dock}')
    print('=' * 78)
    print(f'  ‖v_rel_lin‖                      = {np.linalg.norm(v_rel_lin_dock):.6e}  m/s')
    print(f'  ‖v_rel_ang‖                      = {np.linalg.norm(v_rel_ang_dock):.6e}  rad/s')
    print(f'  v_rel_lin along approach         = {v_rel_lin_along:+.6e}  m/s')
    print(f'  |struct_pos| = {np.linalg.norm(log.struct_pos[k_dock]):.3e} m'
          f'  ‖v_struct_lin‖ = {np.linalg.norm(v_struct_lin):.6e} m/s'
          f'  ‖ω_struct‖ = {np.linalg.norm(v_struct_ang):.6e} rad/s')
    print(f'  v_ref_lin at k_ss_last ({t[k_ss_last]:.3f}s) '
          f'= ‖{np.linalg.norm(v_ref_l_Tstep):.3e}‖ m/s')
    print(f'  v_ref_lin at k_dock    ({t[k_dock]:.3f}s) '
          f'= ‖{np.linalg.norm(v_ref_l_dock):.3e}‖ m/s')
    print(f'  Report: {OUT_PATH}')


if __name__ == '__main__':
    main()
