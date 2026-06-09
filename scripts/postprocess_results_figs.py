"""Post-process the canonical-run logs into figure-ready signals.

READ-ONLY w.r.t. the pipeline: loads the saved sim_log of the canonical run
and exports derived signals (NMPC-planned Ḣ_s, tracking errors, phase mask)
as a tidy CSV plus a small JSON metrics summary. Does NOT re-run any sim.

Inputs:
    results/diag_cooperative_arms_legacy_pid_numerical/sim_log.json
    models/VISPA_crawling_rwa3.xml (MJCF, for anchor site recovery)

Outputs:
    results/diag_cooperative_arms_legacy_pid_numerical/postproc_F3F4.csv
    results/diag_cooperative_arms_legacy_pid_numerical/postproc_metrics.json
    results/diag_cooperative_arms_legacy_pid_numerical/postproc_stance_sanity.txt
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pinocchio as pin

_root = Path(__file__).resolve().parent.parent
LOG_DIR = _root / 'results' / 'diag_cooperative_arms_legacy_pid_numerical'
LOG_PATH = LOG_DIR / 'sim_log.json'
MJCF_PATH = _root / 'models' / 'VISPA_crawling_rwa3.xml'

CSV_OUT = LOG_DIR / 'postproc_F3F4.csv'
JSON_OUT = LOG_DIR / 'postproc_metrics.json'
SANITY_OUT = LOG_DIR / 'postproc_stance_sanity.txt'

# Cross-check threshold: stance gripper must be within this distance of its
# nominal anchor (per gait_plan). If exceeded, the stance-pair mapping
# disagrees with physics and we STOP rather than proceeding.
STANCE_CONSISTENCY_THRESHOLD_MM = 10.0

TAU_W_MAX = 5.0  # canonical (NMPC + AOCS), per-axis Nm


def _load_log():
    if not LOG_PATH.exists():
        sys.exit(f'[FATAL] canonical-run log not found: {LOG_PATH}')
    with open(LOG_PATH) as f:
        sl = json.load(f)
    if 'gait_plan' not in sl:
        sys.exit('[FATAL] sim_log.json is missing the gait_plan field; '
                 're-run the canonical scenario to persist it.')
    return sl


def _load_anchors_struct_frame(struct_pos0, struct_quat0):
    """Replicate sim_loop.py:222-225's R_s-frame anchor reconstruction
    from the MJCF, given the structure pose captured at the first log tick.
    """
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    anchors_a_world, anchors_b_world = [], []
    for i in range(1, 20):
        sa = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f'anchor_{i}a')
        sb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f'anchor_{i}b')
        if sa < 0 or sb < 0:
            break
        anchors_a_world.append(d.site_xpos[sa].copy())
        anchors_b_world.append(d.site_xpos[sb].copy())

    qw, qx, qy, qz = struct_quat0
    R_s0 = pin.Quaternion(qw, qx, qy, qz).toRotationMatrix()
    p_s0 = np.asarray(struct_pos0, dtype=float)
    anchors_a = [R_s0.T @ (a - p_s0) for a in anchors_a_world]
    anchors_b = [R_s0.T @ (b - p_s0) for b in anchors_b_world]
    return anchors_a, anchors_b


def _gait_phase_for_tick(step_idx, phase, gait_plan_phases):
    """Resolve (step_idx, phase) -> gait_plan.phases[idx].

    Convention (verified empirically against the canonical run):
      - gait_plan.phases interleaves DOUBLE / SINGLE_X / DOUBLE / SINGLE_X / ...
        starting with the initial DOUBLE and ending with the trailing DOUBLE
        after the last SS step.
      - The per-tick log only contains SS ticks (phase='SS') and the FINAL
        trailing-DS settle ticks (phase='DS', step_idx = n_steps - 1).
      - Inter-step DS settles run inside _run_ds_passivity_loop and are NOT
        appended to the per-tick log; they appear only as summary entries in
        sim_log['inter_step_settles'].

    Mapping:
      - phase='SS', step_idx=k → gait_plan.phases[2k+1]
      - phase='DS' (trailing only) → gait_plan.phases[-1]
    """
    if phase == 'SS':
        idx = 2 * step_idx + 1
    elif phase == 'DS':
        # Trailing DS (no inter-step DS in per-tick log for canonical run).
        idx = len(gait_plan_phases) - 1
    else:
        raise ValueError(f'Unknown phase: {phase!r}')
    return idx


def _format_xyz(v):
    return f'[{v[0]:+.3f},{v[1]:+.3f},{v[2]:+.3f}]'


def main():
    print('=' * 78)
    print('Canonical-run post-processing — F3 + F4 derived signals')
    print('=' * 78)
    sl = _load_log()
    t = np.asarray(sl['t'])
    n = len(t)

    phase = np.asarray(sl['phase'])
    step_idx = np.asarray(sl['step_idx'], dtype=int)
    swing_arm = np.asarray(sl['swing_arm'])

    lambda_ref = np.asarray(sl['lambda_ref'])   # (n, 12)
    tau_w_cmd = np.asarray(sl['tau_w'])         # (n, 3) — AOCS-commanded
    p_torso = np.asarray(sl['p_torso'])         # (n, 3)
    p_torso_ref = np.asarray(sl['p_torso_ref']) # (n, 3)
    q_torso = np.asarray(sl['q_torso'])         # (n, 4) wxyz
    q_torso_ref = np.asarray(sl['q_torso_ref']) # (n, 4) wxyz
    e_torso_ori_log = np.asarray(sl['e_torso_ori'])  # (n,) degrees, scalar
    p_ee = np.asarray(sl['p_ee'])               # (n, 3) swing arm
    q_ee = np.asarray(sl['q_ee'])               # (n, 4) wxyz
    d_grip_stance = np.asarray(sl['d_grip_stance'])  # (n,) scalar
    struct_pos = np.asarray(sl['struct_pos'])   # (n, 3)
    struct_quat = np.asarray(sl['struct_quat']) # (n, 4) wxyz

    # ── Recover R_s-frame anchor positions ───────────────────────────────
    anchors_a, anchors_b = _load_anchors_struct_frame(
        struct_pos[0], struct_quat[0])
    print(f'\n[anchors] loaded {len(anchors_a)} anchor pairs in R_s frame')
    print(f'  anchor_3a (idx 2): {_format_xyz(anchors_a[2])}')
    print(f'  anchor_4b (idx 3): {_format_xyz(anchors_b[3])}')

    # ── Stance-pair resolution + sanity table ────────────────────────────
    gait_phases = sl['gait_plan']['phases']
    print(f'\n[gait_plan] {len(gait_phases)} phases:')
    for i, p in enumerate(gait_phases):
        tag = ('SS:' + p['swing_arm']) if p['phase'] != 'DOUBLE' else 'DS'
        print(f'  phases[{i:>2}] {p["phase"]:>8s} {tag:>4s} '
              f'anchor=({p["anchor_a_idx"]}a,{p["anchor_b_idx"]}b)  '
              f'swing_from={p["swing_from_idx"]}  swing_to={p["swing_to_idx"]}')

    # Resolve per-tick: stance_a_idx, stance_b_idx, swing_to_idx
    stance_a_idx = np.full(n, -1, dtype=int)
    stance_b_idx = np.full(n, -1, dtype=int)
    swing_to_idx = np.full(n, -1, dtype=int)
    gait_phase_idx_per_tick = np.full(n, -1, dtype=int)
    for k in range(n):
        gp_idx = _gait_phase_for_tick(int(step_idx[k]), str(phase[k]),
                                       gait_phases)
        gp = gait_phases[gp_idx]
        gait_phase_idx_per_tick[k] = gp_idx
        stance_a_idx[k] = gp['anchor_a_idx']
        stance_b_idx[k] = gp['anchor_b_idx']
        swing_to_idx[k] = gp['swing_to_idx']

    # Stance sanity table at every step transition
    sanity_lines = []
    sanity_lines.append('Stance sanity — one row per (step_idx, phase) transition.')
    sanity_lines.append(
        'tick |   t [s] | step | phase | swing | stance_a | stance_b | swing_to '
        '| r_C1=anchor_a (xyz)         | r_C2=anchor_b (xyz)')
    sanity_lines.append('-' * 130)
    prev_key = None
    for k in range(n):
        key = (int(step_idx[k]), str(phase[k]), str(swing_arm[k]))
        if key != prev_key:
            r_C1 = anchors_a[stance_a_idx[k]]
            r_C2 = anchors_b[stance_b_idx[k]]
            sanity_lines.append(
                f'{k:>4d} | {t[k]:>7.2f} | {key[0]:>4d} | {key[1]:>5s} | '
                f'{key[2]:>5s} | {stance_a_idx[k]:>8d} | {stance_b_idx[k]:>8d} '
                f'| {swing_to_idx[k]:>8d} | {_format_xyz(r_C1):>27s} '
                f'| {_format_xyz(r_C2):>27s}')
            prev_key = key
    print('\n' + '\n'.join(sanity_lines))
    SANITY_OUT.write_text('\n'.join(sanity_lines) + '\n')

    # Cross-check: stance gripper must be near its nominal anchor.
    # d_grip_stance is logged per tick as the scalar distance from the
    # stance-arm gripper to "its anchor"; if our stance-pair resolution is
    # consistent with the physics, this should remain small.
    d_grip_stance_mm = d_grip_stance * 1000.0
    stance_max_mm = float(d_grip_stance_mm.max())
    print(f'\n[stance cross-check] max d_grip_stance over run: '
          f'{stance_max_mm:.4f} mm  '
          f'(threshold {STANCE_CONSISTENCY_THRESHOLD_MM} mm)')
    if stance_max_mm > STANCE_CONSISTENCY_THRESHOLD_MM:
        idx_bad = int(np.argmax(d_grip_stance_mm))
        print(f'[FATAL] stance-pair mapping inconsistent with physics at '
              f'tick {idx_bad} (t={t[idx_bad]:.2f}, '
              f'd_grip_stance={d_grip_stance_mm[idx_bad]:.2f} mm). '
              f'Aborting — STOP and re-verify gait_plan ↔ log mapping.')
        sys.exit(1)
    print(f'[stance cross-check] OK — stance gripper stays within '
          f'{stance_max_mm:.4f} mm of nominal anchor at every tick.')

    # ── 1. NMPC-planned Ḣ_s (per axis), reconstructed in R_s ─────────────
    Hdot_s = np.zeros((n, 3))
    for k in range(n):
        r_C1 = anchors_a[stance_a_idx[k]]
        r_C2 = anchors_b[stance_b_idx[k]]
        lr = lambda_ref[k]
        Hdot_s[k] = (np.cross(r_C1, lr[0:3]) + lr[3:6]
                     + np.cross(r_C2, lr[6:9]) + lr[9:12])
    Hdot_s_peak_per_axis = np.abs(Hdot_s).max(axis=0)
    Hdot_s_at_clip = (np.abs(Hdot_s) >= TAU_W_MAX - 1e-6).any(axis=1).sum()
    print(f'\n[1] Hdot_s shape={Hdot_s.shape}, '
          f'per-axis peak |Hdot_s|={Hdot_s_peak_per_axis} Nm, '
          f'ticks at ±{TAU_W_MAX} clip: {int(Hdot_s_at_clip)}')

    # ── tau_w (AOCS commanded), already in log ──
    tau_w_peak_per_axis = np.abs(tau_w_cmd).max(axis=0)
    tau_w_clip_ticks = (np.abs(tau_w_cmd) >= TAU_W_MAX - 1e-6).any(axis=1).sum()
    print(f'[1] tau_w (AOCS cmd) shape={tau_w_cmd.shape}, '
          f'per-axis peak={tau_w_peak_per_axis} Nm, '
          f'ticks at ±{TAU_W_MAX} clip: {int(tau_w_clip_ticks)} '
          f'({100*tau_w_clip_ticks/n:.2f}%)')

    # ── 2. Torso ORIENTATION error (geodesic angle) ──────────────────────
    e_ori_torso_deg_recomp = np.zeros(n)
    for k in range(n):
        qw, qx, qy, qz = q_torso[k]
        R_act = pin.Quaternion(qw, qx, qy, qz).toRotationMatrix()
        qw, qx, qy, qz = q_torso_ref[k]
        R_ref = pin.Quaternion(qw, qx, qy, qz).toRotationMatrix()
        omega = pin.log3(R_act.T @ R_ref)
        e_ori_torso_deg_recomp[k] = float(np.degrees(np.linalg.norm(omega)))
    cross_check_diff = np.max(np.abs(
        e_ori_torso_deg_recomp - e_torso_ori_log))
    print(f'\n[2] e_ori_torso_deg shape=({n},), '
          f'min={e_ori_torso_deg_recomp.min():.4e}, '
          f'max={e_ori_torso_deg_recomp.max():.4f}, '
          f'cross-check vs logged scalar: max|diff|={cross_check_diff:.2e}')
    if cross_check_diff > 1e-3:
        print(f'[warn] recomputed torso ori error diverges from logged by '
              f'{cross_check_diff:.4e} deg (exceeds 1e-3 deg tolerance)')

    # ── 3. Torso POSITION error per component (R_s) ──────────────────────
    e_pos_torso = p_torso - p_torso_ref      # (n, 3) in R_s
    print(f'[3] e_pos_torso shape={e_pos_torso.shape}, per-axis '
          f'min={e_pos_torso.min(axis=0)}, max={e_pos_torso.max(axis=0)}')

    # ── 4. Swing-EE position error toward FIXED docking target ───────────
    # NOTE: target = sched.anchors_{a,b}[swing_to_idx] (fixed in R_s),
    # NOT the moving quintic setpoint sr.p_ee.
    d_ee_to_target = np.zeros(n)
    p_target_per_tick = np.zeros((n, 3))
    for k in range(n):
        arm = str(swing_arm[k])
        if swing_to_idx[k] < 0:
            # DS phase — no active swing; report nan so the SS mask is the
            # one and only filter on the figure.
            d_ee_to_target[k] = np.nan
            p_target_per_tick[k] = np.nan
            continue
        if arm == 'a':
            p_tgt = anchors_a[swing_to_idx[k]]
        elif arm == 'b':
            p_tgt = anchors_b[swing_to_idx[k]]
        else:
            d_ee_to_target[k] = np.nan
            p_target_per_tick[k] = np.nan
            continue
        p_target_per_tick[k] = p_tgt
        d_ee_to_target[k] = float(np.linalg.norm(p_ee[k] - p_tgt))
    d_ee_finite = d_ee_to_target[np.isfinite(d_ee_to_target)]
    print(f'[4] d_ee_to_target shape=({n},), '
          f'min={d_ee_finite.min():.6f} m, max={d_ee_finite.max():.4f} m '
          f'({np.isfinite(d_ee_to_target).sum()} finite ticks)')

    # ── 5. Swing-EE orientation error toward target (identity in R_s) ────
    e_ori_ee_deg = np.zeros(n)
    for k in range(n):
        qw, qx, qy, qz = q_ee[k]
        R_ee = pin.Quaternion(qw, qx, qy, qz).toRotationMatrix()
        # Target rotation = I_3 (anchor_se3 returns eye(3))
        omega = pin.log3(R_ee.T)  # = log3(R_ee.T @ I)
        e_ori_ee_deg[k] = float(np.degrees(np.linalg.norm(omega)))
    print(f'[5] e_ori_ee_deg shape=({n},), '
          f'min={e_ori_ee_deg.min():.4e}, max={e_ori_ee_deg.max():.4f}')

    # ── 6. Phase mask for EE curves ──────────────────────────────────────
    mask_ss = (phase == 'SS')
    print(f'[6] mask_ss: {int(mask_ss.sum())} SS ticks / {n} total '
          f'({100*mask_ss.sum()/n:.1f}%)')

    # ── Export CSV ───────────────────────────────────────────────────────
    header = [
        't', 'phase', 'step_idx', 'mask_ss',
        'Hdot_s_x', 'Hdot_s_y', 'Hdot_s_z',
        'tau_w_x', 'tau_w_y', 'tau_w_z',
        'e_ori_torso_deg',
        'e_pos_torso_x', 'e_pos_torso_y', 'e_pos_torso_z',
        'd_ee_to_target', 'e_ori_ee_deg',
        # Auxiliary for sanity / plotting:
        'swing_arm', 'stance_a_idx', 'stance_b_idx', 'swing_to_idx',
        'gait_phase_idx',
    ]
    with open(CSV_OUT, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for k in range(n):
            w.writerow([
                f'{t[k]:.6f}', str(phase[k]), int(step_idx[k]),
                int(mask_ss[k]),
                f'{Hdot_s[k,0]:.9f}', f'{Hdot_s[k,1]:.9f}', f'{Hdot_s[k,2]:.9f}',
                f'{tau_w_cmd[k,0]:.9f}', f'{tau_w_cmd[k,1]:.9f}', f'{tau_w_cmd[k,2]:.9f}',
                f'{e_ori_torso_deg_recomp[k]:.9f}',
                f'{e_pos_torso[k,0]:.9f}', f'{e_pos_torso[k,1]:.9f}', f'{e_pos_torso[k,2]:.9f}',
                f'{d_ee_to_target[k]:.9f}' if np.isfinite(d_ee_to_target[k]) else 'nan',
                f'{e_ori_ee_deg[k]:.9f}',
                str(swing_arm[k]), int(stance_a_idx[k]), int(stance_b_idx[k]),
                int(swing_to_idx[k]), int(gait_phase_idx_per_tick[k]),
            ])
    print(f'\n[csv] wrote {CSV_OUT}  ({n} rows × {len(header)} cols)')

    # ── Metrics summary ──────────────────────────────────────────────────
    # Per-step d_ee at dock (from dock_events).
    # The dock event's own d_mm field is the canonical sim-side dock distance
    # (from _gripper_distance, MJ world frame, but a norm is frame-invariant).
    # We also report the recomputed d_ee from the last SS tick of the step,
    # which is in R_s — these should agree to numerical precision.
    docks = sl.get('dock_events', [])
    d_ee_at_docks = []
    for ev in docks:
        step = int(ev['step'])
        # Find the last SS tick belonging to this step (before the SS→DS
        # transition). At the dock event's exact time the log tick may have
        # already advanced to DS (swing_to_idx = -1 → d_ee nan).
        ss_mask_step = (step_idx == step) & (phase == 'SS')
        if ss_mask_step.any():
            k_last_ss = int(np.where(ss_mask_step)[0].max())
            d_ee_recomp_mm = (float(d_ee_to_target[k_last_ss] * 1000)
                              if np.isfinite(d_ee_to_target[k_last_ss])
                              else None)
        else:
            k_last_ss = int(np.argmin(np.abs(t - ev['t'])))
            d_ee_recomp_mm = None
        d_ee_at_docks.append({
            'step_idx': step,
            't_dock': float(ev['t']),
            't_last_ss_tick': float(t[k_last_ss]),
            'd_mm_sim_log': float(ev['d_mm']),                  # canonical (sim-side)
            'd_ee_to_target_mm_recomp': d_ee_recomp_mm,         # post-hoc (R_s)
            'ori_deg_sim_log': float(ev['ori_deg']),
            'e_ori_ee_deg_recomp': float(e_ori_ee_deg[k_last_ss]),
            'arm': str(ev['arm']),
            'anchor': int(ev['anchor']),
        })

    e_pos_torso_norm = np.linalg.norm(e_pos_torso, axis=1)

    def _rms(x, mask=None):
        if mask is None:
            return float(np.sqrt(np.mean(x ** 2)))
        x = x[mask]
        return float(np.sqrt(np.mean(x ** 2))) if x.size > 0 else None

    metrics = {
        'canonical_run': str(LOG_DIR),
        'n_ticks': int(n),
        'duration_s': float(t[-1] - t[0]),
        'mask_ss_fraction': float(mask_ss.mean()),
        'e_ori_torso_deg': {
            'rms_full_run': _rms(e_ori_torso_deg_recomp),
            'peak': float(e_ori_torso_deg_recomp.max()),
        },
        'e_pos_torso_norm_m': {
            'rms_full_run': _rms(e_pos_torso_norm),
            'peak': float(e_pos_torso_norm.max()),
        },
        'e_pos_torso_per_axis_peak_m': {
            'x': float(np.abs(e_pos_torso[:, 0]).max()),
            'y': float(np.abs(e_pos_torso[:, 1]).max()),
            'z': float(np.abs(e_pos_torso[:, 2]).max()),
        },
        'd_ee_at_docks_mm': d_ee_at_docks,
        'd_ee_final_m': float(d_ee_to_target[-1])
            if np.isfinite(d_ee_to_target[-1]) else None,
        'e_ori_ee_deg_SS_only': {
            'rms_SS_segments': _rms(e_ori_ee_deg, mask_ss),
            'peak_SS_segments': float(e_ori_ee_deg[mask_ss].max())
                if mask_ss.any() else None,
        },
        'Hdot_s_per_axis_peak_Nm': {
            'x': float(np.abs(Hdot_s[:, 0]).max()),
            'y': float(np.abs(Hdot_s[:, 1]).max()),
            'z': float(np.abs(Hdot_s[:, 2]).max()),
        },
        'Hdot_s_hits_bound': bool((np.abs(Hdot_s) >= TAU_W_MAX - 1e-6).any()),
        'Hdot_s_at_clip_ticks': int(Hdot_s_at_clip),
        'tau_w_per_axis_peak_Nm': {
            'x': float(np.abs(tau_w_cmd[:, 0]).max()),
            'y': float(np.abs(tau_w_cmd[:, 1]).max()),
            'z': float(np.abs(tau_w_cmd[:, 2]).max()),
        },
        'tau_w_at_clip_ticks': int(tau_w_clip_ticks),
        'tau_w_at_clip_fraction': float(tau_w_clip_ticks / n),
        'stance_cross_check_max_d_grip_stance_mm': stance_max_mm,
        'tau_w_max_clip_Nm': TAU_W_MAX,
    }
    with open(JSON_OUT, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'[json] wrote {JSON_OUT}')

    # Print metrics nicely
    print('\n' + '=' * 78)
    print('METRICS SUMMARY')
    print('=' * 78)
    print(f'  e_ori_torso_deg     RMS: {metrics["e_ori_torso_deg"]["rms_full_run"]:.4f}   '
          f'peak: {metrics["e_ori_torso_deg"]["peak"]:.4f}')
    print(f'  |e_pos_torso|       RMS: {metrics["e_pos_torso_norm_m"]["rms_full_run"]:.4e} m   '
          f'peak: {metrics["e_pos_torso_norm_m"]["peak"]:.4e} m')
    print(f'  e_pos_torso per-axis peak (m): x={metrics["e_pos_torso_per_axis_peak_m"]["x"]:.4f}  '
          f'y={metrics["e_pos_torso_per_axis_peak_m"]["y"]:.4f}  '
          f'z={metrics["e_pos_torso_per_axis_peak_m"]["z"]:.4f}')
    print(f'  d_ee at docks (per step, mm)   '
          f'[sim-log "d_mm" | post-hoc R_s recomputed]:')
    for ev in d_ee_at_docks:
        recomp = (f'{ev["d_ee_to_target_mm_recomp"]:.2f}'
                  if ev["d_ee_to_target_mm_recomp"] is not None else '—')
        print(f'    step {ev["step_idx"]} (arm {ev["arm"]}→anchor {ev["anchor"]}): '
              f't_dock={ev["t_dock"]:.2f}s  '
              f'd_mm={ev["d_mm_sim_log"]:.2f}  recomp_mm={recomp}  '
              f'ori_sim={ev["ori_deg_sim_log"]:.3f}°  '
              f'ori_recomp={ev["e_ori_ee_deg_recomp"]:.3f}°')
    print(f'  e_ori_ee_deg (SS)   RMS: {metrics["e_ori_ee_deg_SS_only"]["rms_SS_segments"]:.4f}   '
          f'peak: {metrics["e_ori_ee_deg_SS_only"]["peak_SS_segments"]:.4f}')
    print(f'  Hdot_s per-axis peak (Nm):   x={metrics["Hdot_s_per_axis_peak_Nm"]["x"]:.4f}  '
          f'y={metrics["Hdot_s_per_axis_peak_Nm"]["y"]:.4f}  '
          f'z={metrics["Hdot_s_per_axis_peak_Nm"]["z"]:.4f}   '
          f'(hits ±{TAU_W_MAX}? {metrics["Hdot_s_hits_bound"]})')
    print(f'  tau_w per-axis peak (Nm):    x={metrics["tau_w_per_axis_peak_Nm"]["x"]:.4f}  '
          f'y={metrics["tau_w_per_axis_peak_Nm"]["y"]:.4f}  '
          f'z={metrics["tau_w_per_axis_peak_Nm"]["z"]:.4f}   '
          f'at clip {metrics["tau_w_at_clip_ticks"]}/{n} ticks '
          f'({100*metrics["tau_w_at_clip_fraction"]:.2f}%)')


if __name__ == '__main__':
    main()
