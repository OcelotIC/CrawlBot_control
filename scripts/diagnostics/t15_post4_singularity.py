"""T15-post-4 — Singularity analysis of step 2 swing trajectory.

Read-only diagnostic. For each physics_trace sample across T15
(bug1fix_vel run), loads the Pinocchio robot model, sets state to
the logged q, computes the arm-B EE Jacobian in LOCAL_WORLD_ALIGNED
(same convention as the velocity log fields), and reports SVD /
conditioning / manipulability statistics per tick, per step.

No simulation. No MuJoCo stepping. Just Pinocchio FK + Jacobian.

Outputs (to stdout):
    Section A: model identity + frame/slice conventions
    Section B: per-step summary table (min/max conditioning)
    Section C: step 2 zoom around flailing events (0.1 s spacing)
    Section D: singularity-direction vs reference-velocity alignment
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

import numpy as np
import pinocchio as pin

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)

from crawlbot.core.robot_interface import RobotInterface

URDF = os.path.join(_root, 'models', 'VISPA_crawling_fixed.urdf')
RUN  = os.path.join(_root, 'results', 'M7_1pct_3step_v22_t15_bug1fix_vel')


def unit(v):
    n = np.linalg.norm(v)
    return v if n < 1e-12 else v / n


def manip_full(J):
    """sqrt(det(J J^T)) for J shape (m, n), m <= n."""
    return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))


def log_msg(fp, s):
    print(s)
    fp.write(s + "\n")


def main():
    t_start = time.perf_counter()

    log_path = os.path.join(RUN, 'sim_log.json')
    tr_path = os.path.join(RUN, 'physics_trace.pkl')
    log = json.load(open(log_path))
    tr = pickle.load(open(tr_path, 'rb'))

    t_sim = np.asarray(log['t'])
    step_idx = np.asarray(log['step_idx'])
    phase = np.asarray(log['phase'])
    p_ee_ref = np.asarray(log['p_ee_ref'])  # meters

    out_txt = os.path.join(RUN, 'T15_post4_singularity_output.txt')
    fp = open(out_txt, 'w')

    # ── Section A — model + conventions ────────────────────────
    log_msg(fp, "=" * 76)
    log_msg(fp, "SECTION A  —  Model + frame/slice conventions")
    log_msg(fp, "=" * 76)

    ri = RobotInterface(URDF, gravity='zero')
    m = ri.model
    data = ri.data
    log_msg(fp, f"Pinocchio version: {pin.__version__}")
    log_msg(fp, f"URDF: {URDF}")
    log_msg(fp, f"model.nq = {m.nq}, model.nv = {m.nv}")
    log_msg(fp, f"arm_b_q_slice = {ri.arm_b_q_slice} "
                f"(= q indices {ri.arm_b_q_slice.start}..{ri.arm_b_q_slice.stop-1})")
    log_msg(fp, f"arm_b_v_slice = {ri.arm_b_v_slice} "
                f"(= v indices {ri.arm_b_v_slice.start}..{ri.arm_b_v_slice.stop-1})")
    log_msg(fp, f"frame_tool_b = {ri.frame_tool_b} "
                f"(model.frames[{ri.frame_tool_b}].name = {m.frames[ri.frame_tool_b].name!r})")
    log_msg(fp, f"Jacobian convention: pin.LOCAL_WORLD_ALIGNED "
                f"(matches v_ee_b logging in sim_loop.py:J_tool_b @ v)")

    frame_b = ri.frame_tool_b
    frame_a = ri.frame_tool_a
    # Joint-slice columns of the 6×nv Jacobian
    armB_v = list(range(ri.arm_b_v_slice.start, ri.arm_b_v_slice.stop))
    armA_v = list(range(ri.arm_a_v_slice.start, ri.arm_a_v_slice.stop))
    log_msg(fp, f"Arm B joint columns in J (v-indices): {armB_v}")
    log_msg(fp, f"Arm A joint columns in J (v-indices): {armA_v}")

    # ── Build per-sample Jacobian dataset ───────────────────────
    # Note: physics_trace only covers phase=='SS', and samples
    # at roughly 0.2 s. We use those samples directly; we align
    # them to log ticks by nearest-t lookup.
    tr_ts = np.array([it['t'] for it in tr])
    # Map each tr sample to the nearest log tick (for step_idx tag)
    tr_step_idx = []
    tr_phase = []
    for tt in tr_ts:
        k = int(np.argmin(np.abs(t_sim - tt)))
        tr_step_idx.append(int(step_idx[k]))
        tr_phase.append(phase[k])

    # Compute Jacobian at every tr sample
    per_sample = []
    tr_swing_arm = [log['swing_arm'][int(np.argmin(np.abs(t_sim - tt)))] for tt in tr_ts]

    for i, it in enumerate(tr):
        q = np.asarray(it['q'])
        # Forward kinematics + frame placements + all Jacobians
        pin.forwardKinematics(m, data, q)
        pin.updateFramePlacements(m, data)
        pin.computeJointJacobians(m, data, q)

        # Arm B frame jacobian in LOCAL_WORLD_ALIGNED
        J6 = pin.getFrameJacobian(m, data, frame_b, pin.LOCAL_WORLD_ALIGNED)  # (6, nv)
        Jb = J6[:, armB_v]  # (6, 7)  arm B joint columns
        # SVD full 6x7
        U6, s6, Vt6 = np.linalg.svd(Jb, full_matrices=False)
        # Split trans/rot
        Jt = Jb[:3, :]  # (3, 7)
        Jr = Jb[3:, :]  # (3, 7)
        Ut, st, Vtt = np.linalg.svd(Jt, full_matrices=False)
        Ur, sr, Vtr = np.linalg.svd(Jr, full_matrices=False)

        # Manipulability
        man_full = manip_full(Jb)
        man_t = manip_full(Jt)
        man_r = manip_full(Jr)

        # Singular direction for smallest singular value
        # Full 6x7: task-space direction = U6[:, -1] (rank 6 generically)
        sdir_full = U6[:, -1]
        sdir_t = Ut[:, -1]   # translational direction
        sdir_r = Ur[:, -1]

        per_sample.append({
            't': float(it['t']),
            'step_idx': tr_step_idx[i],
            'phase': tr_phase[i],
            'swing_arm': tr_swing_arm[i],
            's6': s6,
            'st': st,
            'sr': sr,
            'sig_min_full': float(s6.min()),
            'sig_max_full': float(s6.max()),
            'cond_full': float(s6.max() / s6.min()) if s6.min() > 1e-12 else float('inf'),
            'sig_min_trans': float(st.min()),
            'sig_min_rot': float(sr.min()),
            'cond_trans': float(st.max() / st.min()) if st.min() > 1e-12 else float('inf'),
            'manip_full': man_full,
            'manip_trans': man_t,
            'manip_rot': man_r,
            'sdir_full': sdir_full,
            'sdir_trans': sdir_t,
        })

    log_msg(fp, f"\nComputed Jacobian/SVD at {len(per_sample)} physics_trace samples.")

    # Only SS samples
    per_ss = [s for s in per_sample if s['phase'] == 'SS']
    by_step = {0: [], 1: [], 2: []}
    for s in per_ss:
        by_step[s['step_idx']].append(s)

    # ── Section B — per-step summary table ──────────────────────
    log_msg(fp, "\n" + "=" * 76)
    log_msg(fp, "SECTION B  —  Per-step Jacobian conditioning summary (arm B EE, 6x7)")
    log_msg(fp, "=" * 76)
    log_msg(fp, f"\n{'Step':>5} {'N':>4} {'min σ full':>12} {'min σ trans':>13} {'min σ rot':>11} {'max cond full':>14} {'min manip_trans':>17} {'max manip_trans':>17}")
    for s_idx, samples in by_step.items():
        if not samples: continue
        sig_full_min = min(s['sig_min_full'] for s in samples)
        sig_trans_min = min(s['sig_min_trans'] for s in samples)
        sig_rot_min = min(s['sig_min_rot'] for s in samples)
        cond_max = max(s['cond_full'] for s in samples)
        manip_trans_min = min(s['manip_trans'] for s in samples)
        manip_trans_max = max(s['manip_trans'] for s in samples)
        log_msg(fp, f"{s_idx:>5} {len(samples):>4} {sig_full_min:>12.4e} {sig_trans_min:>13.4e} {sig_rot_min:>11.4e} {cond_max:>14.4e} {manip_trans_min:>17.4e} {manip_trans_max:>17.4e}")

    # ── Section C — Step 2 time series, coarse + zoom ───────────
    log_msg(fp, "\n" + "=" * 76)
    log_msg(fp, "SECTION C  —  Step 2 SS conditioning over time")
    log_msg(fp, "=" * 76)
    log_msg(fp, "\nCoarse (~0.5 s spacing):")
    log_msg(fp, f"{'t(s)':>7} {'σ_min full':>12} {'σ_min trans':>13} {'σ_min rot':>11} {'cond full':>11} {'manip full':>12} {'manip trans':>13}")
    s2 = by_step[2]
    # sample ~ every 0.5 s from this sparser grid
    t_prev = -1.0
    coarse_samples = []
    for s in s2:
        if s['t'] - t_prev >= 0.4:
            coarse_samples.append(s)
            t_prev = s['t']
    for s in coarse_samples:
        log_msg(fp,
            f"{s['t']:7.2f} {s['sig_min_full']:12.4e} {s['sig_min_trans']:13.4e} "
            f"{s['sig_min_rot']:11.4e} {s['cond_full']:11.3f} {s['manip_full']:12.4e} {s['manip_trans']:13.4e}")

    # Zoom: physics_trace is ~0.2 s spacing; we show every sample in zoom windows
    log_msg(fp, "\nZoom around flailing events (every physics_trace sample):")
    for lo, hi, label in [(17.4, 19.6, 'flailing #1'), (21.4, 22.6, 'flailing #2'), (26.4, 27.6, 'flailing #3')]:
        log_msg(fp, f"\n  [{label}] t in [{lo:.1f}, {hi:.1f}] s:")
        log_msg(fp, f"  {'t(s)':>7} {'σ_min full':>12} {'σ_min trans':>13} {'σ_min rot':>11} {'cond full':>11} {'manip trans':>13}")
        for s in s2:
            if lo <= s['t'] <= hi:
                log_msg(fp,
                    f"  {s['t']:7.2f} {s['sig_min_full']:12.4e} {s['sig_min_trans']:13.4e} "
                    f"{s['sig_min_rot']:11.4e} {s['cond_full']:11.3f} {s['manip_trans']:13.4e}")

    # ── Section D — Reference velocity vs singular direction ────
    log_msg(fp, "\n" + "=" * 76)
    log_msg(fp, "SECTION D  —  Reference velocity direction vs translational singular direction")
    log_msg(fp, "=" * 76)
    log_msg(fp, "\nPer-tr-sample step 2 SS: |v_ref|, direction, sdir_trans, cos=|dot|")
    log_msg(fp, "  sdir_trans is the left singular vector of J_trans (3x7) for the smallest singular value")
    log_msg(fp, "  v_ref = (p_ee_ref[k+1] - p_ee_ref[k]) / Δt, finite-diff of logged reference")
    log_msg(fp, "  cos(θ) ≈ 1  ⇒  reference commands motion in a near-singular direction")
    log_msg(fp, "")
    log_msg(fp, f"  {'t(s)':>7} {'|v_ref|':>10} {'v_ref (unit)':<28} {'sdir_trans (unit)':<28} {'|cos|':>7}")

    # Helper: look up log-tick finite-diff v_ref at a tr_t
    def v_ref_at(tt):
        k = int(np.argmin(np.abs(t_sim - tt)))
        if k + 1 >= len(t_sim): return np.zeros(3)
        return (p_ee_ref[k+1] - p_ee_ref[k]) / (t_sim[k+1] - t_sim[k])

    aligned_samples = []
    for s in s2:
        vr = v_ref_at(s['t'])
        vr_n = np.linalg.norm(vr)
        uv = unit(vr)
        us = unit(s['sdir_trans'])
        cos_abs = abs(float(np.dot(uv, us))) if vr_n > 1e-6 else 0.0
        if vr_n > 1e-6:
            aligned_samples.append((s['t'], vr_n, uv, us, cos_abs))
        s['v_ref_norm'] = vr_n
        s['cos_align'] = cos_abs

    # Print coarse list
    t_prev = -1.0
    for s in s2:
        if s['t'] - t_prev >= 0.4:
            t_prev = s['t']
            vr_n = s['v_ref_norm']; ca = s['cos_align']
            vr = v_ref_at(s['t']); uv = unit(vr); us = unit(s['sdir_trans'])
            uv_s = f"({uv[0]:+.3f},{uv[1]:+.3f},{uv[2]:+.3f})"
            us_s = f"({us[0]:+.3f},{us[1]:+.3f},{us[2]:+.3f})"
            flag = " *" if ca > 0.5 else ""
            log_msg(fp, f"  {s['t']:7.2f} {vr_n:10.4f} {uv_s:<28} {us_s:<28} {ca:7.3f}{flag}")

    # Count singularity-aligned samples (cos > 0.5 and v_ref meaningful)
    ss_aligned = sum(1 for s in s2 if s['cos_align'] > 0.5 and s.get('v_ref_norm', 0) > 1e-3)
    log_msg(fp, f"\nStep 2 SS: {ss_aligned} of {len(s2)} physics_trace samples have |cos|>0.5 and |v_ref|>1e-3 m/s")

    # Same stats for steps 0 and 1
    for s_idx in [0, 1]:
        s_list = by_step[s_idx]
        for s in s_list:
            vr = v_ref_at(s['t'])
            vr_n = np.linalg.norm(vr)
            us = unit(s['sdir_trans'])
            s['v_ref_norm'] = vr_n
            s['cos_align'] = abs(float(np.dot(unit(vr), us))) if vr_n > 1e-6 else 0.0
        aligned = sum(1 for s in s_list if s['cos_align'] > 0.5 and s.get('v_ref_norm', 0) > 1e-3)
        log_msg(fp, f"Step {s_idx} SS: {aligned} of {len(s_list)} samples have |cos|>0.5 and |v_ref|>1e-3 m/s")

    # Summary table
    log_msg(fp, "\n" + "=" * 76)
    log_msg(fp, "SECTION E  —  High-level summary (min σ_trans, max cond, min manip_trans, max |cos|)")
    log_msg(fp, "=" * 76)
    log_msg(fp, f"{'Step':>5}  {'min σ_full':>12} {'min σ_trans':>13} {'max cond_full':>14} {'min manip_trans':>17} {'max |cos(v_ref,s_min)|':>24}")
    for s_idx, s_list in by_step.items():
        if not s_list: continue
        sig_t_min = min(s['sig_min_trans'] for s in s_list)
        sig_f_min = min(s['sig_min_full'] for s in s_list)
        cond_max = max(s['cond_full'] for s in s_list)
        mant_min = min(s['manip_trans'] for s in s_list)
        cos_max = max(s.get('cos_align', 0.0) for s in s_list
                      if s.get('v_ref_norm', 0) > 1e-3)
        log_msg(fp, f"{s_idx:>5}  {sig_f_min:>12.4e} {sig_t_min:>13.4e} {cond_max:>14.3f} {mant_min:>17.4e} {cos_max:>24.3f}")

    elapsed = time.perf_counter() - t_start
    log_msg(fp, f"\nExecution time: {elapsed:.2f} s")
    fp.close()
    print(f"\nOutput: {out_txt}")


if __name__ == '__main__':
    main()
