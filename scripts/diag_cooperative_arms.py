"""Canonical WBC traversal runner (historically the cooperative-arms probe).

This is the runner that produces the frozen canonical artifacts. The SS
controller it exercises is the Phase-2.1 **two-task** stack (T-MOM linear +
6-D torso-pose + swing-EE + posture, all weighted, no null-space projection);
DS runs the centroidal-DS tasks plus the passivity inequality.

The cooperative-arms split this script was originally written to test has been
superseded and its implementation removed (CLEANUP-6/7), so
`cooperative_arms_mode`, `ss_alpha_torso_ang/lin`, the Option D tube and T-MOM
v1 no longer exist. The `--legacy` and `--alpha_torso_lin` flags survive only
as output-dir / hero-render discriminators — they no longer change control.

Config carried over:
  nmpc_f_max=300, preplanner_f_max=25 (= F-SAT clamp source),
  mapping_bypass_in_ss=False, q_current, F-RATE+F-SAT enabled,
  stop_on_failed_step=True.

Outputs:
    results/diag_cooperative_arms/
        sim_log.json, step_log.json
        step_metrics.txt, nmpc_per_step.txt
        sat_stats.txt, comparison.txt

Run:
    MUJOCO_GL=disabled PYTHONPATH=. python3 \\
        scripts/diag_cooperative_arms.py
    # Regression guard (legacy strict M2 stack):
    MUJOCO_GL=disabled PYTHONPATH=. python3 \\
        scripts/diag_cooperative_arms.py --legacy
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')

import scripts.run_m7_single_step as r_single  # noqa: E402


MJCF = os.path.join(_root, 'models', 'VISPA_crawling_rwa3.xml')

ROBOT_JOINT_RE = re.compile(
    r'(<default class="robot_joint">\s*\n\s*<joint damping=")[^"]+'
    r'(" armature=")[^"]+(")')

ANCHOR_SITE_RE = re.compile(
    r'(<site name="anchor_(\d+)([ab])" class="anchor" pos=")'
    r'([^"]+)(")')

# Structure body inertial (mass + fullinertia). Used to set the
# robot/structure mass ratio programmatically (rule 4: no copy-paste
# MJCF). Canonical structure mass 7110 kg => mass_ratio ~0.01 (1%).
STRUCT_INERTIAL_RE = re.compile(
    r'(<inertial pos="0 0 0" mass=")([0-9.]+)'
    r'("\s*fullinertia=")([^"]+)(")')


def _mjcf_md5(path: str) -> str:
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def _mutate_mjcf(damping: float, armature: float,
                 anchor_dx: float | None = None,
                 mass_ratio: float | None = None):
    with open(MJCF, 'r') as f:
        text = f.read()
    new, n = ROBOT_JOINT_RE.subn(
        rf'\g<1>{damping}\g<2>{armature}\g<3>', text, count=1)
    if n != 1:
        raise RuntimeError('robot_joint default not found')
    # Structure mass/inertia: scale by 0.01/mass_ratio relative to the
    # canonical (1%) values. mass_ratio=0.14 => scale 1/14 => structure
    # mass 7110->507.857 kg + inertia x1/14 (the validated T12 mass,
    # scripts/run_m7_v22_14pct_with_swing_hold.py). hw/tau_w limits are
    # NOT scaled (the stress test: same AOCS box at higher disturbance).
    if mass_ratio is not None and abs(mass_ratio - 0.01) > 1e-9:
        scale = 0.01 / mass_ratio

        def _repl_inertial(m):
            mass_new = float(m.group(2)) * scale
            inertia_new = ' '.join(
                f'{float(v) * scale:.6g}' for v in m.group(4).split())
            return f'{m.group(1)}{mass_new:.6g}{m.group(3)}{inertia_new}{m.group(5)}'
        new, ni = STRUCT_INERTIAL_RE.subn(_repl_inertial, new, count=1)
        if ni != 1:
            raise RuntimeError('structure inertial block not found')
    if anchor_dx is not None and abs(anchor_dx - 0.8) > 1e-9:
        def _repl(m):
            idx = int(m.group(2))
            arm = m.group(3)
            x_new = (idx - 3.5) * anchor_dx
            y = 0.3 if arm == 'a' else -0.3
            return f'{m.group(1)}{x_new:.3f} {y} 0.025{m.group(5)}'
        new, n2 = ANCHOR_SITE_RE.subn(_repl, new)
        if n2 != 12:
            raise RuntimeError(f'expected 12 anchor sites, matched {n2}')
    with open(MJCF, 'w') as f:
        f.write(new)


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"unserializable {type(obj)}")


def _parse_label(label: str):
    if not label:
        return -1, '?'
    parts = label.split('/')
    try:
        step = int(parts[0].replace('step', ''))
    except Exception:
        step = -1
    phase = parts[1] if len(parts) >= 2 else '?'
    return step, phase


def _step_metrics_table(log, diag_log, n_steps, out_path):
    t = np.array(log.t)
    phase = np.array(log.phase, dtype=object)
    sidx = np.array(log.step_idx)
    p_torso = np.array(log.p_torso) if log.p_torso else None
    e_ee_raw = log.e_ee_pos
    e_ee_pos = np.array(e_ee_raw) if e_ee_raw else None
    lqp = np.array(log.lambda_qp) if log.lambda_qp else None

    dock_by_step = {ev.get('step', None): ev for ev in log.dock_events}
    abort_by_step = {ab['step_idx']: ab for ab in log.aborted_steps}

    by_step = defaultdict(list)
    for e in diag_log:
        by_step[e.get('step_idx', -1)].append(e)

    lines = []
    header = (f"{'Step':>4s} {'Outcome':>22s} {'d [mm]':>8s} {'ori [°]':>8s} "
              f"{'recoil [mm]':>12s} {'EE err [mm]':>12s} "
              f"{'|f|max [N]':>11s} {'rb_osc_max':>11s}")
    lines += [header, '-' * len(header)]
    for s in range(n_steps):
        m = (sidx == s) & (phase == 'SS')
        if not m.any():
            lines.append(f"{s:>4d} {'NO_DATA':>22s}")
            continue
        recoil_mm = None
        if p_torso is not None and p_torso.size:
            p0 = p_torso[m][0]
            recoil_mm = float(np.linalg.norm(
                p_torso[m] - p0, axis=1).max() * 1000.0)
        peak_ee_mm = None
        if e_ee_pos is not None and e_ee_pos.size:
            ee = e_ee_pos[m]
            peak_ee_mm = float(
                (np.linalg.norm(ee, axis=1) if ee.ndim == 2
                 else np.abs(ee)).max() * 1000.0)
        fmax_n = None
        if lqp is not None and lqp.size:
            l = lqp[m]
            f1n = np.linalg.norm(l[:, 0:3], axis=1)
            f2n = np.linalg.norm(l[:, 6:9], axis=1)
            fmax_n = float(max(f1n.max(), f2n.max()))
        # outcome
        if s in abort_by_step:
            ab = abort_by_step[s]
            outcome = ab['reason'].upper()
            d_mm = ab.get('d_mm', None)
            ori_deg = ab.get('ori_deg', None)
        elif s in dock_by_step:
            ev = dock_by_step[s]
            outcome = 'DOCK'
            d_mm = ev.get('d_mm', None)
            ori_deg = ev.get('ori_deg', None)
        else:
            outcome = 'UNKNOWN'
            d_mm = ori_deg = None
        # rb_osc
        entries = by_step.get(s, [])
        if len(entries) >= 2:
            r_b = np.array([e['r_b_ref'] for e in entries])
            step_diffs = np.linalg.norm(np.diff(r_b, axis=0), axis=1)
            osc_max_mm = float(step_diffs.max() * 1000.0)
        else:
            osc_max_mm = None
        d = f"{d_mm:.2f}" if d_mm is not None else '--'
        o = f"{ori_deg:.2f}" if ori_deg is not None else '--'
        r = f"{recoil_mm:.1f}" if recoil_mm is not None else '--'
        ee = f"{peak_ee_mm:.1f}" if peak_ee_mm is not None else '--'
        fm = f"{fmax_n:.1f}" if fmax_n is not None else '--'
        om = f"{osc_max_mm:.3f}" if osc_max_mm is not None else '--'
        lines.append(f"{s:>4d} {outcome:>22s} {d:>8s} {o:>8s} "
                     f"{r:>12s} {ee:>12s} {fm:>11s} {om:>11s}")
    text = '\n'.join(lines) + '\n'
    with open(out_path, 'w') as f:
        f.write(text)
    return text


def _nmpc_table(step_log, out_path):
    groups = defaultdict(list)
    for e in step_log:
        groups[_parse_label(e.get('label', ''))].append(e)
    lines = []
    header = (f"{'Step':>4s} {'Phase':>5s} {'N':>4s} "
              f"{'iter_min':>8s} {'iter_max':>8s} {'iter_p95':>8s} "
              f"{'t_max':>7s} {'t_mean':>7s} {'fail':>4s}  status_dist")
    lines += [header, '-' * len(header)]
    for key in sorted(groups.keys()):
        step, ph = key
        entries = groups[key]
        iters = np.array([e['iter'] for e in entries])
        times = np.array([e['time_ms'] for e in entries])
        failed = sum(1 for e in entries
                     if 'Succeeded' not in e['status']
                     and 'Acceptable' not in e['status'])
        statuses = Counter(e['status'] for e in entries)
        sd = ', '.join(f"{s}:{c}" for s, c in statuses.most_common())
        lines.append(
            f"{step:>4d} {ph:>5s} {len(entries):>4d} "
            f"{iters.min():>8d} {iters.max():>8d} "
            f"{int(np.percentile(iters, 95)):>8d} "
            f"{times.max():>7.1f} {times.mean():>7.1f} {failed:>4d}  {sd}")
    text = '\n'.join(lines) + '\n'
    with open(out_path, 'w') as f:
        f.write(text)
    return text


def main(legacy: bool, alpha_torso_lin: float, anchor_dx: float = 0.8,
         mass_ratio: float = 0.01, aocs_mode: str = 'legacy_corrected',
         settle_seconds: float = 20.0, K_theta: float = 1.0,
         K_omega: float = 50.0, tau_w_max: float = 2.5,
         scenario: str = None, baseline_ds_rework: bool = False,
         out_dir_override: str = None,
         ss_alpha_mom: float = 5000.0,
         n_steps: int = 5, ss_two_task: bool = False,
         alpha_torso_pose: float = 5000.0,
         ss_alpha_ee: float = 3e3, ss_alpha_posture: float = 2e1,
         ss_alpha_wrench: float = 1e-2,
         ss_kp_torso: float = 6.0, ss_kd_torso: float = 5.0,
         dock_twist_max: float = None, dock_gate_linear: bool = False,
         weld_radius: float = None,
         ds_mobile_com_magnitude: float = None, dt_ds: float = None,
         dock_hold_passivity_on: bool = False, passivity_W_budget: float = None,
         log_dock_work: bool = False,
         ds_passivity_beta: float = None, qp_envelope_exact: bool = False,
         aocs_active_in_interstep: bool = True,
         interstep_hw_refresh: bool = True,
         interstep_settle_alpha_wrench: float = 0.0,
         interstep_settle_epsilon_v: float = 0.0,
         preplanner_tstep_standoff_gain: float = 0.0,
         preplanner_tstep_standoff_knee: float = 1e9,
         preplanner_tstep_scale_step: int = -1,
         preplanner_tstep_scale_factor: float = 1.0):
    cfg = r_single._make_m7_config()
    cfg.gait_anchor_dx = anchor_dx
    # Sweet-spot config carry-over (these are also already the defaults
    # via _make_m7_config + SimConfig at this point).
    cfg.mapping_bypass_in_ss = False
    cfg.t_ss_margin = 5.0
    cfg.preplanner_f_max = 25.0  # F-SAT clamp source
    # Constant CoM-z standoff (crawl height). Initial config is also
    # re-solved at this standoff (setup) so the z-reference is flat from
    # t=0 — no startup z transient to compete with EE tracking.
    cfg.use_com_z_standoff = True
    cfg.com_z_standoff = -0.35
    # Dock gate: 5mm/5deg (1mm/1deg is infeasible — the EE only transiently
    # approaches to ~1.15mm/~4.8mm then drifts; not a stable sub-mm dock).
    # Velocity criterion kept as a clean-dock guard (no-op at the current
    # 2-4mm/s, but blocks welding during a fast transient).
    cfg.dock_vel_max = 0.01
    # Post-traversal settle: how long to hold both arms welded after the
    # last dock, with the AOCS active. Drives the post-settle drift
    # measurement (residual ω_s, h_w, accumulated attitude). Default 20s
    # matches the prior implicit behavior; bump to ~120s for asymptotic
    # measurement.
    cfg.t_settle_final = float(settle_seconds)
    # Rework knob.
    # AOCS K_theta gain override (active only for legacy_pid_* modes).
    cfg.aocs_K_theta = float(K_theta)
    # AOCS K_omega (ω_s damping) override (legacy_pd_* / legacy_pid_*).
    cfg.aocs_K_omega = float(K_omega)
    # Wheel torque limit override — set NMPC and AOCS together so the
    # decentralized contract stays consistent (NMPC |Ḣ_s|≤τ_w_max,
    # AOCS clips its commanded torque at the same value).
    cfg.tau_w_max = float(tau_w_max)
    cfg.aocs_tau_w_max = float(tau_w_max)
    # STEP5-MARGIN: standoff-keyed dock-margin safety factor on T_step (default off).
    cfg.preplanner_tstep_standoff_gain = float(preplanner_tstep_standoff_gain)
    cfg.preplanner_tstep_standoff_knee = float(preplanner_tstep_standoff_knee)
    cfg.preplanner_tstep_scale_step = int(preplanner_tstep_scale_step)
    cfg.preplanner_tstep_scale_factor = float(preplanner_tstep_scale_factor)
    # AOCS mode override (default 'legacy_corrected' = canonical).
    # legacy_pd_numerical / legacy_pd_model add a PD regulator on ω_s
    # on top of the legacy_corrected feedforward + desat. The two
    # differ only in how ω̇_s is sourced (finite-diff vs Newton-Euler).
    if aocs_mode != 'legacy_corrected':
        cfg.aocs_mode = aocs_mode
        cfg.aocs_use_legacy_corrected = False
        cfg.aocs_use_H_estimator = (aocs_mode == 'H_est')
    # Per-contact wrench feedforward in DS for legacy_pid_* modes — the
    # FD-on-L_com Ḣ-FF misses the welded-loop internal-stress couple.
    # See AOCS code path; AOCS uses tau_struct_ff = −Σ(r_Ci × f_i + τ_i)
    # from λ_qp when this is on and phase=='DS'.
    cfg.aocs_use_wrench_ff_in_ds = True
    # J2 step 4a: re-activate the AOCS inside the inter-step DS loop
    # (default on — restores the free-floating invariant). --no-aocs-in-
    # interstep flips it off for the byte-identical A/B baseline.
    cfg.aocs_active_in_interstep = bool(aocs_active_in_interstep)
    # J2 c_curr: per-tick refresh of the inter-step QP hw_current parameter
    # (default on — _step-consistent). --no-interstep-hw-refresh flips it off
    # for the byte-identical A/B (entry-frozen hw_current).
    cfg.interstep_hw_refresh = bool(interstep_hw_refresh)
    # J2 chatter fix: settle-only α_wrench in the inter-step DS QP (0 = off).
    cfg.interstep_settle_alpha_wrench = float(interstep_settle_alpha_wrench)
    # J2 close (POINT A): dock-tolerance-derived inter-step settle exit ε_v
    # (0 = off = byte-identical 1 mm/s).
    cfg.interstep_settle_epsilon_v = float(interstep_settle_epsilon_v)
    # Stage 3 — trailing-DS torso reference uses the actual welded
    # state instead of the both-tools-at-anchors IK that gave the
    # persistent ~3.86° torso ori error.
    cfg.ds_torso_ref_from_state = True
    # DS internal-stress regularization on λ (welded grasp matrix).
    # In DS the QP's 12-D contact-wrench has a 6-D internal-stress
    # null space — combinations producing zero net wrench on the robot
    # but a couple on the structure. Penalize the projection onto that
    # null space at P4 to keep λ minimum-net-wrench. SS path unchanged
    # (single contact ⇒ no internal-stress subspace).
    cfg.ss_alpha_lambda_int = 1.0
    # DS centroidal-control mode: replace the joint-vel-damping cost
    # with CoM + torso-ori tracking at P1, posture at P3, passivity
    # inequality for energy dissipation. Closes the 8-DOF welded
    # redundancy properly during trailing-DS settle.
    cfg.ds_centroidal_mode = True

    # --- DS-rework baseline override ----------------------------------
    # When --baseline_ds_rework is set, flip every DS-rework feature
    # OFF to reproduce the pre-rework behaviour. Used to generate the
    # comparison baseline for the DS_REWORK_CENTROIDAL_2026-06 memo.
    if baseline_ds_rework:
        cfg.aocs_use_wrench_ff_in_ds = False
        cfg.ds_torso_ref_from_state = False
        cfg.ss_alpha_lambda_int = 0.0
        cfg.ds_centroidal_mode = False
        # Pre-rework: the inter-step loop hardcoded the wheels to 0.
        cfg.aocs_active_in_interstep = False
        # Pre-rework: hw_current was frozen at loop entry (no per-tick refresh).
        cfg.interstep_hw_refresh = False
    # alpha_torso_ang stays at default 500 (set by _make_m7_config).
    # 5 evenly-spaced snapshots per SS for the offline renderer.
    # FRAMES_PER_STEP=0 disables capture entirely (e.g. for tests).
    cfg.frames_per_step = int(os.environ.get('FRAMES_PER_STEP', '5'))

    # Startup-IK regularizers (canonical defaults applied at runner
    # level so other scripts/tests keep the legacy free-rotation IK):
    #  - level torso pitch/roll to the T15-FK rail surface normal
    #    (structure +z), yaw free
    #  - whole-system null-space posture biasing the 14 arm joints
    #    (2×7-DOF) toward a mirror-symmetric template
    #
    # ik_q_nominal is the mirror-symmetric template from the
    # start-pair (2,2) leveled-only solution: solve leveled IK once,
    # take the *less-contorted* arm's joint vector and mirror it onto
    # the other arm via the y-symmetry sign pattern
    # S=[-1,-1,-1,-1,-1,-1,+1]. Reproduced by
    # scripts/diag_qnominal_sweep.py. Sweep result (vs leveled-only
    # ‖q_arm‖=4.05 / max|q|=147° / σ_min·prod=2.43e-2):
    #   zeros  → ‖q‖3.07 max89° prod1.6e-3  (de-contorts but one arm
    #            driven near-singular — rejected)
    #   sym    → ‖q‖3.67 max113° prod2.24e-2 (de-contorts AND keeps
    #            both arms well-conditioned — selected)
    # w_posture is a secondary-task gain; the constrained null-space
    # posture converges to the same fixed point for any w>0, 0.2 is
    # fast enough for the Nelder-Mead inner solves (max_iter=500)
    # while err stays ~1e-6 ≪ the 1e-3 IK-accept threshold.
    cfg.ik_level_axis = np.array([0.0, 0.0, 1.0])
    cfg.ik_q_nominal = np.array([
        0.297781,  1.526727,  0.842257,  1.147432,
        0.751390,  0.251893,  1.332349,
        -0.297781, -1.526727, -0.842257, -1.147432,
        -0.751390, -0.251893,  1.332349])
    cfg.ik_w_posture = 0.2

    if scenario is not None:
        # Route to a per-scenario subdir, using the file stem as the
        # tag. E.g. scenarios/canonical_3step.seq → diag_cooperative_arms_3step.
        _stem = os.path.splitext(os.path.basename(scenario))[0]
        _stem = _stem.replace('canonical_', '')
        if baseline_ds_rework:
            _stem += '_baseline'
        out_dir = os.path.join(_root, 'results',
                               f'diag_cooperative_arms_{_stem}')
    elif legacy:
        out_dir = os.path.join(_root, 'results',
                               'diag_cooperative_arms_legacy')
    elif abs(mass_ratio - 0.01) > 1e-9:
        out_dir = os.path.join(_root, 'results',
                               f'diag_cooperative_arms_{int(round(mass_ratio*100))}pct')
    elif aocs_mode != 'legacy_corrected':
        # Non-default AOCS mode → separate dir for A/B.
        # Add _Kt{val}/_Kw{val}/_Tw{val} suffixes when gains/limits differ from defaults.
        suffix = ''
        if 'pid' in aocs_mode and abs(K_theta - 1.0) > 1e-9:
            suffix += f'_Kt{K_theta:g}'
        if abs(K_omega - 50.0) > 1e-9:
            suffix += f'_Kw{K_omega:g}'
        if abs(tau_w_max - 2.5) > 1e-9:  # non-canonical cap (frozen 2.5)
            suffix += f'_Tw{tau_w_max:g}'
        out_dir = os.path.join(_root, 'results',
                               f'diag_cooperative_arms_{aocs_mode}{suffix}')
    elif abs(alpha_torso_lin - 500.0) > 1e-6:
        out_dir = os.path.join(_root, 'results',
                               'diag_cooperative_arms',
                               f'alpha_lin_{int(alpha_torso_lin)}')
    elif abs(anchor_dx - 0.8) > 1e-9:
        out_dir = os.path.join(_root, 'results',
                               'diag_cooperative_arms',
                               f'dx_{anchor_dx:.2f}')
    else:
        out_dir = os.path.join(_root, 'results', 'diag_cooperative_arms')

    # ── SS centroidal-momentum task (memo SS_CENTROIDAL_MOMENTUM_TASK_2026-06) ──
    # Default OFF reproduces the canonical torso-linear P2 stack (bit-identical).
    cfg.ss_alpha_mom = float(ss_alpha_mom)
    # Phase-2.1: log τ_w + h_w at the 100 Hz QP rate during SS (harmless —
    # adds separate buffers; the 10 Hz log + postproc CSV are unaffected).
    cfg.log_hifreq_ss = True
    # Phase-2.1 reformulation: two-task fully-weighted SS stack.
    cfg.ss_two_task_mode = bool(ss_two_task)
    cfg.alpha_torso_pose = float(alpha_torso_pose)
    # Expose the remaining SS-stack weights for tuning. Defaults equal the
    # config values (ss_alpha_ee=3e3, ss_alpha_posture=2e1, ss_alpha_wrench=1e-2),
    # so a run with no weight flags is behaviour-identical to before.
    cfg.ss_alpha_ee = float(ss_alpha_ee)
    cfg.ss_alpha_posture = float(ss_alpha_posture)
    cfg.ss_alpha_wrench = float(ss_alpha_wrench)
    # SS torso-pose gains (task AGGRESSIVENESS, distinct from weight PRIORITY):
    # a_t_des = a_ff + Kp·e6 + Kd·(v_ref-v_act). Lowering Kp softens the
    # commanded torso accel ⇒ smoother (less wheel-saturating) tracking at a
    # given weight. Defaults equal config (6.0/5.0) ⇒ no-flag = prior behaviour.
    cfg.ss_Kp_torso = float(ss_kp_torso)
    cfg.ss_Kd_torso = float(ss_kd_torso)
    # Fix C (J2 #1): dock-gate parameters. dock_twist_max sets ε_twist for
    # the 6-D weld-relative twist gate ‖Jc·v⁻‖<ε; dock_gate_linear flips
    # back to the legacy linear _gripper_speed gate for A/B. Defaults
    # (None / False) ⇒ the config values (6-D gate at 0.05) are used.
    if dock_twist_max is not None:
        cfg.dock_twist_max = float(dock_twist_max)
    if dock_gate_linear:
        cfg.dock_use_6d_twist = False
    # Fix C ε_pos: dock position tolerance (the gap-couple knob). Default
    # None ⇒ config weld_radius (0.005 m).
    if weld_radius is not None:
        cfg.weld_radius = float(weld_radius)
    # α (J2 #2): CoM-mobile DS. dt_ds lengthens the DS so the DWELL fires;
    # ds_mobile_com_magnitude translates the CoM toward the next anchor.
    if dt_ds is not None:
        cfg.dt_ds = float(dt_ds)
    if ds_mobile_com_magnitude is not None:
        cfg.ds_mobile_com_magnitude = float(ds_mobile_com_magnitude)
    # Dock-floor passivity audit knobs.
    cfg.dock_hold_passivity_on = bool(dock_hold_passivity_on)
    cfg.log_dock_work = bool(log_dock_work)
    if passivity_W_budget is not None:
        cfg.passivity_W_budget = float(passivity_W_budget)
    # Piste A (J2 #3): envelope-coupled budget (β) + exact Ḣ_s box.
    if ds_passivity_beta is not None:
        cfg.ds_passivity_beta = float(ds_passivity_beta)
    cfg.qp_envelope_exact = bool(qp_envelope_exact)
    # Output-dir override (memo §7: new runs → new dirs; prior committed
    # baselines stay read-only). Accepts a name under results/ or an abs path.
    if out_dir_override is not None:
        out_dir = (out_dir_override if os.path.isabs(out_dir_override)
                   else os.path.join(_root, 'results', out_dir_override))

    from crawlbot.simulation.sim_loop import SimulationLoop
    URDF = os.path.join(_root, 'models', 'VISPA_crawling_fixed.urdf')

    sim = SimulationLoop(mjcf_path=MJCF, urdf_path=URDF, config=cfg)
    if scenario is not None:
        sim.setup(sequence_path=scenario)
    else:
        sim.setup(n_steps=n_steps, start_a=2, start_b=2)
    sim._debug_l_com_ref_trace_limit = 5
    sim._debug_physics_trace_limit = 400
    sim._debug_physics_sample_every = 2
    sim.nmpc._nmpc.diag_verbose = False
    sim.nmpc._nmpc.step_log.clear()
    sim.qp_ss.hw_slack_log.clear()
    sim._step2_diag_enabled = True
    sim._step2_diag_log.clear()

    sim._step_q_end_log = []
    _orig_setup_torso = sim._setup_torso_for_step
    def _wrapped_setup_torso(*a, **kw):
        result = _orig_setup_torso(*a, **kw)
        sim._step_q_end_log.append({
            'step_idx': len(sim._step_q_end_log),
            'q_start': sim._step_q_start.tolist(),
            'q_end': sim._step_q_end.tolist(),
        })
        return result
    sim._setup_torso_for_step = _wrapped_setup_torso

    print('\n' + '=' * 70)
    print(f"  Cooperative-arms 5-step (cooperative={not legacy}, "
          f"alpha_torso_lin={alpha_torso_lin})")
    print(f"  Output: {out_dir}")
    print('=' * 70)
    log = sim.run(verbose=True)

    os.makedirs(out_dir, exist_ok=True)
    log.save(os.path.join(out_dir, 'sim_log.json'))
    # Augment sim_log with the GaitPlan so the renderer can derive
    # per-step swing targets / stance pairs for arbitrary scenarios
    # (not just the canonical 5-step). Cheap append to the JSON.
    try:
        _slp = os.path.join(out_dir, 'sim_log.json')
        with open(_slp) as _fh:
            _sl = __import__('json').load(_fh)
        _sl['gait_plan'] = {
            'start_a': int(sim.plan.phases[0].anchor_a_idx),
            'start_b': int(sim.plan.phases[0].anchor_b_idx),
            'phases': [
                {'phase': p.phase.name,
                 'anchor_a_idx': int(p.anchor_a_idx),
                 'anchor_b_idx': int(p.anchor_b_idx),
                 'swing_arm': p.swing_arm,
                 'swing_from_idx': int(p.swing_from_idx),
                 'swing_to_idx': int(p.swing_to_idx)}
                for p in sim.plan.phases],
        }
        with open(_slp, 'w') as _fh:
            __import__('json').dump(_sl, _fh)
    except Exception:
        pass
    with open(os.path.join(out_dir, 'step_log.json'), 'w') as f:
        json.dump(sim._step2_diag_log, f, indent=2)

    ik_trace_drop_so3 = [
        {k: v for k, v in e.items() if k not in ('R_start', 'R_goal')}
        for e in (getattr(sim, '_debug_ik_trace', None) or [])
    ]
    with open(os.path.join(out_dir, 'ik_trace.json'), 'w') as f:
        json.dump(ik_trace_drop_so3, f, indent=2, default=_json_default)

    with open(os.path.join(out_dir, 'step_q_end.json'), 'w') as f:
        json.dump(sim._step_q_end_log, f, indent=2, default=_json_default)

    text_metrics = _step_metrics_table(
        log, sim._step2_diag_log, len(sim.plan.phases) // 2,
        os.path.join(out_dir, 'step_metrics.txt'))
    print('\n=== Per-step outcomes ===\n' + text_metrics)

    step_log = list(sim.nmpc._nmpc.step_log)
    with open(os.path.join(out_dir, 'nmpc_step_log.json'), 'w') as f:
        json.dump(step_log, f, indent=2)
    _nmpc_table(step_log, os.path.join(out_dir, 'nmpc_per_step.txt'))

    tot = int(sim._sat_total_calls)
    clip = int(sim._sat_clipped_calls)
    ratio = (clip / tot * 100.0) if tot > 0 else 0.0
    sat_text = (
        f"F-SAT telemetry\n"
        f"  total saturator calls: {tot}\n"
        f"  cycles clipped:        {clip}  ({ratio:.2f}%)\n"
        f"  max clip magnitude:    {sim._sat_max_clip_mm:.3f} mm\n"
    )
    with open(os.path.join(out_dir, 'sat_stats.txt'), 'w') as f:
        f.write(sat_text)
    print('\n' + sat_text)

    # Stance-thrust correction telemetry
    corr_calls = int(getattr(sim.qp_ss, '_stance_thrust_corr_calls', 0))
    corr_max = float(
        getattr(sim.qp_ss, '_stance_thrust_corr_max_norm', 0.0))
    corr_text = (
        f"Stance-thrust correction telemetry\n"
        f"  total correction applications: {corr_calls}\n"
        f"  max ||Δλ_stance|| (6D wrench): {corr_max:.4f}\n"
    )
    with open(os.path.join(out_dir, 'stance_thrust.txt'), 'w') as f:
        f.write(corr_text)
    print(corr_text)

    # Structure drift summary
    sp = np.array(log.struct_pos) if log.struct_pos else None
    om = np.array(log.omega_s) if log.omega_s else None
    drift_max = float(np.linalg.norm(
        sp - sp[0], axis=1).max() * 1000.0) if sp is not None else float('nan')
    om_max = float(np.linalg.norm(
        om, axis=1).max() * 1e3) if om is not None else float('nan')
    drift_text = (
        f"Structure drift over run\n"
        f"  max ||struct_pos − struct_pos[0]||: {drift_max:6.2f} mm\n"
        f"  max ||omega_s||:                    {om_max:6.2f} mrad/s\n"
    )
    with open(os.path.join(out_dir, 'struct_drift.txt'), 'w') as f:
        f.write(drift_text)
    print(drift_text)

    # ── Auto-render isometric frame sequence ────────────────────
    # Only fires for the canonical default (cooperative + α_lin=500);
    # subprocess so mujoco.Renderer can pick up MUJOCO_GL=osmesa
    # without conflicting with the sim's MUJOCO_GL=disabled.
    canonical = ((not legacy) and abs(alpha_torso_lin - 500.0) < 1e-6
                 and abs(mass_ratio - 0.01) < 1e-9)
    if canonical and cfg.frames_per_step > 0:
        import subprocess
        env = os.environ.copy()
        env['MUJOCO_GL'] = 'osmesa'
        env['PYTHONPATH'] = _root
        print('[render] invoking scripts/render_traversal.py ...')
        r = subprocess.run(
            ['python3', os.path.join(_root, 'scripts',
                                     'render_traversal.py'),
             os.path.join(out_dir, 'sim_log.json')],
            env=env, capture_output=True, text=True)
        # Print only the final line + any error so the parent log
        # stays compact (the renderer logs every frame on its own).
        if r.returncode != 0:
            print('[render] FAILED rc=', r.returncode)
            print(r.stderr[-2000:])
        else:
            tail = r.stdout.strip().splitlines()[-3:]
            for line in tail:
                print('[render]', line)

    return out_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--legacy', action='store_true',
                        help='Naming/render discriminator only. Since CLEANUP-6 '
                             'removed cooperative_arms_mode this no longer changes '
                             'control; it still selects the output dir and gates '
                             'the canonical hero render.')
    parser.add_argument('--alpha_torso_lin', type=float, default=500.0,
                        help='Naming/render discriminator only (default 500). The '
                             'cooperative torso-linear P2 task it used to weight was '
                             'removed in CLEANUP-6; the value no longer reaches the QP.')
    parser.add_argument('--anchor_dx', type=float, default=0.8,
                        help='Anchor-grid pitch [m]; rewrites MJCF anchor sites '
                             'to x=(i-3.5)*dx for i=1..6 (default 0.8 = no-op)')
    parser.add_argument('--mass_ratio', type=float, default=0.01,
                        help='Robot/structure mass ratio; scales structure '
                             'mass+inertia by 0.01/ratio (default 0.01 = no-op; '
                             '0.14 = spec T12 14%%, structure 507.857 kg)')
    parser.add_argument('--aocs_mode', type=str, default='legacy_corrected',
                        choices=['legacy_corrected',
                                 'legacy_pd_numerical', 'legacy_pd_model',
                                 'legacy_pid_numerical', 'legacy_pid_model',
                                 'H_est'],
                        help='AOCS controller (default legacy_corrected). '
                             'legacy_pd_*  add a PD on ω_s. '
                             'legacy_pid_* further add an attitude P term '
                             'to recover the per-traversal net rotation. '
                             '_numerical / _model differ in ω̇_s source.')
    parser.add_argument('--settle_seconds', type=float, default=20.0,
                        help='Post-traversal settle duration [s] (cfg.t_settle_final). '
                             'Drives the post-settle drift measurement. '
                             'Default 20s; ~120s gives asymptotic ω_s/h_w decay '
                             'for PD modes (time constant I_s/K_ω ≈ 30s).')
    parser.add_argument('--K_theta', type=float, default=1.0,
                        help='Attitude P gain [Nm/rad] (legacy_pid_* only). '
                             'Default 1.0 — gentle (~60s recovery time '
                             'constant with K_ω=50). Bump to ~10 to recover '
                             '>97%% of per-traversal rotation in 120s settle.')
    parser.add_argument('--K_omega', type=float, default=50.0,
                        help='ω_s damping gain [Nm·s/rad]. Default 50 '
                             '(legacy, ζ ≈ 0.2 underdamped). '
                             'Pole-placement design (T_s=30s, ζ=0.7, '
                             'worst-axis I_s=1777) gives K_θ=36, K_ω=355.')
    parser.add_argument('--tau_w_max', type=float, default=2.5,
                        help='Wheel torque limit [Nm], per-axis. Default 2.5 '
                             '(frozen canonical; matches MJCF ctrlrange). '
                             'Sets BOTH cfg.tau_w_max (NMPC |Ḣ_s| budget) '
                             'AND cfg.aocs_tau_w_max (AOCS command clip) '
                             'so the two contracts stay in lock-step. '
                             'Use to stress-test wheel sizing.')
    parser.add_argument('--scenario', type=str, default=None,
                        help='Path to a locomotion-sequence .seq file '
                             '(see scenarios/). When set, overrides the '
                             'default 5-step plan with the file-defined '
                             'gait. Anchor names use MJCF site names '
                             '(e.g. anchor_4b).')
    parser.add_argument('--baseline_ds_rework', action='store_true',
                        help='Flip every DS-rework feature OFF '
                             '(wrench-FF / Stage3 ref / internal-stress '
                             '/ centroidal-DS). Reproduces pre-rework '
                             'behaviour for comparison plotting.')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Override the auto-derived output directory '
                             '(name under results/, or an absolute path). '
                             'Use for SS-centroidal-momentum runs so prior '
                             'committed baselines stay read-only (memo §7).')
    parser.add_argument('--ss-alpha-mom', type=float, default=5000.0,
                        help='T-MOM linear weight (default 5000 = working point).')
    parser.add_argument('--n-steps', type=int, default=5,
                        help='Traversal steps (default 5; use 1 for the '
                             'Phase-2.1 single-step swing).')
    parser.add_argument('--ss-two-task', action='store_true',
                        help='Phase-2.1 two-task fully-weighted SS stack '
                             '(T-MOM + 6D torso-pose + EE + posture, no δ).')
    parser.add_argument('--alpha-torso-pose', '--ss-alpha-torso-pose',
                        dest='alpha_torso_pose', type=float, default=5000.0,
                        help='6-D torso-pose task weight in two-task mode '
                             '(default 5000 = working point; '
                             'ss_alpha_mom:alpha_torso_pose is the knob). '
                             'Alias: --ss-alpha-torso-pose.')
    # Remaining SS-stack weights, exposed for tuning. Defaults = config values
    # ⇒ a run with no weight flags reproduces the prior behaviour exactly.
    parser.add_argument('--ss-alpha-ee', type=float, default=3e3,
                        help='SS swing-EE task weight (default 3000 = config).')
    parser.add_argument('--ss-alpha-posture', type=float, default=2e1,
                        help='SS posture task weight (default 20 = config).')
    parser.add_argument('--ss-alpha-wrench', type=float, default=1e-2,
                        help='SS wrench-reg weight (default 0.01 = config).')
    # SS torso-pose GAINS (aggressiveness, separate from the weight knob).
    parser.add_argument('--ss-kp-torso', type=float, default=6.0,
                        help='SS torso-pose proportional gain (default 6.0 = '
                             'config). Lower ⇒ softer commanded torso accel.')
    parser.add_argument('--ss-kd-torso', type=float, default=5.0,
                        help='SS torso-pose derivative gain (default 5.0 = config).')
    # Fix C (J2 #1): 6-D weld-relative dock gate ε_twist + A/B switch.
    parser.add_argument('--dock-twist-max', type=float, default=None,
                        help='Fix C ε_twist: max ‖Jc·v⁻‖ (6-D weld-relative '
                             'twist) for a clean dock (default = config 0.05).')
    parser.add_argument('--dock-gate-linear', action='store_true',
                        help='Fix C A/B: use the legacy LINEAR _gripper_speed '
                             'dock gate instead of the 6-D twist gate.')
    parser.add_argument('--weld-radius', type=float, default=None,
                        help='Fix C ε_pos: dock position tolerance [m] '
                             '(default = config 0.005). The gap-couple knob.')
    # α (J2 #2): CoM-mobile DS.
    parser.add_argument('--ds-mobile-com-magnitude', type=float, default=None,
                        help='α (J2 #2): translate the CoM this far [m] toward '
                             'the next anchor during the DWELL (0=hold).')
    parser.add_argument('--dt-ds', type=float, default=None,
                        help='α (J2 #2): DS phase duration [s] (default 0.5; '
                             '>~1.5 triggers the DWELL where moving-CoM runs).')
    # Dock-floor passivity audit.
    parser.add_argument('--dock-hold-passivity-on', action='store_true',
                        help='dock-floor: force passivity ON in the SS dock-close '
                             'hold window (default OFF = the SS-hold escape).')
    parser.add_argument('--passivity-w-budget', type=float, default=None,
                        help='dock-floor: constant passivity RHS budget '
                             '(dqᵀτ_q+2αT_kin ≤ W; default 0 = strict).')
    parser.add_argument('--log-dock-work', action='store_true',
                        help='dock-floor: trace per-SS-tick dqⱼᵀτ_q + d + passivity.')
    # Piste A (J2 #3).
    parser.add_argument('--ds-passivity-beta', type=float, default=None,
                        help='Piste A LOT A: β for the envelope-coupled passivity '
                             'budget W=β·α·max(0,τ_w_max−‖Ḣ_s(λ_ref)‖∞). 0=strict.')
    parser.add_argument('--qp-envelope-exact', action='store_true',
                        help='Piste A LOT B (FLAG 2): exact origin-referenced Ḣ_s '
                             'envelope box (vs the |M_λ·λ| proxy).')
    parser.add_argument('--no-aocs-in-interstep', action='store_false',
                        dest='aocs_active_in_interstep',
                        help='J2 step 4a A/B: disable the inter-step DS AOCS '
                             '(reverts to the legacy hardcoded-zero wheels). '
                             'Default: AOCS active in the inter-step loop.')
    parser.add_argument('--no-interstep-hw-refresh', action='store_false',
                        dest='interstep_hw_refresh',
                        help='J2 c_curr A/B: freeze the inter-step QP hw_current '
                             'at loop entry (revert the per-tick refresh). '
                             'Default: refresh hw_current per tick (live wheel '
                             'momentum), mirroring the _step QP and c_simple(k).')
    parser.add_argument('--interstep-settle-alpha-wrench', type=float, default=0.0,
                        help='J2 chatter fix: settle-only α_wrench in the inter-step '
                             'DS QP (0=off=byte-identical). >0 makes the λ-cost '
                             'strictly convex ⇒ unique min-norm wrench, breaking the '
                             'period-2 active-set chatter.')
    parser.add_argument('--preplanner-tstep-standoff-gain', type=float, default=0.0,
                        help='STEP5-MARGIN: dock-margin safety factor. If '
                             '|r_com_0|>knee, T_step *= (1 + gain). 0 = off (default).')
    parser.add_argument('--preplanner-tstep-standoff-knee', type=float, default=1e9,
                        help='STEP5-MARGIN: standoff knee above which the gain '
                             'applies. inf = off (default).')
    parser.add_argument('--preplanner-tstep-scale-step', type=int, default=-1,
                        help='TSTEP-DIAG: 0-based step index whose T_step to scale. -1=off.')
    parser.add_argument('--preplanner-tstep-scale-factor', type=float, default=1.0,
                        help='TSTEP-DIAG: multiply that step T_step by this factor.')
    parser.add_argument('--interstep-settle-epsilon-v', type=float, default=0.0,
                        help='J2 close (POINT A): dock-tolerance-derived inter-step '
                             'settle exit target ‖dq_full‖ [m/s] (0=off=byte-identical '
                             '1e-3). Larger ⇒ exits earlier (T<0.5·ε_v²·λ_min). '
                             'Derived 5e-3 = per-tick drift 1/10 of the 5 mm dock gate.')
    args = parser.parse_args()

    with open(MJCF, 'r') as f:
        original = f.read()
    pre_hash = _mjcf_md5(MJCF)
    tag = 'LEGACY' if args.legacy else 'COOP'
    print(f'[COOP-ARMS {tag}] MJCF md5 pre:  {pre_hash}')
    try:
        _mutate_mjcf(damping=0.0, armature=0.05, anchor_dx=args.anchor_dx,
                     mass_ratio=args.mass_ratio)
        main(args.legacy, args.alpha_torso_lin, args.anchor_dx,
             args.mass_ratio, args.aocs_mode, args.settle_seconds,
             args.K_theta, args.K_omega, args.tau_w_max,
             scenario=args.scenario,
             baseline_ds_rework=args.baseline_ds_rework,
             out_dir_override=args.out_dir,
             ss_alpha_mom=args.ss_alpha_mom,
             n_steps=args.n_steps,
             ss_two_task=args.ss_two_task,
             alpha_torso_pose=args.alpha_torso_pose,
             ss_alpha_ee=args.ss_alpha_ee,
             ss_alpha_posture=args.ss_alpha_posture,
             ss_alpha_wrench=args.ss_alpha_wrench,
             ss_kp_torso=args.ss_kp_torso,
             ss_kd_torso=args.ss_kd_torso,
             dock_twist_max=args.dock_twist_max,
             dock_gate_linear=args.dock_gate_linear,
             weld_radius=args.weld_radius,
             ds_mobile_com_magnitude=args.ds_mobile_com_magnitude,
             dt_ds=args.dt_ds,
             dock_hold_passivity_on=args.dock_hold_passivity_on,
             passivity_W_budget=args.passivity_w_budget,
             log_dock_work=args.log_dock_work,
             ds_passivity_beta=args.ds_passivity_beta,
             qp_envelope_exact=args.qp_envelope_exact,
             aocs_active_in_interstep=args.aocs_active_in_interstep,
             interstep_hw_refresh=args.interstep_hw_refresh,
             interstep_settle_alpha_wrench=args.interstep_settle_alpha_wrench,
             interstep_settle_epsilon_v=args.interstep_settle_epsilon_v,
             preplanner_tstep_standoff_gain=args.preplanner_tstep_standoff_gain,
             preplanner_tstep_standoff_knee=args.preplanner_tstep_standoff_knee,
             preplanner_tstep_scale_step=args.preplanner_tstep_scale_step,
             preplanner_tstep_scale_factor=args.preplanner_tstep_scale_factor)
    finally:
        with open(MJCF, 'w') as f:
            f.write(original)
        post_hash = _mjcf_md5(MJCF)
        print(f'[COOP-ARMS {tag}] MJCF md5 post: {post_hash}')
        assert post_hash == pre_hash, 'MJCF restoration failed'
