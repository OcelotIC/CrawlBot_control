"""Cooperative-arms WBC validation — 5-step fail-fast traversal.

Tests cfg.cooperative_arms_mode=True (default in _make_m7_config) against
the sweet-spot baseline (results/diag_nmpc_f300/, cooperative_arms_mode=False).

Sweet-spot config carried over:
  nmpc_f_max=300, preplanner_f_max=25 (= F-SAT clamp source),
  mapping_bypass_in_ss=False, q_current, F-RATE+F-SAT enabled,
  stop_on_failed_step=True.

Rework deltas vs sweet-spot:
  - cooperative_arms_mode=True (split torso 6D into P1 angular +
    P2 linear; EE 6D co-equal at P2; posture P3 projected against
    combined P1+P2 with rcond=1e-4)
  - ss_alpha_torso_ang=500, ss_alpha_torso_lin=500 (defaults)

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
         settle_seconds: float = 20.0, K_theta: float = 1.0):
    cfg = r_single._make_m7_config()
    cfg.gait_anchor_dx = anchor_dx
    # Sweet-spot config carry-over (these are also already the defaults
    # via _make_m7_config + SimConfig at this point).
    cfg.mapping_bypass_in_ss = False
    cfg.t_ss_margin = 5.0
    cfg.r_tube = 0.0
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
    cfg.cooperative_arms_mode = (not legacy)
    cfg.ss_alpha_torso_lin = float(alpha_torso_lin)
    # AOCS K_theta gain override (active only for legacy_pid_* modes).
    cfg.aocs_K_theta = float(K_theta)
    # AOCS mode override (default 'legacy_corrected' = canonical).
    # legacy_pd_numerical / legacy_pd_model add a PD regulator on ω_s
    # on top of the legacy_corrected feedforward + desat. The two
    # differ only in how ω̇_s is sourced (finite-diff vs Newton-Euler).
    if aocs_mode != 'legacy_corrected':
        cfg.aocs_mode = aocs_mode
        cfg.aocs_use_legacy_corrected = False
        cfg.aocs_use_H_estimator = (aocs_mode == 'H_est')
    # alpha_torso_ang stays at default 500 (set by _make_m7_config).
    # 5 evenly-spaced snapshots per SS for the offline renderer.
    # FRAMES_PER_STEP=0 disables capture entirely (e.g. for tests).
    cfg.frames_per_step = int(os.environ.get('FRAMES_PER_STEP', '6'))

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

    if legacy:
        out_dir = os.path.join(_root, 'results',
                               'diag_cooperative_arms_legacy')
    elif abs(mass_ratio - 0.01) > 1e-9:
        out_dir = os.path.join(_root, 'results',
                               f'diag_cooperative_arms_{int(round(mass_ratio*100))}pct')
    elif aocs_mode != 'legacy_corrected':
        # Non-default AOCS mode → separate dir for A/B.
        # Add _Kt{val} suffix when K_theta differs from default (PID modes).
        suffix = ''
        if 'pid' in aocs_mode and abs(K_theta - 1.0) > 1e-9:
            suffix = f'_Kt{K_theta:g}'
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

    from crawlbot.simulation.sim_loop import SimulationLoop
    URDF = os.path.join(_root, 'models', 'VISPA_crawling_fixed.urdf')

    sim = SimulationLoop(mjcf_path=MJCF, urdf_path=URDF, config=cfg)
    sim.setup(n_steps=5, start_a=2, start_b=2)
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
        log, sim._step2_diag_log, 5,
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
                                     'render_traversal.py')],
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
                        help='Disable cooperative_arms_mode (regression guard)')
    parser.add_argument('--alpha_torso_lin', type=float, default=500.0,
                        help='Sensitivity sweep value (default 500)')
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
                        help='Attitude tracking gain [Nm/rad] (legacy_pid_* only). '
                             'Default 1.0 — gentle (60s recovery time constant '
                             'with K_ω=50). Bump to 5-10 for faster recovery.')
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
             args.K_theta)
    finally:
        with open(MJCF, 'w') as f:
            f.write(original)
        post_hash = _mjcf_md5(MJCF)
        print(f'[COOP-ARMS {tag}] MJCF md5 post: {post_hash}')
        assert post_hash == pre_hash, 'MJCF restoration failed'
