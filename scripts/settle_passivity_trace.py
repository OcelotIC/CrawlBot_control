"""DS settle / passivity diagnostic sweep.

For each (damping, armature) variant:

  variants = [
      (0,     0    ),    # the failing case after the prior turn's MJCF edit
      (0.005, 0.005),
      (0.01,  0.01 ),
      (0.05,  0.05 ),    # original committed values
      (0,     0.05 ),    # armature only, no dissipation
  ]

mutate the MJCF default class in place, instantiate SimulationLoop,
and run a self-contained settle loop that mirrors
`SimulationLoop._run_ds_passivity_loop` (sim_loop.py:483-642) but logs
per-tick T_full, T_jj, dq·tau, the passivity LHS, and the constraint
slack. Outputs: results/M7_settle_diag/{variant}/passivity_trace.csv
and the overlay results/M7_settle_diag/passivity_trace.png +
results/M7_settle_diag/settle_damping_sweep.md.

The on-disk MJCF is restored verbatim at the end (try/finally). Per
the task: do not commit. Helper script kept under-_ scratch convention.
"""
from __future__ import annotations

import csv
import os
import re
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')

import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import scripts.run_m7_single_step as r_single
from crawlbot.simulation.sim_loop import SimulationLoop
from crawlbot.core.state_conversions import mujoco_to_pinocchio


URDF = os.path.join(_root, 'models', 'VISPA_crawling_fixed.urdf')
MJCF = os.path.join(_root, 'models', 'VISPA_crawling_rwa3.xml')
OUT_ROOT = os.path.join(_root, 'results', 'M7_settle_diag')

VARIANTS = [
    ('d0_a0',          0.000, 0.000),
    ('d0p005_a0p005',  0.005, 0.005),
    ('d0p01_a0p01',    0.010, 0.010),
    ('d0p05_a0p05',    0.050, 0.050),
    ('d0_a0p05',       0.000, 0.050),
]


# ── MJCF mutation (default class robot_joint only) ─────────────────────
ROBOT_JOINT_RE = re.compile(
    r'(<default class="robot_joint">\s*\n\s*<joint damping=")[^"]+'
    r'(" armature=")[^"]+(")')


def _set_arm_dynamics(damping: float, armature: float):
    with open(MJCF, 'r') as f:
        text = f.read()
    new_text, n = ROBOT_JOINT_RE.subn(
        rf'\g<1>{damping}\g<2>{armature}\g<3>', text, count=1)
    if n != 1:
        raise RuntimeError(
            f'Could not find robot_joint default in {MJCF!r}')
    with open(MJCF, 'w') as f:
        f.write(new_text)


# ── Instrumented settle loop ───────────────────────────────────────────
def _run_settle_with_trace(sim, cc_ds, max_steps, epsilon_v,
                           plateau_window=50, plateau_ratio=0.999,
                           min_steps=0, fallback_Kd=20.0):
    """Mirror of SimulationLoop._run_ds_passivity_loop with per-tick logging.

    Returns (rows, summary). Each row = (k, t_s, T_full, T_jj,
    dq_dot_tau_j, alpha_T_jj, passivity_lhs, constraint_slack,
    abs_tau_max).
    """
    cfg = sim.cfg
    qp = sim.qp_ss
    n_j = sim.robot.n_joints
    dt = cfg.dt_qp
    alpha = float(cfg.alpha_passivity)

    pq0, pv0 = mujoco_to_pinocchio(sim.mj_data.qpos, sim.mj_data.qvel)
    rs0 = sim.robot.update(pq0, pv0)
    eig_H = np.linalg.eigvalsh(rs0.H)
    lambda_min = float(np.min(np.abs(eig_H)))
    T_settle = 0.5 * (epsilon_v ** 2) * lambda_min

    hw_current = (np.zeros(3) if not sim.has_rwa
                  else (cfg.rwa_I_w * sim.mj_data.qvel[6:9]).copy())

    rows = []
    T_history = []
    exit_reason = 'max_steps'

    for k in range(max_steps):
        pq, pv = mujoco_to_pinocchio(sim.mj_data.qpos, sim.mj_data.qvel)
        rs = sim.robot.update(pq, pv)

        # Energies
        T_full = 0.5 * float(rs.v @ rs.H @ rs.v)
        H_jj   = rs.H[6:, 6:]
        dq_j   = rs.dq_joints
        T_jj   = 0.5 * float(dq_j @ H_jj @ dq_j)
        T_history.append(T_full)

        # Exits (after min_steps)
        if k >= min_steps:
            if T_full < T_settle:
                exit_reason = 'target_met'
                break
            if k >= plateau_window:
                T_old = T_history[k - plateau_window]
                if T_full > plateau_ratio * T_old:
                    exit_reason = 'plateau'
                    break

        Jc, Jdc = sim.robot.get_contact_jacobians(True, True)
        try:
            _, _, _, tau, _ = qp.solve(
                q_t=rs.q_torso, dq_t=rs.dq_torso,
                q=rs.q_joints, dq=rs.dq_joints,
                r_com_ref=rs.r_com, v_com_ref=np.zeros(3),
                lambda_ref=np.zeros(12), a_com_ff=np.zeros(3),
                H_robot=rs.H, C_robot=rs.C,
                J_com=rs.J_com, Jdot_dq_com=rs.Jdot_dq_com,
                contact_config=cc_ds,
                J_contacts=Jc, Jdot_dq_contacts=Jdc,
                hw_current=hw_current,
                hw_min=cfg.hw_min, hw_max=cfg.hw_max,
                r_com=rs.r_com, L_com_current=rs.L_com,
                J_torso=rs.J_torso, Jdot_dq_torso=rs.Jdot_dq_torso,
                p_torso=rs.oMf_torso.translation.copy(),
                R_torso=rs.oMf_torso.rotation.copy(),
                p_torso_ref=rs.oMf_torso.translation.copy(),
                R_torso_ref=rs.oMf_torso.rotation.copy(),
                v_torso_ref=np.zeros(6),
                a_torso_ff=np.zeros(6),
                settle_mode=True,
                passivity_active=True)
            tau = np.clip(tau, -cfg.tau_max, cfg.tau_max)
        except Exception:
            tau = np.clip(-fallback_Kd * dq_j,
                          -cfg.tau_max, cfg.tau_max)

        # Passivity LHS = dq_j · tau + 2α·T_jj  (constraint: lhs ≤ 0)
        dq_dot_tau = float(dq_j @ tau)
        passivity_lhs = dq_dot_tau + 2.0 * alpha * T_jj
        # Slack = -lhs (positive = inactive, ~0 = binding, negative = violated)
        slack = -passivity_lhs

        rows.append((k, k * dt, T_full, T_jj, dq_dot_tau,
                     alpha * T_jj, passivity_lhs, slack,
                     float(np.max(np.abs(tau)))))

        sim.mj_data.ctrl[:n_j] = tau
        if sim.has_rwa:
            sim.mj_data.ctrl[n_j:n_j + 3] = 0.0
        mujoco.mj_step(sim.mj_model, sim.mj_data)

    n_steps_run = min(k + 1, max_steps)
    pq_end, pv_end = mujoco_to_pinocchio(sim.mj_data.qpos, sim.mj_data.qvel)
    rs_end = sim.robot.update(pq_end, pv_end)
    T_end = 0.5 * float(rs_end.v @ rs_end.H @ rs_end.v)

    summary = {
        'n_steps_run': n_steps_run,
        'T_start':     T_history[0] if T_history else 0.0,
        'T_end':       T_end,
        'T_settle':    T_settle,
        'lambda_min':  lambda_min,
        'exit_reason': exit_reason,
        'alpha':       alpha,
    }
    return rows, summary


# ── Driver per variant ────────────────────────────────────────────────
def _run_variant(tag: str, damping: float, armature: float):
    print('\n' + '=' * 78)
    print(f' Variant: {tag}  damping={damping}  armature={armature}')
    print('=' * 78)
    _set_arm_dynamics(damping, armature)

    cfg = r_single._make_m7_config()
    cfg.preplanner_a_cruise_max = 0.01
    cfg.preplanner_cruise_ramp_frac = 0.2

    sim = SimulationLoop(mjcf_path=MJCF, urdf_path=URDF, config=cfg)
    sim.setup(n_steps=1, start_a=2, start_b=2)

    # Inter-step DS settle is what fires before SS — instantiate the
    # same contact config as the main loop does for the pre-SS window.
    cc_ds = sim.sched.contact_config_at(sim.plan.t_start[1] + 0.1)

    min_steps = max(0, int(round(cfg.t_settle_inter_min / cfg.dt_qp)))
    rows, summary = _run_settle_with_trace(
        sim, cc_ds,
        max_steps=cfg.n_ds_max_steps,
        epsilon_v=cfg.settle_inter_epsilon_v,
        plateau_window=50,
        plateau_ratio=cfg.settle_plateau_ratio,
        min_steps=min_steps,
        fallback_Kd=cfg.Kd_settle_damping,
    )

    out_dir = os.path.join(OUT_ROOT, tag)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'passivity_trace.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['k', 't_s', 'T_full_J', 'T_jj_J', 'dq_dot_tau_W',
                    'alpha_times_T_jj_W', 'passivity_lhs_W',
                    'constraint_slack_W', 'tau_max_Nm'])
        for row in rows:
            w.writerow([row[0], f'{row[1]:.6f}'] +
                       [f'{x:.6e}' for x in row[2:]])
    summary_path = os.path.join(out_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        for k, v in summary.items():
            f.write(f'{k:18s} = {v}\n')
        f.write(f'(damping, armature) = ({damping}, {armature})\n')
        f.write(f'rows logged          = {len(rows)}\n')

    print(f'  T_settle target = {summary["T_settle"]:.3e} J')
    print(f'  T_start         = {summary["T_start"]:.6e} J')
    print(f'  T_end           = {summary["T_end"]:.6e} J')
    print(f'  exit_reason     = {summary["exit_reason"]}')
    print(f'  n_steps_run     = {summary["n_steps_run"]}')
    print(f'  rows -> {csv_path}')
    return tag, summary, rows


# ── Overlay plot + markdown report ────────────────────────────────────
def _build_overlay(results):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, dpi=140)
    ax_T, ax_lhs = axes
    color_cycle = ['tab:red', 'tab:orange', 'tab:olive', 'tab:blue', 'tab:purple']
    for (tag, summary, rows), color in zip(results, color_cycle):
        if not rows:
            continue
        t = np.array([r_[1] for r_ in rows])
        T = np.array([r_[2] for r_ in rows])
        lhs = np.array([r_[6] for r_ in rows])
        label = f'{tag}  exit={summary["exit_reason"]}  T_end={summary["T_end"]:.2e} J'
        ax_T.plot(t, T, color=color, linewidth=1.6, label=label)
        ax_lhs.plot(t, lhs, color=color, linewidth=1.6, label=tag)

    ax_T.set_ylabel('T_full  [J]')
    ax_T.set_yscale('symlog', linthresh=1e-9)
    ax_T.grid(True, alpha=0.3)
    ax_T.legend(loc='best', fontsize=8)
    ax_T.set_title('DS settle — kinetic energy (5 (damping, armature) variants)')

    ax_lhs.set_xlabel('t [s] (relative to settle entry)')
    ax_lhs.set_ylabel(r'passivity LHS = $\dot q^\top \tau + 2\alpha T_{jj}$  [W]')
    ax_lhs.axhline(0.0, color='k', linewidth=0.6, linestyle=':')
    ax_lhs.set_yscale('symlog', linthresh=1e-9)
    ax_lhs.grid(True, alpha=0.3)
    ax_lhs.legend(loc='best', fontsize=8)

    fig.tight_layout()
    plot_path = os.path.join(OUT_ROOT, 'passivity_trace.png')
    fig.savefig(plot_path)
    plt.close(fig)
    return plot_path


def _build_markdown(results, plot_path):
    md_path = os.path.join(OUT_ROOT, 'settle_damping_sweep.md')
    with open(md_path, 'w') as f:
        f.write('# M7 — DS settle damping/armature sweep\n\n')
        f.write('Diagnostic only. The MJCF file '
                '(`models/VISPA_crawling_rwa3.xml`) is mutated '
                'in-place per variant by writing new `damping` / '
                '`armature` values into the `default class="robot_joint"` '
                'block, then restored to the on-disk state observed at '
                'script entry. The settle loop mirrors '
                '`SimulationLoop._run_ds_passivity_loop` '
                '(`sim_loop.py:483-642`) one-for-one but logs five '
                'extra columns per QP tick. Constraint slack is '
                'computed as `-(dq_j·tau + 2α·T_jj)` from the QP\'s '
                'returned `tau`; positive = constraint inactive, '
                '~0 = binding, negative = the constraint was '
                'violated by the solver. No qpOASES dual values are '
                'extracted (the wrapper does not expose them).\n\n')
        f.write('## Per-variant summary\n\n')
        f.write('| variant | damping | armature | T_settle [J] | T_start [J] | T_end [J] | exit_reason | n_steps |\n')
        f.write('|---|---|---|---|---|---|---|---|\n')
        for (tag, s, _), (_, dmp, arm) in zip(results, VARIANTS):
            f.write(f'| {tag} | {dmp} | {arm} | '
                    f'{s["T_settle"]:.4e} | '
                    f'{s["T_start"]:.4e} | '
                    f'{s["T_end"]:.4e} | '
                    f'{s["exit_reason"]} | '
                    f'{s["n_steps_run"]} |\n')
        f.write(f'\n`alpha_passivity` (cfg.alpha_passivity) used in all '
                f'variants = {results[0][1]["alpha"]:.3g}.\n\n')

        f.write('## Per-variant constraint behaviour (variant `d0_a0`)\n\n')
        d0a0_rows = next((r_ for r_ in results if r_[0] == 'd0_a0'), None)
        if d0a0_rows is not None:
            tag, s, rows = d0a0_rows
            n = len(rows)
            slack = np.array([r_[7] for r_ in rows])
            lhs = np.array([r_[6] for r_ in rows])
            f.write(f'`{tag}` rows logged: {n}.\n\n')
            f.write('Per-tick LHS sign breakdown:\n\n')
            n_violated = int(np.sum(lhs > 1e-9))
            n_binding  = int(np.sum(np.abs(lhs) <= 1e-9))
            n_inactive = int(np.sum(lhs < -1e-9))
            f.write(f'- violated  (lhs > 1e-9 W):   {n_violated} / {n}\n')
            f.write(f'- binding   (|lhs| ≤ 1e-9 W): {n_binding} / {n}\n')
            f.write(f'- inactive  (lhs < -1e-9 W):  {n_inactive} / {n}\n\n')
            f.write(f'`max(lhs)` = {lhs.max():.4e} W,  '
                    f'`min(slack)` = {slack.min():.4e} W,  '
                    f'`max(|lhs|)` = {np.abs(lhs).max():.4e} W.\n\n')

        f.write('## Overlay plot\n\n')
        f.write(f'`{os.path.relpath(plot_path, _root)}` — top: '
                f'`T_full(t)` on a symlog scale; bottom: passivity LHS '
                f'on a symlog scale (zero line marked).\n\n')

        f.write('## Per-variant trace files\n\n')
        f.write('| variant | trace |\n|---|---|\n')
        for tag, _, _ in results:
            rel = os.path.relpath(
                os.path.join(OUT_ROOT, tag, 'passivity_trace.csv'),
                _root)
            f.write(f'| {tag} | `{rel}` |\n')
        f.write('\nColumns: `k, t_s, T_full_J, T_jj_J, dq_dot_tau_W, '
                'alpha_times_T_jj_W, passivity_lhs_W, constraint_slack_W, '
                'tau_max_Nm`.\n')

        f.write('\n## Variant `d0_a0` is the trace requested by item (1)\n\n')
        f.write(f'CSV: `results/M7_settle_diag/d0_a0/passivity_trace.csv` — '
                f'aliased per spec to `results/M7_settle_diag/passivity_trace.csv` '
                f'(also written).\n')

    # Spec also asks for the d0_a0 trace at the top-level results path.
    src = os.path.join(OUT_ROOT, 'd0_a0', 'passivity_trace.csv')
    dst = os.path.join(OUT_ROOT, 'passivity_trace.csv')
    with open(src) as f_in, open(dst, 'w') as f_out:
        f_out.write(f_in.read())

    return md_path


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    with open(MJCF, 'r') as f:
        original_mjcf = f.read()
    results = []
    try:
        for tag, dmp, arm in VARIANTS:
            results.append(_run_variant(tag, dmp, arm))
    finally:
        with open(MJCF, 'w') as f:
            f.write(original_mjcf)
        print(f'\nMJCF restored to entry state (1st robot_joint default '
              f'block) at {MJCF}')

    plot_path = _build_overlay(results)
    md_path = _build_markdown(results, plot_path)

    print('\n' + '=' * 78)
    print(f'Overlay plot : {plot_path}')
    print(f'Markdown     : {md_path}')


if __name__ == '__main__':
    main()
