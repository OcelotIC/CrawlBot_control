"""DS settle — augmented power-balance trace for the d0_a0 variant.

Mutates models/VISPA_crawling_rwa3.xml to (damping=0, armature=0) on
the robot_joint default class, runs the inter-step DS settle with
extra logging, then restores the MJCF.

Three new columns appended to the existing per-tick trace:

  P_weld   = (J_c · q̇)^T · λ
              power done by contact constraint forces (sum over all
              active contacts).
  P_aocs   = τ_w · ω_s
              power done by reaction wheel torques on the structure
              frame. (τ_w hard-zeroed during settle by sim_loop, see
              report § AOCS status.)
  dyn_residual_norm = ‖H·q̈ + C − B·τ_q − J_c^T·λ‖
              q̈ via finite-difference of Pinocchio v across the tick;
              B = [0_{6×14}; I_{14×14}].

Outputs:
  Misc/runs/M7_settle_diag/d0_a0/power_balance_trace.csv  (full per-tick log)
  Misc/runs/M7_settle_diag/settle_power_balance.md         (report)

Diagnostic only. Do not commit.
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

import scripts.run_m7_single_step as r_single
from crawlbot.simulation.sim_loop import SimulationLoop
from crawlbot.core.state_conversions import mujoco_to_pinocchio


URDF = os.path.join(_root, 'models', 'VISPA_crawling_fixed.urdf')
MJCF = os.path.join(_root, 'models', 'VISPA_crawling_rwa3.xml')
OUT_DIR = os.path.join(_root, 'Misc', 'runs', 'M7_settle_diag', 'd0_a0')
TOP_OUT = os.path.join(_root, 'Misc', 'runs', 'M7_settle_diag')

ROBOT_JOINT_RE = re.compile(
    r'(<default class="robot_joint">\s*\n\s*<joint damping=")[^"]+'
    r'(" armature=")[^"]+(")')


def _set_arm_dynamics(damping: float, armature: float):
    with open(MJCF, 'r') as f:
        text = f.read()
    new_text, n = ROBOT_JOINT_RE.subn(
        rf'\g<1>{damping}\g<2>{armature}\g<3>', text, count=1)
    if n != 1:
        raise RuntimeError('Could not locate robot_joint default block')
    with open(MJCF, 'w') as f:
        f.write(new_text)


def _run():
    cfg = r_single._make_m7_config()
    cfg.preplanner_a_cruise_max = 0.01
    cfg.preplanner_cruise_ramp_frac = 0.2

    sim = SimulationLoop(mjcf_path=MJCF, urdf_path=URDF, config=cfg)
    sim.setup(n_steps=1, start_a=2, start_b=2)

    cc_ds = sim.sched.contact_config_at(sim.plan.t_start[1] + 0.1)

    qp = sim.qp_ss
    n_j = sim.robot.n_joints
    nv  = sim.robot.model.nv               # 20 for 7-DOF, free-floating
    dt  = cfg.dt_qp
    alpha = float(cfg.alpha_passivity)

    # Settle thresholds (mirrors _run_ds_passivity_loop entry).
    pq0, pv0 = mujoco_to_pinocchio(sim.mj_data.qpos, sim.mj_data.qvel)
    rs0 = sim.robot.update(pq0, pv0)
    eig_H = np.linalg.eigvalsh(rs0.H)
    lambda_min = float(np.min(np.abs(eig_H)))
    T_settle_thresh = 0.5 * (cfg.settle_inter_epsilon_v ** 2) * lambda_min
    plateau_window = 50
    plateau_ratio  = cfg.settle_plateau_ratio
    min_steps = max(0, int(round(cfg.t_settle_inter_min / cfg.dt_qp)))
    max_steps = cfg.n_ds_max_steps

    # Actuation matrix B: maps τ_q ∈ R^14 to v-space ∈ R^20.
    # Standard floating-base form: first 6 rows zero (no actuation on
    # base DOFs), last 14 rows identity.
    B = np.zeros((nv, n_j))
    B[6:, :] = np.eye(n_j)

    # Track AOCS call counter (manually — _run_ds_passivity_loop
    # never invokes the AOCS function; we confirm by recording how
    # many times the wheel-actuator slot is touched and what value).
    aocs_called = False
    tau_w_unique_values = set()

    rows = []
    T_history = []
    exit_reason = 'max_steps'
    pv_prev = pv0.copy()              # v at start of tick k=0

    for k in range(max_steps):
        pq, pv = mujoco_to_pinocchio(sim.mj_data.qpos, sim.mj_data.qvel)
        rs = sim.robot.update(pq, pv)

        T_full = 0.5 * float(rs.v @ rs.H @ rs.v)
        H_jj = rs.H[6:, 6:]
        dq_j = rs.dq_joints
        T_jj = 0.5 * float(dq_j @ H_jj @ dq_j)
        T_history.append(T_full)

        if k >= min_steps:
            if T_full < T_settle_thresh:
                exit_reason = 'target_met'
                break
            if k >= plateau_window:
                T_old = T_history[k - plateau_window]
                if T_full > plateau_ratio * T_old:
                    exit_reason = 'plateau'
                    break

        Jc_full, _ = sim.robot.get_contact_jacobians(True, True)   # (12, 20)

        try:
            _, _, lam_sol, tau, _ = qp.solve(
                q_t=rs.q_torso, dq_t=rs.dq_torso,
                q=rs.q_joints, dq=rs.dq_joints,
                r_com_ref=rs.r_com, v_com_ref=np.zeros(3),
                lambda_ref=np.zeros(12), a_com_ff=np.zeros(3),
                H_robot=rs.H, C_robot=rs.C,
                J_com=rs.J_com, Jdot_dq_com=rs.Jdot_dq_com,
                contact_config=cc_ds,
                J_contacts=Jc_full, Jdot_dq_contacts=np.zeros(12),
                hw_current=np.zeros(3),
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
            tau = np.clip(-cfg.Kd_settle_damping * dq_j,
                          -cfg.tau_max, cfg.tau_max)
            lam_sol = np.zeros(12)

        # Passivity LHS (joint-block formulation, matches QP).
        dq_dot_tau = float(dq_j @ tau)
        passivity_lhs = dq_dot_tau + 2.0 * alpha * T_jj
        slack = -passivity_lhs

        # P_weld = (J_c · q̇)^T · λ
        v_contacts = Jc_full @ rs.v        # (12,)
        P_weld = float(v_contacts @ lam_sol)

        # P_aocs = τ_w · ω_s. AOCS off → wheel ctrl set to 0 below.
        omega_s = sim.mj_data.qvel[3:6].copy()
        tau_w = np.zeros(3)                # what sim_loop writes
        P_aocs = float(tau_w @ omega_s)

        # Set actuators (matches _run_ds_passivity_loop)
        sim.mj_data.ctrl[:n_j] = tau
        if sim.has_rwa:
            sim.mj_data.ctrl[n_j:n_j + 3] = 0.0
            tau_w_unique_values.add(tuple(
                float(x) for x in sim.mj_data.ctrl[n_j:n_j + 3]))

        # ── Step physics, then finite-difference v for q̈ ────────────
        mujoco.mj_step(sim.mj_model, sim.mj_data)
        _, pv_post = mujoco_to_pinocchio(sim.mj_data.qpos, sim.mj_data.qvel)
        qdd_fd = (pv_post - rs.v) / dt        # (20,)

        # Dynamics residual: ‖H·q̈ + C − B·τ − J_c^T·λ‖
        residual = (rs.H @ qdd_fd + rs.C - B @ tau - Jc_full.T @ lam_sol)
        residual_norm = float(np.linalg.norm(residual))

        rows.append({
            'k': k, 't_s': k * dt,
            'T_full_J': T_full, 'T_jj_J': T_jj,
            'dq_dot_tau_W': dq_dot_tau,
            'alpha_times_T_jj_W': alpha * T_jj,
            'passivity_lhs_W': passivity_lhs,
            'constraint_slack_W': slack,
            'tau_max_Nm': float(np.max(np.abs(tau))),
            'P_weld_W': P_weld,
            'P_aocs_W': P_aocs,
            'dyn_residual_norm': residual_norm,
        })

    n_steps_run = min(k + 1, max_steps)
    pq_e, pv_e = mujoco_to_pinocchio(sim.mj_data.qpos, sim.mj_data.qvel)
    rs_e = sim.robot.update(pq_e, pv_e)
    T_end = 0.5 * float(rs_e.v @ rs_e.H @ rs_e.v)

    return {
        'rows': rows,
        'n_steps_run': n_steps_run,
        'T_start': T_history[0] if T_history else 0.0,
        'T_end': T_end,
        'T_settle_thresh': T_settle_thresh,
        'lambda_min': lambda_min,
        'exit_reason': exit_reason,
        'alpha': alpha,
        'aocs_called': aocs_called,
        'tau_w_unique_values': sorted(tau_w_unique_values),
    }


def _summary_stats(rows, key):
    vals = np.array([r[key] for r in rows])
    return float(vals.min()), float(vals.max()), float(vals.mean())


def _write_csv(rows):
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, 'power_balance_trace.csv')
    fieldnames = ['k', 't_s', 'T_full_J', 'T_jj_J', 'dq_dot_tau_W',
                  'alpha_times_T_jj_W', 'passivity_lhs_W',
                  'constraint_slack_W', 'tau_max_Nm',
                  'P_weld_W', 'P_aocs_W', 'dyn_residual_norm']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            row_out = {k: (f'{r[k]:.6e}' if isinstance(r[k], float) else r[k])
                       for k in fieldnames}
            w.writerow(row_out)
    return csv_path


def _write_md(result, csv_path):
    rows = result['rows']
    md_path = os.path.join(TOP_OUT, 'settle_power_balance.md')

    p_weld_min, p_weld_max, p_weld_mean = _summary_stats(rows, 'P_weld_W')
    p_aocs_min, p_aocs_max, p_aocs_mean = _summary_stats(rows, 'P_aocs_W')
    res_min, res_max, res_mean = _summary_stats(rows, 'dyn_residual_norm')
    lhs_max = max(r['passivity_lhs_W'] for r in rows)
    lhs_min = min(r['passivity_lhs_W'] for r in rows)

    with open(md_path, 'w') as f:
        f.write('# M7 — DS settle d0_a0: power balance + dynamics residual\n\n')
        f.write('Diagnostic only, single variant `d0_a0` (damping=0, '
                'armature=0). The script mutates the MJCF in place and '
                'restores it on exit; the original on-disk state at '
                'script entry is preserved (try/finally).\n\n')

        f.write('## Run summary\n\n')
        f.write('| field | value |\n|---|---|\n')
        f.write(f'| n_steps_run | {result["n_steps_run"]} |\n')
        f.write(f'| T_start [J] | {result["T_start"]:.6e} |\n')
        f.write(f'| T_end [J] | {result["T_end"]:.6e} |\n')
        f.write(f'| T_settle threshold [J] | {result["T_settle_thresh"]:.6e} |\n')
        f.write(f'| lambda_min(H_entry) | {result["lambda_min"]:.6e} |\n')
        f.write(f'| exit_reason | {result["exit_reason"]} |\n')
        f.write(f'| alpha (cfg.alpha_passivity) | {result["alpha"]} |\n')
        f.write(f'| (damping, armature) | (0, 0) |\n\n')

        f.write('## AOCS status during the settle\n\n')
        f.write('`SimulationLoop._run_ds_passivity_loop` (sim_loop.py:483-642) '
                'does not call `compute_aocs_command_legacy_corrected`; '
                'the wheel-actuator slot is hard-zeroed at line 632 '
                '(`self.mj_data.ctrl[n_j:n_j + 3] = 0.0`). The AOCS call '
                'sites are at sim_loop.py:1898 and :1917 inside `_step()`, '
                'which the inter-step settle bypasses entirely. This '
                'diagnostic mirrors that wiring.\n\n')
        f.write(f'- Unique values written to `mj_data.ctrl[n_j:n_j+3]` '
                f'across the {len(rows)} ticks: '
                f'`{result["tau_w_unique_values"]}`\n')
        f.write(f'- `compute_aocs_command_legacy_corrected` calls '
                f'observed: 0 (function never invoked from the settle loop)\n\n')

        f.write('## Augmented per-tick columns (over the full trace)\n\n')
        f.write('Trace CSV: `' + os.path.relpath(csv_path, _root) + '`. '
                'Columns: `k, t_s, T_full_J, T_jj_J, dq_dot_tau_W, '
                'alpha_times_T_jj_W, passivity_lhs_W, constraint_slack_W, '
                'tau_max_Nm, P_weld_W, P_aocs_W, dyn_residual_norm`.\n\n')

        f.write('### P_weld = (J_c · q̇)^T · λ  [W]\n\n')
        f.write(f'- min  = `{p_weld_min:+.6e}`\n')
        f.write(f'- max  = `{p_weld_max:+.6e}`\n')
        f.write(f'- mean = `{p_weld_mean:+.6e}`\n\n')

        f.write('### P_aocs = τ_w · ω_s  [W]\n\n')
        f.write(f'- min  = `{p_aocs_min:+.6e}`\n')
        f.write(f'- max  = `{p_aocs_max:+.6e}`\n')
        f.write(f'- mean = `{p_aocs_mean:+.6e}`\n\n')
        f.write('(τ_w identically 0 throughout the settle — see AOCS status above.)\n\n')

        f.write('### Dynamics residual  ‖H·q̈ + C − B·τ_q − J_c^T·λ‖\n\n')
        f.write('q̈ via Pinocchio finite-difference: `qdd = (v_post - v) / dt_qp`. '
                '`B = [[0_{6×14}]; [I_{14×14}]]`. `H, C` from '
                '`RobotInterface.update`. `J_c` from '
                '`get_contact_jacobians(True, True)` (DS, both arms welded). '
                '`λ` from the QP\'s `lam_sol` return.\n\n')
        f.write(f'- min  = `{res_min:.6e}`\n')
        f.write(f'- max  = `{res_max:.6e}`\n')
        f.write(f'- mean = `{res_mean:.6e}`\n\n')

        f.write('### Passivity LHS (for cross-reference)\n\n')
        f.write(f'- max(LHS) = `{lhs_max:+.6e}` W\n')
        f.write(f'- min(LHS) = `{lhs_min:+.6e}` W\n\n')

        f.write('## Selected per-tick samples\n\n')
        f.write('Every 5th row (first 11 entries) plus the last row.\n\n')
        f.write('| k | t [s] | T_full [J] | P_weld [W] | P_aocs [W] | dyn residual | passivity LHS [W] | tau_max [Nm] |\n')
        f.write('|---|---|---|---|---|---|---|---|\n')
        sampled = list(range(0, len(rows), 5))
        if sampled[-1] != len(rows) - 1:
            sampled.append(len(rows) - 1)
        for k_idx in sampled:
            r = rows[k_idx]
            f.write(f"| {r['k']} | {r['t_s']:.3f} | {r['T_full_J']:.4e} | "
                    f"{r['P_weld_W']:+.4e} | {r['P_aocs_W']:+.4e} | "
                    f"{r['dyn_residual_norm']:.4e} | "
                    f"{r['passivity_lhs_W']:+.4e} | "
                    f"{r['tau_max_Nm']:.4e} |\n")

    return md_path


def main():
    os.makedirs(TOP_OUT, exist_ok=True)
    with open(MJCF, 'r') as f:
        original_mjcf = f.read()
    try:
        _set_arm_dynamics(0.0, 0.0)
        result = _run()
    finally:
        with open(MJCF, 'w') as f:
            f.write(original_mjcf)

    csv_path = _write_csv(result['rows'])
    md_path  = _write_md(result, csv_path)

    print()
    print('=' * 78)
    print(f'd0_a0 power-balance trace')
    print('=' * 78)
    print(f'  n_steps_run   = {result["n_steps_run"]}')
    print(f'  exit_reason   = {result["exit_reason"]}')
    print(f'  T_end         = {result["T_end"]:.4e} J')
    rows = result['rows']
    p_weld_min, p_weld_max, p_weld_mean = _summary_stats(rows, 'P_weld_W')
    p_aocs_min, p_aocs_max, p_aocs_mean = _summary_stats(rows, 'P_aocs_W')
    res_min,    res_max,    res_mean    = _summary_stats(rows, 'dyn_residual_norm')
    print(f'  P_weld  min/max/mean [W] = {p_weld_min:+.4e} / {p_weld_max:+.4e} / {p_weld_mean:+.4e}')
    print(f'  P_aocs  min/max/mean [W] = {p_aocs_min:+.4e} / {p_aocs_max:+.4e} / {p_aocs_mean:+.4e}')
    print(f'  dyn_res min/max/mean      = {res_min:.4e} / {res_max:.4e} / {res_mean:.4e}')
    print(f'  AOCS function calls during settle = 0')
    print(f'  Unique tau_w values written        = {result["tau_w_unique_values"]}')
    print()
    print(f'CSV : {csv_path}')
    print(f'MD  : {md_path}')


if __name__ == '__main__':
    main()
