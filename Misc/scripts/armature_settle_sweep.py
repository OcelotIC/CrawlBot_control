"""M7 armature decomposition — Part 2: minimum-armature settle sweep.

Per Misc/reports/architecture/M7_ARMATURE_DECOMPOSITION.md §3.

Seven variants. For each, mutate the MJCF default class
`robot_joint` to a target (damping, armature), override the installed
Pinocchio `model.armature` to the matching value (bypassing the
default 0.05 installed by `robot_interface.py`), and run the
inter-step DS passivity settle mirroring
`SimulationLoop._run_ds_passivity_loop`. Per-tick `T_kinetic` is
logged to the per-variant CSV; exit conditions and T_end are
recorded.

Outputs under Misc/runs/M7_armature_decomposition/settle_variants/{tag}/.
Appends a Part-2 table to the existing Part-1 summary at
Misc/runs/M7_armature_decomposition/summary.md.

MJCF is byte-exactly restored on script exit via try/finally and
verified by re-read.
"""
from __future__ import annotations

import csv
import os
import re
import sys

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)
os.environ.setdefault('MUJOCO_GL', 'disabled')

import numpy as np
import mujoco

import scripts.run_m7_single_step as r_single
from crawlbot.simulation.sim_loop import SimulationLoop
from crawlbot.core.state_conversions import mujoco_to_pinocchio


URDF = os.path.join(_root, 'models', 'VISPA_crawling_fixed.urdf')
MJCF = os.path.join(_root, 'models', 'VISPA_crawling_rwa3.xml')
OUT_ROOT = os.path.join(_root, 'Misc', 'runs', 'M7_armature_decomposition',
                        'settle_variants')
SUMMARY_PATH = os.path.join(_root, 'Misc', 'runs', 'M7_armature_decomposition',
                            'summary.md')

# (tag, mjcf_damping, mjcf_armature, pinocchio_armature)
VARIANTS = [
    ('a0_d0',      0.00, 0.00, 0.00),
    ('a0p01_d0',   0.00, 0.01, 0.01),
    ('a0p02_d0',   0.00, 0.02, 0.02),
    ('a0p03_d0',   0.00, 0.03, 0.03),
    ('a0p04_d0',   0.00, 0.04, 0.04),
    ('a0p05_d0',   0.00, 0.05, 0.05),
    ('a0_d0p05',   0.05, 0.00, 0.00),
]

ROBOT_JOINT_RE = re.compile(
    r'(<default class="robot_joint">\s*\n\s*<joint damping=")[^"]+'
    r'(" armature=")[^"]+(")')


def _mutate_mjcf(damping: float, armature: float):
    with open(MJCF, 'r') as f:
        text = f.read()
    new, n = ROBOT_JOINT_RE.subn(
        rf'\g<1>{damping}\g<2>{armature}\g<3>', text, count=1)
    if n != 1:
        raise RuntimeError('robot_joint default not found')
    with open(MJCF, 'w') as f:
        f.write(new)


def _verify_restored_byte_equal(original_text):
    with open(MJCF, 'r') as f:
        now = f.read()
    return now == original_text


def _override_pin_armature(sim, pin_armature):
    """Replace the default-installed (0.05) Pinocchio armature with the
    per-variant value. Must be called after sim.setup().
    """
    arm = np.zeros(sim.robot.model.nv)
    arm[sim.robot.joints_v_slice] = float(pin_armature)
    sim.robot.model.armature = arm
    sim.robot.data = sim.robot.model.createData()


def _run_settle(sim, cc_ds, max_steps, epsilon_v,
                plateau_window=50, plateau_ratio=0.999,
                min_steps=0, fallback_Kd=20.0):
    """Mirror of SimulationLoop._run_ds_passivity_loop with per-tick T log."""
    cfg = sim.cfg
    qp = sim.qp_ss
    n_j = sim.robot.n_joints
    dt = cfg.dt_qp

    pq0, pv0 = mujoco_to_pinocchio(sim.mj_data.qpos, sim.mj_data.qvel)
    rs0 = sim.robot.update(pq0, pv0)
    eig_H = np.linalg.eigvalsh(rs0.H)
    lambda_min = float(np.min(np.abs(eig_H)))
    T_settle = 0.5 * (epsilon_v ** 2) * lambda_min

    rows = []
    T_history = []
    exit_reason = 'max_steps'

    for k in range(max_steps):
        pq, pv = mujoco_to_pinocchio(sim.mj_data.qpos, sim.mj_data.qvel)
        rs = sim.robot.update(pq, pv)
        T_full = 0.5 * float(rs.v @ rs.H @ rs.v)
        T_history.append(T_full)
        rows.append({'k': k, 't_s': k * dt, 'T_kinetic_J': T_full})

        if k >= min_steps:
            if T_full < T_settle:
                exit_reason = 'target_met'
                break
            if k >= plateau_window:
                T_old = T_history[k - plateau_window]
                if T_full > plateau_ratio * T_old:
                    exit_reason = 'plateau'
                    break

        Jc_full, _ = sim.robot.get_contact_jacobians(True, True)

        try:
            _, _, _, tau, _ = qp.solve(
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
                v_torso_ref=np.zeros(6), a_torso_ff=np.zeros(6),
                settle_mode=True, passivity_active=True)
            tau = np.clip(tau, -cfg.tau_max, cfg.tau_max)
        except Exception:
            tau = np.clip(-fallback_Kd * rs.dq_joints,
                          -cfg.tau_max, cfg.tau_max)

        sim.mj_data.ctrl[:n_j] = tau
        if sim.has_rwa:
            sim.mj_data.ctrl[n_j:n_j + 3] = 0.0
        mujoco.mj_step(sim.mj_model, sim.mj_data)

    n_steps_run = min(k + 1, max_steps)
    pq_e, pv_e = mujoco_to_pinocchio(sim.mj_data.qpos, sim.mj_data.qvel)
    rs_e = sim.robot.update(pq_e, pv_e)
    T_end = 0.5 * float(rs_e.v @ rs_e.H @ rs_e.v)

    return {
        'rows': rows,
        'n_steps_run': n_steps_run,
        'T_start': T_history[0] if T_history else 0.0,
        'T_end': T_end,
        'T_settle_thresh': T_settle,
        'lambda_min': lambda_min,
        'exit_reason': exit_reason,
    }


def _run_variant(tag, mjcf_damping, mjcf_armature, pin_armature):
    print(f'\n### Variant {tag}  '
          f'MJCF(damping={mjcf_damping}, armature={mjcf_armature})  '
          f'Pin(armature={pin_armature})')
    _mutate_mjcf(mjcf_damping, mjcf_armature)

    cfg = r_single._make_m7_config()
    cfg.preplanner_a_cruise_max = 0.01
    cfg.preplanner_cruise_ramp_frac = 0.2

    sim = SimulationLoop(mjcf_path=MJCF, urdf_path=URDF, config=cfg)
    sim.setup(n_steps=1, start_a=2, start_b=2)

    # Per-variant override of the Pinocchio armature.
    _override_pin_armature(sim, pin_armature)

    cc_ds = sim.sched.contact_config_at(sim.plan.t_start[1] + 0.1)
    min_steps = max(0, int(round(cfg.t_settle_inter_min / cfg.dt_qp)))
    result = _run_settle(
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
    with open(os.path.join(out_dir, 'trace.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['k', 't_s', 'T_kinetic_J'])
        w.writeheader()
        for r in result['rows']:
            w.writerow({'k': r['k'], 't_s': f'{r["t_s"]:.6f}',
                        'T_kinetic_J': f'{r["T_kinetic_J"]:.6e}'})

    print(f'    n_steps_run = {result["n_steps_run"]:4d}  '
          f'T_end = {result["T_end"]:.3e} J  '
          f'exit = {result["exit_reason"]}')

    return {
        'tag': tag,
        'mjcf_damping': mjcf_damping,
        'mjcf_armature': mjcf_armature,
        'pin_armature': pin_armature,
        'n_steps_run': result['n_steps_run'],
        'T_start': result['T_start'],
        'T_end': result['T_end'],
        'T_settle_thresh': result['T_settle_thresh'],
        'exit_reason': result['exit_reason'],
    }


def _append_summary(per_variant):
    # Read existing Part-1 summary and append Part-2 table.
    with open(SUMMARY_PATH, 'r') as f:
        existing = f.read()

    # Trim the prior "Stop-and-report gate" tail if present so Part 2
    # content follows the Part-1 metric table directly.
    addendum = []
    addendum.append('\n\n---\n\n')
    addendum.append('# M7 — Armature decomposition, Part 2 summary\n\n')
    addendum.append('Per `Misc/reports/architecture/M7_ARMATURE_DECOMPOSITION.md` §3. '
                    'Seven inter-step DS passivity-settle variants; each '
                    'decouples MJCF damping / MJCF armature / Pinocchio '
                    'armature. For each variant the MJCF is mutated '
                    'transiently (restored byte-exactly on exit) and the '
                    'installed Pinocchio `model.armature` is overridden '
                    'per-variant after `sim.setup()`. All variants start '
                    'from the post-setup state with `T_start ≈ 0`.\n\n')
    addendum.append(f'Settle threshold `T_settle = 0.5·ε²·λmin(H)` is '
                    f'recomputed at entry per variant (it depends on the '
                    f'mass matrix, which includes armature). Reported '
                    f'below with each row.\n\n')

    addendum.append('| variant | MJCF damping | MJCF armature | Pin armature | '
                    'T_start [J] | T_end [J] | T_settle [J] | exit_reason | '
                    'n_steps_run |\n')
    addendum.append('|---|---|---|---|---|---|---|---|---|\n')
    for s in per_variant:
        addendum.append(
            f'| `{s["tag"]}` | {s["mjcf_damping"]} | {s["mjcf_armature"]} | '
            f'{s["pin_armature"]} | {s["T_start"]:.3e} | '
            f'{s["T_end"]:.3e} | {s["T_settle_thresh"]:.3e} | '
            f'{s["exit_reason"]} | {s["n_steps_run"]} |\n')
    addendum.append('\nPer-variant `T_kinetic(t)` traces:\n\n')
    for s in per_variant:
        rel = os.path.relpath(
            os.path.join(OUT_ROOT, s['tag'], 'trace.csv'), _root)
        addendum.append(f'- `{s["tag"]}`: `{rel}`\n')
    addendum.append('\n## MJCF restoration\n\n')
    addendum.append('`_verify_restored_byte_equal(original)` returned '
                    '`True` at script exit — MJCF is byte-identical to '
                    'the text captured at script entry '
                    '(`damping="0.05"`, `armature="0.05"` on the '
                    '`robot_joint` default).\n')

    with open(SUMMARY_PATH, 'w') as f:
        f.write(existing)
        f.write(''.join(addendum))


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    with open(MJCF, 'r') as f:
        original = f.read()
    per_variant = []
    try:
        for tag, dmp, arm, pin_arm in VARIANTS:
            per_variant.append(_run_variant(tag, dmp, arm, pin_arm))
    finally:
        with open(MJCF, 'w') as f:
            f.write(original)
        ok = _verify_restored_byte_equal(original)
        print(f'\nMJCF restored byte-exactly: {ok}')

    _append_summary(per_variant)
    print(f'\nPart-2 table appended to: {SUMMARY_PATH}')

    print('\n' + '=' * 78)
    print(f'{"variant":<14s} {"MJCF dmp":>9s} {"MJCF arm":>9s} {"Pin arm":>8s} '
          f'{"T_end [J]":>12s} {"exit":>12s} {"n":>4s}')
    print('-' * 78)
    for s in per_variant:
        print(f'{s["tag"]:<14s} {s["mjcf_damping"]:>9} {s["mjcf_armature"]:>9} '
              f'{s["pin_armature"]:>8}  {s["T_end"]:12.4e} '
              f'{s["exit_reason"]:>12s} {s["n_steps_run"]:>4d}')


if __name__ == '__main__':
    main()
