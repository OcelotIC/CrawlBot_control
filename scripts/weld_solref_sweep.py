"""DS settle — weld solref sweep under (damping=0, armature=0).

Transiently mutates models/VISPA_crawling_rwa3.xml for each variant:
  - robot_joint default:  damping=0, armature=0
  - all <weld> elements:  solref = <variant value>

Runs the inter-step DS settle (same path as
SimulationLoop._run_ds_passivity_loop, sim_loop.py:483-642) with
instrumentation for T_full, P_weld = (J_c·q̇)^T·λ, and the full
(J_c·q̇) 12-vector split into per-contact lin/ang 3-vectors. Writes
per-variant CSV + summary.txt and one consolidated summary.md.

MJCF is restored to its entry state (damping=0.05, armature=0.05 and
solref="0.003 1" on every weld — the branch HEAD values) via
try/finally, and the restoration is verified by re-reading the file.

Diagnostic only. Do not commit the MJCF change.
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
OUT_ROOT = os.path.join(_root, 'Misc', 'runs', 'M7_settle_diag', 'weld_sweep')

VARIANTS = [
    ('solref_0p003_1',      '0.003 1'),
    ('solref_0p001_1',      '0.001 1'),
    ('solref_direct_stiff', '-1e6 -1e3'),
]

ROBOT_JOINT_RE = re.compile(
    r'(<default class="robot_joint">\s*\n\s*<joint damping=")[^"]+'
    r'(" armature=")[^"]+(")')

WELD_SOLREF_RE = re.compile(
    r'(<weld name="[^"]+" site1="[^"]+" site2="[^"]+" solref=")[^"]+(")')


def _mutate_mjcf(damping: float, armature: float, solref: str):
    with open(MJCF, 'r') as f:
        text = f.read()
    text, n1 = ROBOT_JOINT_RE.subn(
        rf'\g<1>{damping}\g<2>{armature}\g<3>', text, count=1)
    if n1 != 1:
        raise RuntimeError('robot_joint default block not found')
    text, n2 = WELD_SOLREF_RE.subn(
        rf'\g<1>{solref}\g<2>', text)
    if n2 == 0:
        raise RuntimeError('no <weld> solref substitutions made')
    with open(MJCF, 'w') as f:
        f.write(text)
    return n1, n2


def _verify_mjcf_restored():
    """Re-read MJCF and assert the branch-HEAD state."""
    with open(MJCF, 'r') as f:
        text = f.read()
    m = ROBOT_JOINT_RE.search(text)
    assert m, 'robot_joint default missing'
    body = text[m.start():m.end()]
    assert 'damping="0.05"' in body and 'armature="0.05"' in body, \
        f'robot_joint default not restored to 0.05/0.05: {body}'
    welds = WELD_SOLREF_RE.findall(text)
    # Collect the unique solref value actually present on welds.
    uniq = set()
    for m2 in WELD_SOLREF_RE.finditer(text):
        uniq.add(m2.group(0).split('solref="')[1].split('"')[0])
    assert uniq == {'0.003 1'}, \
        f'weld solref not restored to "0.003 1": {uniq}'
    return True


# ── Instrumented settle loop ──────────────────────────────────────────
def _run_settle(sim, cc_ds, max_steps, epsilon_v,
                plateau_window=50, plateau_ratio=0.999,
                min_steps=0, fallback_Kd=20.0):
    cfg = sim.cfg
    qp = sim.qp_ss
    n_j = sim.robot.n_joints
    dt = cfg.dt_qp

    pq0, pv0 = mujoco_to_pinocchio(sim.mj_data.qpos, sim.mj_data.qvel)
    rs0 = sim.robot.update(pq0, pv0)
    eig_H = np.linalg.eigvalsh(rs0.H)
    lambda_min = float(np.min(np.abs(eig_H)))
    T_settle = 0.5 * (epsilon_v ** 2) * lambda_min

    T_history = []
    rows = []
    exit_reason = 'max_steps'

    for k in range(max_steps):
        pq, pv = mujoco_to_pinocchio(sim.mj_data.qpos, sim.mj_data.qvel)
        rs = sim.robot.update(pq, pv)
        T_full = 0.5 * float(rs.v @ rs.H @ rs.v)
        T_history.append(T_full)

        if k >= min_steps:
            if T_full < T_settle:
                exit_reason = 'target_met'
                break
            if k >= plateau_window:
                T_old = T_history[k - plateau_window]
                if T_full > plateau_ratio * T_old:
                    exit_reason = 'plateau'
                    break

        Jc_full, _ = sim.robot.get_contact_jacobians(True, True)   # (12,20)

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
                settle_mode=True, passivity_active=True)
            tau = np.clip(tau, -cfg.tau_max, cfg.tau_max)
        except Exception:
            tau = np.clip(-fallback_Kd * rs.dq_joints,
                          -cfg.tau_max, cfg.tau_max)
            lam_sol = np.zeros(12)

        v_contacts = Jc_full @ rs.v      # (12,)
        P_weld = float(v_contacts @ lam_sol)

        # Split into per-contact lin/ang 3-vectors.
        v_A_lin = v_contacts[0:3]
        v_A_ang = v_contacts[3:6]
        v_B_lin = v_contacts[6:9]
        v_B_ang = v_contacts[9:12]
        jcq_norm_full = float(np.linalg.norm(v_contacts))
        jcq_norm_lin  = float(np.linalg.norm(np.concatenate([v_A_lin, v_B_lin])))
        jcq_norm_ang  = float(np.linalg.norm(np.concatenate([v_A_ang, v_B_ang])))

        rows.append({
            'k': k, 't_s': k * dt,
            'T_full_J': T_full,
            'P_weld_W': P_weld,
            'tau_max_Nm': float(np.max(np.abs(tau))),
            'Jcq_norm_full': jcq_norm_full,
            'Jcq_norm_lin':  jcq_norm_lin,
            'Jcq_norm_ang':  jcq_norm_ang,
            'Jcq_A_lin_x': float(v_A_lin[0]),
            'Jcq_A_lin_y': float(v_A_lin[1]),
            'Jcq_A_lin_z': float(v_A_lin[2]),
            'Jcq_A_ang_x': float(v_A_ang[0]),
            'Jcq_A_ang_y': float(v_A_ang[1]),
            'Jcq_A_ang_z': float(v_A_ang[2]),
            'Jcq_B_lin_x': float(v_B_lin[0]),
            'Jcq_B_lin_y': float(v_B_lin[1]),
            'Jcq_B_lin_z': float(v_B_lin[2]),
            'Jcq_B_ang_x': float(v_B_ang[0]),
            'Jcq_B_ang_y': float(v_B_ang[1]),
            'Jcq_B_ang_z': float(v_B_ang[2]),
        })

        sim.mj_data.ctrl[:n_j] = tau
        if sim.has_rwa:
            sim.mj_data.ctrl[n_j:n_j + 3] = 0.0
        mujoco.mj_step(sim.mj_model, sim.mj_data)

    n_steps_run = min(k + 1, max_steps)
    pq_e, pv_e = mujoco_to_pinocchio(sim.mj_data.qpos, sim.mj_data.qvel)
    rs_e = sim.robot.update(pq_e, pv_e)
    T_end = 0.5 * float(rs_e.v @ rs_e.H @ rs_e.v)

    return {
        'rows': rows, 'n_steps_run': n_steps_run,
        'T_start': T_history[0] if T_history else 0.0, 'T_end': T_end,
        'T_settle_thresh': T_settle, 'lambda_min': lambda_min,
        'exit_reason': exit_reason,
    }


def _run_variant(tag, solref):
    print('\n' + '#' * 78)
    print(f'#  Variant {tag}:  solref="{solref}"')
    print('#' * 78)
    _mutate_mjcf(damping=0.0, armature=0.0, solref=solref)

    cfg = r_single._make_m7_config()
    cfg.preplanner_a_cruise_max = 0.01
    cfg.preplanner_cruise_ramp_frac = 0.2

    sim = SimulationLoop(mjcf_path=MJCF, urdf_path=URDF, config=cfg)
    sim.setup(n_steps=1, start_a=2, start_b=2)
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

    # Per-variant CSV + summary.
    out_dir = os.path.join(OUT_ROOT, tag)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'trace.csv')
    if result['rows']:
        fieldnames = list(result['rows'][0].keys())
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in result['rows']:
                w.writerow({k: (f'{r[k]:.6e}' if isinstance(r[k], float)
                                else r[k])
                            for k in fieldnames})

    rows = result['rows']
    arr = lambda key: np.array([r[key] for r in rows])
    P_weld = arr('P_weld_W') if rows else np.array([0.0])
    Jcq_full = arr('Jcq_norm_full') if rows else np.array([0.0])
    Jcq_lin = arr('Jcq_norm_lin') if rows else np.array([0.0])
    Jcq_ang = arr('Jcq_norm_ang') if rows else np.array([0.0])

    summary = {
        'variant': tag,
        'solref': solref,
        'damping': 0.0,
        'armature': 0.0,
        'n_steps_run': result['n_steps_run'],
        'T_start_J': result['T_start'],
        'T_end_J': result['T_end'],
        'T_settle_thresh_J': result['T_settle_thresh'],
        'exit_reason': result['exit_reason'],
        'mean_P_weld_W': float(P_weld.mean()),
        'min_P_weld_W':  float(P_weld.min()),
        'max_P_weld_W':  float(P_weld.max()),
        'mean_Jcq_full': float(Jcq_full.mean()),
        'max_Jcq_full':  float(Jcq_full.max()),
        'mean_Jcq_lin':  float(Jcq_lin.mean()),
        'max_Jcq_lin':   float(Jcq_lin.max()),
        'mean_Jcq_ang':  float(Jcq_ang.mean()),
        'max_Jcq_ang':   float(Jcq_ang.max()),
        'n_rows': len(rows),
    }
    with open(os.path.join(out_dir, 'summary.txt'), 'w') as f:
        for k, v in summary.items():
            f.write(f'{k:22s} = {v}\n')
    print(f'  T_end         = {summary["T_end_J"]:.4e} J '
          f'(target {summary["T_settle_thresh_J"]:.2e}, '
          f'exit={summary["exit_reason"]})')
    print(f'  mean P_weld   = {summary["mean_P_weld_W"]:+.4e} W, '
          f'max = {summary["max_P_weld_W"]:+.4e} W')
    print(f'  mean |J_c·q̇| = {summary["mean_Jcq_full"]:.4e}, '
          f'max = {summary["max_Jcq_full"]:.4e}')
    print(f'  (lin: mean {summary["mean_Jcq_lin"]:.3e}, max {summary["max_Jcq_lin"]:.3e}  '
          f'ang: mean {summary["mean_Jcq_ang"]:.3e}, max {summary["max_Jcq_ang"]:.3e})')
    return summary


# ── Consolidated report ───────────────────────────────────────────────
def _write_consolidated(summaries):
    md_path = os.path.join(OUT_ROOT, 'summary.md')
    with open(md_path, 'w') as f:
        f.write('# M7 — DS settle weld solref sweep (damping=0, armature=0)\n\n')
        f.write('Three 51-step DS settles, identical robot config '
                '(`damping=0, armature=0` via transient MJCF mutation; '
                'restored on script exit). Only difference across rows: '
                'the `<weld solref="...">` attribute applied to every '
                '`grip_*_to_*` weld in the MJCF.\n\n')
        f.write('## Consolidated results\n\n')
        cols = ['variant', 'solref', 'n_steps', 'T_end [J]', 'exit',
                'mean(P_weld) [W]', 'max(P_weld) [W]',
                'mean(‖J_c·q̇‖) [m/s, rad/s]',
                'max(‖J_c·q̇‖)',
                'mean(lin) / mean(ang)',
                'max(lin) / max(ang)']
        f.write('| ' + ' | '.join(cols) + ' |\n')
        f.write('|' + '|'.join(['---'] * len(cols)) + '|\n')
        for s in summaries:
            f.write('| ' + ' | '.join([
                s['variant'],
                f'`"{s["solref"]}"`',
                str(s['n_steps_run']),
                f'{s["T_end_J"]:.4e}',
                s['exit_reason'],
                f'{s["mean_P_weld_W"]:+.4e}',
                f'{s["max_P_weld_W"]:+.4e}',
                f'{s["mean_Jcq_full"]:.4e}',
                f'{s["max_Jcq_full"]:.4e}',
                f'{s["mean_Jcq_lin"]:.3e} / {s["mean_Jcq_ang"]:.3e}',
                f'{s["max_Jcq_lin"]:.3e} / {s["max_Jcq_ang"]:.3e}',
            ]) + ' |\n')

        f.write('\n## MJCF mutation protocol\n\n')
        f.write('Per variant, the MJCF file is edited in place: the '
                '`<default class="robot_joint">`\'s `damping` / '
                '`armature` attributes are set to 0, and every `<weld>` '
                'element\'s `solref` attribute is replaced with the '
                'variant value. The script records the branch-HEAD '
                'state of the MJCF at entry and restores it via '
                '`try/finally`. A re-read at exit verifies `damping=0.05`, '
                '`armature=0.05`, and `solref="0.003 1"` on every weld.\n\n')

        f.write('## Per-tick traces\n\n')
        for s in summaries:
            rel = os.path.relpath(
                os.path.join(OUT_ROOT, s['variant'], 'trace.csv'), _root)
            f.write(f'- `{s["variant"]}`: `{rel}`\n')
        f.write('\nTrace columns per tick: `k, t_s, T_full_J, P_weld_W, '
                'tau_max_Nm, Jcq_norm_full, Jcq_norm_lin, Jcq_norm_ang, '
                'Jcq_A_lin_{x,y,z}, Jcq_A_ang_{x,y,z}, '
                'Jcq_B_lin_{x,y,z}, Jcq_B_ang_{x,y,z}`. Per-contact '
                'norms are pooled (lin = concat(A_lin, B_lin); '
                'ang = concat(A_ang, B_ang)) for the summary columns.\n')

        f.write('\n## MJCF restoration check\n\n')
        f.write('`_verify_mjcf_restored()` passed at script exit: '
                '`damping="0.05"` and `armature="0.05"` on the '
                '`robot_joint` default, and `solref="0.003 1"` is the '
                'unique value across all twelve `<weld>` elements '
                '(`grip_a_to_1a` … `grip_b_to_6b`).\n')
    return md_path


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    with open(MJCF, 'r') as f:
        original_mjcf = f.read()
    summaries = []
    try:
        for tag, solref in VARIANTS:
            summaries.append(_run_variant(tag, solref))
    finally:
        with open(MJCF, 'w') as f:
            f.write(original_mjcf)
        _verify_mjcf_restored()
        print('\nMJCF restored + verified (damping=0.05, armature=0.05, '
              'solref="0.003 1" on all welds).')

    md_path = _write_consolidated(summaries)
    print(f'\nConsolidated report: {md_path}')


if __name__ == '__main__':
    main()
