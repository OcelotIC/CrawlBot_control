"""d0_a0 settle — MuJoCo constraint ground-truth at tick k=10.

Mutates the MJCF to (damping=0, armature=0) transiently; runs 11 ticks
of the inter-step DS settle mirroring SimulationLoop._run_ds_passivity_loop.
At tick k=10 captures, side by side:

  - Pinocchio J_c(q), QP τ_q, QP λ, Pinocchio v      (our side)
  - MuJoCo efc_J, efc_force, qfrc_actuator, etc.     (ground truth)

Writes Misc/runs/M7_settle_diag/mujoco_constraint_ground_truth.md.
MJCF restored via try/finally and verified by re-read.
"""
from __future__ import annotations

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
OUT  = os.path.join(_root, 'Misc', 'runs', 'M7_settle_diag',
                    'mujoco_constraint_ground_truth.md')

TARGET_K = 10

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


def _verify_restored():
    with open(MJCF, 'r') as f:
        text = f.read()
    m = ROBOT_JOINT_RE.search(text)
    assert m, 'robot_joint default missing'
    body = text[m.start():m.end()]
    assert 'damping="0.05"' in body and 'armature="0.05"' in body, \
        f'MJCF not restored: {body}'
    return True


def _run_and_capture():
    """Run 11 settle ticks, capture tick-10 state pre- and post-step.

    Returns a dict with every field the report needs.
    """
    cfg = r_single._make_m7_config()
    cfg.preplanner_a_cruise_max = 0.01
    cfg.preplanner_cruise_ramp_frac = 0.2

    sim = SimulationLoop(mjcf_path=MJCF, urdf_path=URDF, config=cfg)
    sim.setup(n_steps=1, start_a=2, start_b=2)
    cc_ds = sim.sched.contact_config_at(sim.plan.t_start[1] + 0.1)

    qp = sim.qp_ss
    n_j  = sim.robot.n_joints                 # 14
    mj_nv = sim.mj_model.nv                   # 29
    dt   = cfg.dt_qp

    snap = {'dt_qp': dt, 'n_j': n_j, 'mj_nv': mj_nv}

    for k in range(TARGET_K + 1):
        pq, pv = mujoco_to_pinocchio(sim.mj_data.qpos, sim.mj_data.qvel)
        rs = sim.robot.update(pq, pv)

        Jc_pin, _ = sim.robot.get_contact_jacobians(True, True)    # (12, 20)

        try:
            _, _, lam_qp, tau, _ = qp.solve(
                q_t=rs.q_torso, dq_t=rs.dq_torso,
                q=rs.q_joints, dq=rs.dq_joints,
                r_com_ref=rs.r_com, v_com_ref=np.zeros(3),
                lambda_ref=np.zeros(12), a_com_ff=np.zeros(3),
                H_robot=rs.H, C_robot=rs.C,
                J_com=rs.J_com, Jdot_dq_com=rs.Jdot_dq_com,
                contact_config=cc_ds,
                J_contacts=Jc_pin, Jdot_dq_contacts=np.zeros(12),
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
            tau = np.clip(-cfg.Kd_settle_damping * rs.dq_joints,
                          -cfg.tau_max, cfg.tau_max)
            lam_qp = np.zeros(12)

        # Pre-step capture for tick TARGET_K
        if k == TARGET_K:
            snap['k']         = k
            snap['t_s']       = k * dt
            # Pinocchio side
            snap['q_pin']     = rs.q.copy()
            snap['v_pin']     = rs.v.copy()
            snap['H_pin']     = rs.H.copy()
            snap['C_pin']     = rs.C.copy()
            snap['Jc_pin']    = Jc_pin.copy()
            snap['tau_qp']    = tau.copy()             # (14,)
            snap['lam_qp']    = lam_qp.copy()          # (12,)
            snap['T_pin_pre'] = 0.5 * float(rs.v @ rs.H @ rs.v)

            # MuJoCo pre-step state
            snap['mj_qpos_pre'] = sim.mj_data.qpos.copy()
            snap['mj_qvel_pre'] = sim.mj_data.qvel.copy()
            # ctrl hasn't been written yet — capture current
            snap['mj_ctrl_pre_write'] = sim.mj_data.ctrl.copy()
            snap['mj_qfrc_applied_pre'] = sim.mj_data.qfrc_applied.copy()

        # Write control (mirrors _run_ds_passivity_loop)
        sim.mj_data.ctrl[:n_j] = tau
        if sim.has_rwa:
            sim.mj_data.ctrl[n_j:n_j + 3] = 0.0

        if k == TARGET_K:
            snap['mj_ctrl_post_write'] = sim.mj_data.ctrl.copy()

        mujoco.mj_step(sim.mj_model, sim.mj_data)

        if k == TARGET_K:
            # Post-step MuJoCo state (efc_* was populated during mj_step)
            nefc = int(sim.mj_data.nefc)
            efc_J_flat = sim.mj_data.efc_J[:nefc * mj_nv].copy()
            efc_J = efc_J_flat.reshape(nefc, mj_nv)
            efc_force = sim.mj_data.efc_force[:nefc].copy()
            snap['nefc'] = nefc
            snap['mj_efc_J'] = efc_J
            snap['mj_efc_force'] = efc_force
            snap['mj_qfrc_actuator'] = sim.mj_data.qfrc_actuator.copy()
            snap['mj_qfrc_constraint'] = sim.mj_data.qfrc_constraint.copy()
            snap['mj_qfrc_applied_post'] = sim.mj_data.qfrc_applied.copy()
            snap['mj_qpos_post'] = sim.mj_data.qpos.copy()
            snap['mj_qvel_post'] = sim.mj_data.qvel.copy()

            # Pinocchio post-step (derived)
            pq_post, pv_post = mujoco_to_pinocchio(
                sim.mj_data.qpos, sim.mj_data.qvel)
            rs_post = sim.robot.update(pq_post, pv_post)
            snap['v_pin_post'] = rs_post.v.copy()
            snap['T_pin_post'] = 0.5 * float(rs_post.v @ rs_post.H @ rs_post.v)
            # MuJoCo-side kinetic energy via mj_fullM
            M_mj = np.zeros((mj_nv, mj_nv))
            mujoco.mj_fullM(sim.mj_model, M_mj, sim.mj_data.qM)
            snap['mj_M_post'] = M_mj.copy()
            snap['T_mj_post'] = 0.5 * float(
                snap['mj_qvel_post'] @ M_mj @ snap['mj_qvel_post'])
            break

    # Also compute T_mj_pre using the pre-step M. Go back and recompute
    # at that state: reset qpos/qvel to pre-step copies temporarily.
    tmp_qpos = sim.mj_data.qpos.copy()
    tmp_qvel = sim.mj_data.qvel.copy()
    sim.mj_data.qpos[:] = snap['mj_qpos_pre']
    sim.mj_data.qvel[:] = snap['mj_qvel_pre']
    mujoco.mj_forward(sim.mj_model, sim.mj_data)
    M_mj_pre = np.zeros((mj_nv, mj_nv))
    mujoco.mj_fullM(sim.mj_model, M_mj_pre, sim.mj_data.qM)
    snap['mj_M_pre'] = M_mj_pre.copy()
    snap['T_mj_pre'] = 0.5 * float(
        snap['mj_qvel_pre'] @ M_mj_pre @ snap['mj_qvel_pre'])
    sim.mj_data.qpos[:] = tmp_qpos
    sim.mj_data.qvel[:] = tmp_qvel

    return snap


def _fmt_vec(v, prec=6, max_n=None):
    if max_n is not None and len(v) > max_n:
        head = ', '.join(f'{x:+.{prec}e}' for x in v[:max_n])
        return f'[{head}, ... ({len(v)} elements)]'
    return '[' + ', '.join(f'{x:+.{prec}e}' for x in v) + ']'


def _fmt_mat(M, prec=6, max_rows=6, max_cols=10):
    r, c = M.shape
    lines = []
    nr = min(r, max_rows)
    nc = min(c, max_cols)
    for i in range(nr):
        lines.append('  '.join(f'{M[i, j]:+.{prec}e}' for j in range(nc))
                     + (f'  ... ({c} cols)' if c > max_cols else ''))
    if r > max_rows:
        lines.append(f'  ... ({r} rows)')
    return '\n'.join(lines)


def main():
    with open(MJCF, 'r') as f:
        original = f.read()
    snap = None
    try:
        _mutate_mjcf(damping=0.0, armature=0.0)
        snap = _run_and_capture()
    finally:
        with open(MJCF, 'w') as f:
            f.write(original)
        _verify_restored()
        print('MJCF restored + verified (damping=0.05, armature=0.05).')

    assert snap is not None
    k     = snap['k']
    t_s   = snap['t_s']
    dt    = snap['dt_qp']
    n_j   = snap['n_j']
    mj_nv = snap['mj_nv']

    # ── Item 3: power via both Jacobians ─────────────────────────────
    # Pinocchio side
    v_pin   = snap['v_pin']
    Jc_pin  = snap['Jc_pin']
    lam_qp  = snap['lam_qp']
    tau_qp  = snap['tau_qp']
    v_c_pin = Jc_pin @ v_pin                 # (12,)
    P_weld_pin = float(v_c_pin @ lam_qp)

    # MuJoCo side
    v_mj     = snap['mj_qvel_pre']
    efc_J    = snap['mj_efc_J']
    efc_force= snap['mj_efc_force']
    v_c_mj   = efc_J @ v_mj                  # (nefc,)
    P_weld_mj = float(v_c_mj @ efc_force)

    # ── Item 4: QP → MuJoCo handoff check ────────────────────────────
    ctrl_joints_post_write = snap['mj_ctrl_post_write'][:n_j]
    ctrl_matches_tau = bool(np.array_equal(ctrl_joints_post_write, tau_qp))
    ctrl_max_diff    = float(np.max(np.abs(ctrl_joints_post_write - tau_qp)))
    # qfrc_actuator (post-step) should equal gear × ctrl for motor actuators
    # with gain=1. The first 6 DOFs are the free-joint base; no actuators.
    # The next few are wheels (no commanded torque set to 0), then joints.
    qfrc_act = snap['mj_qfrc_actuator']
    qfrc_applied_post = snap['mj_qfrc_applied_post']

    # ── Item 5: power balance check ──────────────────────────────────
    dT_pin_over_dt = (snap['T_pin_post'] - snap['T_pin_pre']) / dt
    dT_mj_over_dt  = (snap['T_mj_post']  - snap['T_mj_pre'])  / dt
    # P_tau_input (Pinocchio convention): v_pin[6:] · τ_q
    P_tau_pin = float(v_pin[6:] @ tau_qp)
    # P_tau_input (MuJoCo convention): v_mj · qfrc_actuator
    P_tau_mj  = float(v_mj @ qfrc_act)
    identity_pin = P_tau_pin + P_weld_pin
    identity_mj  = P_tau_mj  + P_weld_mj

    # ── Write report ─────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        f.write('# M7 — MuJoCo constraint ground truth at d0_a0 settle, tick k=10\n\n')
        f.write(f't = {t_s:.4f} s, dt = {dt:.3f} s. MJCF mutated to '
                '(damping=0, armature=0) and restored on script exit.\n\n')

        f.write('## Shapes\n\n')
        f.write(f'- `mj_model.nv` = {mj_nv}\n')
        f.write(f'- `mj_model.neq` = 12 (welds defined)\n')
        f.write(f'- `mj_data.nefc` at tick {k} = {snap["nefc"]}\n')
        f.write(f'- Pinocchio `v` = {v_pin.shape}\n')
        f.write(f'- Pinocchio `J_c` = {Jc_pin.shape}\n\n')

        f.write('## 1. MuJoCo `efc_J` and `efc_force` (ground truth)\n\n')
        f.write('`mj_data.efc_J` reshaped to `(nefc, nv)`. First 6 rows × 10 cols:\n\n')
        f.write('```\n' + _fmt_mat(efc_J, prec=4) + '\n```\n\n')
        f.write(f'`mj_data.efc_force` (length {snap["nefc"]}):\n\n')
        f.write(f'`{_fmt_vec(efc_force)}`\n\n')

        f.write('## 2. Pinocchio-based `J_c` at the same `q`\n\n')
        f.write('First 6 rows × 10 cols (shape `(12, 20)`):\n\n')
        f.write('```\n' + _fmt_mat(Jc_pin, prec=4) + '\n```\n\n')

        f.write('## 3. Power check — both Jacobians\n\n')
        f.write('Pinocchio-side:\n\n')
        f.write(f'- `J_c_pin · v_pin`  = `{_fmt_vec(v_c_pin)}`\n')
        f.write(f'- `λ_qp`             = `{_fmt_vec(lam_qp)}`\n')
        f.write(f'- **`P_weld_pin = (J_c_pin · v_pin)^T · λ_qp = {P_weld_pin:+.6e} W`**\n\n')
        f.write('MuJoCo-side:\n\n')
        f.write(f'- `J_c_mj · v_mj`    = `{_fmt_vec(v_c_mj)}`\n')
        f.write(f'- `efc_force`        = `{_fmt_vec(efc_force)}`\n')
        f.write(f'- **`P_weld_mj = (J_c_mj · v_mj)^T · efc_force = {P_weld_mj:+.6e} W`**\n\n')

        f.write('## 4. QP → MuJoCo handoff check\n\n')
        f.write(f'- `τ_qp` (14)                         = `{_fmt_vec(tau_qp)}`\n')
        f.write(f'- `mj_data.ctrl[:14]` (post-write)    = `{_fmt_vec(ctrl_joints_post_write)}`\n')
        f.write(f'- **exact equality? `{ctrl_matches_tau}`**  '
                f'(max abs diff = {ctrl_max_diff:.3e})\n\n')
        f.write(f'- `mj_data.qfrc_applied` (pre-step, 29)  = `{_fmt_vec(snap["mj_qfrc_applied_pre"], max_n=12)}`\n')
        f.write(f'- `mj_data.qfrc_applied` (post-step, 29) = `{_fmt_vec(qfrc_applied_post, max_n=12)}`\n')
        f.write(f'- all-zero? pre: `{bool(np.all(snap["mj_qfrc_applied_pre"] == 0))}`, '
                f'post: `{bool(np.all(qfrc_applied_post == 0))}`\n\n')
        f.write(f'- `mj_data.qfrc_actuator` (29) = `{_fmt_vec(qfrc_act, max_n=14)}`\n')
        f.write(f'- `mj_data.qfrc_constraint` (29) = '
                f'`{_fmt_vec(snap["mj_qfrc_constraint"], max_n=14)}`\n\n')

        f.write('## 5. Numerical power balance at tick 10\n\n')
        f.write('Pinocchio side (using `H_pin`, `v_pin`, `τ_qp`, `λ_qp`):\n\n')
        f.write(f'- `T_pin_pre` = `{snap["T_pin_pre"]:.6e}` J\n')
        f.write(f'- `T_pin_post`= `{snap["T_pin_post"]:.6e}` J\n')
        f.write(f'- `ΔT_pin / dt` = `{dT_pin_over_dt:+.6e}` W\n')
        f.write(f'- `P_tau_pin = v_pin[6:] · τ_qp` = `{P_tau_pin:+.6e}` W\n')
        f.write(f'- `P_weld_pin`                    = `{P_weld_pin:+.6e}` W\n')
        f.write(f'- `P_tau_pin + P_weld_pin`        = `{identity_pin:+.6e}` W\n')
        f.write(f'- identity residual = `ΔT_pin/dt − (P_tau_pin + P_weld_pin)` '
                f'= `{dT_pin_over_dt - identity_pin:+.6e}` W\n\n')
        f.write('MuJoCo side (using `M_mj`, `qvel_mj`, `qfrc_actuator`, `efc_force`):\n\n')
        f.write(f'- `T_mj_pre`  = `{snap["T_mj_pre"]:.6e}` J\n')
        f.write(f'- `T_mj_post` = `{snap["T_mj_post"]:.6e}` J\n')
        f.write(f'- `ΔT_mj / dt` = `{dT_mj_over_dt:+.6e}` W\n')
        f.write(f'- `P_tau_mj = v_mj · qfrc_actuator` = `{P_tau_mj:+.6e}` W\n')
        f.write(f'- `P_weld_mj`                        = `{P_weld_mj:+.6e}` W\n')
        f.write(f'- `P_tau_mj + P_weld_mj`             = `{identity_mj:+.6e}` W\n')
        f.write(f'- identity residual = `ΔT_mj/dt − (P_tau_mj + P_weld_mj)` '
                f'= `{dT_mj_over_dt - identity_mj:+.6e}` W\n\n')

        f.write('## Auxiliary — raw numbers\n\n')
        f.write(f'Pinocchio `v` (20): `{_fmt_vec(v_pin)}`\n\n')
        f.write(f'MuJoCo `qvel_pre` (29): `{_fmt_vec(v_mj)}`\n\n')
        f.write(f'MuJoCo `qvel_post` (29): `{_fmt_vec(snap["mj_qvel_post"])}`\n\n')

    # Console summary
    print()
    print('=' * 78)
    print(f'MuJoCo constraint ground truth — d0_a0 settle, tick k={k} '
          f'(t={t_s:.3f} s)')
    print('=' * 78)
    print(f'  nefc                                 = {snap["nefc"]}')
    print(f'  P_weld_pin = (J_c_pin·v_pin)^T·λ_qp = {P_weld_pin:+.6e} W')
    print(f'  P_weld_mj  = (efc_J·v_mj )^T·efc_f  = {P_weld_mj:+.6e} W')
    print(f'  ctrl[:14] == τ_qp ?                  = {ctrl_matches_tau}  (Δmax {ctrl_max_diff:.2e})')
    print(f'  P_tau_pin  (v_pin_j · τ_qp)          = {P_tau_pin:+.6e} W')
    print(f'  P_tau_mj   (v_mj · qfrc_actuator)    = {P_tau_mj:+.6e} W')
    print(f'  ΔT_pin / dt                          = {dT_pin_over_dt:+.6e} W')
    print(f'  ΔT_mj  / dt                          = {dT_mj_over_dt:+.6e} W')
    print(f'  Pin identity residual                = {dT_pin_over_dt - identity_pin:+.6e} W')
    print(f'  MJ  identity residual                = {dT_mj_over_dt - identity_mj:+.6e} W')
    print()
    print(f'Report: {OUT}')


if __name__ == '__main__':
    main()
