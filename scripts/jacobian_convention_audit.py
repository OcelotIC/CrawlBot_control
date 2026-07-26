"""Jacobian / velocity convention audit at d0_a0 settle tick k=10.

Re-runs the first 11 ticks of the inter-step DS settle (same
deterministic trajectory as the prior ground-truth report) with
damping=0, armature=0 via transient MJCF mutation. At k=10 captures:

  • Pinocchio J_c (LOCAL_WORLD_ALIGNED), v_pin, λ_qp from the QP
  • MuJoCo efc_J (12 × 29), efc_force, qvel, mj_data.energy
  • mj_model.dof_jntid, mj_model.jnt_type for column-ordering audit
  • mj_model.eq_data and mj_model.eq_type for weld-constraint ordering

Computes 8 power variants (2×4 swap combinations per side) and
compares Pinocchio T = ½v·H·v to MuJoCo's mj_data.energy[0].

Writes Misc/runs/M7_settle_diag/jacobian_convention_audit.md.
MJCF restored via try/finally and verified by re-read.
"""
from __future__ import annotations

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
OUT  = os.path.join(_root, 'Misc', 'runs', 'M7_settle_diag',
                    'jacobian_convention_audit.md')
TARGET_K = 10

ROBOT_JOINT_RE = re.compile(
    r'(<default class="robot_joint">\s*\n\s*<joint damping=")[^"]+'
    r'(" armature=")[^"]+(")')


def _mutate_mjcf(damping, armature):
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
    assert m
    body = text[m.start():m.end()]
    assert 'damping="0.05"' in body and 'armature="0.05"' in body, body
    return True


def _run_and_capture():
    cfg = r_single._make_m7_config()
    cfg.preplanner_a_cruise_max = 0.01
    cfg.preplanner_cruise_ramp_frac = 0.2

    sim = SimulationLoop(mjcf_path=MJCF, urdf_path=URDF, config=cfg)
    sim.setup(n_steps=1, start_a=2, start_b=2)
    cc_ds = sim.sched.contact_config_at(sim.plan.t_start[1] + 0.1)

    qp = sim.qp_ss
    n_j = sim.robot.n_joints
    mj_nv = sim.mj_model.nv
    dt = cfg.dt_qp
    snap = {}

    # Pre-capture model-level metadata (doesn't change per tick).
    mj_model = sim.mj_model
    snap['mj_nv'] = mj_nv
    snap['mj_nq'] = mj_model.nq
    snap['mj_neq'] = mj_model.neq
    # Column meaning: for each v-index, which joint does it belong to?
    dof_jntid = np.array(mj_model.dof_jntid).copy()
    snap['dof_jntid'] = dof_jntid
    snap['jnt_names'] = [mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j)
                         for j in range(mj_model.njnt)]
    snap['jnt_type'] = np.array(mj_model.jnt_type).copy()
    snap['jnt_dofadr'] = np.array(mj_model.jnt_dofadr).copy()
    # Equality constraints
    snap['eq_names'] = [mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_EQUALITY, e)
                        for e in range(mj_model.neq)]
    snap['eq_type']  = np.array(mj_model.eq_type).copy()

    for k in range(TARGET_K + 1):
        pq, pv = mujoco_to_pinocchio(sim.mj_data.qpos, sim.mj_data.qvel)
        rs = sim.robot.update(pq, pv)
        Jc_pin, _ = sim.robot.get_contact_jacobians(True, True)

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

        if k == TARGET_K:
            snap['k'] = k
            snap['t_s'] = k * dt
            snap['v_pin'] = rs.v.copy()
            snap['H_pin'] = rs.H.copy()
            snap['Jc_pin'] = Jc_pin.copy()
            snap['lam_qp'] = lam_qp.copy()
            snap['tau_qp'] = tau.copy()
            snap['T_pin'] = 0.5 * float(rs.v @ rs.H @ rs.v)
            # Pre-step MuJoCo state. mj_energyPos + mj_energyVel
            # explicitly populate d.energy = [potential, kinetic] (MuJoCo
            # convention: index 0 = potential, index 1 = kinetic).
            mujoco.mj_forward(sim.mj_model, sim.mj_data)
            mujoco.mj_energyPos(sim.mj_model, sim.mj_data)
            mujoco.mj_energyVel(sim.mj_model, sim.mj_data)
            snap['mj_qpos'] = sim.mj_data.qpos.copy()
            snap['mj_qvel'] = sim.mj_data.qvel.copy()
            snap['mj_energy_pre'] = sim.mj_data.energy.copy()  # (2,)
            # Manual cross-check: M from full mass, then ½v·M·v.
            M_mj = np.zeros((mj_nv, mj_nv))
            mujoco.mj_fullM(sim.mj_model, M_mj, sim.mj_data.qM)
            snap['T_mj_manual_pre'] = 0.5 * float(
                sim.mj_data.qvel @ M_mj @ sim.mj_data.qvel)

        sim.mj_data.ctrl[:n_j] = tau
        if sim.has_rwa:
            sim.mj_data.ctrl[n_j:n_j + 3] = 0.0
        mujoco.mj_step(sim.mj_model, sim.mj_data)

        if k == TARGET_K:
            nefc = int(sim.mj_data.nefc)
            snap['nefc'] = nefc
            snap['mj_efc_J'] = sim.mj_data.efc_J[:nefc * mj_nv] \
                                .reshape(nefc, mj_nv).copy()
            snap['mj_efc_force'] = sim.mj_data.efc_force[:nefc].copy()
            mujoco.mj_energyPos(sim.mj_model, sim.mj_data)
            mujoco.mj_energyVel(sim.mj_model, sim.mj_data)
            snap['mj_energy_post'] = sim.mj_data.energy.copy()
            M_mj_post = np.zeros((mj_nv, mj_nv))
            mujoco.mj_fullM(sim.mj_model, M_mj_post, sim.mj_data.qM)
            snap['T_mj_manual_post'] = 0.5 * float(
                sim.mj_data.qvel @ M_mj_post @ sim.mj_data.qvel)
            break

    return snap


def _swap_base6(v):
    """Swap linear (0:3) and angular (3:6) blocks of the first 6 DOFs."""
    out = v.copy()
    out[0:3], out[3:6] = v[3:6].copy(), v[0:3].copy()
    return out


def _swap_base_two_free(v_mj):
    """Swap lin/ang for BOTH free joints in MuJoCo qvel (structure 0:6 and
    torso 9:15). Arm DOFs (indices 15:29) and wheel DOFs (6:9) unchanged."""
    out = v_mj.copy()
    out[0:3], out[3:6] = v_mj[3:6].copy(), v_mj[0:3].copy()
    out[9:12], out[12:15] = v_mj[12:15].copy(), v_mj[9:12].copy()
    return out


def _swap_lam_per_contact(lam):
    """Swap [0:3]<->[3:6] within each 6-tuple of a stacked wrench vector."""
    out = lam.copy()
    n = lam.size // 6
    for i in range(n):
        base = 6 * i
        out[base:base+3] = lam[base+3:base+6].copy()
        out[base+3:base+6] = lam[base:base+3].copy()
    return out


def _fmt(v, prec=6, max_n=None):
    if max_n is not None and len(v) > max_n:
        return ('[' + ', '.join(f'{x:+.{prec}e}' for x in v[:max_n])
                + f', ... ({len(v)} total)]')
    return '[' + ', '.join(f'{x:+.{prec}e}' for x in v) + ']'


def main():
    with open(MJCF, 'r') as f:
        original = f.read()
    snap = None
    try:
        _mutate_mjcf(0.0, 0.0)
        snap = _run_and_capture()
    finally:
        with open(MJCF, 'w') as f:
            f.write(original)
        _verify_restored()

    # ── Item 3: eight P_weld variants ───────────────────────────────
    v_pin = snap['v_pin']
    Jc_pin = snap['Jc_pin']
    lam_qp = snap['lam_qp']

    v_mj = snap['mj_qvel']
    efc_J = snap['mj_efc_J']
    efc_f = snap['mj_efc_force']

    def P_pin(v, lam):
        return float((Jc_pin @ v) @ lam)

    def P_mj(v, lam):
        return float((efc_J @ v) @ lam)

    pin_variants = {
        'a (as-is)':              P_pin(v_pin,            lam_qp),
        'b (swap v base lin/ang)':P_pin(_swap_base6(v_pin),lam_qp),
        'c (swap λ per contact)': P_pin(v_pin,            _swap_lam_per_contact(lam_qp)),
        'd (both swapped)':       P_pin(_swap_base6(v_pin),
                                        _swap_lam_per_contact(lam_qp)),
    }
    mj_variants = {
        'a (as-is)':              P_mj(v_mj,                    efc_f),
        'b (swap v base lin/ang)':P_mj(_swap_base_two_free(v_mj),efc_f),
        'c (swap λ per contact)': P_mj(v_mj,                    _swap_lam_per_contact(efc_f)),
        'd (both swapped)':       P_mj(_swap_base_two_free(v_mj),
                                       _swap_lam_per_contact(efc_f)),
    }

    # ── Item 4: kinetic energy cross-check ─────────────────────────
    # MuJoCo stores d.energy = [potential, kinetic] (header comment
    # in mujoco/mjdata.h). energy[1] is kinetic.
    T_pin = snap['T_pin']
    T_mj_pre  = float(snap['mj_energy_pre'][1])
    T_mj_post = float(snap['mj_energy_post'][1])
    U_mj_pre  = float(snap['mj_energy_pre'][0])
    U_mj_post = float(snap['mj_energy_post'][0])
    T_mj_manual_pre  = float(snap['T_mj_manual_pre'])
    T_mj_manual_post = float(snap['T_mj_manual_post'])

    # ── Write report ───────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        f.write('# M7 — Jacobian/velocity convention audit (d0_a0, k=10)\n\n')
        f.write(f't = {snap["t_s"]:.4f} s. MJCF mutated to (damping=0, '
                'armature=0) transiently and restored on exit.\n\n')

        f.write('## 1. Pinocchio side — convention (source inspection)\n\n')
        f.write('- `crawlbot/core/robot_interface.py:310-313` — '
                '`J_tool_a/b = pin.getFrameJacobian(model, data, fid, '
                '**pin.LOCAL_WORLD_ALIGNED**)`. Each tool Jacobian is '
                '(6, nv). Rows are `[v_lin(3), v_ang(3)]` (linear-first, '
                'axes aligned with Pinocchio\'s working "world" = the '
                'structure frame).\n')
        f.write('- `get_contact_jacobians(active_A, active_B)` '
                '(`robot_interface.py:421-443`) `np.vstack`s the two '
                'tool Jacobians in `[A, B]` order → shape `(12, 20)`.\n')
        f.write('- `crawlbot/core/state_conversions.py:43-104` — '
                '`mujoco_to_pinocchio` converts MuJoCo world-frame '
                'free-joint velocities to Pinocchio structure-frame, '
                'storing `pin_v[0:3] = v_torso_local_linear`, '
                '`pin_v[3:6] = omega_torso_local_angular`. '
                'Linear-first, structure-frame axes, at torso frame.\n')
        f.write('- λ layout (`crawlbot/solvers/wholebody_qp.py:340-355`): '
                'the dynamics equation `H·q̈ + C = B·τ + J_c^T·λ` uses '
                'the stacked `J_contacts` as-is; therefore λ has the '
                'same row ordering as `J_contacts` — `[force(3), '
                'torque(3)]` per contact, structure-frame axes.\n\n')

        f.write('## 2. MuJoCo side — convention (model + data)\n\n')
        # Free-joint and arm-joint column mapping
        jnt_names = snap['jnt_names']
        jnt_type  = snap['jnt_type']
        jnt_dofadr = snap['jnt_dofadr']
        f.write(f'`mj_model.nv = {snap["mj_nv"]}`, '
                f'`mj_model.nq = {snap["mj_nq"]}`, '
                f'`mj_model.neq = {snap["mj_neq"]}`. '
                f'At k={snap["k"]}, `mj_data.nefc = {snap["nefc"]}`.\n\n')
        f.write('Column (v-space) assignment from `mj_model.jnt_dofadr` '
                'and `jnt_type`:\n\n')
        f.write('| jid | name | type | dof_start | dof_count |\n')
        f.write('|---|---|---|---|---|\n')
        type_names = {
            mujoco.mjtJoint.mjJNT_FREE.value:  'free(6)',
            mujoco.mjtJoint.mjJNT_BALL.value:  'ball(3)',
            mujoco.mjtJoint.mjJNT_SLIDE.value: 'slide(1)',
            mujoco.mjtJoint.mjJNT_HINGE.value: 'hinge(1)',
        }
        for jid, (name, t, adr) in enumerate(
                zip(jnt_names, jnt_type, jnt_dofadr)):
            end = (jnt_dofadr[jid+1] if jid+1 < len(jnt_dofadr)
                   else snap['mj_nv'])
            f.write(f'| {jid} | `{name}` | {type_names.get(int(t), str(int(t)))}'
                    f' | {int(adr)} | {int(end - adr)} |\n')
        f.write('\nFor `mjJNT_FREE`, MuJoCo documents qvel ordering as '
                '`[lin_x, lin_y, lin_z, ang_x, ang_y, ang_z]` in the world '
                'frame (linear-first).\n\n')
        f.write('Equality-constraint types (for each of the 12 '
                '`<weld>` elements):\n\n')
        eq_names = snap['eq_names']
        eq_type  = snap['eq_type']
        eq_type_names = {
            mujoco.mjtEq.mjEQ_CONNECT.value:   'connect(3)',
            mujoco.mjtEq.mjEQ_WELD.value:      'weld(6)',
            mujoco.mjtEq.mjEQ_JOINT.value:     'joint(1)',
            mujoco.mjtEq.mjEQ_TENDON.value:    'tendon(1)',
        }
        f.write('| eid | name | type |\n|---|---|---|\n')
        for eid, (n, t) in enumerate(zip(eq_names, eq_type)):
            f.write(f'| {eid} | `{n}` | '
                    f'{eq_type_names.get(int(t), str(int(t)))} |\n')
        f.write('\nFor `mjEQ_WELD`, MuJoCo generates 6 `efc` rows per '
                'weld. Per MuJoCo source/docs, the efc row ordering for '
                'a weld is **[rotation(3), translation(3)]** — angular '
                'first, then positional. `efc_force` units per row are '
                'the Lagrange multipliers matching the corresponding '
                '`efc_J` row (in the frame MuJoCo used internally; see '
                'the cross-convention sweep below).\n\n')

        f.write('## 3. Eight P_weld variants\n\n')
        f.write('For each side, four swap variants of the inner product '
                '`(J · v) · λ`:\n')
        f.write('- **a (as-is)**: the 6-tuples enter in native ordering.\n'
                '- **b (swap v base lin/ang)**: for Pinocchio, swap '
                '`v_pin[0:3] ↔ v_pin[3:6]` (torso free-joint block). '
                'For MuJoCo, swap BOTH `qvel[0:3]↔[3:6]` (structure '
                'free-joint) and `qvel[9:12]↔[12:15]` (torso free-joint).\n'
                '- **c (swap λ per contact)**: within each 6-tuple of '
                '`λ_qp` (Pin) or `efc_force` (MJ), swap the first 3 '
                'components with the last 3.\n'
                '- **d (both swapped)**: apply both b and c.\n\n')
        f.write('| variant | Pin P_weld [W] | MuJoCo P_weld [W] | '
                'rel agreement |\n|---|---|---|---|\n')
        for key in pin_variants.keys():
            p_pin = pin_variants[key]
            p_mj  = mj_variants[key]
            if abs(p_pin) + abs(p_mj) > 0:
                rel = 2 * abs(p_pin - p_mj) / (abs(p_pin) + abs(p_mj))
            else:
                rel = 0.0
            f.write(f'| {key} | `{p_pin:+.6e}` | `{p_mj:+.6e}` | '
                    f'`{rel:.3e}` |\n')
        f.write('\nBest cross-side match (smallest relative disagreement):')
        best_key = min(pin_variants.keys(),
                       key=lambda k: (
                           2 * abs(pin_variants[k] - mj_variants[k])
                           / max(abs(pin_variants[k]) + abs(mj_variants[k]),
                                 1e-30)))
        f.write(f'\n\n- Variant `{best_key}` — Pin = '
                f'`{pin_variants[best_key]:+.6e}` W, '
                f'MJ = `{mj_variants[best_key]:+.6e}` W.\n\n')

        f.write('## 4. Kinetic energy cross-check\n\n')
        f.write('MuJoCo convention (per `mjdata.h`): `mj_data.energy = '
                '[potential, kinetic]`. `mj_energyPos` and `mj_energyVel` '
                'are called explicitly before reading.\n\n')
        f.write(f'- `T_pin = ½ v_pin^T H_pin v_pin` = `{T_pin:.6e}` J\n')
        f.write(f'- `mj_data.energy[1]` (pre-step, kinetic)  = `{T_mj_pre:.6e}` J\n')
        f.write(f'- `mj_data.energy[1]` (post-step, kinetic) = `{T_mj_post:.6e}` J\n')
        f.write(f'- `mj_data.energy[0]` (pre-step, potential) = `{U_mj_pre:.6e}` J\n')
        f.write(f'- `mj_data.energy[0]` (post-step, potential)= `{U_mj_post:.6e}` J\n')
        f.write(f'- manual `0.5·qvel·M·qvel` (pre-step)  = `{T_mj_manual_pre:.6e}` J\n')
        f.write(f'- manual `0.5·qvel·M·qvel` (post-step) = `{T_mj_manual_post:.6e}` J\n')
        denom = max(abs(T_pin), abs(T_mj_pre), 1e-30)
        rel = abs(T_pin - T_mj_pre) / denom
        f.write(f'- relative diff `|T_pin − T_mj_kin_pre| / max(|T_pin|, |T_mj_kin_pre|)` '
                f'= `{rel:.3e}`\n')
        f.write('\n## 5. MJCF restoration\n\n')
        f.write('Verified on exit: `damping="0.05"`, `armature="0.05"`.\n')

    # Console summary
    print()
    print('=' * 78)
    print(f'Jacobian/velocity convention audit — d0_a0 k={snap["k"]} t={snap["t_s"]:.3f}s')
    print('=' * 78)
    print(f'  T_pin = {T_pin:+.6e} J   '
          f'mj kinetic (energy[1]) pre = {T_mj_pre:+.6e} J   post = {T_mj_post:+.6e} J')
    print(f'                                mj potential (energy[0]) pre = '
          f'{U_mj_pre:+.3e} J   post = {U_mj_post:+.3e} J')
    print(f'                                manual ½qvel·M·qvel pre = '
          f'{T_mj_manual_pre:+.6e} J   post = {T_mj_manual_post:+.6e} J')
    print(f'  {"variant":<28s} {"Pin P_weld [W]":>18s} {"MJ  P_weld [W]":>18s}  rel')
    print('  ' + '-' * 72)
    for key in pin_variants:
        p_pin = pin_variants[key]
        p_mj  = mj_variants[key]
        rel = (2 * abs(p_pin - p_mj) / (abs(p_pin) + abs(p_mj))
               if abs(p_pin) + abs(p_mj) > 0 else 0)
        print(f'  {key:<28s} {p_pin:+18.6e} {p_mj:+18.6e}  {rel:.3e}')
    print()
    print(f'Report: {OUT}')


if __name__ == '__main__':
    main()
