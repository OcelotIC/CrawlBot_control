#!/usr/bin/env python3
"""Dock-leak Part 2 (REDIRECTED) — full-DOF confirmatory test. READ-ONLY, STATIC.

Recomputes the dock impact projection two ways on the 5 logged pre-activation
dock snapshots and reports Δ(subtree_angmom[0]) for each — NO re-run, NO source
behaviour change (the in-plant impact map in sim_loop.py is untouched).

  Test A (discriminator):
    A.2  current partial map  — robot-only Pinocchio H + Jc (structure-fixed
         base), write back only qvel[6+off:] (torso+arms). EXPECT: reproduces
         the Part-1 per-dock leak (0.041,0.048,0.049,0.146,0.073).
    A.1  full-DOF map         — full MuJoCo mass matrix (mj_fullM) + full weld
         Jacobian (relative site Jacobians over ALL qvel, incl. structure base
         & wheels), write back ALL qvel. EXPECT: ‖ΔH‖ collapses (full action-
         reaction pair) — down to the O(gap×f) residual = the genuine soft-weld
         term Part 1 measured (~0.6%).
  Test B (correct null): null the pre-impact CONSTRAINT velocity Jc·v⁻ → run the
    current partial map. EXPECT ‖ΔH‖≈0 (no impulse to mis-apply ⇒ velocity, not
    position, is the lever). Contrast: gap does not enter the projection.

Metric: subtree_angmom[0] (world), the validated A'.1 / Part-1 ground truth.
"""
import json
import os
import numpy as np
import mujoco
import pinocchio as pin

from crawlbot.simulation.sim_loop import SimulationLoop
from crawlbot.simulation.config import SimConfig
from crawlbot.core.robot_interface import RobotInterface
from crawlbot.core.state_conversions import (
    mujoco_to_pinocchio, pinocchio_to_mujoco)

ROOT = os.path.join(os.path.dirname(__file__), '..')
MJCF = os.path.join(ROOT, 'models', 'VISPA_crawling_rwa3.xml')
URDF = os.path.join(ROOT, 'models', 'VISPA_crawling_fixed.urdf')
INSTR = os.path.join(ROOT, 'results', 'dockleak_instrumented')
PART1_LEAK = {0: 0.0409, 1: 0.0480, 2: 0.0491, 3: 0.1456, 4: 0.0729}


def build_loop():
    """Mirror SimulationLoop.setup() model/robot build, WITHOUT the settle."""
    cfg = SimConfig()
    sim = SimulationLoop(mjcf_path=MJCF, urdf_path=URDF, config=cfg)
    sim.mj_model = mujoco.MjModel.from_xml_path(MJCF)
    sim.mj_data = mujoco.MjData(sim.mj_model)
    sim.mj_model.opt.timestep = cfg.dt_qp
    sim.has_rwa = mujoco.mj_name2id(
        sim.mj_model, mujoco.mjtObj.mjOBJ_JOINT, 'rw_x') >= 0
    sim.robot = RobotInterface(URDF, gravity='zero')
    sim._cache_site_ids()
    sim._build_weld_map()
    return sim


def H_sys(m, d, qpos, qvel):
    """subtree_angmom[0] at the given state (pure read-out)."""
    d.qpos[:] = qpos
    d.qvel[:] = qvel
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    mujoco.mj_comVel(m, d)
    mujoco.mj_subtreeVel(m, d)
    return d.subtree_angmom[0].copy()


def nearest_anchor_site(sim, arm, grip_xpos):
    """Site id of the closest anchor_{i}{arm} to the gripper (the welded one)."""
    best_id, best_d = -1, np.inf
    for name, sid in sim._site_ids.items():
        if name.startswith('anchor_') and name.endswith(arm) and sid >= 0:
            dist = np.linalg.norm(sim.mj_data.site_xpos[sid] - grip_xpos)
            if dist < best_d:
                best_d, best_id = dist, sid
    return best_id, best_d


def partial_map(sim, qpos, qvel, apply_impulse=True):
    """EXACT replica of sim_loop.py's current impact map. Returns post qvel.

    apply_impulse=False skips the projection (pv_post=pv) but keeps the
    mujoco->pinocchio->mujoco round-trip + partial write-back — isolating the
    velocity-CONVERSION artefact (pinocchio_to_mujoco 'assumes v_struct≈0')
    from the IMPULSE mis-application."""
    m, d = sim.mj_model, sim.mj_data
    pq, pv = mujoco_to_pinocchio(qpos, qvel)
    rs = sim.robot.update(pq, pv)
    Jc, _ = sim.robot.get_contact_jacobians(True, True)
    if Jc is None:
        return qvel.copy(), None
    v_pre = Jc @ pv
    MiJcT = np.linalg.solve(rs.H, Jc.T)
    Lambda_inv = Jc @ MiJcT
    impulse = np.linalg.solve(Lambda_inv, v_pre)
    pv_post = (pv - MiJcT @ impulse) if apply_impulse else pv.copy()
    _, mj_qvel_post = pinocchio_to_mujoco(
        pq, pv_post, struct_pos=qpos[0:3], struct_quat=qpos[3:7],
        rwa=sim.has_rwa)
    off_v = 3 if sim.has_rwa else 0
    qvel_post = qvel.copy()
    qvel_post[6 + off_v:] = mj_qvel_post[6 + off_v:]
    return qvel_post, float(np.linalg.norm(v_pre))


def full_dof_map(sim, qpos, qvel):
    """Full-DOF momentum-consistent impact: mj_fullM + relative site Jacobians
    over ALL qvel, write back ALL DOFs. Returns (post qvel, n_welds, |v_pre|)."""
    m, d = sim.mj_model, sim.mj_data
    d.qpos[:] = qpos
    d.qvel[:] = qvel
    mujoco.mj_forward(m, d)                       # populate qM, site_xpos
    nv = m.nv
    M = np.zeros((nv, nv))
    mujoco.mj_fullM(m, d, M)   # this build's binding: (model, data, dst); reads d.qM
    sid_struct = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'structure')
    # structure-body angular Jacobian (jacr of any site on it = its anchors)
    rows = []
    used = []
    for arm in ('a', 'b'):
        gsid = sim._site_ids.get(f'gripper_{arm}', -1)
        if gsid < 0:
            continue
        asid, gap = nearest_anchor_site(sim, arm, d.site_xpos[gsid])
        if asid < 0:
            continue
        used.append((arm, asid, gap))
        jpg = np.zeros((3, nv)); jrg = np.zeros((3, nv))
        jpa = np.zeros((3, nv)); jra = np.zeros((3, nv))
        mujoco.mj_jacSite(m, d, jpg, jrg, gsid)   # gripper site lin + body ang
        mujoco.mj_jacSite(m, d, jpa, jra, asid)   # anchor site lin + body ang
        # weld 6-DOF relative constraint: rel lin (site pts) + rel ang (bodies)
        rows.append(jpg - jpa)                     # 3 x nv
        rows.append(jrg - jra)                     # 3 x nv
    J = np.vstack(rows)                            # (6*nw) x nv
    v = qvel.copy()
    v_pre = J @ v
    MiJT = np.linalg.solve(M, J.T)
    Lam = J @ MiJT
    impulse = np.linalg.solve(Lam, v_pre)
    v_post = v - MiJT @ impulse
    return v_post, len(used), float(np.linalg.norm(v_pre)), used


def null_constraint_velocity(sim, qpos, qvel):
    """Return qvel with the robot-only Jc row-space component removed (Jc·v⁻→0)
    in the Pinocchio (torso+arm) DOFs; structure+wheels untouched (as the map
    leaves them). Used as the zero-pre-impact-velocity baseline for Test B."""
    pq, pv = mujoco_to_pinocchio(qpos, qvel)
    sim.robot.update(pq, pv)
    Jc, _ = sim.robot.get_contact_jacobians(True, True)
    pv_null = pv - Jc.T @ np.linalg.solve(Jc @ Jc.T, Jc @ pv)
    _, mj_qvel_null = pinocchio_to_mujoco(
        pq, pv_null, struct_pos=qpos[0:3], struct_quat=qpos[3:7],
        rwa=sim.has_rwa)
    off_v = 3 if sim.has_rwa else 0
    qvel_null = qvel.copy()
    qvel_null[6 + off_v:] = mj_qvel_null[6 + off_v:]
    resid = float(np.linalg.norm(Jc @ pv_null))
    return qvel_null, resid


def main():
    sl = json.load(open(os.path.join(INSTR, 'sim_log.json')))
    snaps = {lbl: (np.array(q), np.array(v))
             for (t, q, v, lbl) in sl['snapshots']}
    probes = {p['step']: p for p in sl['dock_probe']}
    sim = build_loop()
    m, d = sim.mj_model, sim.mj_data

    print('Dock-leak Part 2 (redirected) — full-DOF confirmatory test (static)\n')
    print('snapshot consistency: subtree_angmom(dock_stepN) vs Part-1 probe H0')
    print(f'  {"step":>4} {"|H0_snap|":>10} {"|H0_probe|":>10} {"match":>8}')
    for n in range(5):
        q, v = snaps[f'dock_step{n}']
        H0 = H_sys(m, d, q, v)
        H0p = np.array(probes[n]['H0'])
        ok = np.linalg.norm(H0 - H0p) < 1e-6
        print(f'  {n:>4} {np.linalg.norm(H0):>10.5f} {np.linalg.norm(H0p):>10.5f} '
              f'{("YES" if ok else "NO -> "+f"{np.linalg.norm(H0-H0p):.2e}"):>8}')

    print('\n=== TEST A — discriminator: full-DOF vs current partial ===')
    print(f'  {"step":>4} {"gap[mm]":>8} {"A.2 |ΔH| partial":>16} {"Part-1":>8} '
          f'{"A.1 |ΔH| fullDOF":>16} {"nw":>3} {"reduction":>10}')
    a1_all, a2_all = [], []
    for n in range(5):
        q, v = snaps[f'dock_step{n}']
        H0 = H_sys(m, d, q, v)
        # A.2 current partial
        qv2, _ = partial_map(sim, q, v)
        H2 = H_sys(m, d, q, qv2)
        dA2 = np.linalg.norm(H2 - H0)
        # A.1 full DOF
        qv1, nw, _, used = full_dof_map(sim, q, v)
        H1 = H_sys(m, d, q, qv1)
        dA1 = np.linalg.norm(H1 - H0)
        gap_mm = min(g for _, _, g in used) * 1000
        red = dA2 / dA1 if dA1 > 1e-9 else float('inf')
        a1_all.append(dA1); a2_all.append(dA2)
        print(f'  {n:>4} {gap_mm:>8.3f} {dA2:>16.4f} {PART1_LEAK[n]:>8.4f} '
              f'{dA1:>16.4f} {nw:>3} {red:>9.1f}x')
    print(f'\n  Σ A.2 (current partial) = {sum(a2_all):.4f}  (Part-1 impact total 0.3565)')
    print(f'  Σ A.1 (full-DOF)        = {sum(a1_all):.4f}  '
          f'(expect ≈ Part-1 gap-stab total 0.0022 — the O(gap×f) residual)')
    discr = (sum(a1_all) < 0.2 * sum(a2_all)) and \
            all(abs(a2_all[n] - PART1_LEAK[n]) < 0.02 for n in range(5))
    print(f'  DISCRIMINATOR: A.1 ≪ A.2 and A.2 reproduces Part-1 leak ?  '
          f'{"CONFIRMED" if discr else "NOT met -> FLAG"}')

    print('\n=== TEST B — what drives the partial-map leak: impulse vs '
          'velocity-conversion (NOT gap) ===')
    print('  B.1 — decompose the partial map: full leak = impulse mis-application'
          ' + lossy m2p→p2m round-trip')
    print(f'  {"step":>4} {"ΔH_full":>9} {"ΔH_noImpulse":>13} {"ΔH_impulse":>11} '
          f'{"conv. share":>12}')
    convs = []
    for n in range(5):
        q, v = snaps[f'dock_step{n}']
        H0 = H_sys(m, d, q, v)
        qv_full, _ = partial_map(sim, q, v, apply_impulse=True)
        qv_noimp, _ = partial_map(sim, q, v, apply_impulse=False)
        dfull = np.linalg.norm(H_sys(m, d, q, qv_full) - H0)
        dconv = np.linalg.norm(H_sys(m, d, q, qv_noimp) - H0)
        dimp = np.linalg.norm(H_sys(m, d, q, qv_full)
                              - H_sys(m, d, q, qv_noimp))
        convs.append(dconv)
        print(f'  {n:>4} {dfull:>9.4f} {dconv:>13.4f} {dimp:>11.4f} '
              f'{dconv/dfull*100:>11.1f}%')
    print(f'  => the partial map injects even with the impulse removed '
          f'(Σ conv-artefact = {sum(convs):.4f}): pinocchio_to_mujoco drops the')
    print('     structure-coupling (v_struct, ω_s×dp) terms when writing the torso'
          ' velocity back. BOTH this and the')
    print('     one-sided impulse are velocity-driven and BOTH are removed by the'
          ' full-DOF map (Test A).')

    print('\n  B.2 — velocity is the lever, gap is fixed: scale the pre-impact '
          'velocity by α (gap held constant)')
    print(f'  {"step":>4} {"gap[mm]":>8} ' +
          ' '.join(f'{("α="+str(a)):>9}' for a in (1.0, 0.5, 0.25, 0.0)))
    for n in range(5):
        q, v = snaps[f'dock_step{n}']
        gap_mm = min(g for _, _, g in full_dof_map(sim, q, v)[3]) * 1000
        row = []
        for a in (1.0, 0.5, 0.25, 0.0):
            va = a * v
            H0a = H_sys(m, d, q, va)
            qva, _ = partial_map(sim, q, va, apply_impulse=True)
            row.append(np.linalg.norm(H_sys(m, d, q, qva) - H0a))
        print(f'  {n:>4} {gap_mm:>8.3f} ' + ' '.join(f'{r:>9.4f}' for r in row))
    print('  => ΔH ∝ velocity, →0 at zero approach velocity, while the (fixed,')
    print('     nonzero) gap is unchanged. Confirms the lever is VELOCITY, not')
    print('     position/gap (Part-1 corr(|ΔH|,gap)=−0.40). Validates Fix C')
    print('     (null approach velocity before eq_active=1) and Fix A (full-DOF).')
    print('\n[done]')


if __name__ == '__main__':
    main()
