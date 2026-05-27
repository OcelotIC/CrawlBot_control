"""Standalone q_nominal x w_posture sweep for the leveled startup IK.

Fast iteration on the (start 2,2) anchor pair. Compares the two
q_nominal candidates from the plan (zeros vs a mirror-symmetric
template built from the less-contorted arm of the leveled-only
solution) across w_posture in {0.02, 0.05, 0.1, 0.2}.

To keep the sweep tractable the per-cell solve uses solve_ik directly
on a fixed torso seed (the leveled-only solution's torso position),
i.e. it skips the outer Nelder-Mead. This isolates the *relative*
effect of q_nominal/w_posture on contortion + leveling + convergence
(the quantities the plan's selection criterion ranks). The chosen
winner is then re-validated with the full manipulability_config.

Run:
  MUJOCO_GL=osmesa PYTHONPATH=. python3 scripts/diag_qnominal_sweep.py
"""
import sys
import time
import itertools
import numpy as np
import pinocchio as pin
import mujoco

sys.path.insert(0, '/home/user/CrawlBot_control')
from crawlbot.core.robot_interface import RobotInterface
from crawlbot.core.ik import (
    solve_ik, manipulability_config, _get_tool_frames, _arm_v_slice,
)
from crawlbot.simulation.sim_loop import read_anchors_from_mujoco
from crawlbot.planning.contact_scheduler import ContactScheduler

MJCF = '/home/user/CrawlBot_control/models/VISPA_crawling_rwa3.xml'
URDF = '/home/user/CrawlBot_control/models/VISPA_crawling_fixed.urdf'
LEVEL_AXIS = np.array([0.0, 0.0, 1.0])


def _tilt_deg(q):
    """Angle (deg) between torso body-z and the level axis (0 = leveled)."""
    qx, qy, qz, qw = q[3], q[4], q[5], q[6]
    R = pin.Quaternion(qw, qx, qy, qz).toRotationMatrix()
    u = R[:, 2]
    c = abs(float(np.clip(u @ LEVEL_AXIS, -1.0, 1.0)))
    return np.degrees(np.arccos(c))


def _sigma_prod(model, data, q, fa, fb, sa_sl, sb_sl):
    pin.forwardKinematics(model, data, q)
    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)
    Ja = pin.getFrameJacobian(model, data, fa, pin.LOCAL)[:, sa_sl]
    Jb = pin.getFrameJacobian(model, data, fb, pin.LOCAL)[:, sb_sl]
    sa = float(np.linalg.svd(Ja, compute_uv=False)[-1])
    sb = float(np.linalg.svd(Jb, compute_uv=False)[-1])
    return sa, sb, sa * sb


def _tool_err(model, data, q, fa, fb, Ta, Tb):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    ea = pin.log6(data.oMf[fa].inverse() * Ta).vector
    eb = pin.log6(data.oMf[fb].inverse() * Tb).vector
    return np.linalg.norm(ea) + np.linalg.norm(eb)


def main():
    m = mujoco.MjModel.from_xml_path(MJCF)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    ma, mb = read_anchors_from_mujoco(m, d)
    p0 = d.qpos[0:3].copy()
    w, x, y, z = d.qpos[3:7]
    Rs0 = pin.Quaternion(w, x, y, z).toRotationMatrix()
    aa = [Rs0.T @ (a - p0) for a in ma]
    ab = [Rs0.T @ (b - p0) for b in mb]
    sched = ContactScheduler(anchors_a=aa, anchors_b=ab, dt_ds=0.5, dt_ss=0.0)
    Ta = sched.anchor_se3('a', 2)
    Tb = sched.anchor_se3('b', 2)

    robot = RobotInterface(URDF, gravity='zero')
    model = robot.model
    data = model.createData()
    fa, fb = _get_tool_frames(model)
    sa_sl = _arm_v_slice(model, fa)
    sb_sl = _arm_v_slice(model, fb)
    # config-space arm index ranges (v slice + free-flyer offset 1)
    qa = slice(sa_sl.start + 1, sa_sl.stop + 1)
    qb = slice(sb_sl.start + 1, sb_sl.stop + 1)
    targets = {fa: Ta, fb: Tb}

    # 1) leveled-only solution -> identify the less-contorted arm
    t0 = time.time()
    q_lvl, err_lvl = solve_ik(model, pin.neutral(model), targets,
                              max_iter=2000, level_axis=LEVEL_AXIS)
    t_lvl = time.time() - t0
    qA = q_lvl[qa].copy()
    qB = q_lvl[qb].copy()
    nA, nB = np.linalg.norm(qA), np.linalg.norm(qB)
    template = qB if nB <= nA else qA          # less-contorted arm
    tmpl_is_B = nB <= nA
    print(f"[leveled-only] err={err_lvl:.2e} tilt={_tilt_deg(q_lvl):.2f} "
          f"t={t_lvl:.0f}s")
    print(f"  arm A |q|={nA:.3f} ({np.degrees(np.max(np.abs(qA))):.0f}deg max)"
          f"  arm B |q|={nB:.3f} ({np.degrees(np.max(np.abs(qB))):.0f}deg max)"
          f"  -> template = arm {'B' if tmpl_is_B else 'A'}")

    # 2) brute-force the y-mirror sign pattern S so the template arm
    #    can be mirrored onto the other arm (anchors are y-symmetric).
    rng = np.random.default_rng(0)
    My = np.diag([1.0, -1.0, 1.0])
    best_S, best_res = None, np.inf
    test_thetas = [rng.uniform(-1.0, 1.0, 7) for _ in range(6)]
    for S in itertools.product([1.0, -1.0], repeat=7):
        S = np.array(S)
        res = 0.0
        for th in test_thetas:
            q = pin.neutral(model)
            q[qb] = th
            q[qa] = S * th
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            pa = data.oMf[fa].translation
            pb = data.oMf[fb].translation
            res += np.linalg.norm(pa - My @ pb)
        if res < best_res:
            best_res, best_S = res, S
    print(f"[mirror] S={best_S.astype(int).tolist()} residual={best_res:.3e}")

    # 3) build the mirror-symmetric q_nominal (both arms in the
    #    less-contorted shape: template on its own arm, S*template on
    #    the other).
    q_sym = np.zeros(14)
    if tmpl_is_B:
        q_sym[7:14] = template            # arm B keeps template
        q_sym[0:7] = best_S * template    # arm A = mirrored template
    else:
        q_sym[0:7] = template
        q_sym[7:14] = best_S * template
    print(f"[q_sym] |q|={np.linalg.norm(q_sym):.3f} "
          f"max={np.degrees(np.max(np.abs(q_sym))):.0f}deg")

    # 4) fast sweep: solve_ik on a fixed torso seed (leveled-only torso)
    seed = pin.neutral(model)
    seed[:3] = q_lvl[:3]
    cands = {'zeros': np.zeros(14), 'sym': q_sym}
    wlist = [0.02, 0.05, 0.1, 0.2, 0.5]
    print("\n cand   w     | tilt   |q_arm| max|q|  sigA   sigB   prod   "
          "err     t(s)")
    print(" " + "-" * 76)
    base_sa, base_sb, base_prod = _sigma_prod(
        model, data, q_lvl, fa, fb, sa_sl, sb_sl)
    print(f" LVLonly  -    | {_tilt_deg(q_lvl):5.2f}  "
          f"{np.linalg.norm(q_lvl[7:]):6.3f} "
          f"{np.degrees(np.max(np.abs(q_lvl[7:]))):5.0f}  "
          f"{base_sa:.3f}  {base_sb:.3f}  {base_prod:.3e}  "
          f"{err_lvl:.1e}  {t_lvl:.0f}")
    rows = []
    for name, qn in cands.items():
        for wv in wlist:
            t0 = time.time()
            q, e = solve_ik(model, seed, targets, max_iter=2000,
                            q_nominal=qn, w_posture=wv,
                            level_axis=LEVEL_AXIS)
            dt = time.time() - t0
            sA, sB, pr = _sigma_prod(model, data, q, fa, fb, sa_sl, sb_sl)
            te = _tool_err(model, data, q, fa, fb, Ta, Tb)
            row = dict(name=name, w=wv, tilt=_tilt_deg(q),
                       nq=np.linalg.norm(q[7:]),
                       mx=np.degrees(np.max(np.abs(q[7:]))),
                       sA=sA, sB=sB, pr=pr, err=te, t=dt)
            rows.append(row)
            print(f" {name:5s} {wv:4.2f} | {row['tilt']:5.2f}  "
                  f"{row['nq']:6.3f} {row['mx']:5.0f}  "
                  f"{sA:.3f}  {sB:.3f}  {pr:.3e}  {te:.1e}  {dt:.0f}")

    # 5) pick winner: converged (err<1mm-equiv ~1e-3) AND tilt<1deg AND
    #    sigma product not collapsed >2x vs leveled-only baseline,
    #    minimise |q_arm|.
    ok = [r for r in rows
          if r['err'] < 1e-3 and r['tilt'] < 1.0
          and r['pr'] > base_prod / 2.0]
    print("\n[selection] feasible cells:",
          [(r['name'], r['w']) for r in ok] or "NONE")
    if ok:
        win = min(ok, key=lambda r: r['nq'])
        print(f"[winner] q_nominal={win['name']} w_posture={win['w']}  "
              f"|q_arm|={win['nq']:.3f} (vs leveled-only "
              f"{np.linalg.norm(q_lvl[7:]):.3f}) prod={win['pr']:.3e}")
        if win['name'] == 'sym':
            print("[q_nominal vector for runner]")
            print(repr(np.round(q_sym, 6).tolist()))
    else:
        print("[winner] none meet criteria -> recommend leveling-only "
              "(q_nominal disabled)")


if __name__ == '__main__':
    main()
