"""Validate the chosen sym + w_posture through the FULL
manipulability_config (Nelder-Mead) path — the path the runner uses
via sim_loop.setup() for the (start 2,2) pair. The sweep used a
fixed-seed solve_ik; this confirms the production path converges and
de-contorts with the same numbers.
"""
import sys
import time
import numpy as np
import pinocchio as pin
import mujoco

sys.path.insert(0, '/home/user/CrawlBot_control')
from crawlbot.core.robot_interface import RobotInterface
from crawlbot.core.ik import manipulability_config, _get_tool_frames, _arm_v_slice
from crawlbot.simulation.sim_loop import read_anchors_from_mujoco
from crawlbot.planning.contact_scheduler import ContactScheduler

MJCF = '/home/user/CrawlBot_control/models/VISPA_crawling_rwa3.xml'
URDF = '/home/user/CrawlBot_control/models/VISPA_crawling_fixed.urdf'
Z = np.array([0.0, 0.0, 1.0])
Q_SYM = np.array([
    0.297781,  1.526727,  0.842257,  1.147432,
    0.751390,  0.251893,  1.332349,
    -0.297781, -1.526727, -0.842257, -1.147432,
    -0.751390, -0.251893,  1.332349])
W = 0.2


def tilt_deg(q):
    qx, qy, qz, qw = q[3], q[4], q[5], q[6]
    R = pin.Quaternion(qw, qx, qy, qz).toRotationMatrix()
    return np.degrees(np.arccos(abs(np.clip(R[:, 2] @ Z, -1, 1))))


m = mujoco.MjModel.from_xml_path(MJCF)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
ma, mb = read_anchors_from_mujoco(m, d)
p0 = d.qpos[0:3].copy()
w0, x0, y0, z0 = d.qpos[3:7]
Rs0 = pin.Quaternion(w0, x0, y0, z0).toRotationMatrix()
aa = [Rs0.T @ (a - p0) for a in ma]
ab = [Rs0.T @ (b - p0) for b in mb]
sched = ContactScheduler(anchors_a=aa, anchors_b=ab, dt_ds=0.5, dt_ss=0.0)
Ta, Tb = sched.anchor_se3('a', 2), sched.anchor_se3('b', 2)

robot = RobotInterface(URDF, gravity='zero')
model = robot.model
data = model.createData()
fa, fb = _get_tool_frames(model)
sa_sl, sb_sl = _arm_v_slice(model, fa), _arm_v_slice(model, fb)


def report(tag, q, w):
    pin.forwardKinematics(model, data, q)
    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)
    Ja = pin.getFrameJacobian(model, data, fa, pin.LOCAL)[:, sa_sl]
    Jb = pin.getFrameJacobian(model, data, fb, pin.LOCAL)[:, sb_sl]
    sA = float(np.linalg.svd(Ja, compute_uv=False)[-1])
    sB = float(np.linalg.svd(Jb, compute_uv=False)[-1])
    ea = np.linalg.norm(pin.log6(data.oMf[fa].inverse() * Ta).vector)
    eb = np.linalg.norm(pin.log6(data.oMf[fb].inverse() * Tb).vector)
    print(f"{tag:14s} tilt={tilt_deg(q):5.2f}  "
          f"|q_arm|={np.linalg.norm(q[7:]):.3f}  "
          f"max|q|={np.degrees(np.max(np.abs(q[7:]))):5.1f}  "
          f"sigA={sA:.3f} sigB={sB:.3f} prod={sA*sB:.3e}  "
          f"toolerr={ea+eb:.2e}  w_yoshikawa={w:.3e}")


t0 = time.time()
q_leg, w_leg = manipulability_config(model, Ta, Tb)
print(f"[legacy   ] {time.time()-t0:.0f}s")
report("legacy", q_leg, w_leg)

t0 = time.time()
q_new, w_new = manipulability_config(model, Ta, Tb, level_axis=Z,
                                     q_nominal=Q_SYM, w_posture=W)
print(f"[lvl+sym  ] {time.time()-t0:.0f}s")
report("lvl+sym(w=.2)", q_new, w_new)
