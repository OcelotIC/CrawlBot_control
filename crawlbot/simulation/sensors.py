"""
SensorSuite — every MuJoCo READ the control stack consumes.

The counterpart of ``plant.py``: the plant is the only writer of MuJoCo state,
this is the only reader of it on the command path. Each method is one physical
measurement channel, named after what a real SpaceServicer would publish:

    joint_state()          nav + gyros + encoders -> Pinocchio (q, v),
                           relative twist in platform frame (spec §3,
                           "Relative Twist and Pinocchio Interface")
    omega_struct()         platform gyro  ω_s  (body frame)
    struct_quat()          platform attitude (star tracker), wxyz
    struct_pos()           platform position (navigation), world frame
    wheel_momentum()       wheel tachometers x I_w  ->  h_w
    gripper_distance()     gripper <-> anchor proximity (dock sensor)
    weld_relative_twist()  gripper <-> anchor relative twist (dock sensor)
    anchor_sites_world()   handhold map, read once at setup

Under ROS 2 each channel becomes a topic (sensor_msgs/JointState, Imu, ...);
here they read ``plant.data`` directly. Nothing in this class writes MuJoCo
state — the dock-sensor Jacobians use ``mj_jacSite``, a pure query.

Behaviour is byte-identical to the pre-extraction reads in ``sim_loop.py``
(refactor/sim-loop-split, extraction 2): each method returns exactly the
expression it replaced, copies where the original copied and views where the
original took a view (``omega_struct_view``).
"""

import numpy as np

try:
    import mujoco
except ImportError:
    mujoco = None

from crawlbot.core.state_conversions import mujoco_to_pinocchio
from crawlbot.planning.contact_scheduler import read_anchors_from_mujoco


class SensorSuite:
    """Read-only measurement channels over a ``MujocoPlant``."""

    def __init__(self, plant, rwa_I_w):
        self._plant = plant
        self._rwa_I_w = rwa_I_w

    # ── Proprioception + platform navigation ─────────────────────────────

    def joint_state(self):
        """Pinocchio ``(q, v)``: torso pose and twist relative to the
        platform, in platform frame, plus arm joints (``mujoco_to_pinocchio``)."""
        d = self._plant.data
        return mujoco_to_pinocchio(d.qpos, d.qvel)

    def omega_struct(self):
        """Platform angular velocity ω_s, body frame (copy)."""
        return self._plant.data.qvel[3:6].copy()

    def omega_struct_view(self):
        """ω_s as a live VIEW into ``qvel`` — used only where the original
        code took a view (the ``_step`` AOCS block). It changes after the
        next ``plant.step()``; do not hold on to it."""
        return self._plant.data.qvel[3:6]

    def struct_quat(self):
        """Platform attitude quaternion, wxyz (MuJoCo convention), copy."""
        return self._plant.data.qpos[3:7].copy()

    def struct_pos(self):
        """Platform position, world frame, copy."""
        return self._plant.data.qpos[0:3].copy()

    def wheel_momentum(self):
        """Reaction-wheel momentum h_w = I_w · ω_wheels [N·m·s]."""
        return (self._rwa_I_w * self._plant.data.qvel[6:9]).copy()

    def raw_state(self):
        """(qpos, qvel) copies — for snapshots / offline rendering only."""
        d = self._plant.data
        return d.qpos.copy(), d.qvel.copy()

    # ── Docking sensor ───────────────────────────────────────────────────

    def gripper_distance(self, arm, anchor_idx):
        site_ids = self._plant.site_ids
        grip_sid = site_ids.get(f'gripper_{arm}', -1)
        anch_sid = site_ids.get(f'anchor_{anchor_idx+1}{arm}', -1)
        if grip_sid < 0 or anch_sid < 0:
            return np.inf
        d = self._plant.data
        return float(np.linalg.norm(
            d.site_xpos[grip_sid] - d.site_xpos[anch_sid]))

    def weld_relative_twist(self, arm, anchor_idx):
        """6-D weld-relative twist Jc·v⁻ for one gripper↔anchor pair.

        Returns the 6-vector [relative linear; relative angular] velocity
        of the swing gripper site w.r.t. its target anchor site, over ALL
        qvel (incl. structure base + wheels). This is the exact quantity
        the inelastic impact map projects out at dock.

        Reuses the Fix-A relative-site weld-Jacobian construction
        (``MujocoPlant.apply_dock_impact``): J = [jpg-jpa; jrg-jra] via
        ``mj_jacSite`` for the gripper/anchor sites, twist = J @ qvel.
        Unlike the impact (which iterates ``eq_active``), the gate
        runs BEFORE the weld engages, so the pair is addressed by name.
        Assumes the caller has already run ``plant.forward()`` so the site
        Jacobians are current. Returns zeros if either site is missing.
        """
        model, data = self._plant.model, self._plant.data
        site_ids = self._plant.site_ids
        nv = model.nv
        gsid = site_ids.get(f'gripper_{arm}', -1)
        asid = site_ids.get(f'anchor_{anchor_idx + 1}{arm}', -1)
        if gsid < 0 or asid < 0:
            return np.zeros(6)
        jpg = np.zeros((3, nv)); jrg = np.zeros((3, nv))
        jpa = np.zeros((3, nv)); jra = np.zeros((3, nv))
        mujoco.mj_jacSite(model, data, jpg, jrg, gsid)
        mujoco.mj_jacSite(model, data, jpa, jra, asid)
        J = np.vstack([jpg - jpa, jrg - jra])      # (6, nv)
        return J @ data.qvel                       # 6-D weld-relative twist

    # ── Handhold map (setup) ─────────────────────────────────────────────

    def anchor_sites_world(self):
        """World-frame anchor-site positions ``(anchors_a, anchors_b)``."""
        return read_anchors_from_mujoco(self._plant.model, self._plant.data)
