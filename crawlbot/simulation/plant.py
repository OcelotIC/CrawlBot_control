"""
MujocoPlant — the simulated robot + platform, and nothing else.

Everything that WRITES MuJoCo state lives here: stepping the physics, toggling
the gripper welds, the inelastic dock-impact velocity projection, the actuator
command, and the diagnostic arm-joint lock. The controller never touches
``mj_data`` directly; it gets a measurement from ``sensors.py`` and hands back
torques through ``apply_joint_torques`` / ``apply_wheel_torques``.

This is the seam a ROS 2 port cuts along: under ROS the plant is either this
class behind a bridge node, or the real hardware — in both cases the controller
side sees only measurements in and commands out.

Behaviour is byte-identical to the pre-extraction ``sim_loop.py`` blocks
(refactor/sim-loop-split, extraction 1): every block was moved verbatim, in order, with the
same numpy expressions, so floating-point operation order is unchanged.
"""

import numpy as np

try:
    import mujoco
except ImportError:
    mujoco = None


class MujocoPlant:
    """Owns ``MjModel`` / ``MjData``; the only writer of MuJoCo state."""

    def __init__(self, mjcf_path: str, dt: float):
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = dt
        # Detect RWA model (3 reaction wheels)
        rw_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'rw_x')
        self.has_rwa = rw_jid >= 0
        # Number of arm joints driven through ctrl[:n_joints]; set by the
        # owner once the Pinocchio model is loaded (the MJCF carries the
        # wheel actuators too, so nu alone does not give it).
        self.n_joints = 0
        # (arm, anchor_idx) -> equality id, and site name -> site id.
        self.weld_map = {}
        self.site_ids = {}

    # ── Physics ──────────────────────────────────────────────────────────

    def forward(self):
        mujoco.mj_forward(self.model, self.data)

    def step(self, lock_arm_joints: bool = False):
        """Advance the physics by one ``dt``.

        ``lock_arm_joints`` is the diagnostic arm-joint lock
        (``SimulationLoop._diag_lock_arm_joints``): zero ``qvel`` of the arm
        joints after the step, to measure the contact/weld/MJ baseline drift
        with the robot "frozen". Off on the canonical. NB the DS passivity
        loop has never applied it, even with the hook on — kept as found.
        """
        mujoco.mj_step(self.model, self.data)
        if lock_arm_joints:
            # Re-freeze arm joints after the physics step. The arm
            # joints live in the tail of qvel immediately after the
            # structure(6) + RWA(3 if present) + torso(6) block.
            off_rw = 3 if self.has_rwa else 0
            arm_v_start = 6 + off_rw + 6
            arm_v_end = arm_v_start + self.n_joints
            self.data.qvel[arm_v_start:arm_v_end] = 0.0

    def set_state(self, qpos, qvel):
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel

    # ── Actuators ────────────────────────────────────────────────────────

    def apply_joint_torques(self, tau):
        self.data.ctrl[:self.n_joints] = tau

    def apply_wheel_torques(self, tau_w):
        self.data.ctrl[self.n_joints:self.n_joints + 3] = tau_w

    def zero_ctrl(self):
        self.data.ctrl[:] = 0.0

    # ── Welds ────────────────────────────────────────────────────────────

    def build_weld_map(self):
        self.weld_map = {}
        for i in range(self.model.neq):
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_EQUALITY, i)
            if name and name.startswith('grip_'):
                parts = name.split('_to_')
                arm = parts[0].split('_')[1]
                anchor_idx = int(parts[1][0]) - 1
                self.weld_map[(arm, anchor_idx)] = i

    def deactivate_all_welds(self):
        for eq_id in range(self.model.neq):
            self.data.eq_active[eq_id] = 0

    def activate_weld(self, arm, anchor_idx):
        key = (arm, anchor_idx)
        if key in self.weld_map:
            self.data.eq_active[self.weld_map[key]] = 1

    def deactivate_weld(self, arm, anchor_idx):
        key = (arm, anchor_idx)
        if key in self.weld_map:
            self.data.eq_active[self.weld_map[key]] = 0

    def cache_site_ids(self):
        for name in ['gripper_a', 'gripper_b']:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            self.site_ids[name] = sid
        # Cache ALL anchor sites present in the model (not a hardcoded
        # count). A hardcoded range(5) silently dropped anchor_6, so the
        # dock gate read d=inf and never fired on any step targeting the
        # 6th anchor (e.g. step 4 -> dock_timeout despite the EE reaching
        # the anchor). Discovering the count from the model keeps the
        # cache consistent with the gait/MJCF for any anchor grid.
        for arm in ['a', 'b']:
            idx = 0
            while True:
                name = f'anchor_{idx+1}{arm}'
                sid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_SITE, name)
                if sid < 0:
                    break
                self.site_ids[name] = sid
                idx += 1

    # ── Dock impact ──────────────────────────────────────────────────────

    def apply_dock_impact(self, verbose: bool):
        """Inelastic impact: FULL-DOF momentum-consistent velocity jump.

        Called right after the swing weld is activated (and ``forward()``).
        Fix A (dock-leak Part 3). The previous map projected in
        the robot-only Pinocchio space (structure as fixed
        base) and wrote back only qvel[6+off:] via the
        setup-only pinocchio_to_mujoco conversion — a one-sided
        impulse that injected ~0.2 N·m·s of spurious system
        angular momentum at the docks (dock-leak Parts 1–2:
        the conversion also drops the structure-coupling terms
        since it assumes v_struct≈0, false at a dock). Replace
        with the full-DOF projection validated offline in
        Part-2 A.1 (leak 0.3565→0.0011 over the 5 docks),
        computed ENTIRELY in MuJoCo DOF (no Pinocchio round-
        trip) and written back to ALL qvel, so the constraint
        impulse is a full action-reaction pair and conserves
        subtree_angmom to the O(gap·f) couple residual. The
        weld is already active, so every active
        gripper↔anchor relation is in the constraint set.
        """
        nv = self.model.nv
        M_full = np.zeros((nv, nv))
        mujoco.mj_fullM(self.model, self.data, M_full)
        # Full-DOF weld constraint Jacobian: relative twist of
        # each welded gripper↔anchor site pair over ALL qvel
        # (incl. structure base + wheels) — the same relative-
        # site weld relation Part-2 A.1 validated. Active welds
        # are read from eq_active (exact welded pairs).
        inv_weld = {eq: key for key, eq in self.weld_map.items()}
        rows = []
        for eq_id in range(self.model.neq):
            if not self.data.eq_active[eq_id] or eq_id not in inv_weld:
                continue
            arm, a_idx = inv_weld[eq_id]
            gsid = self.site_ids.get(f'gripper_{arm}', -1)
            asid = self.site_ids.get(f'anchor_{a_idx + 1}{arm}', -1)
            if gsid < 0 or asid < 0:
                continue
            jpg = np.zeros((3, nv)); jrg = np.zeros((3, nv))
            jpa = np.zeros((3, nv)); jra = np.zeros((3, nv))
            mujoco.mj_jacSite(self.model, self.data, jpg, jrg, gsid)
            mujoco.mj_jacSite(self.model, self.data, jpa, jra, asid)
            rows.append(jpg - jpa)   # relative linear twist
            rows.append(jrg - jra)   # relative angular twist
        if rows:
            J = np.vstack(rows)        # (6·n_weld) × nv
            v_minus = self.data.qvel.copy()
            v_pre = J @ v_minus
            MiJT = np.linalg.solve(M_full, J.T)
            impulse = np.linalg.solve(J @ MiJT, v_pre)
            v_plus = v_minus - MiJT @ impulse
            if verbose:
                mujoco.mj_subtreeVel(self.model, self.data)
                H_pre = float(np.linalg.norm(
                    self.data.subtree_angmom[0]))
            self.data.qvel[:] = v_plus   # write back ALL DOFs
            mujoco.mj_forward(self.model, self.data)
            if verbose:
                mujoco.mj_subtreeVel(self.model, self.data)
                H_post = float(np.linalg.norm(
                    self.data.subtree_angmom[0]))
                print(f"  Impact(fullDOF): ||dv||="
                      f"{np.linalg.norm(v_plus - v_minus):.4f}, "
                      f"||J·v-||={np.linalg.norm(v_pre):.4f}, "
                      f"|H_sys| {H_pre:.4f}->{H_post:.4f}")
