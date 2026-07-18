# PORT-AUDIT — what it takes to swap in a new robot

**Read-only modularity audit.** Scope: swap the current VISPA dual-arm crawler for a different
robot (new URDF + meshes, different DOF count, different gripper). No code was changed and no
simulation was run to produce this. Every finding is `file:line` with the verbatim source. Findings
are classified into five bins (A–E); a port checklist follows.

Conventions used below: **PLANT** = the MuJoCo model the physics runs on; **CONTROLLER** = the
Pinocchio model the controller computes on. They are **two separate files of different scope**.

---

## BIN A — Model assets & loading

### a1. The two robot descriptions and where they load

| role | file | scope | dims | load site (verbatim) |
|---|---|---|---|---|
| **PLANT** (MuJoCo) | `models/VISPA_crawling_rwa3.xml` | structure(7110 kg) + **3 reaction wheels** + torso(40 kg) + 2×7-DOF arms | nq=31 / nv=29 / nu=17 | `sim_loop.py:208` `self.mj_model = mujoco.MjModel.from_xml_path(self.mjcf_path)` |
| **CONTROLLER** (Pinocchio) | `models/VISPA_crawling_fixed.urdf` | **robot only**: torso `Link_0` + 2×7-DOF arms (no structure, no wheels) | nq=21 / nv=20 (free-flyer) | `robot_interface.py:161` `self.model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())`, constructed at `sim_loop.py:241-242` `self.robot = RobotInterface(self.urdf_path, gravity='zero')` |

- Two descriptions, not one. The MJCF DOF layout is documented in its own header
  (`VISPA_crawling_rwa3.xml:12-19`): `qpos[0:7]` structure freejoint, `qpos[7:10]` 3 wheels,
  `qpos[10:17]` torso freejoint, `qpos[17:31]` 14 arm joints.
- The URDF is robot-only and mesh-based: `VISPA_crawling_fixed.urdf:15-16` "*Each arm is now 7-DOF →
  total actuated joints = 14 (Pinocchio nq = 21, nv = 20 with free-flyer base)*"; meshes at
  `VISPA_crawling_fixed.urdf:27` `<mesh filename="meshes/torso.stl"/>` (MJCF uses primitive geoms
  only). Corroborated: `docs/force_estimator_note.md:330` "*Le modèle Pinocchio (URDF) ne contient
  que le robot (torso + 2 bras).*"
- Model **paths** are not centralized — each runner re-declares them as module constants:
  `scripts/diag_cooperative_arms.py:49` `MJCF = os.path.join(_root, 'models', 'VISPA_crawling_rwa3.xml')`
  and `scripts/diag_cooperative_arms.py:497` `URDF = os.path.join(_root, 'models', 'VISPA_crawling_fixed.urdf')`
  (same pair duplicated in `run_m4_baseline.py:32`, `bisect_qp_cascade.py:49`, `diag_pure_pd.py:40`,
  `diag_torso_tracking.py:40`, `diag_platform_rotation.py:41`, `diag_loopfree_mapping_step2.py:39`,
  `diag_step2_bypass_off.py:299`, `test_force_estimator.py:48`, `lutze_baseline/sim_lutze.py:432`, …).

### a2. Quantities that MUST be kept consistent by hand between MJCF and URDF

There is no shared source — every one of these is written into **both** files (or into one file and a
code literal) and will silently disagree if only one is edited:

1. **Link masses.** e.g. torso `mass="40.0"` (`VISPA_crawling_rwa3.xml:142`) vs `<mass value="40.0"/>`
   (`VISPA_crawling_fixed.urdf:36`); `Link_1` `mass="2.328"` (`VISPA_crawling_rwa3.xml:154`) vs
   `<mass value="2.328"/>` (`VISPA_crawling_fixed.urdf:53`); every arm link.
2. **Link inertias.** torso `fullinertia="0.17980580 0.17982988 0.24241079 0 0 0"`
   (`VISPA_crawling_rwa3.xml:143`) vs `ixx="0.17980580" iyy="0.17982988" izz="0.24241079"`
   (`VISPA_crawling_fixed.urdf:37-39`); every arm link.
3. **Joint names with `_a`/`_b` suffix.** The controller derives arm membership *only* from the name
   suffix: `robot_interface.py:39-42` `if name.endswith('_a'): arm_a_joints.append(i) elif
   name.endswith('_b'): arm_b_joints.append(i)`. MJCF joints `Joint_1_a`…`Joint_6_a` + `Joint_swivel_a`
   (`VISPA_crawling_rwa3.xml:153-211`) must match URDF joint names exactly.
4. **Frame names looked up by string.** `tool_a`, `tool_b` (`robot_interface.py:204-205`
   `self.frame_tool_a = self.model.getFrameId("tool_a")`), torso `Link_0` (`robot_interface.py:206`
   `self.frame_torso = self.model.getFrameId("Link_0")`). Both files must define these.
5. **Joint / actuator ORDER.** The MJCF actuator order must equal the Pinocchio joint order so the
   controller's τ vector lines up element-for-element: `VISPA_crawling_rwa3.xml:296-300` "*Order must
   match Pinocchio joint order so `ctrl[:n_joints]` and the controller's (n_joints,) tau vector line
   up element-for-element*"; actuators at `VISPA_crawling_rwa3.xml:304-318`.
6. **Armature (rotor inertia).** MJCF sets `armature="0.05"` per arm joint
   (`VISPA_crawling_rwa3.xml:50`); the URDF has no armature field, so it is re-installed in code to
   match: `robot_interface.py:198-200` `_arm[slices['joints_v']] = 0.05; self.model.armature = _arm`.
   A different rotor inertia must be edited in **both** the MJCF and this literal.
7. **Torso mass.** A dedicated override exists because the descriptions can disagree:
   `robot_interface.py:164` "*Override torso mass if specified (URDF value may be incorrect)*"
   (`torso_mass` param, `robot_interface.py:156,165-171`); and the plant value is hard-asserted:
   `sim_loop.py:218` `assert abs(self.mj_model.body_mass[tid] - 40.0) < 1.0`.
8. **Gravity.** Both zeroed: `VISPA_crawling_rwa3.xml:33` `gravity="0 0 0"`; `robot_interface.py:174-175`
   `if gravity == 'zero' or gravity == 'micro': self.model.gravity = pin.Motion.Zero()`.

### a3. Scene attachment + contact/weld constraints

- **Both** torso and structure are free-floating (no fixed base): torso `<freejoint name="root"/>`
  (`VISPA_crawling_rwa3.xml:141`); structure `<freejoint name="structure_free"/>`
  (`VISPA_crawling_rwa3.xml:81`).
- Grasps are MuJoCo **weld equality constraints** between a gripper site and an anchor site, all 12
  pre-declared and toggled at runtime: `VISPA_crawling_rwa3.xml:329-345`
  `<weld name="grip_a_to_1a" site1="gripper_a" site2="anchor_1a" solref="0.003 1" active="false"/>` …
  `grip_b_to_6b`; start pair active (`grip_a_to_3a`:333, `grip_b_to_3b`:341, `active="true"`).
- Activation logic parses the weld names and flips `eq_active`:
  - `sim_loop.py:1199-1208` `_build_weld_map`: `if name and name.startswith('grip_'): parts =
    name.split('_to_'); arm = parts[0].split('_')[1]; anchor_idx = int(parts[1][0]) - 1`.
  - `sim_loop.py:1210-1217` `_deactivate_all_welds` / `_activate_weld` set
    `self.mj_data.eq_active[...] = 0|1`; called at init `sim_loop.py:369-372`.
- ⚠ The anchor-index parser reads a **single character** — `int(parts[1][0]) - 1`
  (`sim_loop.py:1207`) — so anchor indices ≥ 10 are mis-parsed.

---

## BIN B — Hardcoded names & index slicing

### b1. Hardcoded frame/body/site/joint/equality names

| kind | name(s) | file:line |
|---|---|---|
| Pinocchio frame | `tool_a`, `tool_b` | `robot_interface.py:204-205`; `ik.py:22` `return model.getFrameId("tool_a"), model.getFrameId("tool_b")` |
| Pinocchio frame | `Link_0` (torso) | `robot_interface.py:206` |
| Pinocchio joint suffix | `_a` / `_b` | `robot_interface.py:39-42`; raises if absent `robot_interface.py:44-46` "*Could not detect arm joints from URDF names (expected Joint_*_a and Joint_*_b)*" |
| MuJoCo joint | `rw_x` (RWA presence) | `sim_loop.py:213` `rw_jid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, 'rw_x')` |
| MuJoCo body | `torso` | `sim_loop.py:217` |
| MuJoCo body | `structure` | `sim_loop.py:222` |
| MuJoCo site | `gripper_a`, `gripper_b` | `sim_loop.py:1225` |
| MuJoCo site | `anchor_{i}a` / `anchor_{i}b` | `contact_scheduler.py:83-88` (loop `range(1,20)`); `sim_loop.py:1237` `name = f'anchor_{idx+1}{arm}'` |
| MuJoCo equality | prefix `grip_`, `_to_` split | `sim_loop.py:1204-1207` |
| Pinocchio joint idx | torso = joint 1 | `com_to_torso_mapping.py:35` `TORSO_JOINT_IDX = 1` |

### b2. Index slices / DOF-count assumptions

- **MJCF↔Pinocchio bridge** (`state_conversions.py`) — the two layouts differ and are bridged by
  hand: MuJoCo `[struct(7), rwa(3), torso(7), joints(N)]` (`state_conversions.py:11-15,29-34`) →
  Pinocchio `[torso(7), joints(N)]` in the structure frame (`:51`). RWA presence is inferred by
  **parity**: `state_conversions.py:58` `rwa = (len(mj_qpos) - 14) % 2 == 1` — assumes base = 14 and
  an **even** total arm-DOF; an odd arm-DOF count misdetects the wheel block.
- **Structure pose/twist** by fixed slice (structure is MuJoCo body 0, freejoint first):
  `sim_loop.py:229` `self._struct_quat_init = self.mj_data.qpos[3:7].copy()`; `sim_loop.py:234-235`
  `qpos[0:3]`, `qpos[3:7]`; also `sim_loop.py:360-361`; struct ω `qvel[3:6]`
  (`sim_loop.py:717,816,3471`).
- **RWA wheel velocities** at fixed slice `qvel[6:9]` (`sim_loop.py:709` `cfg.rwa_I_w *
  self.mj_data.qvel[6:9]`; `sim_loop.py:759`) — assumes structure(6 nv) + exactly 3 wheels.
- **Wheel torque write** `ctrl[n_joints:n_joints+3]` (`sim_loop.py:3455`
  `self.mj_data.ctrl[self.robot.n_joints:self.robot.n_joints + 3] = tau_w_cmd`) and `tau_w_cmd =
  np.zeros(3)` (`sim_loop.py:3454`) — exactly 3 wheels.
- **Arm-block offset** `off_rw = 3 if self.has_rwa else 0; arm_v_start = 6 + off_rw + 6`
  (`sim_loop.py:3466-3467`).
- **Joint torque write** `ctrl[:n_joints]` (`sim_loop.py:557,801,3278`); Pinocchio torque
  `tau_applied[6:6 + self.robot.n_joints] = tau` (`sim_loop.py:3478`).
- **Free-flyer base slices** (correct for any arm count, but assume a single 6-DOF free-flyer torso):
  `robot_interface.py:365-366` `q[0:7]`, `v[0:6]`; `:425` `J_base = J_full[:, :6]`; `:427-428`
  `H[:6,:6]`, `H[:6, arm_slice]`; `:314` `L_com = h_centroidal[3:6]`; ik base cols `ik.py:197,491`
  `J[:, :6]` / `J[:, :3]`.
- **Stale comment**: `sim_loop.py:13` "*AOCS wheel torque -> ctrl[12:15]*" (actual slice is now
  `ctrl[14:17]` at n_joints=14).
- DOF-generic by design (no literal): `n_j = self.robot.n_joints` (`sim_loop.py:529,685`); joint
  detection derives all slices from the model (`robot_interface.py:29-70,180-189`).

### b3. Literals `7` / `14` / model-mirroring constants

- `wholebody_qp.py:206` `tau_max: np.ndarray = field(default_factory=lambda: 50.0 * np.ones(14))`
  — hardcoded 14 (overridden by `sim_loop.py:1140`, but the config default is 14-specific).
- Structure-inertia **fallback literal** `np.array([597.0, 1493.0, 1777.0])` (`sim_loop.py:225`),
  mirroring MJCF `fullinertia="597 1493 1777 …"` (`VISPA_crawling_rwa3.xml:83`).
- Torso-mass assert literal `40.0` (`sim_loop.py:218`).
- **Solver dims fixed for 2 contacts × 6-wrench + 3 wheels**: `nc_max: int = 2` (`wholebody_qp.py:64`;
  `sim_loop.py:1139` `nc_max=2`); `self._dim_lambda = 6 * nc_max` (`wholebody_qp.py:243`); NMPC
  `nx=9` / `nu=12` (`centroidal_nmpc.py:12` "*State vector (nx=9)*", `:21` "*Control vector (nu=12,
  for nc_max=2)*"; `nmpc_solver.py:21`); momentum map `M[:, 3:6] = np.eye(3)` / `M[:, 9:12] =
  np.eye(3)` (`contact_phase.py:130,135`). nx=9 = r_com(3)+L(3)+h_w(3-wheel); nu=12 = 2×[f(3),τ(3)].
- h_w as R³ throughout config: `hw_min`/`hw_max`/`hw_qp_tight`/`h_max_tight` = `np.full(3, …)`
  (`config.py:71-72,78,238`).
- Reasoning-only literals in docstrings: `ik.py:375-377` "*2*7=14 arm DOFs = 17 DOFs to satisfy 12
  constraints (2 × SE3), leaving a 5-dim null space*".
- **Deprecated** module constants still importable: `robot_interface.py:76-86`
  `FRAME_TORSO = 4 / FRAME_TOOL_A = 18 / FRAME_TOOL_B = 32 / JOINT_6A_ID = 7 / JOINT_6B_ID = 13 /
  N_JOINTS = 12 / NQ = 19 / NV = 18` — overwritten in `__init__` (`robot_interface.py:213-220`) but
  imported by `ik.py:17` (`from crawlbot.core.robot_interface import FRAME_TOOL_A, FRAME_TOOL_B, …`).

### b3-note. The good news (already DOF-generic)

The Pinocchio side derives every dimension and slice from the model at construction:
`_detect_arm_slices` (`robot_interface.py:29-70`) returns per-arm and combined q/v slices and
`n_joints` by name; `state_conversions` infers `n_joints` from vector length
(`state_conversions.py:37-40,61`); `ik.py` caches model-derived slices (`ik.py:24-52`). A different
**per-arm DOF count** (e.g. 6-DOF arms) is explicitly supported here (`robot_interface.py:9-10`,
`state_conversions.py:8-9`). What is *not* generic is the **3-wheel / 2-arm / structure-frame**
scaffolding in bins B2/B3 above.

---

## BIN C — Robot-parameter config

**There is no single robot-parameter source.** Robot facts are split across four places:

1. **The two model files** (BIN A) — masses, inertias, joint limits (`VISPA_crawling_rwa3.xml:50`
   `range="-3.14159 3.14159"`), geometry, actuator ranges (`VISPA_crawling_rwa3.xml:304` `ctrlrange="-50 50"`).
2. **`SimConfig`** (`config.py`) — the tuning knobs (weights/gains/limits), not geometry.
3. **Code literals** mirroring the model — armature `robot_interface.py:199`, torso-mass assert
   `sim_loop.py:218`, inertia fallback `sim_loop.py:225`.
4. **Per-script path constants** — `MJCF`/`URDF` duplicated across every runner (BIN A a1).

**Joint torque limit** is the clearest scatter — three different values in four places:

| value | where | meaning |
|---|---|---|
| **20** | `config.py:32` `tau_max: float = 20.0`; used at `sim_loop.py:795,798` (SS clip) and `sim_loop.py:1140` `tau_max=cfg.tau_max * np.ones(self.robot.n_joints)` (QP bound) | controller joint limit |
| **10** | `robot_interface.py:155` `tau_max=10.0` default — **not** overridden at construction (`sim_loop.py:241` passes none) ⇒ `state.tau_max = 10` | RobotState field (`robot_interface.py:387`) |
| **50** | `wholebody_qp.py:206` `50.0 * np.ones(14)` (WBC default, overridden by sim_loop) | QP default |
| **±50** | `VISPA_crawling_rwa3.xml:304-318` `ctrlrange="-50 50"` | plant actuator clamp |

(RWA cap is likewise triplicated — `config.py:80,84` + `VISPA_crawling_rwa3.xml:323-325` — per CLAUDE.md.)

**q0 / nominal posture**: the initial dock pose is computed by IK (`manipulability_config`,
`sim_loop.py:314`) seeded from `pin.neutral`; the only literal "home" is the MJCF keyframe
`VISPA_crawling_rwa3.xml:378-380` (all zeros). Startup-IK posture bias `ik_q_nominal` defaults None
(`config.py:443`). **Gripper/tool offset**: `tool_a`/`tool_b` sit at `pos="0 0 0"` on `Link_6`
(`VISPA_crawling_rwa3.xml:206,276`) ⇒ tool frame ≡ wrist frame, no offset parameter.

⇒ **A new robot is not a new config file.** It is: new MJCF + new URDF (mutually consistent per
a2) + edits to the armature literal (`robot_interface.py:199`), the torso-mass assert
(`sim_loop.py:218`), the inertia fallback (`sim_loop.py:225`), the `MJCF`/`URDF` constants in each
runner script, **and** a full `SimConfig` re-tune (BIN D).

---

## BIN D — Tuned quantities requiring re-validation (not code, but inventoried)

All are mass/inertia-ratio-specific (tuned at ~71 kg robot on a 7110 kg structure) and must be
re-validated, not copied:

- **Add-5 weight set**: α torso-pose 2000 (`config.py:351`), α swing-EE 1000 (`config.py:319`),
  α momentum 400 (`config.py:336`), w hw-slack 800 (`wholebody_qp.py:181`), α posture 20
  (`config.py:320`), α wrench 1.0 (`config.py:321`), α torque-min 5 + α accel-reg 1.0 (QP literal
  `sim_loop.py:1149` `alpha_torque=5e0, alpha_reg=1e0`).
- **Feasibility gate** (CLAUDE.md Rule 14): torque-min ≳ 5× the accel-reg floor — a hard gate that a
  new inertia scale shifts.
- **NMPC/QP budgets fed by robot properties**: hw_max ±5 (`config.py:72`, = wheel `I_w·ω_max` =
  0.01·500 from `VISPA_crawling_rwa3.xml:108`), tau_w_max 2.5 (`config.py:80`), nmpc_p_max 50
  (`config.py:292`), preplanner f_max 25 / tau_max 8 (`config.py:254-255`).
- **Swing** clearance 0.03 m (`config.py:409`) + timing fractions (`config.py:536,545`).
- **Task PD gains** ss_Kp/Kd_com/torso/ee (`config.py:399-406`).
- **CoM-z standoff** −0.35 m (`config.py:458`) — from `scripts/diag_standoff_feasibility.py` for this
  arm reach.
- **Dock gate** weld_radius 5 mm (`config.py:35`), ori 5° (`config.py:42`), twist 0.05 (`config.py:58`)
  — the docking-mechanism capture radius.

---

## BIN E — Silent assumptions a new robot/gripper breaks without erroring

1. **Anchor & dock orientation = Identity in the structure frame.** `contact_scheduler.py:349`
   `return pin.SE3(np.eye(3), pos)`; IK targets `ik.py:703-704,840-841` `pin.SE3(np.eye(3), …)`; swing
   target `swing_planner.py:94-95` `self._R_end: np.ndarray = np.eye(3)` "*Target orientation:
   identity (tool aligned with structure frame)*"; dock ORI gate measures the angle to Identity
   (`config.py:38-42` "*The anchor frame is Identity in the structure frame … angle between the
   gripper's rotation matrix and I*"). A gripper/anchor whose mated pose ≠ identity docks at the wrong
   orientation **and** the ori gate never passes — no error.
2. **Tool-frame mounting convention.** `tool_a`/`tool_b` at `Link_6` origin with no offset
   (`VISPA_crawling_rwa3.xml:206,276`); "docked" ⇔ tool frame coincident with the anchor at identity.
   A gripper with a TCP offset or a non-identity mating frame silently mis-docks.
3. **Swing clearance direction `away_normal` is a fixed constant** `[0,0,-1]`
   (`swing_planner.py:43-44` "*Structure surface is at z ≈ +0.025 … robot hangs below → away = −z*").
   It is **duplicated with a disagreeing sign**: `ik.py:1402` default `[0.0, 0.0, -1.0]`
   (solve_ik_waypoints) vs `ik.py:1282` default `[0.0, 0.0, 1.0]` (check_path_feasibility, whose
   comment `ik.py:1281` wrongly claims "*DEFAULT_AWAY_NORMAL = [0, 0, 1] in swing_planner*"). A
   different mounting surface pushes the clearance bump **into** the structure. Silent.
4. **Capture-gate geometry** `d < 5 mm ∧ ori < 5° ∧ ‖Jc·v‖ < 0.05` (`config.py:35,42,58`) is the
   docking-mechanism capture radius for this hardware; a coarser/finer gripper silently over- or
   under-triggers.
5. **Pinocchio-world ≡ structure body frame, treated as quasi-static (inertial).** All controller
   quantities live in the structure frame; the torso is transformed into it (`state_conversions.py:85-90`)
   and anchors stored structure-local (`sim_loop.py:232-238`). Non-inertial corrections are derived but
   **disabled** because they destabilize the tuned gains: `robot_interface.py:268-273` "*Non-inertial
   corrections … are NOT applied … The controller was designed assuming a quasi-static (inertial)
   structure frame.*" (also `omega_struct` "*stored but not used*", `:252-254`). A lighter or
   faster-rotating structure violates this silently.
6. **Exactly 2 arms sharing one free-flyer torso.** `nc_max=2` (`wholebody_qp.py:64`); detection
   *requires* both `_a` and `_b` present or raises (`robot_interface.py:44-46`); the GJM base-recoil
   term assumes two arms on one base (`robot_interface.py:401-431`). A 1- or 3-arm robot needs
   structural change, not config.
7. **Exactly 3 orthogonal reaction wheels (Hsw=I₃), h_w ∈ R³.** `VISPA_crawling_rwa3.xml:3`; NMPC
   state carries h_w(3) (nx=9); `ctrl[n_j:n_j+3]`; `qvel[6:9]`. A 4-wheel pyramid / CMG cluster (the
   "small-CMG class" noted in CLAUDE.md remaining work) changes the AOCS map and the NMPC state dim.
8. **RWA detected by DOF parity** `(len(mj_qpos) - 14) % 2 == 1` (`state_conversions.py:58`) — an odd
   total arm-DOF count misdetects the wheel block.
9. **Single-digit anchor indices** `int(parts[1][0]) - 1` (`sim_loop.py:1207`) and the
   `anchor_{i}{a|b}` / `grip_{arm}_to_{idx}{arm}` naming scheme on the structure body.

---

## PORT CHECKLIST — steps for a third party with their own URDF

Code/model work (in dependency order). Each step cites the bin that motivates it.

1. **Provide two consistent descriptions** (BIN A a1/a2). Author the new URDF (robot-only:
   free-flyer torso + arms, meshes) *and* the new MJCF (full scene: structure + wheels + torso +
   arms). Keep masses, inertias, joint names (`_a`/`_b` suffix), frame names (`tool_a`/`tool_b`/torso),
   and joint↔actuator order identical between them.
2. **Preserve the naming contract** (BIN B b1): torso frame `Link_0`, tools `tool_a`/`tool_b`, arm
   joints ending `_a`/`_b`, MuJoCo bodies `torso`/`structure`, wheels `rw_x/y/z`, gripper sites
   `gripper_a`/`gripper_b`, anchor sites `anchor_{i}{a|b}` (i < 10), welds `grip_{arm}_to_{i}{arm}`.
   Renaming any of these requires editing the lookups, not just the model.
3. **Fix the frame reference** (BIN A a2, BIN E 1/2): mount `tool_a`/`tool_b` so that "docked" means
   tool-at-anchor with **identity** orientation in the structure frame — or change the identity
   assumption in `contact_scheduler.py:349`, `swing_planner.py:95`, `ik.py:703-704/840-841`, and the
   ori gate `config.py:38-42`.
4. **Set the clearance normal** (BIN E 3): point `away_normal` off the new mounting surface —
   reconcile `swing_planner.py:44`, `ik.py:1402`, and `ik.py:1281-1282` (currently sign-inconsistent).
5. **Update the code literals that mirror the model** (BIN B b3, BIN C): armature
   `robot_interface.py:199`, torso-mass assert `sim_loop.py:218`, structure-inertia fallback
   `sim_loop.py:225`, and the `MJCF`/`URDF` path constants in every runner script you use.
6. **If the AOCS hardware differs** (BIN B b2/b3, BIN E 7): the 3-wheel assumption is wired into
   `qvel[6:9]` (`sim_loop.py:709,759`), `ctrl[n_j:n_j+3]` (`sim_loop.py:3455`), the NMPC state
   `nx=9` (`centroidal_nmpc.py:12`), and the R³ h_w config fields. A different wheel/CMG count is a
   code change across the NMPC + AOCS + state-conversion path, not a config edit.
7. **If arm count ≠ 2 or the base topology differs** (BIN E 6): `nc_max`, the arm-detection raise,
   and the GJM must be generalized. (A different *per-arm* DOF count, e.g. 6-DOF arms, already works —
   BIN B b3-note.)
8. **Confirm the parity heuristic** (BIN B b2, BIN E 8): if total arm-DOF is odd, fix RWA detection
   `state_conversions.py:58`.

Re-validation campaign (**separate from the code work above** — no amount of code correctness makes
these transfer; they are tuned to this robot's mass/inertia ratios):

9. **Re-tune and re-gate `SimConfig`** (BIN D): the Add-5 weights, the torque-min ≳ 5×floor
   feasibility gate, the NMPC/preplanner force/torque/momentum budgets, the task PD gains, the CoM-z
   standoff, the swing clearance/timing, and the dock capture gate — each re-derived for the new
   inertia scale and re-validated by the diagnostic suite (per CLAUDE.md: every sim ends in
   `run_diagnostics()`; at-weld dock metric only; one variable at a time).
10. **Re-verify the quasi-static-structure assumption holds** (BIN E 5) for the new
    robot/structure mass ratio before trusting the tuned gains.
