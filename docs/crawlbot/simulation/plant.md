# `crawlbot.simulation.plant`

**File**: [`crawlbot/simulation/plant.py`](../../../crawlbot/simulation/plant.py) — **200 lines** — canonical coverage **93 %**

**The simulated robot + platform, and nothing else.** Extraction 1 of the
`sim_loop.py` split (branch `refactor/sim-loop-split`), preparing the ROS 2 seam.

---

## 1. The rule this module enforces

`MujocoPlant` is the **only writer of MuJoCo state**. Everything that changes
`MjData` lives here:

| write | method | was in `sim_loop.py` |
|---|---|---|
| model load, `opt.timestep`, RWA detection | `__init__` | `setup()` head |
| `mj_step` + diagnostic arm-joint lock | `step(lock_arm_joints)` | `_step` QP sub-loop, `_run_ds_passivity_loop` |
| `mj_forward` | `forward()` | ~8 call sites |
| initial `qpos` / `qvel` | `set_state` | `setup()` |
| actuator `ctrl` | `apply_joint_torques`, `apply_wheel_torques`, `zero_ctrl` | `_step`, DS loop, `_settle_setup` |
| gripper welds (`eq_active`) | `build_weld_map`, `activate_weld`, `deactivate_weld`, `deactivate_all_welds` | weld-management block |
| site-id cache | `cache_site_ids` | weld-management block |
| inelastic dock impact (writes all `qvel`) | `apply_dock_impact(verbose)` | `run()` post-dock block |

The read side is [`sensors.py`](sensors.md), which queries `plant.data` and the
`site_ids` cache built here. `SimulationLoop.mj_model` / `.mj_data` remain as **read-only properties** —
`tick_logging.py` and the diagnostic drivers read them — but no controller code
writes through them any more. Under ROS 2 this class is what sits behind the
bridge node (or is replaced by hardware): the controller sees measurements in and
commands out.

## 2. Why the move is byte-identical

Every block was moved verbatim — same numpy expressions, same call order — so the
floating-point operation sequence is unchanged. Verified with
`gate/local_ref.py check` (every replay file bit-identical to the host-local
reference, JSON floats by `float.hex`) and `gate/dock_check.py`.

`set_state(mj_qpos, 0.0)` and `apply_wheel_torques(0.0)` broadcast the scalar
exactly as the original `qvel[:] = 0.0` / `ctrl[n:n+3] = 0.0` did.

## 3. The dock impact

Fix A (dock-leak Part 3): after the swing weld is activated, the velocity is
projected onto the weld constraint manifold in **full MuJoCo DOF**:

```
J      = [ J_grip - J_anchor ]  stacked over every active weld   (6·n_weld × nv)
v+     = v- - M⁻¹Jᵀ (J M⁻¹ Jᵀ)⁻¹ J v-
```

written back to all of `qvel` (structure base and wheels included), so the
impulse is a full action–reaction pair and conserves `subtree_angmom` to the
O(gap·f) residual. The robot-only Pinocchio projection it replaced injected
~0.2 N·m·s per dock (leak 0.3565 → 0.0011 over five docks, Part-2 A.1).

## 4. Traps kept as found

- **The DS passivity loop never applied the diagnostic arm lock**, even with
  `_diag_lock_arm_joints` on — only `_step` did. `step()` takes the lock as an
  argument so each caller keeps its historical behaviour. Irrelevant on the
  canonical (lock off); worth resolving before the lock is used again.
- The companion **torque** zeroing of the lock (`tau := 0` before
  `apply_joint_torques`) is still on the controller side, in `_step`.

## Public API

| symbol | signature | canonical? | code |
|---|---|---|---|
| **`MujocoPlant`** |  |  | [L27](../../../crawlbot/simulation/plant.py#L27) |
| `.forward` | `()` | **yes** | [L47](../../../crawlbot/simulation/plant.py#L47) |
| `.step` | `(lock_arm_joints=False)` | **yes** | [L50](../../../crawlbot/simulation/plant.py#L50) |
| `.set_state` | `(qpos, qvel)` | **yes** | [L69](../../../crawlbot/simulation/plant.py#L69) |
| `.apply_joint_torques` | `(tau)` | **yes** | [L75](../../../crawlbot/simulation/plant.py#L75) |
| `.apply_wheel_torques` | `(tau_w)` | **yes** | [L78](../../../crawlbot/simulation/plant.py#L78) |
| `.zero_ctrl` | `()` | **yes** | [L81](../../../crawlbot/simulation/plant.py#L81) |
| `.build_weld_map` | `()` | **yes** | [L86](../../../crawlbot/simulation/plant.py#L86) |
| `.deactivate_all_welds` | `()` | **yes** | [L97](../../../crawlbot/simulation/plant.py#L97) |
| `.activate_weld` | `(arm, anchor_idx)` | **yes** | [L101](../../../crawlbot/simulation/plant.py#L101) |
| `.deactivate_weld` | `(arm, anchor_idx)` | **yes** | [L106](../../../crawlbot/simulation/plant.py#L106) |
| `.cache_site_ids` | `()` | **yes** | [L111](../../../crawlbot/simulation/plant.py#L111) |
| `.apply_dock_impact` | `(verbose)` | **yes** | [L134](../../../crawlbot/simulation/plant.py#L134) |

## Code map

| unit | source |
|---|---|
| `class MujocoPlant` | [L27-199](../../../crawlbot/simulation/plant.py#L27-L199) |
| `MujocoPlant.forward` | [L47-48](../../../crawlbot/simulation/plant.py#L47-L48) |
| `MujocoPlant.step` | [L50-67](../../../crawlbot/simulation/plant.py#L50-L67) |
| `MujocoPlant.set_state` | [L69-71](../../../crawlbot/simulation/plant.py#L69-L71) |
| `MujocoPlant.apply_joint_torques` | [L75-76](../../../crawlbot/simulation/plant.py#L75-L76) |
| `MujocoPlant.apply_wheel_torques` | [L78-79](../../../crawlbot/simulation/plant.py#L78-L79) |
| `MujocoPlant.zero_ctrl` | [L81-82](../../../crawlbot/simulation/plant.py#L81-L82) |
| `MujocoPlant.build_weld_map` | [L86-95](../../../crawlbot/simulation/plant.py#L86-L95) |
| `MujocoPlant.deactivate_all_welds` | [L97-99](../../../crawlbot/simulation/plant.py#L97-L99) |
| `MujocoPlant.activate_weld` | [L101-104](../../../crawlbot/simulation/plant.py#L101-L104) |
| `MujocoPlant.deactivate_weld` | [L106-109](../../../crawlbot/simulation/plant.py#L106-L109) |
| `MujocoPlant.cache_site_ids` | [L111-130](../../../crawlbot/simulation/plant.py#L111-L130) |
| `MujocoPlant.apply_dock_impact` | [L134-199](../../../crawlbot/simulation/plant.py#L134-L199) |

---

## See also

- the loop that drives it: [`sim_loop.md`](sim_loop.md)
- package overview: [`simulation.md`](simulation.md)
