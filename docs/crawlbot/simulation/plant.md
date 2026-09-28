# `crawlbot.simulation.plant`

**File**: [`crawlbot/simulation/plant.py`](../../../crawlbot/simulation/plant.py) — **202 lines** — canonical coverage **93 %**

**The simulated robot + platform, and nothing else.** Extraction 1 of the
`sim_loop.py` split (branch `refactor/sim-loop-split`), preparing the ROS 2 seam.

---

## 1. The rule this module enforces

`MujocoPlant` is the **only writer of MuJoCo state**. Everything that changes
`MjData` lives here:

| write | method | was in `sim_loop.py` |
|---|---|---|
| model load, `opt.timestep`, RWA check (raises if the MJCF has no wheels) | `__init__` | `setup()` head |
| `mj_step` + diagnostic arm-joint lock | `step(lock_arm_joints)` | `_step` QP sub-loop, `_run_ds_passivity_loop` (now `_qp_substep`, `_ds_tick`) |
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
| `.forward` | `()` | **yes** | [L50](../../../crawlbot/simulation/plant.py#L50) |
| `.step` | `(lock_arm_joints=False)` | **yes** | [L53](../../../crawlbot/simulation/plant.py#L53) |
| `.set_state` | `(qpos, qvel)` | **yes** | [L71](../../../crawlbot/simulation/plant.py#L71) |
| `.apply_joint_torques` | `(tau)` | **yes** | [L77](../../../crawlbot/simulation/plant.py#L77) |
| `.apply_wheel_torques` | `(tau_w)` | **yes** | [L80](../../../crawlbot/simulation/plant.py#L80) |
| `.zero_ctrl` | `()` | **yes** | [L83](../../../crawlbot/simulation/plant.py#L83) |
| `.build_weld_map` | `()` | **yes** | [L88](../../../crawlbot/simulation/plant.py#L88) |
| `.deactivate_all_welds` | `()` | **yes** | [L99](../../../crawlbot/simulation/plant.py#L99) |
| `.activate_weld` | `(arm, anchor_idx)` | **yes** | [L103](../../../crawlbot/simulation/plant.py#L103) |
| `.deactivate_weld` | `(arm, anchor_idx)` | **yes** | [L108](../../../crawlbot/simulation/plant.py#L108) |
| `.cache_site_ids` | `()` | **yes** | [L113](../../../crawlbot/simulation/plant.py#L113) |
| `.apply_dock_impact` | `(verbose)` | **yes** | [L136](../../../crawlbot/simulation/plant.py#L136) |

## Code map

| unit | source |
|---|---|
| `class MujocoPlant` | [L27-201](../../../crawlbot/simulation/plant.py#L27-L201) |
| `MujocoPlant.forward` | [L50-51](../../../crawlbot/simulation/plant.py#L50-L51) |
| `MujocoPlant.step` | [L53-69](../../../crawlbot/simulation/plant.py#L53-L69) |
| `MujocoPlant.set_state` | [L71-73](../../../crawlbot/simulation/plant.py#L71-L73) |
| `MujocoPlant.apply_joint_torques` | [L77-78](../../../crawlbot/simulation/plant.py#L77-L78) |
| `MujocoPlant.apply_wheel_torques` | [L80-81](../../../crawlbot/simulation/plant.py#L80-L81) |
| `MujocoPlant.zero_ctrl` | [L83-84](../../../crawlbot/simulation/plant.py#L83-L84) |
| `MujocoPlant.build_weld_map` | [L88-97](../../../crawlbot/simulation/plant.py#L88-L97) |
| `MujocoPlant.deactivate_all_welds` | [L99-101](../../../crawlbot/simulation/plant.py#L99-L101) |
| `MujocoPlant.activate_weld` | [L103-106](../../../crawlbot/simulation/plant.py#L103-L106) |
| `MujocoPlant.deactivate_weld` | [L108-111](../../../crawlbot/simulation/plant.py#L108-L111) |
| `MujocoPlant.cache_site_ids` | [L113-132](../../../crawlbot/simulation/plant.py#L113-L132) |
| `MujocoPlant.apply_dock_impact` | [L136-201](../../../crawlbot/simulation/plant.py#L136-L201) |

---

## See also

- the loop that drives it: [`sim_loop.md`](sim_loop.md)
- package overview: [`simulation.md`](simulation.md)
