# `crawlbot.simulation.sensors`

**File**: [`crawlbot/simulation/sensors.py`](../../../crawlbot/simulation/sensors.py) — **129 lines** — canonical coverage **92 %**

**Every MuJoCo read the control stack consumes.** Extraction 2 of the
`sim_loop.py` split (branch `refactor/sim-loop-split`).

---

## 1. The rule this module enforces

[`plant.py`](plant.md) is the only **writer** of MuJoCo state; `SensorSuite` is
the only **reader** of it on the command path. After this extraction
`sim_loop.py` contains no `mj_data` access at all outside the read-only
`mj_data` property kept for telemetry.

Each method is one measurement channel, named after what the real robot would
publish — which is what makes it the ROS 2 seam:

| channel | method | physical source | frame |
|---|---|---|---|
| torso pose/twist rel. platform + arm joints | `joint_state()` → Pinocchio `(q, v)` | nav + gyros + encoders | platform (spec §3, *Relative Twist*) |
| platform angular velocity ω_s | `omega_struct()` / `omega_struct_view()` | platform gyro | platform body |
| platform attitude | `struct_quat()` | star tracker | wxyz (MuJoCo) |
| platform position | `struct_pos()` | navigation | world |
| wheel momentum h_w = I_w·ω_w | `wheel_momentum()` | wheel tachometers | platform body |
| gripper↔anchor distance | `gripper_distance(arm, idx)` | dock proximity sensor | — |
| gripper↔anchor 6-D relative twist | `weld_relative_twist(arm, idx)` | dock sensor | world (MuJoCo DOF) |
| handhold map | `anchor_sites_world()` | known map, read once at setup | world |
| raw `(qpos, qvel)` | `raw_state()` | — snapshots / rendering only | — |

`joint_state()` is `mujoco_to_pinocchio`: it forms the **relative** twist of the
torso w.r.t. the platform from inertial nav + body gyros, so every Pinocchio
quantity downstream is platform-frame and relative (spec §3).

## 2. Why the move is byte-identical

Each method returns exactly the expression it replaced. Two details matter:

- **Copy vs view.** The `_step` AOCS block read `omega_s = qvel[3:6]` as a
  **view** (and passed it to `H_estimator.update`); every other site copied.
  `omega_struct_view()` preserves the view where the original took one, so no
  consumer that might retain the array sees different data.
- `wheel_momentum()` is `(I_w · qvel[6:9]).copy()`; the three variants that
  existed (`(I·v).copy()`, `I·v.copy()`, `I·view`) are the same element-wise
  product and therefore the same bits.

Verified with `gate/local_ref.py check` + `gate/dock_check.py`.

## 3. What still reads MuJoCo directly, and why

- **`tick_logging.py`** reads `self.mj_data` (and calls `mj_forward`) for
  telemetry. It is not on the command path; routing it through `SensorSuite`
  is a separate, logging-only change.
- **Model parameters** (`body_mass`, `body_inertia`, name → id lookups) are read
  from `mj_model` in `setup()`. They are constants of the model, not
  measurements.

## Public API

| symbol | signature | canonical? | code |
|---|---|---|---|
| **`SensorSuite`** |  |  | [L40](../../../crawlbot/simulation/sensors.py#L40) |
| `.joint_state` | `()` | **yes** | [L49](../../../crawlbot/simulation/sensors.py#L49) |
| `.omega_struct` | `()` | **yes** | [L55](../../../crawlbot/simulation/sensors.py#L55) |
| `.omega_struct_view` | `()` | **yes** | [L59](../../../crawlbot/simulation/sensors.py#L59) |
| `.struct_quat` | `()` | **yes** | [L65](../../../crawlbot/simulation/sensors.py#L65) |
| `.struct_pos` | `()` | **yes** | [L69](../../../crawlbot/simulation/sensors.py#L69) |
| `.wheel_momentum` | `()` | **yes** | [L73](../../../crawlbot/simulation/sensors.py#L73) |
| `.raw_state` | `()` | **yes** | [L77](../../../crawlbot/simulation/sensors.py#L77) |
| `.gripper_distance` | `(arm, anchor_idx)` | **yes** | [L84](../../../crawlbot/simulation/sensors.py#L84) |
| `.weld_relative_twist` | `(arm, anchor_idx)` | **yes** | [L94](../../../crawlbot/simulation/sensors.py#L94) |
| `.anchor_sites_world` | `()` | **yes** | [L126](../../../crawlbot/simulation/sensors.py#L126) |

## Code map

| unit | source |
|---|---|
| `class SensorSuite` | [L40-128](../../../crawlbot/simulation/sensors.py#L40-L128) |
| `SensorSuite.joint_state` | [L49-53](../../../crawlbot/simulation/sensors.py#L49-L53) |
| `SensorSuite.omega_struct` | [L55-57](../../../crawlbot/simulation/sensors.py#L55-L57) |
| `SensorSuite.omega_struct_view` | [L59-63](../../../crawlbot/simulation/sensors.py#L59-L63) |
| `SensorSuite.struct_quat` | [L65-67](../../../crawlbot/simulation/sensors.py#L65-L67) |
| `SensorSuite.struct_pos` | [L69-71](../../../crawlbot/simulation/sensors.py#L69-L71) |
| `SensorSuite.wheel_momentum` | [L73-75](../../../crawlbot/simulation/sensors.py#L73-L75) |
| `SensorSuite.raw_state` | [L77-80](../../../crawlbot/simulation/sensors.py#L77-L80) |
| `SensorSuite.gripper_distance` | [L84-92](../../../crawlbot/simulation/sensors.py#L84-L92) |
| `SensorSuite.weld_relative_twist` | [L94-122](../../../crawlbot/simulation/sensors.py#L94-L122) |
| `SensorSuite.anchor_sites_world` | [L126-128](../../../crawlbot/simulation/sensors.py#L126-L128) |

---

## See also

- the writer side: [`plant.md`](plant.md)
- the loop that consumes it: [`sim_loop.md`](sim_loop.md)
- quaternion conventions: `crawlbot/core/state_conversions.py`
