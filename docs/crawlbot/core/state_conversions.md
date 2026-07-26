# `crawlbot.core.state_conversions`

**File**: `crawlbot/core/state_conversions.py` — **165 lines** — canonical coverage **100 %**

> Module docstring: *"State conversions between MuJoCo (world frame) and Pinocchio (structure frame)."*

The MuJoCo <-> Pinocchio bridge. **The only module in the package with 100 %
canonical coverage** — every conversion in the controller goes through it.

---

## Public API

| symbol | signature | canonical? |
|---|---|---|
| `mujoco_to_pinocchio` | `(mj_qpos, mj_qvel)` | **yes** |
| `pinocchio_to_mujoco` | `(pin_q, pin_v, struct_pos=None, struct_quat=None, rwa=False)` | **yes** |
| `quat_wxyz_to_euler_deg` | `(qw, qx, qy, qz)` | **yes** |

### Module constants

| name | value |
|---|---|
| `_MJ_STRUCT_NQ` | `7` |
| `_MJ_STRUCT_NV` | `6` |
| `_MJ_RWA_NQ` | `3` |
| `_MJ_RWA_NV` | `3` |
| `_MJ_TORSO_NQ` | `7` |
| `_MJ_TORSO_NV` | `6` |

---

---

## 1. Two representations of the same robot

| side | frame | dimensions |
|---|---|---|
| MuJoCo (plant) | **world**, structure included | `nq=31 / nv=29 / nu=17` (7-DOF arms + 3 wheels) |
| Pinocchio (controller) | **structure frame R_s** | `nq=21 / nv=20 / nu=14` |

The controller never reasons in world coordinates. Everything — the NMPC state,
the QP tasks, the momentum bookkeeping — lives in R_s, the frame attached to the
host structure. `mujoco_to_pinocchio` is therefore not a reshuffle: it performs
the change of frame, and it is where the two dimensionalities are reconciled
(the structure's own 6 DOF and the 3 wheels exist only on the MuJoCo side).

## 2. ⚠ Quaternion conventions — never assume

```
Pinocchio : (x, y, z, w)          MuJoCo : (w, x, y, z)
```

Explicit project rule: *"Do not assume quaternion conventions — verify in
`state_conversions.py`."* This file is **the reference**, not a convenience: a
conversion re-written inline elsewhere is how a sign error enters and survives,
because a wrong quaternion still normalises and still integrates.

## 3. The three functions

| function | role |
|---|---|
| `mujoco_to_pinocchio(...)` | MuJoCo world state -> Pinocchio state in R_s |
| `pinocchio_to_mujoco(...)` | the return trip, to apply commands |
| `quat_wxyz_to_euler_deg(w,x,y,z)` | Euler angles in degrees, for logging |

`quat_wxyz_to_euler_deg` feeds the `struct_euler_deg` log channel, from which
`theta_s` is derived. The canonical peak **0.540 deg** is the norm of that
3-vector — worth knowing when reading the figures, since the per-axis peaks
(0.13 / 0.27 / 0.53) are all smaller.

100 % coverage: all three run on every tick of the canonical.

## See also

- package overview: [`core.md`](core.md)
