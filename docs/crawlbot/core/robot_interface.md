# `crawlbot.core.robot_interface`

**File**: `crawlbot/core/robot_interface.py` — **460 lines** — canonical coverage **87 %**

> Module docstring: *"RobotInterface — Pinocchio wrapper for the VISPA crawling controller."*

Pinocchio wrapper. One `update(q, v)` call produces **every** quantity the
controller needs for that tick, packaged in a single `RobotState`.

The design point is that both stages, the AOCS and the observer all read from
one consistent snapshot — no module re-derives a Jacobian on its own.

---

## Public API

| symbol | signature | canonical? |
|---|---|---|
| **`RobotState`** *(dataclass)* |  |  |
|   `q` |  | _field_ |
|   `v` |  | _field_ |
|   `q_joints` |  | _field_ |
|   `dq_joints` |  | _field_ |
|   `q_torso` |  | _field_ |
|   `dq_torso` |  | _field_ |
|   `H` |  | _field_ |
|   `C` |  | _field_ |
|   `C_matrix` |  | _field_ |
|   `r_com` |  | _field_ |
|   `v_com` |  | _field_ |
|   `J_com` |  | _field_ |
|   `Jdot_dq_com` |  | _field_ |
|   `h_centroidal` |  | _field_ |
|   `L_com` |  | _field_ |
|   `oMf_tool_a` |  | _field_ |
|   `oMf_tool_b` |  | _field_ |
|   `J_tool_a` |  | _field_ |
|   `J_tool_b` |  | _field_ |
|   `Jdot_dq_tool_a` |  | _field_ |
|   `Jdot_dq_tool_b` |  | _field_ |
|   `oMf_torso` |  | _field_ |
|   `J_torso` |  | _field_ |
|   `Jdot_dq_torso` |  | _field_ |
|   `q_min` |  | _field_ |
|   `q_max` |  | _field_ |
|   `tau_max` |  | _field_ |
|   `total_mass` |  | _field_ |
| **`RobotInterface`** |  |  |
| `.update` | `(q, v, omega_struct=None)` | **yes** |
| `.state` | `()` | **yes** |
| `.compute_gjm` | `(swing_arm)` | not exercised |
| `.get_contact_jacobians` | `(active_A, active_B)` | **yes** |
| `.neutral_configuration` | `()` | not exercised |

### Module constants

| name | value |
|---|---|
| `FRAME_TORSO` | `4` |
| `FRAME_TOOL_A` | `18` |
| `FRAME_TOOL_B` | `32` |
| `JOINT_6A_ID` | `7` |
| `JOINT_6B_ID` | `13` |
| `N_JOINTS` | `12` |
| `NQ` | `19` |
| `NV` | `18` |

---

---

## 1. What `update()` computes

A single Pinocchio pass fills a 27-field `RobotState`:

| group | fields |
|---|---|
| configuration | `q`, `v`, `q_joints`, `dq_joints`, `q_torso`, `dq_torso` |
| dynamics | `H` (mass matrix), `C` (Coriolis vector), `C_matrix` (Coriolis matrix) |
| centroidal | `r_com`, `v_com`, `J_com`, `Jdot_dq_com`, `h_centroidal`, `L_com` |
| end-effectors | `oMf_tool_a/b`, `J_tool_a/b`, `Jdot_dq_tool_a/b` |
| torso | `oMf_torso`, `J_torso`, `Jdot_dq_torso` |
| limits | `q_min`, `q_max`, `tau_max`, `total_mass` |

`C_matrix` (the full matrix, not just `C @ v`) is computed because the GMO
observer needs `C^T v` — see `estimation/contact_estimator.md`. It is the one
quantity here that exists for the observer rather than for control.

`gravity='zero'` is the project regime: microgravity.

Computing all of this in one pass is what allows the QP to run at 100 Hz —
recomputing Jacobians per consumer would not fit in the budget.

## 2. ⚠ The trap: module constants are rebound at construction

`FRAME_TORSO`, `FRAME_TOOL_A`, `FRAME_TOOL_B`, `JOINT_6A_ID`, `JOINT_6B_ID`,
`N_JOINTS`, `NQ`, `NV` are declared at module level with values inherited from
the **6-DOF** model, then overwritten via `global` in `__init__` (`:157-158`,
`:213-218`) from the model actually loaded.

Measured:

```
at import          : NQ=19  NV=18  N_JOINTS=12      (stale 6-DOF values)
after construction : NQ=21  NV=20  N_JOINTS=14      (the real 7-DOF model)
```

Consequence: **`from crawlbot.core.robot_interface import NQ` freezes 19
forever** — a `global` rebinding does not propagate into a module that imported
the name by value.

```python
import crawlbot.core.robot_interface as ri
ri.NQ                                          # correct AFTER construction
from crawlbot.core.robot_interface import NQ   # 19, permanently
```

Prefer instance attributes: `robot.model.nq`, `robot.n_joints`,
`robot.frame_torso`. They are always right.

## 3. DOF-generic by design

The module does not hard-code "6 DOF per arm": it detects the joint slice from
the loaded model. That is what allowed the move to 7 DOF per arm
(`nq=21 / nv=20 / nu=14`) without rewriting the controller — the frame IDs and
joint counts follow the URDF.

The stale module-level defaults in section 2 are the residue of that transition.

## 4. Unexercised

`compute_gjm` (generalized momentum map) and `neutral_configuration`.

## See also

- package overview: [`core.md`](core.md)
