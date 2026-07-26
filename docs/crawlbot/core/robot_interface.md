# `crawlbot.core.robot_interface`

**File**: [`crawlbot/core/robot_interface.py`](../../../crawlbot/core/robot_interface.py) — **460 lines** — canonical coverage **87 %**

> Module docstring: *"RobotInterface — Pinocchio wrapper for the VISPA crawling controller."*

Pinocchio wrapper. One `update(q, v)` call produces **every** quantity the
controller needs for that tick, packaged in a single `RobotState`.

The design point is that both stages, the AOCS and the observer all read from
one consistent snapshot — no module re-derives a Jacobian on its own.

---

## Public API

| symbol | signature | canonical? | code |
|---|---|---|---|
| **`RobotState`** *(dataclass)* |  |  | [L89](../../../crawlbot/core/robot_interface.py#L89) |
|   `q` | `` | _field_ | [L95](../../../crawlbot/core/robot_interface.py#L95) |
|   `v` | `` | _field_ | [L96](../../../crawlbot/core/robot_interface.py#L96) |
|   `q_joints` | `` | _field_ | [L97](../../../crawlbot/core/robot_interface.py#L97) |
|   `dq_joints` | `` | _field_ | [L98](../../../crawlbot/core/robot_interface.py#L98) |
|   `q_torso` | `` | _field_ | [L99](../../../crawlbot/core/robot_interface.py#L99) |
|   `dq_torso` | `` | _field_ | [L100](../../../crawlbot/core/robot_interface.py#L100) |
|   `H` | `` | _field_ | [L103](../../../crawlbot/core/robot_interface.py#L103) |
|   `C` | `` | _field_ | [L104](../../../crawlbot/core/robot_interface.py#L104) |
|   `C_matrix` | `` | _field_ | [L105](../../../crawlbot/core/robot_interface.py#L105) |
|   `r_com` | `` | _field_ | [L108](../../../crawlbot/core/robot_interface.py#L108) |
|   `v_com` | `` | _field_ | [L109](../../../crawlbot/core/robot_interface.py#L109) |
|   `J_com` | `` | _field_ | [L110](../../../crawlbot/core/robot_interface.py#L110) |
|   `Jdot_dq_com` | `` | _field_ | [L111](../../../crawlbot/core/robot_interface.py#L111) |
|   `h_centroidal` | `` | _field_ | [L114](../../../crawlbot/core/robot_interface.py#L114) |
|   `L_com` | `` | _field_ | [L115](../../../crawlbot/core/robot_interface.py#L115) |
|   `oMf_tool_a` | `` | _field_ | [L118](../../../crawlbot/core/robot_interface.py#L118) |
|   `oMf_tool_b` | `` | _field_ | [L119](../../../crawlbot/core/robot_interface.py#L119) |
|   `J_tool_a` | `` | _field_ | [L120](../../../crawlbot/core/robot_interface.py#L120) |
|   `J_tool_b` | `` | _field_ | [L121](../../../crawlbot/core/robot_interface.py#L121) |
|   `Jdot_dq_tool_a` | `` | _field_ | [L122](../../../crawlbot/core/robot_interface.py#L122) |
|   `Jdot_dq_tool_b` | `` | _field_ | [L123](../../../crawlbot/core/robot_interface.py#L123) |
|   `oMf_torso` | `` | _field_ | [L126](../../../crawlbot/core/robot_interface.py#L126) |
|   `J_torso` | `` | _field_ | [L127](../../../crawlbot/core/robot_interface.py#L127) |
|   `Jdot_dq_torso` | `` | _field_ | [L128](../../../crawlbot/core/robot_interface.py#L128) |
|   `q_min` | `` | _field_ | [L131](../../../crawlbot/core/robot_interface.py#L131) |
|   `q_max` | `` | _field_ | [L132](../../../crawlbot/core/robot_interface.py#L132) |
|   `tau_max` | `` | _field_ | [L133](../../../crawlbot/core/robot_interface.py#L133) |
|   `total_mass` | `` | _field_ | [L136](../../../crawlbot/core/robot_interface.py#L136) |
| **`RobotInterface`** |  |  | [L139](../../../crawlbot/core/robot_interface.py#L139) |
| `.update` | `(q, v, omega_struct=None)` | **yes** | [L242](../../../crawlbot/core/robot_interface.py#L242) |
| `.state` | `()` | **yes** | [L393](../../../crawlbot/core/robot_interface.py#L393) |
| `.compute_gjm` | `(swing_arm)` | not exercised | [L401](../../../crawlbot/core/robot_interface.py#L401) |
| `.get_contact_jacobians` | `(active_A, active_B)` | **yes** | [L433](../../../crawlbot/core/robot_interface.py#L433) |
| `.neutral_configuration` | `()` | not exercised | [L457](../../../crawlbot/core/robot_interface.py#L457) |

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

## Code map

| unit | source |
|---|---|
| `class RobotState` | [L89-136](../../../crawlbot/core/robot_interface.py#L89-L136) |
| `class RobotInterface` | [L139-459](../../../crawlbot/core/robot_interface.py#L139-L459) |
| `RobotInterface.update` | [L242-390](../../../crawlbot/core/robot_interface.py#L242-L390) |
| `RobotInterface.state` | [L393-397](../../../crawlbot/core/robot_interface.py#L393-L397) |
| `RobotInterface.compute_gjm` | [L401-431](../../../crawlbot/core/robot_interface.py#L401-L431) |
| `RobotInterface.get_contact_jacobians` | [L433-455](../../../crawlbot/core/robot_interface.py#L433-L455) |
| `RobotInterface.neutral_configuration` | [L457-459](../../../crawlbot/core/robot_interface.py#L457-L459) |

---

## See also

- package overview: [`core.md`](core.md)
