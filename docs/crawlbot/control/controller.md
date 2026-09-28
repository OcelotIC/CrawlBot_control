# `crawlbot.control.controller`

**File**: [`crawlbot/control/controller.py`](../../../crawlbot/control/controller.py) — **210 lines** — canonical coverage **75 %**

**The two-stage controller driven as one block.** Built by extractions 3b–3c of
the `sim_loop.py` split (branch `refactor/sim-loop-split`).

---

## 1. The contract — deliberately not the QP's signature

`WholeBodyQP.solve()` takes ~40 arguments, and several of them (`settle_mode`,
`passivity_active`, the torso / EE reference sextuplets) are caller-side
decisions that `docs/architecture/unified_planner_architecture.md` §4.3 removes
from the solver interface. Freezing the controller API on that signature would
lock the debt in. The controller's boundary is instead the one that document
targets (§4.1–4.2):

| boundary | type | unified-architecture counterpart |
|---|---|---|
| measurements in | [`SensorSuite`](../simulation/sensors.md) | `state: RobotState` |
| references in, **by time** | a `ReferenceSource` (`com_at`, `L_com_at`, …) | `gait_trajectory.at(t)` (§3.2) |
| stage 1 → stage 2 | `NMPCPlan` | `NMPCOutput` (§4.1) |
| commands out | wheel / joint torques | `(tau, …)` (§4.2) |

The `ReferenceSource` in use today is `PlannerReferences` in
`simulation/sim_loop.py`, an adapter over TorsoPlanner, the coarse pre-planner
and SwingPlanner. When the unified planner lands it replaces the adapter, not
the controller. Each typed boundary is also what a ROS 2 message will carry.

The mode flags that still exist (`settle_mode`, `phase`) are passed explicitly
and named as what §4.3 deletes; they are not hidden in the interface.

## 2. Stage 1 — `plan()`  (10 Hz)

1. `refs.com_at(t, settle_mode)` — CoM reference at the horizon end
   `t + N·dt_nmpc` (coarse pre-planner trajectory outside settle mode, with the
   `torso_early_finish_fraction` time compression).
2. Measure → `robot.update(q, v, ω_s)`.
3. `refs.L_com_at(t + N·dt/2)` — momentum feedforward (zeroed under `pure_pd`).
4. `CentroidalNMPC.solve` with the live wheel momentum h_w for the conservation
   box; on failure, the receding-horizon **shifted fallback** (M5 Fix 2) — never
   a jump to the raw reference.
5. First two knots of the planned CoM trajectory, for linear interpolation at
   QP rate (M5 Fix 1b).

## 3. `DiagHooks`

The four runtime diagnostic switches (`pure_pd`, `freeze_ref`, `disable_aocs`,
`lock_arm_joints`) as one record shared by the loop and the controller.
`SimulationLoop._diag_*` remain as properties onto it, because drivers set them
on the loop object.

## 4. Why each move is byte-identical

Blocks are moved by text slicing with asserted renames; call order is kept —
including the order of `robot.update` calls, which matters because
`RobotInterface` caches the last state for `get_contact_jacobians`. Verified per
commit with `gate/local_ref.py check` + `gate/dock_check.py`.

## Public API

| symbol | signature | canonical? | code |
|---|---|---|---|
| **`DiagHooks`** *(dataclass)* |  |  | [L25](../../../crawlbot/control/controller.py#L25) |
|   `pure_pd` | `False` | _field_ | [L35](../../../crawlbot/control/controller.py#L35) |
|   `freeze_ref` | `False` | _field_ | [L36](../../../crawlbot/control/controller.py#L36) |
|   `disable_aocs` | `False` | _field_ | [L37](../../../crawlbot/control/controller.py#L37) |
|   `lock_arm_joints` | `False` | _field_ | [L38](../../../crawlbot/control/controller.py#L38) |
| **`NMPCPlan`** *(dataclass)* |  |  | [L42](../../../crawlbot/control/controller.py#L42) |
|   `rs` | `` | _field_ | [L54](../../../crawlbot/control/controller.py#L54) |
|   `L_com_now` | `` | _field_ | [L55](../../../crawlbot/control/controller.py#L55) |
|   `cref_r` | `` | _field_ | [L56](../../../crawlbot/control/controller.py#L56) |
|   `cref_v` | `` | _field_ | [L57](../../../crawlbot/control/controller.py#L57) |
|   `rp` | `` | _field_ | [L58](../../../crawlbot/control/controller.py#L58) |
|   `vp` | `` | _field_ | [L59](../../../crawlbot/control/controller.py#L59) |
|   `lr` | `` | _field_ | [L60](../../../crawlbot/control/controller.py#L60) |
|   `af` | `` | _field_ | [L61](../../../crawlbot/control/controller.py#L61) |
|   `rp_k0` | `` | _field_ | [L62](../../../crawlbot/control/controller.py#L62) |
|   `rp_k1` | `` | _field_ | [L63](../../../crawlbot/control/controller.py#L63) |
|   `vp_k0` | `` | _field_ | [L64](../../../crawlbot/control/controller.py#L64) |
|   `vp_k1` | `` | _field_ | [L65](../../../crawlbot/control/controller.py#L65) |
|   `ok` | `` | _field_ | [L66](../../../crawlbot/control/controller.py#L66) |
|   `status_code` | `` | _field_ | [L67](../../../crawlbot/control/controller.py#L67) |
|   `cost` | `` | _field_ | [L68](../../../crawlbot/control/controller.py#L68) |
|   `info` | `` | _field_ | [L69](../../../crawlbot/control/controller.py#L69) |
|   `t_ms` | `` | _field_ | [L70](../../../crawlbot/control/controller.py#L70) |
|   `L_com_ref` | `` | _field_ | [L71](../../../crawlbot/control/controller.py#L71) |
|   `t_mid` | `` | _field_ | [L72](../../../crawlbot/control/controller.py#L72) |
| **`WholeBodyController`** |  |  | [L75](../../../crawlbot/control/controller.py#L75) |
| `.plan` | `(t, phase, step_idx, cc_nmpc, settle_mode, refs, hw)` | **yes** | [L86](../../../crawlbot/control/controller.py#L86) |

## Code map

| unit | source |
|---|---|
| `class DiagHooks` | [L25-38](../../../crawlbot/control/controller.py#L25-L38) |
| `class NMPCPlan` | [L42-72](../../../crawlbot/control/controller.py#L42-L72) |
| `class WholeBodyController` | [L75-209](../../../crawlbot/control/controller.py#L75-L209) |
| `WholeBodyController.plan` | [L86-209](../../../crawlbot/control/controller.py#L86-L209) |

---

## See also

- the AOCS block: [`attitude.md`](attitude.md)
- the solvers it drives: [`solvers/centroidal_nmpc.md`](../solvers/centroidal_nmpc.md),
  [`solvers/wholebody_qp.md`](../solvers/wholebody_qp.md)
- package overview: [`control.md`](control.md)
