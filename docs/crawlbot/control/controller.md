# `crawlbot.control.controller`

**File**: [`crawlbot/control/controller.py`](../../../crawlbot/control/controller.py) — **589 lines** — canonical coverage **82 %**

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

## 3. Stage 2 — `begin_tracking()` / `track()` / `after_step()`  (100 Hz)

Per NMPC tick `begin_tracking(plan, hw)` creates a `QPCarry` — the state the QP
sub-steps thread (`qp_ok`, the h_w carry, λ_ref and a_ff, and the last τ_w /
transport magnitude for telemetry) — and restarts the AOCS history
(`AttitudeController.reset_for_nmpc_tick`; the AOCS owns it since A0). Then per
sub-step:

1. `track()` — measure → `robot.update` → contact Jacobians
   (`contact_jacobians(rs, …)`, a pure function of that `rs`); interpolate the
   NMPC knots; **torso reference**: `refs.torso_at(tq)` shaped by
   [`TorsoReferenceShaper`](torso_reference.md) (in DS the δ-mapping and the
   post-dock blend; in SS — the two-task stack — the raw quintic; then the
   freeze diagnostics); **swing reference**
   `refs.swing_at(tq)`; passivity decision; `WholeBodyQP.solve`; clip to
   τ_max; diagnostic lock; AOCS wheel torque. Returns `TrackOut` — the command,
   plus `tau_raw` (pre-clip) and the values the loop's traces read.
2. the loop applies the command and steps the plant;
3. `after_step()` — post-step measurement, GMO update, h_w carry.

`settle()` is the DS passivity-loop tick (settle-mode QP with the passivity
inequality, fallback joint damping, inter-step AOCS), returning the wheel command
explicitly (0.0 when the inter-step AOCS is off, else τ_w).

**No wheel-less variant.** The reaction wheels are part of the plant
(`MujocoPlant` raises if the MJCF has none). The `has_rwa == False` branches —
h_w integrated from L̇_com (`hw -= …`), the NMPC h_w from the loop carry, the
un-commanded wheels — were never executed on the canonical or any scenario and
were removed (decision: Idriss, 2026-09-28).

**No reference events.** The `on_ss_entry` / `on_dock` events fed the post-dock
DS blend of the δ-mapping; both were retired with it (R2b). The torso reference
is now a pure function of the planner's (`torso_reference.md`).

**Kept as found:** the AOCS history restarts at zero each NMPC tick — see
[`attitude.md`](attitude.md) §1b and §3. `begin_settle()` seeds it at the entry
of an inter-step DS settle.

## 4. `DiagHooks`

The four runtime diagnostic switches (`pure_pd`, `freeze_ref`, `disable_aocs`,
`lock_arm_joints`) as one record shared by the loop and the controller.
`SimulationLoop._diag_*` remain as properties onto it, because drivers set them
on the loop object.

## 5. Why each move is byte-identical

Blocks are moved by text slicing with asserted renames; call order is kept —
including the order of `robot.update` calls, which matters because
`RobotInterface` caches the last state for `get_contact_jacobians`. Verified per
commit with `gate/local_ref.py check` + `gate/dock_check.py`.

## Public API

| symbol | signature | canonical? | code |
|---|---|---|---|
| **`DiagHooks`** *(dataclass)* |  |  | [L28](../../../crawlbot/control/controller.py#L28) |
|   `pure_pd` | `False` | _field_ | [L38](../../../crawlbot/control/controller.py#L38) |
|   `freeze_ref` | `False` | _field_ | [L39](../../../crawlbot/control/controller.py#L39) |
|   `disable_aocs` | `False` | _field_ | [L40](../../../crawlbot/control/controller.py#L40) |
|   `lock_arm_joints` | `False` | _field_ | [L41](../../../crawlbot/control/controller.py#L41) |
| `ee_data` | `(rs, arm)` | **yes** | [L44](../../../crawlbot/control/controller.py#L44) |
| **`ControlIntent`** *(dataclass)* |  |  | [L53](../../../crawlbot/control/controller.py#L53) |
|   `t` | `` | _field_ | [L65](../../../crawlbot/control/controller.py#L65) |
|   `phase` | `` | _field_ | [L66](../../../crawlbot/control/controller.py#L66) |
|   `step_idx` | `` | _field_ | [L67](../../../crawlbot/control/controller.py#L67) |
|   `contact` | `` | _field_ | [L68](../../../crawlbot/control/controller.py#L68) |
|   `contact_nmpc` | `` | _field_ | [L69](../../../crawlbot/control/controller.py#L69) |
|   `stance_anchors` | `` | _field_ | [L70](../../../crawlbot/control/controller.py#L70) |
|   `swing_arm` | `` | _field_ | [L71](../../../crawlbot/control/controller.py#L71) |
|   `ss_end` | `` | _field_ | [L72](../../../crawlbot/control/controller.py#L72) |
|   `settle_mode` | `False` | _field_ | [L73](../../../crawlbot/control/controller.py#L73) |
|   `passivity_hold` | `False` | _field_ | [L74](../../../crawlbot/control/controller.py#L74) |
|   `passivity_override` | `None` | _field_ | [L75](../../../crawlbot/control/controller.py#L75) |
|   `ds_centroidal_active` | `False` | _field_ | [L76](../../../crawlbot/control/controller.py#L76) |
| **`NMPCPlan`** *(dataclass)* |  |  | [L80](../../../crawlbot/control/controller.py#L80) |
|   `rs` | `` | _field_ | [L92](../../../crawlbot/control/controller.py#L92) |
|   `L_com_now` | `` | _field_ | [L93](../../../crawlbot/control/controller.py#L93) |
|   `cref_r` | `` | _field_ | [L94](../../../crawlbot/control/controller.py#L94) |
|   `cref_v` | `` | _field_ | [L95](../../../crawlbot/control/controller.py#L95) |
|   `rp` | `` | _field_ | [L96](../../../crawlbot/control/controller.py#L96) |
|   `vp` | `` | _field_ | [L97](../../../crawlbot/control/controller.py#L97) |
|   `lr` | `` | _field_ | [L98](../../../crawlbot/control/controller.py#L98) |
|   `af` | `` | _field_ | [L99](../../../crawlbot/control/controller.py#L99) |
|   `rp_k0` | `` | _field_ | [L100](../../../crawlbot/control/controller.py#L100) |
|   `rp_k1` | `` | _field_ | [L101](../../../crawlbot/control/controller.py#L101) |
|   `vp_k0` | `` | _field_ | [L102](../../../crawlbot/control/controller.py#L102) |
|   `vp_k1` | `` | _field_ | [L103](../../../crawlbot/control/controller.py#L103) |
|   `ok` | `` | _field_ | [L104](../../../crawlbot/control/controller.py#L104) |
|   `status_code` | `` | _field_ | [L105](../../../crawlbot/control/controller.py#L105) |
|   `cost` | `` | _field_ | [L106](../../../crawlbot/control/controller.py#L106) |
|   `info` | `` | _field_ | [L107](../../../crawlbot/control/controller.py#L107) |
|   `t_ms` | `` | _field_ | [L108](../../../crawlbot/control/controller.py#L108) |
|   `L_com_ref` | `` | _field_ | [L109](../../../crawlbot/control/controller.py#L109) |
|   `t_mid` | `` | _field_ | [L110](../../../crawlbot/control/controller.py#L110) |
| **`QPCarry`** *(dataclass)* |  |  | [L114](../../../crawlbot/control/controller.py#L114) |
|   `tau_last` | `` | _field_ | [L124](../../../crawlbot/control/controller.py#L124) |
|   `tau_w_last` | `` | _field_ | [L125](../../../crawlbot/control/controller.py#L125) |
|   `transport_mag_last` | `` | _field_ | [L126](../../../crawlbot/control/controller.py#L126) |
|   `qp_ok` | `` | _field_ | [L127](../../../crawlbot/control/controller.py#L127) |
|   `hw` | `` | _field_ | [L128](../../../crawlbot/control/controller.py#L128) |
|   `lr` | `` | _field_ | [L129](../../../crawlbot/control/controller.py#L129) |
|   `af` | `` | _field_ | [L130](../../../crawlbot/control/controller.py#L130) |
|   `lambda_qp_sol` | `None` | _field_ | [L131](../../../crawlbot/control/controller.py#L131) |
|   `p_torso_ref_used` | `None` | _field_ | [L132](../../../crawlbot/control/controller.py#L132) |
| **`TrackOut`** *(dataclass)* |  |  | [L136](../../../crawlbot/control/controller.py#L136) |
|   `tau` | `` | _field_ | [L144](../../../crawlbot/control/controller.py#L144) |
|   `tau_w` | `` | _field_ | [L145](../../../crawlbot/control/controller.py#L145) |
|   `tau_raw` | `` | _field_ | [L146](../../../crawlbot/control/controller.py#L146) |
|   `rs` | `` | _field_ | [L147](../../../crawlbot/control/controller.py#L147) |
|   `qdd_t` | `` | _field_ | [L148](../../../crawlbot/control/controller.py#L148) |
|   `lambda_qp` | `` | _field_ | [L149](../../../crawlbot/control/controller.py#L149) |
|   `qp_ok` | `` | _field_ | [L150](../../../crawlbot/control/controller.py#L150) |
|   `passivity_active` | `` | _field_ | [L151](../../../crawlbot/control/controller.py#L151) |
|   `rp_interp` | `` | _field_ | [L152](../../../crawlbot/control/controller.py#L152) |
|   `p_torso_ref_used` | `` | _field_ | [L153](../../../crawlbot/control/controller.py#L153) |
| **`WholeBodyController`** |  |  | [L156](../../../crawlbot/control/controller.py#L156) |
| `.plan` | `(intent, refs)` | **yes** | [L174](../../../crawlbot/control/controller.py#L174) |
| `.begin_tracking` | `(plan, hw)` | **yes** | [L298](../../../crawlbot/control/controller.py#L298) |
| `.track` | `(carry, qs, tq, intent, refs, plan)` | **yes** | [L308](../../../crawlbot/control/controller.py#L308) |
| `.after_step` | `(carry, tau)` | **yes** | [L486](../../../crawlbot/control/controller.py#L486) |
| `.begin_settle` | `()` | **yes** | [L509](../../../crawlbot/control/controller.py#L509) |
| `.settle` | `(rs, cc_ds, hw_current, fallback_Kd)` | **yes** | [L513](../../../crawlbot/control/controller.py#L513) |

## Code map

| unit | source |
|---|---|
| `class DiagHooks` | [L28-41](../../../crawlbot/control/controller.py#L28-L41) |
| `ee_data()` | [L44-49](../../../crawlbot/control/controller.py#L44-L49) |
| `class ControlIntent` | [L53-76](../../../crawlbot/control/controller.py#L53-L76) |
| `class NMPCPlan` | [L80-110](../../../crawlbot/control/controller.py#L80-L110) |
| `class QPCarry` | [L114-132](../../../crawlbot/control/controller.py#L114-L132) |
| `class TrackOut` | [L136-153](../../../crawlbot/control/controller.py#L136-L153) |
| `class WholeBodyController` | [L156-588](../../../crawlbot/control/controller.py#L156-L588) |
| `WholeBodyController.plan` | [L174-294](../../../crawlbot/control/controller.py#L174-L294) |
| `WholeBodyController.begin_tracking` | [L298-306](../../../crawlbot/control/controller.py#L298-L306) |
| `WholeBodyController.track` | [L308-484](../../../crawlbot/control/controller.py#L308-L484) |
| `WholeBodyController.after_step` | [L486-505](../../../crawlbot/control/controller.py#L486-L505) |
| `WholeBodyController.begin_settle` | [L509-511](../../../crawlbot/control/controller.py#L509-L511) |
| `WholeBodyController.settle` | [L513-588](../../../crawlbot/control/controller.py#L513-L588) |

---

## See also

- the AOCS block: [`attitude.md`](attitude.md)
- the solvers it drives: [`solvers/centroidal_nmpc.md`](../solvers/centroidal_nmpc.md),
  [`solvers/wholebody_qp.md`](../solvers/wholebody_qp.md)
- package overview: [`control.md`](control.md)
