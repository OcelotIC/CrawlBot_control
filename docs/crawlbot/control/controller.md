# `crawlbot.control.controller`

**File**: [`crawlbot/control/controller.py`](../../../crawlbot/control/controller.py) — **757 lines** — canonical coverage **80 %**

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
   NMPC knots; **torso reference**: `refs.torso_at(tq)` then, outside two-task
   SS, the CoM→torso δ-mapping with the F-SAT jitter guard and the post-dock DS
   blend (two-task SS takes the raw quintic); **swing reference**
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

**Events.** The mapping layer's reference-shaping state (SS-entry torso position,
post-dock DS blend, F-SAT and δ caches) lives here; the loop signals the two
transitions that touch it with `on_ss_entry(p)` and `on_dock(t)`.

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
| **`DiagHooks`** *(dataclass)* |  |  | [L27](../../../crawlbot/control/controller.py#L27) |
|   `pure_pd` | `False` | _field_ | [L37](../../../crawlbot/control/controller.py#L37) |
|   `freeze_ref` | `False` | _field_ | [L38](../../../crawlbot/control/controller.py#L38) |
|   `disable_aocs` | `False` | _field_ | [L39](../../../crawlbot/control/controller.py#L39) |
|   `lock_arm_joints` | `False` | _field_ | [L40](../../../crawlbot/control/controller.py#L40) |
| `ee_data` | `(rs, arm)` | **yes** | [L43](../../../crawlbot/control/controller.py#L43) |
| **`ControlIntent`** *(dataclass)* |  |  | [L52](../../../crawlbot/control/controller.py#L52) |
|   `t` | `` | _field_ | [L64](../../../crawlbot/control/controller.py#L64) |
|   `phase` | `` | _field_ | [L65](../../../crawlbot/control/controller.py#L65) |
|   `step_idx` | `` | _field_ | [L66](../../../crawlbot/control/controller.py#L66) |
|   `contact` | `` | _field_ | [L67](../../../crawlbot/control/controller.py#L67) |
|   `contact_nmpc` | `` | _field_ | [L68](../../../crawlbot/control/controller.py#L68) |
|   `stance_anchors` | `` | _field_ | [L69](../../../crawlbot/control/controller.py#L69) |
|   `swing_arm` | `` | _field_ | [L70](../../../crawlbot/control/controller.py#L70) |
|   `ss_end` | `` | _field_ | [L71](../../../crawlbot/control/controller.py#L71) |
|   `settle_mode` | `False` | _field_ | [L72](../../../crawlbot/control/controller.py#L72) |
|   `passivity_hold` | `False` | _field_ | [L73](../../../crawlbot/control/controller.py#L73) |
|   `passivity_override` | `None` | _field_ | [L74](../../../crawlbot/control/controller.py#L74) |
|   `ds_centroidal_active` | `False` | _field_ | [L75](../../../crawlbot/control/controller.py#L75) |
| **`NMPCPlan`** *(dataclass)* |  |  | [L79](../../../crawlbot/control/controller.py#L79) |
|   `rs` | `` | _field_ | [L91](../../../crawlbot/control/controller.py#L91) |
|   `L_com_now` | `` | _field_ | [L92](../../../crawlbot/control/controller.py#L92) |
|   `cref_r` | `` | _field_ | [L93](../../../crawlbot/control/controller.py#L93) |
|   `cref_v` | `` | _field_ | [L94](../../../crawlbot/control/controller.py#L94) |
|   `rp` | `` | _field_ | [L95](../../../crawlbot/control/controller.py#L95) |
|   `vp` | `` | _field_ | [L96](../../../crawlbot/control/controller.py#L96) |
|   `lr` | `` | _field_ | [L97](../../../crawlbot/control/controller.py#L97) |
|   `af` | `` | _field_ | [L98](../../../crawlbot/control/controller.py#L98) |
|   `rp_k0` | `` | _field_ | [L99](../../../crawlbot/control/controller.py#L99) |
|   `rp_k1` | `` | _field_ | [L100](../../../crawlbot/control/controller.py#L100) |
|   `vp_k0` | `` | _field_ | [L101](../../../crawlbot/control/controller.py#L101) |
|   `vp_k1` | `` | _field_ | [L102](../../../crawlbot/control/controller.py#L102) |
|   `ok` | `` | _field_ | [L103](../../../crawlbot/control/controller.py#L103) |
|   `status_code` | `` | _field_ | [L104](../../../crawlbot/control/controller.py#L104) |
|   `cost` | `` | _field_ | [L105](../../../crawlbot/control/controller.py#L105) |
|   `info` | `` | _field_ | [L106](../../../crawlbot/control/controller.py#L106) |
|   `t_ms` | `` | _field_ | [L107](../../../crawlbot/control/controller.py#L107) |
|   `L_com_ref` | `` | _field_ | [L108](../../../crawlbot/control/controller.py#L108) |
|   `t_mid` | `` | _field_ | [L109](../../../crawlbot/control/controller.py#L109) |
| **`QPCarry`** *(dataclass)* |  |  | [L113](../../../crawlbot/control/controller.py#L113) |
|   `tau_last` | `` | _field_ | [L123](../../../crawlbot/control/controller.py#L123) |
|   `tau_w_last` | `` | _field_ | [L124](../../../crawlbot/control/controller.py#L124) |
|   `transport_mag_last` | `` | _field_ | [L125](../../../crawlbot/control/controller.py#L125) |
|   `qp_ok` | `` | _field_ | [L126](../../../crawlbot/control/controller.py#L126) |
|   `hw` | `` | _field_ | [L127](../../../crawlbot/control/controller.py#L127) |
|   `lr` | `` | _field_ | [L128](../../../crawlbot/control/controller.py#L128) |
|   `af` | `` | _field_ | [L129](../../../crawlbot/control/controller.py#L129) |
|   `lambda_qp_sol` | `None` | _field_ | [L130](../../../crawlbot/control/controller.py#L130) |
|   `p_torso_ref_used` | `None` | _field_ | [L131](../../../crawlbot/control/controller.py#L131) |
| **`TrackOut`** *(dataclass)* |  |  | [L135](../../../crawlbot/control/controller.py#L135) |
|   `tau` | `` | _field_ | [L143](../../../crawlbot/control/controller.py#L143) |
|   `tau_w` | `` | _field_ | [L144](../../../crawlbot/control/controller.py#L144) |
|   `tau_raw` | `` | _field_ | [L145](../../../crawlbot/control/controller.py#L145) |
|   `rs` | `` | _field_ | [L146](../../../crawlbot/control/controller.py#L146) |
|   `qdd_t` | `` | _field_ | [L147](../../../crawlbot/control/controller.py#L147) |
|   `lambda_qp` | `` | _field_ | [L148](../../../crawlbot/control/controller.py#L148) |
|   `qp_ok` | `` | _field_ | [L149](../../../crawlbot/control/controller.py#L149) |
|   `passivity_active` | `` | _field_ | [L150](../../../crawlbot/control/controller.py#L150) |
|   `rp_interp` | `` | _field_ | [L151](../../../crawlbot/control/controller.py#L151) |
|   `p_torso_ref_used` | `` | _field_ | [L152](../../../crawlbot/control/controller.py#L152) |
| **`WholeBodyController`** |  |  | [L155](../../../crawlbot/control/controller.py#L155) |
| `.plan` | `(intent, refs)` | **yes** | [L209](../../../crawlbot/control/controller.py#L209) |
| `.begin_tracking` | `(plan, hw)` | **yes** | [L333](../../../crawlbot/control/controller.py#L333) |
| `.track` | `(carry, qs, tq, intent, refs, plan)` | **yes** | [L343](../../../crawlbot/control/controller.py#L343) |
| `.after_step` | `(carry, tau)` | **yes** | [L634](../../../crawlbot/control/controller.py#L634) |
| `.on_ss_entry` | `(p_torso_entry)` | **yes** | [L657](../../../crawlbot/control/controller.py#L657) |
| `.on_dock` | `(t)` | **yes** | [L665](../../../crawlbot/control/controller.py#L665) |
| `.begin_settle` | `()` | **yes** | [L677](../../../crawlbot/control/controller.py#L677) |
| `.settle` | `(rs, cc_ds, hw_current, fallback_Kd)` | **yes** | [L681](../../../crawlbot/control/controller.py#L681) |

## Code map

| unit | source |
|---|---|
| `class DiagHooks` | [L27-40](../../../crawlbot/control/controller.py#L27-L40) |
| `ee_data()` | [L43-48](../../../crawlbot/control/controller.py#L43-L48) |
| `class ControlIntent` | [L52-75](../../../crawlbot/control/controller.py#L52-L75) |
| `class NMPCPlan` | [L79-109](../../../crawlbot/control/controller.py#L79-L109) |
| `class QPCarry` | [L113-131](../../../crawlbot/control/controller.py#L113-L131) |
| `class TrackOut` | [L135-152](../../../crawlbot/control/controller.py#L135-L152) |
| `class WholeBodyController` | [L155-756](../../../crawlbot/control/controller.py#L155-L756) |
| `WholeBodyController.plan` | [L209-329](../../../crawlbot/control/controller.py#L209-L329) |
| `WholeBodyController.begin_tracking` | [L333-341](../../../crawlbot/control/controller.py#L333-L341) |
| `WholeBodyController.track` | [L343-632](../../../crawlbot/control/controller.py#L343-L632) |
| `WholeBodyController.after_step` | [L634-653](../../../crawlbot/control/controller.py#L634-L653) |
| `WholeBodyController.on_ss_entry` | [L657-663](../../../crawlbot/control/controller.py#L657-L663) |
| `WholeBodyController.on_dock` | [L665-673](../../../crawlbot/control/controller.py#L665-L673) |
| `WholeBodyController.begin_settle` | [L677-679](../../../crawlbot/control/controller.py#L677-L679) |
| `WholeBodyController.settle` | [L681-756](../../../crawlbot/control/controller.py#L681-L756) |

---

## See also

- the AOCS block: [`attitude.md`](attitude.md)
- the solvers it drives: [`solvers/centroidal_nmpc.md`](../solvers/centroidal_nmpc.md),
  [`solvers/wholebody_qp.md`](../solvers/wholebody_qp.md)
- package overview: [`control.md`](control.md)
