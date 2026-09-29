# `crawlbot.simulation.sim_loop`

**File**: [`crawlbot/simulation/sim_loop.py`](../../../crawlbot/simulation/sim_loop.py) — **2404 lines** — canonical coverage **88 %**

> Module docstring: *"SimulationLoop — Closed-loop MuJoCo simulation with two-stage controller."*

**The closed loop.** DS/SS state machine, orchestration of planners and
solvers, logging. The AOCS is in [`control/attitude.py`](../control/attitude.md).
Every MuJoCo **write** (step, welds, dock impact,
actuators, diagnostic lock) now goes through [`plant.py`](plant.md), and every
MuJoCo **read** on the command path through [`sensors.py`](sensors.md) —
extractions 1–2 of the `refactor/sim-loop-split` chantier. The largest file in the repository and
the one carrying the most architectural history.

---

## Public API

| symbol | signature | canonical? | code |
|---|---|---|---|
| **`_DSSettle`** *(dataclass)* |  |  | [L103](../../../crawlbot/simulation/sim_loop.py#L103) |
|   `contact_config` | `` | _field_ | [L125](../../../crawlbot/simulation/sim_loop.py#L125) |
|   `max_steps` | `` | _field_ | [L126](../../../crawlbot/simulation/sim_loop.py#L126) |
|   `epsilon_v` | `` | _field_ | [L127](../../../crawlbot/simulation/sim_loop.py#L127) |
|   `plateau_window` | `50` | _field_ | [L128](../../../crawlbot/simulation/sim_loop.py#L128) |
|   `plateau_ratio` | `0.999` | _field_ | [L129](../../../crawlbot/simulation/sim_loop.py#L129) |
|   `min_steps` | `0` | _field_ | [L130](../../../crawlbot/simulation/sim_loop.py#L130) |
|   `fallback_Kd` | `20.0` | _field_ | [L131](../../../crawlbot/simulation/sim_loop.py#L131) |
|   `t_log` | `None` | _field_ | [L132](../../../crawlbot/simulation/sim_loop.py#L132) |
|   `T_log` | `None` | _field_ | [L133](../../../crawlbot/simulation/sim_loop.py#L133) |
|   `t_log_step_offset` | `0` | _field_ | [L134](../../../crawlbot/simulation/sim_loop.py#L134) |
|   `log_obj` | `None` | _field_ | [L135](../../../crawlbot/simulation/sim_loop.py#L135) |
|   `log_step_idx` | `-1` | _field_ | [L136](../../../crawlbot/simulation/sim_loop.py#L136) |
|   `log_just_landed_arm` | `''` | _field_ | [L137](../../../crawlbot/simulation/sim_loop.py#L137) |
|   `log_anchor_a_idx` | `-1` | _field_ | [L138](../../../crawlbot/simulation/sim_loop.py#L138) |
|   `log_anchor_b_idx` | `-1` | _field_ | [L139](../../../crawlbot/simulation/sim_loop.py#L139) |
|   `log_t_abs` | `0.0` | _field_ | [L140](../../../crawlbot/simulation/sim_loop.py#L140) |
| **`_DSRun`** *(dataclass)* |  |  | [L144](../../../crawlbot/simulation/sim_loop.py#L144) |
|   `req` | `` | _field_ | [L146](../../../crawlbot/simulation/sim_loop.py#L146) |
|   `lambda_min` | `` | _field_ | [L147](../../../crawlbot/simulation/sim_loop.py#L147) |
|   `T_settle` | `` | _field_ | [L148](../../../crawlbot/simulation/sim_loop.py#L148) |
|   `T_start` | `` | _field_ | [L149](../../../crawlbot/simulation/sim_loop.py#L149) |
|   `hw_current` | `` | _field_ | [L150](../../../crawlbot/simulation/sim_loop.py#L150) |
|   `T_history` | `field(default_factory=list)` | _field_ | [L151](../../../crawlbot/simulation/sim_loop.py#L151) |
|   `exit_reason` | `'max_steps'` | _field_ | [L152](../../../crawlbot/simulation/sim_loop.py#L152) |
|   `k` | `0` | _field_ | [L153](../../../crawlbot/simulation/sim_loop.py#L153) |
| **`_NMPCTick`** *(dataclass)* |  |  | [L157](../../../crawlbot/simulation/sim_loop.py#L157) |
|   `t` | `` | _field_ | [L167](../../../crawlbot/simulation/sim_loop.py#L167) |
|   `phase` | `` | _field_ | [L168](../../../crawlbot/simulation/sim_loop.py#L168) |
|   `step_idx` | `` | _field_ | [L169](../../../crawlbot/simulation/sim_loop.py#L169) |
|   `swing_arm` | `` | _field_ | [L170](../../../crawlbot/simulation/sim_loop.py#L170) |
|   `stance_arm` | `` | _field_ | [L171](../../../crawlbot/simulation/sim_loop.py#L171) |
|   `cc_ss` | `` | _field_ | [L172](../../../crawlbot/simulation/sim_loop.py#L172) |
|   `target_anchor` | `` | _field_ | [L173](../../../crawlbot/simulation/sim_loop.py#L173) |
|   `stance_a` | `` | _field_ | [L174](../../../crawlbot/simulation/sim_loop.py#L174) |
|   `stance_b` | `` | _field_ | [L175](../../../crawlbot/simulation/sim_loop.py#L175) |
|   `hw` | `` | _field_ | [L176](../../../crawlbot/simulation/sim_loop.py#L176) |
|   `L_com_prev` | `` | _field_ | [L177](../../../crawlbot/simulation/sim_loop.py#L177) |
|   `log` | `` | _field_ | [L178](../../../crawlbot/simulation/sim_loop.py#L178) |
|   `ss_end` | `None` | _field_ | [L179](../../../crawlbot/simulation/sim_loop.py#L179) |
|   `settle_mode` | `False` | _field_ | [L180](../../../crawlbot/simulation/sim_loop.py#L180) |
|   `passivity_hold` | `False` | _field_ | [L181](../../../crawlbot/simulation/sim_loop.py#L181) |
|   `passivity_override` | `None` | _field_ | [L182](../../../crawlbot/simulation/sim_loop.py#L182) |
|   `ds_centroidal_active` | `False` | _field_ | [L183](../../../crawlbot/simulation/sim_loop.py#L183) |
| **`_NMPCRun`** *(dataclass)* |  |  | [L187](../../../crawlbot/simulation/sim_loop.py#L187) |
|   `t` | `` | _field_ | [L189](../../../crawlbot/simulation/sim_loop.py#L189) |
|   `phase` | `` | _field_ | [L190](../../../crawlbot/simulation/sim_loop.py#L190) |
|   `step_idx` | `` | _field_ | [L191](../../../crawlbot/simulation/sim_loop.py#L191) |
|   `swing_arm` | `` | _field_ | [L192](../../../crawlbot/simulation/sim_loop.py#L192) |
|   `stance_arm` | `` | _field_ | [L193](../../../crawlbot/simulation/sim_loop.py#L193) |
|   `stance_a` | `` | _field_ | [L194](../../../crawlbot/simulation/sim_loop.py#L194) |
|   `stance_b` | `` | _field_ | [L195](../../../crawlbot/simulation/sim_loop.py#L195) |
|   `target_anchor` | `` | _field_ | [L196](../../../crawlbot/simulation/sim_loop.py#L196) |
|   `log` | `` | _field_ | [L197](../../../crawlbot/simulation/sim_loop.py#L197) |
|   `ss_end` | `` | _field_ | [L198](../../../crawlbot/simulation/sim_loop.py#L198) |
|   `settle_mode` | `` | _field_ | [L199](../../../crawlbot/simulation/sim_loop.py#L199) |
|   `L_com_prev` | `` | _field_ | [L200](../../../crawlbot/simulation/sim_loop.py#L200) |
|   `intent` | `` | _field_ | [L201](../../../crawlbot/simulation/sim_loop.py#L201) |
|   `plan` | `` | _field_ | [L202](../../../crawlbot/simulation/sim_loop.py#L202) |
|   `carry` | `` | _field_ | [L203](../../../crawlbot/simulation/sim_loop.py#L203) |
|   `vp` | `` | _field_ | [L204](../../../crawlbot/simulation/sim_loop.py#L204) |
|   `cref_r` | `` | _field_ | [L205](../../../crawlbot/simulation/sim_loop.py#L205) |
|   `nmpc_ok` | `` | _field_ | [L206](../../../crawlbot/simulation/sim_loop.py#L206) |
|   `nmpc_status_code` | `` | _field_ | [L207](../../../crawlbot/simulation/sim_loop.py#L207) |
|   `nmpc_cost_val` | `` | _field_ | [L208](../../../crawlbot/simulation/sim_loop.py#L208) |
|   `info_n` | `` | _field_ | [L209](../../../crawlbot/simulation/sim_loop.py#L209) |
|   `t_nmpc_ms` | `` | _field_ | [L210](../../../crawlbot/simulation/sim_loop.py#L210) |
|   `t_qp_start` | `` | _field_ | [L211](../../../crawlbot/simulation/sim_loop.py#L211) |
|   `qs` | `0` | _field_ | [L212](../../../crawlbot/simulation/sim_loop.py#L212) |
| **`PlannerReferences`** |  |  | [L215](../../../crawlbot/simulation/sim_loop.py#L215) |
| `.com_at` | `(t, settle_mode)` | **yes** | [L231](../../../crawlbot/simulation/sim_loop.py#L231) |
| `.L_com_at` | `(t_mid)` | **yes** | [L290](../../../crawlbot/simulation/sim_loop.py#L290) |
| `.torso_at` | `(tq, phase, ss_end)` | **yes** | [L294](../../../crawlbot/simulation/sim_loop.py#L294) |
| `.swing_at` | `(tq, phase, ss_end)` | **yes** | [L309](../../../crawlbot/simulation/sim_loop.py#L309) |
| **`SimulationLoop`** |  |  | [L317](../../../crawlbot/simulation/sim_loop.py#L317) |
| `.mj_model` | `()` | **yes** | [L417](../../../crawlbot/simulation/sim_loop.py#L417) |
| `.mj_data` | `()` | **yes** | [L421](../../../crawlbot/simulation/sim_loop.py#L421) |
| `._sat_total_calls` | `()` | **yes** | [L427](../../../crawlbot/simulation/sim_loop.py#L427) |
| `._sat_clipped_calls` | `()` | **yes** | [L431](../../../crawlbot/simulation/sim_loop.py#L431) |
| `._sat_max_clip_mm` | `()` | **yes** | [L435](../../../crawlbot/simulation/sim_loop.py#L435) |
| `._diag_pure_pd` | `()` | not exercised | [L440](../../../crawlbot/simulation/sim_loop.py#L440) |
| `._diag_pure_pd` | `(v)` | not exercised | [L444](../../../crawlbot/simulation/sim_loop.py#L444) |
| `._diag_freeze_ref` | `()` | not exercised | [L448](../../../crawlbot/simulation/sim_loop.py#L448) |
| `._diag_freeze_ref` | `(v)` | not exercised | [L452](../../../crawlbot/simulation/sim_loop.py#L452) |
| `._diag_disable_aocs` | `()` | not exercised | [L456](../../../crawlbot/simulation/sim_loop.py#L456) |
| `._diag_disable_aocs` | `(v)` | not exercised | [L460](../../../crawlbot/simulation/sim_loop.py#L460) |
| `._diag_lock_arm_joints` | `()` | **yes** | [L464](../../../crawlbot/simulation/sim_loop.py#L464) |
| `._diag_lock_arm_joints` | `(v)` | not exercised | [L468](../../../crawlbot/simulation/sim_loop.py#L468) |
| `.setup` | `(n_steps=3, start_a=2, start_b=2, sequence_path=None)` | **yes** | [L473](../../../crawlbot/simulation/sim_loop.py#L473) |
| `._settle_setup` | `(start_a, start_b)` | **yes** | [L777](../../../crawlbot/simulation/sim_loop.py#L777) |
| `._run_ds_passivity_loop` | `(**kw)` | **yes** | [L862](../../../crawlbot/simulation/sim_loop.py#L862) |
| `._ds_begin` | `(r)` | **yes** | [L871](../../../crawlbot/simulation/sim_loop.py#L871) |
| `._ds_tick` | `(st)` | **yes** | [L896](../../../crawlbot/simulation/sim_loop.py#L896) |
| `._ds_end` | `(st, k_last)` | **yes** | [L951](../../../crawlbot/simulation/sim_loop.py#L951) |
| `._build_qp` | `(ae, ap, aw, kpc, kdc, kpt, kdt, kpe, kde, kpe_ang=5.0, ...)` | **yes** | [L969](../../../crawlbot/simulation/sim_loop.py#L969) |
| `._gripper_distance` | `(arm, anchor_idx)` | **yes** | [L1024](../../../crawlbot/simulation/sim_loop.py#L1024) |
| `._gripper_speed` | `(arm)` | not exercised | [L1028](../../../crawlbot/simulation/sim_loop.py#L1028) |
| `._gripper_ori_err_deg` | `(arm, anchor_idx)` | **yes** | [L1043](../../../crawlbot/simulation/sim_loop.py#L1043) |
| `._dock_gate` | `(swing_arm, target_idx, log, t, step_idx)` | **yes** | [L1058](../../../crawlbot/simulation/sim_loop.py#L1058) |
| `._setup_torso_for_step` | `(t_ss_start, swing_arm, stance_a, stance_b, target_arm, ...)` | **yes** | [L1095](../../../crawlbot/simulation/sim_loop.py#L1095) |
| `._run_preplanner` | `(t_plan_start, stance_arm, stance_a, stance_b, r_com_0, ...)` | **yes** | [L1310](../../../crawlbot/simulation/sim_loop.py#L1310) |
| `._capture_snapshot` | `(log, t, label)` | **yes** | [L1415](../../../crawlbot/simulation/sim_loop.py#L1415) |
| `.run` | `(verbose=True)` | **yes** | [L1420](../../../crawlbot/simulation/sim_loop.py#L1420) |
| `._drive` | `(program)` | **yes** | [L1430](../../../crawlbot/simulation/sim_loop.py#L1430) |
| `._begin` | `(req)` | **yes** | [L1459](../../../crawlbot/simulation/sim_loop.py#L1459) |
| `._gait_program` | `(verbose=True)` | **yes** | [L1467](../../../crawlbot/simulation/sim_loop.py#L1467) |
| `._swing_query_time` | `(t_raw, phase, ss_end)` | **yes** | [L2013](../../../crawlbot/simulation/sim_loop.py#L2013) |
| `._step` | `(t, phase, step_idx, swing_arm, stance_arm, cc_ss, targe...)` | not exercised | [L2031](../../../crawlbot/simulation/sim_loop.py#L2031) |
| `._nmpc_begin` | `(r)` | **yes** | [L2050](../../../crawlbot/simulation/sim_loop.py#L2050) |
| `._nmpc_tick` | `(st)` | **yes** | [L2127](../../../crawlbot/simulation/sim_loop.py#L2127) |
| `._qp_substep` | `(st)` | **yes** | [L2136](../../../crawlbot/simulation/sim_loop.py#L2136) |
| `._nmpc_handoff` | `(st)` | **yes** | [L2309](../../../crawlbot/simulation/sim_loop.py#L2309) |
| `._get_ee_data` | `(rs, arm)` | **yes** | [L2356](../../../crawlbot/simulation/sim_loop.py#L2356) |
| `._print_summary` | `(log)` | **yes** | [L2363](../../../crawlbot/simulation/sim_loop.py#L2363) |
| `.plot` | `(log, save_path=None, cfg=None)` | not exercised | [L2402](../../../crawlbot/simulation/sim_loop.py#L2402) |

---

---

## 1. Two phases, not three

`DS` (double support) and `SS` (single support). Explicit project rule: *do not
implement a three-phase state machine (DS/SS/EXT) — the architecture is two-phase
per spec 7.1.*

```
setup()                          plant, sensors, planners, controller, settle
  |
  +-- run() = _drive(_gait_program())
        _gait_program()   the SEQUENCER (coroutine) — decides what runs next:
          yield _DSSettle          DS: passive settle (energy exit)
          [yield _NMPCTick ...]    DWELL, if the planned DS is long
          _setup_torso_for_step()  docking IK + torso phase
            +-- _run_preplanner()  T_step + feasible CoM trajectory
          yield _NMPCTick ...      SS: the swing, then HOLD if not docked
          _dock_gate() -> plant.activate_weld() + plant.apply_dock_impact()
          ... trailing DS: yield _NMPCTick ...
        _drive()          THE loop — one control period dt_qp per iteration
```

### The single loop (refactor/sim-loop-split, extraction 4)

Before this extraction there were three physics loops: the DS passivity loop,
`_step`'s QP sub-loop, and `run()`'s traversal `while` wrapped around both.
Now `_drive` is the only place the plant is stepped. Each iteration advances the
active request by one dt_qp — `_ds_tick` (measure, exit checks, settle QP,
plant step, log row) or `_qp_substep` (controller.track, traces, plant I/O,
plant step, after_step) — or completes it (`_ds_end`, `_nmpc_handoff`). The
result goes back into the sequencer through the `yield`.

The sequencer is `run()`'s former body **verbatim**: only the two call kinds
became `yield`s, so every time expression (`t += dt_nmpc`, `t += n·dt_qp`,
`log_t_abs + (k+1)·dt`) is unchanged. That is what made a bit-identical proof
possible, and it is also the ROS 2 shape: `_drive`'s body is a timer callback,
and the sequencer is resumed when a request completes.

Kept as found: an exit check that fires at DS iteration `k` returns without
applying control, yet `n_steps = k + 1` counts it (as the old `for k … break`
did).

**Verification.** The canonical replay cannot see DWELL, SKIP, TIMEOUT,
`stop_on_failed_step` or the `diag_*_on_abort` overrides, so each was forced in
a shortened scenario and replayed on the pre-extraction tree and on this one:
every output file bit-identical, stdout identical line for line (wall-clock
masked). Harness: `gate/_run/scenarios.py` (local).

## 2. Per-step sequence, and why the order is forced

1. **Docking IK** gives the target configuration and therefore `r_com_goal`.
2. **Pre-planner** needs that goal to compute `T_step` — so it must come second.
3. **`set_step_duration(T_step)`** installs the duration and cascades the
   timeline — so planners must be configured third.
4. **Torso and swing phases** are built over `[t_ss_start, t_ss_start + T_step]`,
   sharing one horizon. This is what keeps the two references synchronised.
5. Only then can `_step()` run.

Point 4 is a project rule in itself: *do not freeze references or add
threshold-based switches to handle trajectory coordination failures — fix the
synchronisation instead.* The shared horizon is that fix.

## 3. The docking gate

Rule: *do not activate welds on position alone — require both `d < 5 mm` AND
`ori < 5 deg`.* `_dock_gate` applies both; `plant.activate_weld` runs only after, followed by
the full-DOF inelastic impact (`plant.apply_dock_impact`, see [`plant.md`](plant.md)).

**Rule 10 — the metric is the one at weld time.** Docking precision is the
`d_mm` recorded in `dock_events`, never the minimum over the swing. A closer
pass *before* docking is a fly-by artefact: on step 2 the minimum over swing was
3.0 mm while the actual at-weld distance was **4.89 mm**. Reporting the first
would overstate precision by 40 %.

Canonical result: 6/6 at 4.02 / 4.89 / 4.99 / 4.97 / 4.95 / 4.62 mm, worst
margin **0.01 mm** against a 5 mm capture radius.

## 4. DS: passivity rather than a cost

The DS settle (`_DSSettle`, ticked by `_ds_tick`) dissipates residual energy through a passivity
**inequality** in the QP rather than a damping cost. The distinction matters: a
cost trades against the other tasks and can be outvoted; an inequality cannot.
It guarantees the energy budget is non-increasing whatever the task weights do.

## 5. The NMPC period (formerly `_step()`, the largest block)

1014 lines before CLEANUP-31 lifted the logging tail into `_log_ss_tick`, the
single-support counterpart of the long-standing `_log_ds_tick` (203 lines against
its 210 — the asymmetry was drift, not design). 851 after, then 878 once
CLEANUP-34 added the four phase banners — deliberately longer and much easier to
navigate.

Those banners are the fastest way in. Four blocks, in order: **read state and
references** → **STAGE 1, centroidal NMPC** (once per `_step`, dt 0.1 s) →
**STAGE 2, whole-body QP sub-loop** (dt 0.01 s) → **hand off to telemetry**.
Since extraction 3 the first three delegate to the controller (below).

The cut was chosen by measurement, not by eye. For every statement boundary in
`_step`, count the locals assigned before it and read after it; that number is
the signature any helper extracted there would need. Expressed as a fraction
through the method, so it does not rot the next time lines move:

```
  0 %    0
 10 %   17  #################
 20 %   25  #########################   <- NMPC + QP + integration core
 55 %   27  ###########################
 75 %   14  ##############
100 %    3  ###                          <- the tail decays to nothing
```

Three regions. The tail's monotone decay to 3 is the signature of a block that
only records.

### Why `TickState` exists

Naively, the tail's live-in set is 21 locals plus 13 of `_step`'s own arguments —
a **29-parameter** helper, which is the `solve()` debt (below) rebuilt somewhere
else. Two corrections shrink it:

- a name the tail **re-assigns before reading** is not an input. `L_dot_est` and
  `R_err` look like inputs — the head assigns both — but the tail recomputes them
  from `rs_f`. Five names dropped out this way.
- `cfg` is only ever `self.cfg`, and `log` is the destination rather than tick
  state.

What remains crosses as one record, built once at the boundary. Field names come
from where each value is **logged**, not from the head's abbreviations: `lr` is
the NMPC contact-wrench reference (`log.lambda_ref`) so it is `lambda_ref`; `vp`
is the planned CoM velocity (`log.v_com_ref`) so it is `v_com_ref`; `cref_r` is
`r_com_ref`.

One behaviour is now explicit rather than implicit: `p_torso_ref_used` is bound
only when the QP sub-loop ran, which the old code discovered by catching
`NameError` inside the logging block. As a field defaulting to `None` it is an
ordinary `is None` test — identical behaviour, said out loud.

The extraction is logging-only, so the gate settles it: artifact identity
**byte-exact** over 2077 rows x 132 928 fields, all six docks delta +0.0000.

### The core, cut (refactor/sim-loop-split, extraction 3c)

The ~25-local core now lives in [`WholeBodyController`](../control/controller.md):
the state it threads across QP sub-steps crosses as one `QPCarry` (created per
NMPC tick by `begin_tracking`), the per-tick request as one `ControlIntent`, and
stage 1's output as one `NMPCPlan`. What `_step` keeps is orchestration — build
the intent, `plan()`, then per sub-step `track()` → diagnostic traces →
`plant.apply_*` / `plant.step()` → `after_step()` — plus the hand-off to
telemetry.

The diagnostic traces (dock-work, step-2 B+C, physics trace) are pure reads and
stay in the loop, executed before `plant.step()` as before, since `rs` fields may
be views on Pinocchio data that the next `robot.update` overwrites.

`WholeBodyQP.solve()` still takes **40 parameters**, but only the controller calls
it now; the controller's own boundary is the unified-architecture one (§4.2), so
that signature can shrink without touching the loop (A1).

`run()`'s 600 lines of nesting are now the sequencer `_gait_program`; the
physics loop is `_drive` (§1).

## 6. The `use_m2_stack` trap

`SimConfig.use_m2_stack` **looks dead** — its `WholeBodyQPConfig` twin was
removed in CLEANUP-8 — but it gates two paths unrelated to the task stack:

| site | what it gates |
|---|---|
| `torso_reference.py:102-104` | torso-reference routing (delta-mapping vs raw quintic) |
| `controller.py:408-409` | `passivity_active` — **the DS passivity constraint** |

Deleting it would silently disable DS passivity. Same name, opposite fates.

## 7. Diagnostic hooks — live, keep

`_diag_freeze_ref`, `_diag_lock_arm_joints`, `_diag_pure_pd`: unexercised by the
canonical but used by scripts in `Misc/scripts/`. A third class of "unexercised"
distinct from both sediment and fallback.

## 8. Logging conventions worth knowing

- **`nmpc_ok = 0` means "not called", not "failed".** The NMPC runs only in SS
  and the terminal settle. On the canonical that is **1368 of 2077 ticks**, so a
  whole-column read gives a misleading 34 % success rate against a true
  **100 % (709/709)**.
- The CoM reference **snaps to the measured CoM** at SS->DS entry: `_log_ds_tick`
  writes `e_com = 0` with `ref := measured` (`sim_loop.py:1038-1041`). Logging
  convention; decision pending.
- The exported torso reference is **continuous** since the terminal-hold fix —
  logging only, control proven byte-identical.
- `H_rO`, `H_dot_est` and `gmo_contact_state` **carry no signal** — see
  `aocs/force_estimator.md` and `estimation/contact_estimator.md`.

Unexercised: `_gripper_speed`, `_planned_arm_config`, `plot`.

## Code map

| unit | source |
|---|---|
| `class _DSSettle` | [L103-140](../../../crawlbot/simulation/sim_loop.py#L103-L140) |
| `class _DSRun` | [L144-153](../../../crawlbot/simulation/sim_loop.py#L144-L153) |
| `class _NMPCTick` | [L157-183](../../../crawlbot/simulation/sim_loop.py#L157-L183) |
| `class _NMPCRun` | [L187-212](../../../crawlbot/simulation/sim_loop.py#L187-L212) |
| `class PlannerReferences` | [L215-314](../../../crawlbot/simulation/sim_loop.py#L215-L314) |
| `PlannerReferences.com_at` | [L231-288](../../../crawlbot/simulation/sim_loop.py#L231-L288) |
| `PlannerReferences.L_com_at` | [L290-292](../../../crawlbot/simulation/sim_loop.py#L290-L292) |
| `PlannerReferences.torso_at` | [L294-307](../../../crawlbot/simulation/sim_loop.py#L294-L307) |
| `PlannerReferences.swing_at` | [L309-314](../../../crawlbot/simulation/sim_loop.py#L309-L314) |
| `class SimulationLoop` | [L317-2403](../../../crawlbot/simulation/sim_loop.py#L317-L2403) |
| `SimulationLoop.mj_model` | [L417-418](../../../crawlbot/simulation/sim_loop.py#L417-L418) |
| `SimulationLoop.mj_data` | [L421-422](../../../crawlbot/simulation/sim_loop.py#L421-L422) |
| `SimulationLoop._sat_total_calls` | [L427-428](../../../crawlbot/simulation/sim_loop.py#L427-L428) |
| `SimulationLoop._sat_clipped_calls` | [L431-432](../../../crawlbot/simulation/sim_loop.py#L431-L432) |
| `SimulationLoop._sat_max_clip_mm` | [L435-436](../../../crawlbot/simulation/sim_loop.py#L435-L436) |
| `SimulationLoop._diag_pure_pd` | [L440-441](../../../crawlbot/simulation/sim_loop.py#L440-L441) |
| `SimulationLoop._diag_pure_pd` | [L444-445](../../../crawlbot/simulation/sim_loop.py#L444-L445) |
| `SimulationLoop._diag_freeze_ref` | [L448-449](../../../crawlbot/simulation/sim_loop.py#L448-L449) |
| `SimulationLoop._diag_freeze_ref` | [L452-453](../../../crawlbot/simulation/sim_loop.py#L452-L453) |
| `SimulationLoop._diag_disable_aocs` | [L456-457](../../../crawlbot/simulation/sim_loop.py#L456-L457) |
| `SimulationLoop._diag_disable_aocs` | [L460-461](../../../crawlbot/simulation/sim_loop.py#L460-L461) |
| `SimulationLoop._diag_lock_arm_joints` | [L464-465](../../../crawlbot/simulation/sim_loop.py#L464-L465) |
| `SimulationLoop._diag_lock_arm_joints` | [L468-469](../../../crawlbot/simulation/sim_loop.py#L468-L469) |
| `SimulationLoop.setup` | [L473-775](../../../crawlbot/simulation/sim_loop.py#L473-L775) |
| `SimulationLoop._settle_setup` | [L777-860](../../../crawlbot/simulation/sim_loop.py#L777-L860) |
| `SimulationLoop._run_ds_passivity_loop` | [L862-869](../../../crawlbot/simulation/sim_loop.py#L862-L869) |
| `SimulationLoop._ds_begin` | [L871-894](../../../crawlbot/simulation/sim_loop.py#L871-L894) |
| `SimulationLoop._ds_tick` | [L896-949](../../../crawlbot/simulation/sim_loop.py#L896-L949) |
| `SimulationLoop._ds_end` | [L951-967](../../../crawlbot/simulation/sim_loop.py#L951-L967) |
| `SimulationLoop._build_qp` | [L969-1022](../../../crawlbot/simulation/sim_loop.py#L969-L1022) |
| `SimulationLoop._gripper_distance` | [L1024-1026](../../../crawlbot/simulation/sim_loop.py#L1024-L1026) |
| `SimulationLoop._gripper_speed` | [L1028-1041](../../../crawlbot/simulation/sim_loop.py#L1028-L1041) |
| `SimulationLoop._gripper_ori_err_deg` | [L1043-1056](../../../crawlbot/simulation/sim_loop.py#L1043-L1056) |
| `SimulationLoop._dock_gate` | [L1058-1090](../../../crawlbot/simulation/sim_loop.py#L1058-L1090) |
| `SimulationLoop._setup_torso_for_step` | [L1095-1308](../../../crawlbot/simulation/sim_loop.py#L1095-L1308) |
| `SimulationLoop._run_preplanner` | [L1310-1411](../../../crawlbot/simulation/sim_loop.py#L1310-L1411) |
| `SimulationLoop._capture_snapshot` | [L1415-1418](../../../crawlbot/simulation/sim_loop.py#L1415-L1418) |
| `SimulationLoop.run` | [L1420-1426](../../../crawlbot/simulation/sim_loop.py#L1420-L1426) |
| `SimulationLoop._drive` | [L1430-1457](../../../crawlbot/simulation/sim_loop.py#L1430-L1457) |
| `SimulationLoop._begin` | [L1459-1465](../../../crawlbot/simulation/sim_loop.py#L1459-L1465) |
| `SimulationLoop._gait_program` | [L1467-2009](../../../crawlbot/simulation/sim_loop.py#L1467-L2009) |
| `SimulationLoop._swing_query_time` | [L2013-2029](../../../crawlbot/simulation/sim_loop.py#L2013-L2029) |
| `SimulationLoop._step` | [L2031-2048](../../../crawlbot/simulation/sim_loop.py#L2031-L2048) |
| `SimulationLoop._nmpc_begin` | [L2050-2125](../../../crawlbot/simulation/sim_loop.py#L2050-L2125) |
| `SimulationLoop._nmpc_tick` | [L2127-2134](../../../crawlbot/simulation/sim_loop.py#L2127-L2134) |
| `SimulationLoop._qp_substep` | [L2136-2306](../../../crawlbot/simulation/sim_loop.py#L2136-L2306) |
| `SimulationLoop._nmpc_handoff` | [L2309-2354](../../../crawlbot/simulation/sim_loop.py#L2309-L2354) |
| `SimulationLoop._get_ee_data` | [L2356-2359](../../../crawlbot/simulation/sim_loop.py#L2356-L2359) |
| `SimulationLoop._print_summary` | [L2363-2394](../../../crawlbot/simulation/sim_loop.py#L2363-L2394) |
| `SimulationLoop.plot` | [L2402-2403](../../../crawlbot/simulation/sim_loop.py#L2402-L2403) |

---

## See also

- package overview: [`simulation.md`](simulation.md)
