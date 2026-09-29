# `crawlbot.simulation.sim_loop`

**File**: [`crawlbot/simulation/sim_loop.py`](../../../crawlbot/simulation/sim_loop.py) — **2418 lines** — canonical coverage **88 %**

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
| **`_DSSettle`** *(dataclass)* |  |  | [L105](../../../crawlbot/simulation/sim_loop.py#L105) |
|   `contact_config` | `` | _field_ | [L127](../../../crawlbot/simulation/sim_loop.py#L127) |
|   `max_steps` | `` | _field_ | [L128](../../../crawlbot/simulation/sim_loop.py#L128) |
|   `epsilon_v` | `` | _field_ | [L129](../../../crawlbot/simulation/sim_loop.py#L129) |
|   `plateau_window` | `50` | _field_ | [L130](../../../crawlbot/simulation/sim_loop.py#L130) |
|   `plateau_ratio` | `0.999` | _field_ | [L131](../../../crawlbot/simulation/sim_loop.py#L131) |
|   `min_steps` | `0` | _field_ | [L132](../../../crawlbot/simulation/sim_loop.py#L132) |
|   `fallback_Kd` | `20.0` | _field_ | [L133](../../../crawlbot/simulation/sim_loop.py#L133) |
|   `t_log` | `None` | _field_ | [L134](../../../crawlbot/simulation/sim_loop.py#L134) |
|   `T_log` | `None` | _field_ | [L135](../../../crawlbot/simulation/sim_loop.py#L135) |
|   `t_log_step_offset` | `0` | _field_ | [L136](../../../crawlbot/simulation/sim_loop.py#L136) |
|   `log_obj` | `None` | _field_ | [L137](../../../crawlbot/simulation/sim_loop.py#L137) |
|   `log_step_idx` | `-1` | _field_ | [L138](../../../crawlbot/simulation/sim_loop.py#L138) |
|   `log_just_landed_arm` | `''` | _field_ | [L139](../../../crawlbot/simulation/sim_loop.py#L139) |
|   `log_anchor_a_idx` | `-1` | _field_ | [L140](../../../crawlbot/simulation/sim_loop.py#L140) |
|   `log_anchor_b_idx` | `-1` | _field_ | [L141](../../../crawlbot/simulation/sim_loop.py#L141) |
|   `log_t_abs` | `0.0` | _field_ | [L142](../../../crawlbot/simulation/sim_loop.py#L142) |
| **`_DSRun`** *(dataclass)* |  |  | [L146](../../../crawlbot/simulation/sim_loop.py#L146) |
|   `req` | `` | _field_ | [L148](../../../crawlbot/simulation/sim_loop.py#L148) |
|   `lambda_min` | `` | _field_ | [L149](../../../crawlbot/simulation/sim_loop.py#L149) |
|   `T_settle` | `` | _field_ | [L150](../../../crawlbot/simulation/sim_loop.py#L150) |
|   `T_start` | `` | _field_ | [L151](../../../crawlbot/simulation/sim_loop.py#L151) |
|   `hw_current` | `` | _field_ | [L152](../../../crawlbot/simulation/sim_loop.py#L152) |
|   `T_history` | `field(default_factory=list)` | _field_ | [L153](../../../crawlbot/simulation/sim_loop.py#L153) |
|   `exit_reason` | `'max_steps'` | _field_ | [L154](../../../crawlbot/simulation/sim_loop.py#L154) |
|   `k` | `0` | _field_ | [L155](../../../crawlbot/simulation/sim_loop.py#L155) |
| **`_NMPCTick`** *(dataclass)* |  |  | [L159](../../../crawlbot/simulation/sim_loop.py#L159) |
|   `t` | `` | _field_ | [L169](../../../crawlbot/simulation/sim_loop.py#L169) |
|   `phase` | `` | _field_ | [L170](../../../crawlbot/simulation/sim_loop.py#L170) |
|   `step_idx` | `` | _field_ | [L171](../../../crawlbot/simulation/sim_loop.py#L171) |
|   `swing_arm` | `` | _field_ | [L172](../../../crawlbot/simulation/sim_loop.py#L172) |
|   `stance_arm` | `` | _field_ | [L173](../../../crawlbot/simulation/sim_loop.py#L173) |
|   `cc_ss` | `` | _field_ | [L174](../../../crawlbot/simulation/sim_loop.py#L174) |
|   `target_anchor` | `` | _field_ | [L175](../../../crawlbot/simulation/sim_loop.py#L175) |
|   `stance_a` | `` | _field_ | [L176](../../../crawlbot/simulation/sim_loop.py#L176) |
|   `stance_b` | `` | _field_ | [L177](../../../crawlbot/simulation/sim_loop.py#L177) |
|   `hw` | `` | _field_ | [L178](../../../crawlbot/simulation/sim_loop.py#L178) |
|   `L_com_prev` | `` | _field_ | [L179](../../../crawlbot/simulation/sim_loop.py#L179) |
|   `log` | `` | _field_ | [L180](../../../crawlbot/simulation/sim_loop.py#L180) |
|   `ss_end` | `None` | _field_ | [L181](../../../crawlbot/simulation/sim_loop.py#L181) |
|   `settle_mode` | `False` | _field_ | [L182](../../../crawlbot/simulation/sim_loop.py#L182) |
|   `passivity_hold` | `False` | _field_ | [L183](../../../crawlbot/simulation/sim_loop.py#L183) |
|   `passivity_override` | `None` | _field_ | [L184](../../../crawlbot/simulation/sim_loop.py#L184) |
|   `ds_centroidal_active` | `False` | _field_ | [L185](../../../crawlbot/simulation/sim_loop.py#L185) |
| **`_NMPCRun`** *(dataclass)* |  |  | [L189](../../../crawlbot/simulation/sim_loop.py#L189) |
|   `t` | `` | _field_ | [L191](../../../crawlbot/simulation/sim_loop.py#L191) |
|   `phase` | `` | _field_ | [L192](../../../crawlbot/simulation/sim_loop.py#L192) |
|   `step_idx` | `` | _field_ | [L193](../../../crawlbot/simulation/sim_loop.py#L193) |
|   `swing_arm` | `` | _field_ | [L194](../../../crawlbot/simulation/sim_loop.py#L194) |
|   `stance_arm` | `` | _field_ | [L195](../../../crawlbot/simulation/sim_loop.py#L195) |
|   `stance_a` | `` | _field_ | [L196](../../../crawlbot/simulation/sim_loop.py#L196) |
|   `stance_b` | `` | _field_ | [L197](../../../crawlbot/simulation/sim_loop.py#L197) |
|   `target_anchor` | `` | _field_ | [L198](../../../crawlbot/simulation/sim_loop.py#L198) |
|   `log` | `` | _field_ | [L199](../../../crawlbot/simulation/sim_loop.py#L199) |
|   `ss_end` | `` | _field_ | [L200](../../../crawlbot/simulation/sim_loop.py#L200) |
|   `settle_mode` | `` | _field_ | [L201](../../../crawlbot/simulation/sim_loop.py#L201) |
|   `L_com_prev` | `` | _field_ | [L202](../../../crawlbot/simulation/sim_loop.py#L202) |
|   `intent` | `` | _field_ | [L203](../../../crawlbot/simulation/sim_loop.py#L203) |
|   `plan` | `` | _field_ | [L204](../../../crawlbot/simulation/sim_loop.py#L204) |
|   `carry` | `` | _field_ | [L205](../../../crawlbot/simulation/sim_loop.py#L205) |
|   `vp` | `` | _field_ | [L206](../../../crawlbot/simulation/sim_loop.py#L206) |
|   `cref_r` | `` | _field_ | [L207](../../../crawlbot/simulation/sim_loop.py#L207) |
|   `nmpc_ok` | `` | _field_ | [L208](../../../crawlbot/simulation/sim_loop.py#L208) |
|   `nmpc_status_code` | `` | _field_ | [L209](../../../crawlbot/simulation/sim_loop.py#L209) |
|   `nmpc_cost_val` | `` | _field_ | [L210](../../../crawlbot/simulation/sim_loop.py#L210) |
|   `info_n` | `` | _field_ | [L211](../../../crawlbot/simulation/sim_loop.py#L211) |
|   `t_nmpc_ms` | `` | _field_ | [L212](../../../crawlbot/simulation/sim_loop.py#L212) |
|   `t_qp_start` | `` | _field_ | [L213](../../../crawlbot/simulation/sim_loop.py#L213) |
|   `qs` | `0` | _field_ | [L214](../../../crawlbot/simulation/sim_loop.py#L214) |
| **`PlannerReferences`** |  |  | [L217](../../../crawlbot/simulation/sim_loop.py#L217) |
| `.com_at` | `(t, settle_mode)` | **yes** | [L233](../../../crawlbot/simulation/sim_loop.py#L233) |
| `.L_com_at` | `(t_mid)` | **yes** | [L292](../../../crawlbot/simulation/sim_loop.py#L292) |
| `.torso_at` | `(tq, phase, ss_end)` | **yes** | [L296](../../../crawlbot/simulation/sim_loop.py#L296) |
| `.swing_at` | `(tq, phase, ss_end)` | **yes** | [L311](../../../crawlbot/simulation/sim_loop.py#L311) |
| **`SimulationLoop`** |  |  | [L319](../../../crawlbot/simulation/sim_loop.py#L319) |
| `.mj_model` | `()` | **yes** | [L419](../../../crawlbot/simulation/sim_loop.py#L419) |
| `.mj_data` | `()` | **yes** | [L423](../../../crawlbot/simulation/sim_loop.py#L423) |
| `._sat_total_calls` | `()` | **yes** | [L429](../../../crawlbot/simulation/sim_loop.py#L429) |
| `._sat_clipped_calls` | `()` | **yes** | [L433](../../../crawlbot/simulation/sim_loop.py#L433) |
| `._sat_max_clip_mm` | `()` | **yes** | [L437](../../../crawlbot/simulation/sim_loop.py#L437) |
| `._diag_pure_pd` | `()` | not exercised | [L442](../../../crawlbot/simulation/sim_loop.py#L442) |
| `._diag_pure_pd` | `(v)` | not exercised | [L446](../../../crawlbot/simulation/sim_loop.py#L446) |
| `._diag_freeze_ref` | `()` | not exercised | [L450](../../../crawlbot/simulation/sim_loop.py#L450) |
| `._diag_freeze_ref` | `(v)` | not exercised | [L454](../../../crawlbot/simulation/sim_loop.py#L454) |
| `._diag_disable_aocs` | `()` | not exercised | [L458](../../../crawlbot/simulation/sim_loop.py#L458) |
| `._diag_disable_aocs` | `(v)` | not exercised | [L462](../../../crawlbot/simulation/sim_loop.py#L462) |
| `._diag_lock_arm_joints` | `()` | **yes** | [L466](../../../crawlbot/simulation/sim_loop.py#L466) |
| `._diag_lock_arm_joints` | `(v)` | not exercised | [L470](../../../crawlbot/simulation/sim_loop.py#L470) |
| `.setup` | `(n_steps=3, start_a=2, start_b=2, sequence_path=None)` | **yes** | [L475](../../../crawlbot/simulation/sim_loop.py#L475) |
| `._settle_setup` | `(start_a, start_b)` | **yes** | [L791](../../../crawlbot/simulation/sim_loop.py#L791) |
| `._run_ds_passivity_loop` | `(**kw)` | **yes** | [L876](../../../crawlbot/simulation/sim_loop.py#L876) |
| `._ds_begin` | `(r)` | **yes** | [L885](../../../crawlbot/simulation/sim_loop.py#L885) |
| `._ds_tick` | `(st)` | **yes** | [L910](../../../crawlbot/simulation/sim_loop.py#L910) |
| `._ds_end` | `(st, k_last)` | **yes** | [L965](../../../crawlbot/simulation/sim_loop.py#L965) |
| `._build_qp` | `(ae, ap, aw, kpc, kdc, kpt, kdt, kpe, kde, kpe_ang=5.0, ...)` | **yes** | [L983](../../../crawlbot/simulation/sim_loop.py#L983) |
| `._gripper_distance` | `(arm, anchor_idx)` | **yes** | [L1038](../../../crawlbot/simulation/sim_loop.py#L1038) |
| `._gripper_speed` | `(arm)` | not exercised | [L1042](../../../crawlbot/simulation/sim_loop.py#L1042) |
| `._gripper_ori_err_deg` | `(arm, anchor_idx)` | **yes** | [L1057](../../../crawlbot/simulation/sim_loop.py#L1057) |
| `._dock_gate` | `(swing_arm, target_idx, log, t, step_idx)` | **yes** | [L1072](../../../crawlbot/simulation/sim_loop.py#L1072) |
| `._setup_torso_for_step` | `(t_ss_start, swing_arm, stance_a, stance_b, target_arm, ...)` | **yes** | [L1109](../../../crawlbot/simulation/sim_loop.py#L1109) |
| `._run_preplanner` | `(t_plan_start, stance_arm, stance_a, stance_b, r_com_0, ...)` | **yes** | [L1324](../../../crawlbot/simulation/sim_loop.py#L1324) |
| `._capture_snapshot` | `(log, t, label)` | **yes** | [L1429](../../../crawlbot/simulation/sim_loop.py#L1429) |
| `.run` | `(verbose=True)` | **yes** | [L1434](../../../crawlbot/simulation/sim_loop.py#L1434) |
| `._drive` | `(program)` | **yes** | [L1444](../../../crawlbot/simulation/sim_loop.py#L1444) |
| `._begin` | `(req)` | **yes** | [L1473](../../../crawlbot/simulation/sim_loop.py#L1473) |
| `._gait_program` | `(verbose=True)` | **yes** | [L1481](../../../crawlbot/simulation/sim_loop.py#L1481) |
| `._swing_query_time` | `(t_raw, phase, ss_end)` | **yes** | [L2027](../../../crawlbot/simulation/sim_loop.py#L2027) |
| `._step` | `(t, phase, step_idx, swing_arm, stance_arm, cc_ss, targe...)` | not exercised | [L2045](../../../crawlbot/simulation/sim_loop.py#L2045) |
| `._nmpc_begin` | `(r)` | **yes** | [L2064](../../../crawlbot/simulation/sim_loop.py#L2064) |
| `._nmpc_tick` | `(st)` | **yes** | [L2141](../../../crawlbot/simulation/sim_loop.py#L2141) |
| `._qp_substep` | `(st)` | **yes** | [L2150](../../../crawlbot/simulation/sim_loop.py#L2150) |
| `._nmpc_handoff` | `(st)` | **yes** | [L2323](../../../crawlbot/simulation/sim_loop.py#L2323) |
| `._get_ee_data` | `(rs, arm)` | **yes** | [L2370](../../../crawlbot/simulation/sim_loop.py#L2370) |
| `._print_summary` | `(log)` | **yes** | [L2377](../../../crawlbot/simulation/sim_loop.py#L2377) |
| `.plot` | `(log, save_path=None, cfg=None)` | not exercised | [L2416](../../../crawlbot/simulation/sim_loop.py#L2416) |

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
| `controller.py:408-410` | torso-reference routing (delta-mapping vs raw quintic) |
| `controller.py:554-555` | `passivity_active` — **the DS passivity constraint** |

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
| `class _DSSettle` | [L105-142](../../../crawlbot/simulation/sim_loop.py#L105-L142) |
| `class _DSRun` | [L146-155](../../../crawlbot/simulation/sim_loop.py#L146-L155) |
| `class _NMPCTick` | [L159-185](../../../crawlbot/simulation/sim_loop.py#L159-L185) |
| `class _NMPCRun` | [L189-214](../../../crawlbot/simulation/sim_loop.py#L189-L214) |
| `class PlannerReferences` | [L217-316](../../../crawlbot/simulation/sim_loop.py#L217-L316) |
| `PlannerReferences.com_at` | [L233-290](../../../crawlbot/simulation/sim_loop.py#L233-L290) |
| `PlannerReferences.L_com_at` | [L292-294](../../../crawlbot/simulation/sim_loop.py#L292-L294) |
| `PlannerReferences.torso_at` | [L296-309](../../../crawlbot/simulation/sim_loop.py#L296-L309) |
| `PlannerReferences.swing_at` | [L311-316](../../../crawlbot/simulation/sim_loop.py#L311-L316) |
| `class SimulationLoop` | [L319-2417](../../../crawlbot/simulation/sim_loop.py#L319-L2417) |
| `SimulationLoop.mj_model` | [L419-420](../../../crawlbot/simulation/sim_loop.py#L419-L420) |
| `SimulationLoop.mj_data` | [L423-424](../../../crawlbot/simulation/sim_loop.py#L423-L424) |
| `SimulationLoop._sat_total_calls` | [L429-430](../../../crawlbot/simulation/sim_loop.py#L429-L430) |
| `SimulationLoop._sat_clipped_calls` | [L433-434](../../../crawlbot/simulation/sim_loop.py#L433-L434) |
| `SimulationLoop._sat_max_clip_mm` | [L437-438](../../../crawlbot/simulation/sim_loop.py#L437-L438) |
| `SimulationLoop._diag_pure_pd` | [L442-443](../../../crawlbot/simulation/sim_loop.py#L442-L443) |
| `SimulationLoop._diag_pure_pd` | [L446-447](../../../crawlbot/simulation/sim_loop.py#L446-L447) |
| `SimulationLoop._diag_freeze_ref` | [L450-451](../../../crawlbot/simulation/sim_loop.py#L450-L451) |
| `SimulationLoop._diag_freeze_ref` | [L454-455](../../../crawlbot/simulation/sim_loop.py#L454-L455) |
| `SimulationLoop._diag_disable_aocs` | [L458-459](../../../crawlbot/simulation/sim_loop.py#L458-L459) |
| `SimulationLoop._diag_disable_aocs` | [L462-463](../../../crawlbot/simulation/sim_loop.py#L462-L463) |
| `SimulationLoop._diag_lock_arm_joints` | [L466-467](../../../crawlbot/simulation/sim_loop.py#L466-L467) |
| `SimulationLoop._diag_lock_arm_joints` | [L470-471](../../../crawlbot/simulation/sim_loop.py#L470-L471) |
| `SimulationLoop.setup` | [L475-789](../../../crawlbot/simulation/sim_loop.py#L475-L789) |
| `SimulationLoop._settle_setup` | [L791-874](../../../crawlbot/simulation/sim_loop.py#L791-L874) |
| `SimulationLoop._run_ds_passivity_loop` | [L876-883](../../../crawlbot/simulation/sim_loop.py#L876-L883) |
| `SimulationLoop._ds_begin` | [L885-908](../../../crawlbot/simulation/sim_loop.py#L885-L908) |
| `SimulationLoop._ds_tick` | [L910-963](../../../crawlbot/simulation/sim_loop.py#L910-L963) |
| `SimulationLoop._ds_end` | [L965-981](../../../crawlbot/simulation/sim_loop.py#L965-L981) |
| `SimulationLoop._build_qp` | [L983-1036](../../../crawlbot/simulation/sim_loop.py#L983-L1036) |
| `SimulationLoop._gripper_distance` | [L1038-1040](../../../crawlbot/simulation/sim_loop.py#L1038-L1040) |
| `SimulationLoop._gripper_speed` | [L1042-1055](../../../crawlbot/simulation/sim_loop.py#L1042-L1055) |
| `SimulationLoop._gripper_ori_err_deg` | [L1057-1070](../../../crawlbot/simulation/sim_loop.py#L1057-L1070) |
| `SimulationLoop._dock_gate` | [L1072-1104](../../../crawlbot/simulation/sim_loop.py#L1072-L1104) |
| `SimulationLoop._setup_torso_for_step` | [L1109-1322](../../../crawlbot/simulation/sim_loop.py#L1109-L1322) |
| `SimulationLoop._run_preplanner` | [L1324-1425](../../../crawlbot/simulation/sim_loop.py#L1324-L1425) |
| `SimulationLoop._capture_snapshot` | [L1429-1432](../../../crawlbot/simulation/sim_loop.py#L1429-L1432) |
| `SimulationLoop.run` | [L1434-1440](../../../crawlbot/simulation/sim_loop.py#L1434-L1440) |
| `SimulationLoop._drive` | [L1444-1471](../../../crawlbot/simulation/sim_loop.py#L1444-L1471) |
| `SimulationLoop._begin` | [L1473-1479](../../../crawlbot/simulation/sim_loop.py#L1473-L1479) |
| `SimulationLoop._gait_program` | [L1481-2023](../../../crawlbot/simulation/sim_loop.py#L1481-L2023) |
| `SimulationLoop._swing_query_time` | [L2027-2043](../../../crawlbot/simulation/sim_loop.py#L2027-L2043) |
| `SimulationLoop._step` | [L2045-2062](../../../crawlbot/simulation/sim_loop.py#L2045-L2062) |
| `SimulationLoop._nmpc_begin` | [L2064-2139](../../../crawlbot/simulation/sim_loop.py#L2064-L2139) |
| `SimulationLoop._nmpc_tick` | [L2141-2148](../../../crawlbot/simulation/sim_loop.py#L2141-L2148) |
| `SimulationLoop._qp_substep` | [L2150-2320](../../../crawlbot/simulation/sim_loop.py#L2150-L2320) |
| `SimulationLoop._nmpc_handoff` | [L2323-2368](../../../crawlbot/simulation/sim_loop.py#L2323-L2368) |
| `SimulationLoop._get_ee_data` | [L2370-2373](../../../crawlbot/simulation/sim_loop.py#L2370-L2373) |
| `SimulationLoop._print_summary` | [L2377-2408](../../../crawlbot/simulation/sim_loop.py#L2377-L2408) |
| `SimulationLoop.plot` | [L2416-2417](../../../crawlbot/simulation/sim_loop.py#L2416-L2417) |

---

## See also

- package overview: [`simulation.md`](simulation.md)
