# `crawlbot.simulation.sim_loop`

**File**: [`crawlbot/simulation/sim_loop.py`](../../../crawlbot/simulation/sim_loop.py) — **2393 lines** — canonical coverage **88 %**

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
| **`_DSSettle`** *(dataclass)* |  |  | [L102](../../../crawlbot/simulation/sim_loop.py#L102) |
|   `contact_config` | `` | _field_ | [L124](../../../crawlbot/simulation/sim_loop.py#L124) |
|   `max_steps` | `` | _field_ | [L125](../../../crawlbot/simulation/sim_loop.py#L125) |
|   `epsilon_v` | `` | _field_ | [L126](../../../crawlbot/simulation/sim_loop.py#L126) |
|   `plateau_window` | `50` | _field_ | [L127](../../../crawlbot/simulation/sim_loop.py#L127) |
|   `plateau_ratio` | `0.999` | _field_ | [L128](../../../crawlbot/simulation/sim_loop.py#L128) |
|   `min_steps` | `0` | _field_ | [L129](../../../crawlbot/simulation/sim_loop.py#L129) |
|   `fallback_Kd` | `20.0` | _field_ | [L130](../../../crawlbot/simulation/sim_loop.py#L130) |
|   `t_log` | `None` | _field_ | [L131](../../../crawlbot/simulation/sim_loop.py#L131) |
|   `T_log` | `None` | _field_ | [L132](../../../crawlbot/simulation/sim_loop.py#L132) |
|   `t_log_step_offset` | `0` | _field_ | [L133](../../../crawlbot/simulation/sim_loop.py#L133) |
|   `log_obj` | `None` | _field_ | [L134](../../../crawlbot/simulation/sim_loop.py#L134) |
|   `log_step_idx` | `-1` | _field_ | [L135](../../../crawlbot/simulation/sim_loop.py#L135) |
|   `log_just_landed_arm` | `''` | _field_ | [L136](../../../crawlbot/simulation/sim_loop.py#L136) |
|   `log_anchor_a_idx` | `-1` | _field_ | [L137](../../../crawlbot/simulation/sim_loop.py#L137) |
|   `log_anchor_b_idx` | `-1` | _field_ | [L138](../../../crawlbot/simulation/sim_loop.py#L138) |
|   `log_t_abs` | `0.0` | _field_ | [L139](../../../crawlbot/simulation/sim_loop.py#L139) |
| **`_DSRun`** *(dataclass)* |  |  | [L143](../../../crawlbot/simulation/sim_loop.py#L143) |
|   `req` | `` | _field_ | [L145](../../../crawlbot/simulation/sim_loop.py#L145) |
|   `lambda_min` | `` | _field_ | [L146](../../../crawlbot/simulation/sim_loop.py#L146) |
|   `T_settle` | `` | _field_ | [L147](../../../crawlbot/simulation/sim_loop.py#L147) |
|   `T_start` | `` | _field_ | [L148](../../../crawlbot/simulation/sim_loop.py#L148) |
|   `hw_current` | `` | _field_ | [L149](../../../crawlbot/simulation/sim_loop.py#L149) |
|   `T_history` | `field(default_factory=list)` | _field_ | [L150](../../../crawlbot/simulation/sim_loop.py#L150) |
|   `exit_reason` | `'max_steps'` | _field_ | [L151](../../../crawlbot/simulation/sim_loop.py#L151) |
|   `k` | `0` | _field_ | [L152](../../../crawlbot/simulation/sim_loop.py#L152) |
| **`_NMPCTick`** *(dataclass)* |  |  | [L156](../../../crawlbot/simulation/sim_loop.py#L156) |
|   `t` | `` | _field_ | [L166](../../../crawlbot/simulation/sim_loop.py#L166) |
|   `phase` | `` | _field_ | [L167](../../../crawlbot/simulation/sim_loop.py#L167) |
|   `step_idx` | `` | _field_ | [L168](../../../crawlbot/simulation/sim_loop.py#L168) |
|   `swing_arm` | `` | _field_ | [L169](../../../crawlbot/simulation/sim_loop.py#L169) |
|   `stance_arm` | `` | _field_ | [L170](../../../crawlbot/simulation/sim_loop.py#L170) |
|   `cc_ss` | `` | _field_ | [L171](../../../crawlbot/simulation/sim_loop.py#L171) |
|   `target_anchor` | `` | _field_ | [L172](../../../crawlbot/simulation/sim_loop.py#L172) |
|   `stance_a` | `` | _field_ | [L173](../../../crawlbot/simulation/sim_loop.py#L173) |
|   `stance_b` | `` | _field_ | [L174](../../../crawlbot/simulation/sim_loop.py#L174) |
|   `hw` | `` | _field_ | [L175](../../../crawlbot/simulation/sim_loop.py#L175) |
|   `L_com_prev` | `` | _field_ | [L176](../../../crawlbot/simulation/sim_loop.py#L176) |
|   `log` | `` | _field_ | [L177](../../../crawlbot/simulation/sim_loop.py#L177) |
|   `ss_end` | `None` | _field_ | [L178](../../../crawlbot/simulation/sim_loop.py#L178) |
|   `settle_mode` | `False` | _field_ | [L179](../../../crawlbot/simulation/sim_loop.py#L179) |
|   `passivity_hold` | `False` | _field_ | [L180](../../../crawlbot/simulation/sim_loop.py#L180) |
|   `passivity_override` | `None` | _field_ | [L181](../../../crawlbot/simulation/sim_loop.py#L181) |
|   `ds_centroidal_active` | `False` | _field_ | [L182](../../../crawlbot/simulation/sim_loop.py#L182) |
| **`_NMPCRun`** *(dataclass)* |  |  | [L186](../../../crawlbot/simulation/sim_loop.py#L186) |
|   `t` | `` | _field_ | [L188](../../../crawlbot/simulation/sim_loop.py#L188) |
|   `phase` | `` | _field_ | [L189](../../../crawlbot/simulation/sim_loop.py#L189) |
|   `step_idx` | `` | _field_ | [L190](../../../crawlbot/simulation/sim_loop.py#L190) |
|   `swing_arm` | `` | _field_ | [L191](../../../crawlbot/simulation/sim_loop.py#L191) |
|   `stance_arm` | `` | _field_ | [L192](../../../crawlbot/simulation/sim_loop.py#L192) |
|   `stance_a` | `` | _field_ | [L193](../../../crawlbot/simulation/sim_loop.py#L193) |
|   `stance_b` | `` | _field_ | [L194](../../../crawlbot/simulation/sim_loop.py#L194) |
|   `target_anchor` | `` | _field_ | [L195](../../../crawlbot/simulation/sim_loop.py#L195) |
|   `log` | `` | _field_ | [L196](../../../crawlbot/simulation/sim_loop.py#L196) |
|   `ss_end` | `` | _field_ | [L197](../../../crawlbot/simulation/sim_loop.py#L197) |
|   `settle_mode` | `` | _field_ | [L198](../../../crawlbot/simulation/sim_loop.py#L198) |
|   `L_com_prev` | `` | _field_ | [L199](../../../crawlbot/simulation/sim_loop.py#L199) |
|   `intent` | `` | _field_ | [L200](../../../crawlbot/simulation/sim_loop.py#L200) |
|   `plan` | `` | _field_ | [L201](../../../crawlbot/simulation/sim_loop.py#L201) |
|   `carry` | `` | _field_ | [L202](../../../crawlbot/simulation/sim_loop.py#L202) |
|   `vp` | `` | _field_ | [L203](../../../crawlbot/simulation/sim_loop.py#L203) |
|   `cref_r` | `` | _field_ | [L204](../../../crawlbot/simulation/sim_loop.py#L204) |
|   `nmpc_ok` | `` | _field_ | [L205](../../../crawlbot/simulation/sim_loop.py#L205) |
|   `nmpc_status_code` | `` | _field_ | [L206](../../../crawlbot/simulation/sim_loop.py#L206) |
|   `nmpc_cost_val` | `` | _field_ | [L207](../../../crawlbot/simulation/sim_loop.py#L207) |
|   `info_n` | `` | _field_ | [L208](../../../crawlbot/simulation/sim_loop.py#L208) |
|   `t_nmpc_ms` | `` | _field_ | [L209](../../../crawlbot/simulation/sim_loop.py#L209) |
|   `t_qp_start` | `` | _field_ | [L210](../../../crawlbot/simulation/sim_loop.py#L210) |
|   `qs` | `0` | _field_ | [L211](../../../crawlbot/simulation/sim_loop.py#L211) |
| **`PlannerReferences`** |  |  | [L214](../../../crawlbot/simulation/sim_loop.py#L214) |
| `.com_at` | `(t, settle_mode)` | **yes** | [L230](../../../crawlbot/simulation/sim_loop.py#L230) |
| `.L_com_at` | `(t_mid)` | **yes** | [L289](../../../crawlbot/simulation/sim_loop.py#L289) |
| `.torso_at` | `(tq, phase, ss_end)` | **yes** | [L293](../../../crawlbot/simulation/sim_loop.py#L293) |
| `.swing_at` | `(tq, phase, ss_end)` | **yes** | [L308](../../../crawlbot/simulation/sim_loop.py#L308) |
| **`SimulationLoop`** |  |  | [L316](../../../crawlbot/simulation/sim_loop.py#L316) |
| `.mj_model` | `()` | **yes** | [L419](../../../crawlbot/simulation/sim_loop.py#L419) |
| `.mj_data` | `()` | **yes** | [L423](../../../crawlbot/simulation/sim_loop.py#L423) |
| `._sat_total_calls` | `()` | **yes** | [L430](../../../crawlbot/simulation/sim_loop.py#L430) |
| `._sat_clipped_calls` | `()` | **yes** | [L434](../../../crawlbot/simulation/sim_loop.py#L434) |
| `._sat_max_clip_mm` | `()` | **yes** | [L438](../../../crawlbot/simulation/sim_loop.py#L438) |
| `._diag_pure_pd` | `()` | not exercised | [L443](../../../crawlbot/simulation/sim_loop.py#L443) |
| `._diag_pure_pd` | `(v)` | not exercised | [L447](../../../crawlbot/simulation/sim_loop.py#L447) |
| `._diag_freeze_ref` | `()` | not exercised | [L451](../../../crawlbot/simulation/sim_loop.py#L451) |
| `._diag_freeze_ref` | `(v)` | not exercised | [L455](../../../crawlbot/simulation/sim_loop.py#L455) |
| `._diag_disable_aocs` | `()` | not exercised | [L459](../../../crawlbot/simulation/sim_loop.py#L459) |
| `._diag_disable_aocs` | `(v)` | not exercised | [L463](../../../crawlbot/simulation/sim_loop.py#L463) |
| `._diag_lock_arm_joints` | `()` | **yes** | [L467](../../../crawlbot/simulation/sim_loop.py#L467) |
| `._diag_lock_arm_joints` | `(v)` | not exercised | [L471](../../../crawlbot/simulation/sim_loop.py#L471) |
| `.setup` | `(n_steps=3, start_a=2, start_b=2, sequence_path=None)` | **yes** | [L476](../../../crawlbot/simulation/sim_loop.py#L476) |
| `._settle_setup` | `(start_a, start_b)` | **yes** | [L771](../../../crawlbot/simulation/sim_loop.py#L771) |
| `._run_ds_passivity_loop` | `(**kw)` | **yes** | [L856](../../../crawlbot/simulation/sim_loop.py#L856) |
| `._ds_begin` | `(r)` | **yes** | [L865](../../../crawlbot/simulation/sim_loop.py#L865) |
| `._ds_tick` | `(st)` | **yes** | [L890](../../../crawlbot/simulation/sim_loop.py#L890) |
| `._ds_end` | `(st, k_last)` | **yes** | [L953](../../../crawlbot/simulation/sim_loop.py#L953) |
| `._build_qp` | `(ae, ap, aw, kpc, kdc, kpt, kdt, kpe, kde, kpe_ang=5.0, ...)` | **yes** | [L971](../../../crawlbot/simulation/sim_loop.py#L971) |
| `._gripper_distance` | `(arm, anchor_idx)` | **yes** | [L1025](../../../crawlbot/simulation/sim_loop.py#L1025) |
| `._gripper_speed` | `(arm)` | not exercised | [L1029](../../../crawlbot/simulation/sim_loop.py#L1029) |
| `._gripper_ori_err_deg` | `(arm, anchor_idx)` | **yes** | [L1044](../../../crawlbot/simulation/sim_loop.py#L1044) |
| `._dock_gate` | `(swing_arm, target_idx, log, t, step_idx)` | **yes** | [L1059](../../../crawlbot/simulation/sim_loop.py#L1059) |
| `._setup_torso_for_step` | `(t_ss_start, swing_arm, stance_a, stance_b, target_arm, ...)` | **yes** | [L1096](../../../crawlbot/simulation/sim_loop.py#L1096) |
| `._run_preplanner` | `(t_plan_start, stance_arm, stance_a, stance_b, r_com_0, ...)` | **yes** | [L1303](../../../crawlbot/simulation/sim_loop.py#L1303) |
| `._capture_snapshot` | `(log, t, label)` | **yes** | [L1408](../../../crawlbot/simulation/sim_loop.py#L1408) |
| `.run` | `(verbose=True)` | **yes** | [L1413](../../../crawlbot/simulation/sim_loop.py#L1413) |
| `._drive` | `(program)` | **yes** | [L1423](../../../crawlbot/simulation/sim_loop.py#L1423) |
| `._begin` | `(req)` | **yes** | [L1452](../../../crawlbot/simulation/sim_loop.py#L1452) |
| `._gait_program` | `(verbose=True)` | **yes** | [L1460](../../../crawlbot/simulation/sim_loop.py#L1460) |
| `._swing_query_time` | `(t_raw, phase, ss_end)` | **yes** | [L1998](../../../crawlbot/simulation/sim_loop.py#L1998) |
| `._step` | `(t, phase, step_idx, swing_arm, stance_arm, cc_ss, targe...)` | not exercised | [L2016](../../../crawlbot/simulation/sim_loop.py#L2016) |
| `._nmpc_begin` | `(r)` | **yes** | [L2035](../../../crawlbot/simulation/sim_loop.py#L2035) |
| `._nmpc_tick` | `(st)` | **yes** | [L2112](../../../crawlbot/simulation/sim_loop.py#L2112) |
| `._qp_substep` | `(st)` | **yes** | [L2121](../../../crawlbot/simulation/sim_loop.py#L2121) |
| `._nmpc_handoff` | `(st)` | **yes** | [L2298](../../../crawlbot/simulation/sim_loop.py#L2298) |
| `._get_ee_data` | `(rs, arm)` | **yes** | [L2345](../../../crawlbot/simulation/sim_loop.py#L2345) |
| `._print_summary` | `(log)` | **yes** | [L2352](../../../crawlbot/simulation/sim_loop.py#L2352) |
| `.plot` | `(log, save_path=None, cfg=None)` | not exercised | [L2391](../../../crawlbot/simulation/sim_loop.py#L2391) |

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

## 6. `use_m2_stack` — frozen and removed (R2c)

`SimConfig.use_m2_stack` looked dead but gated two live paths: the
torso-reference routing (retired with the δ-mapping, R2b) and `passivity_active`,
**the DS passivity constraint**. Every run of the paper set it `True`
(`_make_m7_config`), so R2c froze it there: `passivity_active` is now
`phase == 'DS' or passivity_hold` (`controller.py:406`). Behaviour change only
for a bare `SimConfig()`, whose default was `False` (DS passivity off).

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
| `class _DSSettle` | [L102-139](../../../crawlbot/simulation/sim_loop.py#L102-L139) |
| `class _DSRun` | [L143-152](../../../crawlbot/simulation/sim_loop.py#L143-L152) |
| `class _NMPCTick` | [L156-182](../../../crawlbot/simulation/sim_loop.py#L156-L182) |
| `class _NMPCRun` | [L186-211](../../../crawlbot/simulation/sim_loop.py#L186-L211) |
| `class PlannerReferences` | [L214-313](../../../crawlbot/simulation/sim_loop.py#L214-L313) |
| `PlannerReferences.com_at` | [L230-287](../../../crawlbot/simulation/sim_loop.py#L230-L287) |
| `PlannerReferences.L_com_at` | [L289-291](../../../crawlbot/simulation/sim_loop.py#L289-L291) |
| `PlannerReferences.torso_at` | [L293-306](../../../crawlbot/simulation/sim_loop.py#L293-L306) |
| `PlannerReferences.swing_at` | [L308-313](../../../crawlbot/simulation/sim_loop.py#L308-L313) |
| `class SimulationLoop` | [L316-2392](../../../crawlbot/simulation/sim_loop.py#L316-L2392) |
| `SimulationLoop.mj_model` | [L419-420](../../../crawlbot/simulation/sim_loop.py#L419-L420) |
| `SimulationLoop.mj_data` | [L423-424](../../../crawlbot/simulation/sim_loop.py#L423-L424) |
| `SimulationLoop._sat_total_calls` | [L430-431](../../../crawlbot/simulation/sim_loop.py#L430-L431) |
| `SimulationLoop._sat_clipped_calls` | [L434-435](../../../crawlbot/simulation/sim_loop.py#L434-L435) |
| `SimulationLoop._sat_max_clip_mm` | [L438-439](../../../crawlbot/simulation/sim_loop.py#L438-L439) |
| `SimulationLoop._diag_pure_pd` | [L443-444](../../../crawlbot/simulation/sim_loop.py#L443-L444) |
| `SimulationLoop._diag_pure_pd` | [L447-448](../../../crawlbot/simulation/sim_loop.py#L447-L448) |
| `SimulationLoop._diag_freeze_ref` | [L451-452](../../../crawlbot/simulation/sim_loop.py#L451-L452) |
| `SimulationLoop._diag_freeze_ref` | [L455-456](../../../crawlbot/simulation/sim_loop.py#L455-L456) |
| `SimulationLoop._diag_disable_aocs` | [L459-460](../../../crawlbot/simulation/sim_loop.py#L459-L460) |
| `SimulationLoop._diag_disable_aocs` | [L463-464](../../../crawlbot/simulation/sim_loop.py#L463-L464) |
| `SimulationLoop._diag_lock_arm_joints` | [L467-468](../../../crawlbot/simulation/sim_loop.py#L467-L468) |
| `SimulationLoop._diag_lock_arm_joints` | [L471-472](../../../crawlbot/simulation/sim_loop.py#L471-L472) |
| `SimulationLoop.setup` | [L476-769](../../../crawlbot/simulation/sim_loop.py#L476-L769) |
| `SimulationLoop._settle_setup` | [L771-854](../../../crawlbot/simulation/sim_loop.py#L771-L854) |
| `SimulationLoop._run_ds_passivity_loop` | [L856-863](../../../crawlbot/simulation/sim_loop.py#L856-L863) |
| `SimulationLoop._ds_begin` | [L865-888](../../../crawlbot/simulation/sim_loop.py#L865-L888) |
| `SimulationLoop._ds_tick` | [L890-951](../../../crawlbot/simulation/sim_loop.py#L890-L951) |
| `SimulationLoop._ds_end` | [L953-969](../../../crawlbot/simulation/sim_loop.py#L953-L969) |
| `SimulationLoop._build_qp` | [L971-1023](../../../crawlbot/simulation/sim_loop.py#L971-L1023) |
| `SimulationLoop._gripper_distance` | [L1025-1027](../../../crawlbot/simulation/sim_loop.py#L1025-L1027) |
| `SimulationLoop._gripper_speed` | [L1029-1042](../../../crawlbot/simulation/sim_loop.py#L1029-L1042) |
| `SimulationLoop._gripper_ori_err_deg` | [L1044-1057](../../../crawlbot/simulation/sim_loop.py#L1044-L1057) |
| `SimulationLoop._dock_gate` | [L1059-1091](../../../crawlbot/simulation/sim_loop.py#L1059-L1091) |
| `SimulationLoop._setup_torso_for_step` | [L1096-1301](../../../crawlbot/simulation/sim_loop.py#L1096-L1301) |
| `SimulationLoop._run_preplanner` | [L1303-1404](../../../crawlbot/simulation/sim_loop.py#L1303-L1404) |
| `SimulationLoop._capture_snapshot` | [L1408-1411](../../../crawlbot/simulation/sim_loop.py#L1408-L1411) |
| `SimulationLoop.run` | [L1413-1419](../../../crawlbot/simulation/sim_loop.py#L1413-L1419) |
| `SimulationLoop._drive` | [L1423-1450](../../../crawlbot/simulation/sim_loop.py#L1423-L1450) |
| `SimulationLoop._begin` | [L1452-1458](../../../crawlbot/simulation/sim_loop.py#L1452-L1458) |
| `SimulationLoop._gait_program` | [L1460-1994](../../../crawlbot/simulation/sim_loop.py#L1460-L1994) |
| `SimulationLoop._swing_query_time` | [L1998-2014](../../../crawlbot/simulation/sim_loop.py#L1998-L2014) |
| `SimulationLoop._step` | [L2016-2033](../../../crawlbot/simulation/sim_loop.py#L2016-L2033) |
| `SimulationLoop._nmpc_begin` | [L2035-2110](../../../crawlbot/simulation/sim_loop.py#L2035-L2110) |
| `SimulationLoop._nmpc_tick` | [L2112-2119](../../../crawlbot/simulation/sim_loop.py#L2112-L2119) |
| `SimulationLoop._qp_substep` | [L2121-2295](../../../crawlbot/simulation/sim_loop.py#L2121-L2295) |
| `SimulationLoop._nmpc_handoff` | [L2298-2343](../../../crawlbot/simulation/sim_loop.py#L2298-L2343) |
| `SimulationLoop._get_ee_data` | [L2345-2348](../../../crawlbot/simulation/sim_loop.py#L2345-L2348) |
| `SimulationLoop._print_summary` | [L2352-2383](../../../crawlbot/simulation/sim_loop.py#L2352-L2383) |
| `SimulationLoop.plot` | [L2391-2392](../../../crawlbot/simulation/sim_loop.py#L2391-L2392) |

---

## See also

- package overview: [`simulation.md`](simulation.md)
