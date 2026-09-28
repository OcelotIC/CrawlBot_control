# `crawlbot.simulation.sim_loop`

**File**: [`crawlbot/simulation/sim_loop.py`](../../../crawlbot/simulation/sim_loop.py) — **2271 lines** — canonical coverage **87 %**

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
| **`PlannerReferences`** |  |  | [L97](../../../crawlbot/simulation/sim_loop.py#L97) |
| `.com_at` | `(t, settle_mode)` | **yes** | [L113](../../../crawlbot/simulation/sim_loop.py#L113) |
| `.L_com_at` | `(t_mid)` | **yes** | [L172](../../../crawlbot/simulation/sim_loop.py#L172) |
| `.torso_at` | `(tq, phase, ss_end)` | **yes** | [L176](../../../crawlbot/simulation/sim_loop.py#L176) |
| `.swing_at` | `(tq, phase, ss_end)` | **yes** | [L191](../../../crawlbot/simulation/sim_loop.py#L191) |
| **`SimulationLoop`** |  |  | [L199](../../../crawlbot/simulation/sim_loop.py#L199) |
| `.mj_model` | `()` | **yes** | [L300](../../../crawlbot/simulation/sim_loop.py#L300) |
| `.mj_data` | `()` | **yes** | [L304](../../../crawlbot/simulation/sim_loop.py#L304) |
| `._sat_total_calls` | `()` | **yes** | [L310](../../../crawlbot/simulation/sim_loop.py#L310) |
| `._sat_clipped_calls` | `()` | **yes** | [L314](../../../crawlbot/simulation/sim_loop.py#L314) |
| `._sat_max_clip_mm` | `()` | **yes** | [L318](../../../crawlbot/simulation/sim_loop.py#L318) |
| `._diag_pure_pd` | `()` | not exercised | [L323](../../../crawlbot/simulation/sim_loop.py#L323) |
| `._diag_pure_pd` | `(v)` | not exercised | [L327](../../../crawlbot/simulation/sim_loop.py#L327) |
| `._diag_freeze_ref` | `()` | not exercised | [L331](../../../crawlbot/simulation/sim_loop.py#L331) |
| `._diag_freeze_ref` | `(v)` | not exercised | [L335](../../../crawlbot/simulation/sim_loop.py#L335) |
| `._diag_disable_aocs` | `()` | not exercised | [L339](../../../crawlbot/simulation/sim_loop.py#L339) |
| `._diag_disable_aocs` | `(v)` | not exercised | [L343](../../../crawlbot/simulation/sim_loop.py#L343) |
| `._diag_lock_arm_joints` | `()` | **yes** | [L347](../../../crawlbot/simulation/sim_loop.py#L347) |
| `._diag_lock_arm_joints` | `(v)` | not exercised | [L351](../../../crawlbot/simulation/sim_loop.py#L351) |
| `.setup` | `(n_steps=3, start_a=2, start_b=2, sequence_path=None)` | **yes** | [L356](../../../crawlbot/simulation/sim_loop.py#L356) |
| `._settle_setup` | `(start_a, start_b)` | **yes** | [L673](../../../crawlbot/simulation/sim_loop.py#L673) |
| `._run_ds_passivity_loop` | `(contact_config, max_steps, epsilon_v, plateau_window, p...)` | **yes** | [L758](../../../crawlbot/simulation/sim_loop.py#L758) |
| `._build_qp` | `(ae, ap, aw, kpc, kdc, kpt, kdt, kpe, kde, kpe_ang=5.0, ...)` | **yes** | [L927](../../../crawlbot/simulation/sim_loop.py#L927) |
| `._gripper_distance` | `(arm, anchor_idx)` | **yes** | [L982](../../../crawlbot/simulation/sim_loop.py#L982) |
| `._gripper_speed` | `(arm)` | not exercised | [L986](../../../crawlbot/simulation/sim_loop.py#L986) |
| `._gripper_ori_err_deg` | `(arm, anchor_idx)` | **yes** | [L1001](../../../crawlbot/simulation/sim_loop.py#L1001) |
| `._dock_gate` | `(swing_arm, target_idx, log, t, step_idx)` | **yes** | [L1016](../../../crawlbot/simulation/sim_loop.py#L1016) |
| `._setup_torso_for_step` | `(t_ss_start, swing_arm, stance_a, stance_b, target_arm, ...)` | **yes** | [L1053](../../../crawlbot/simulation/sim_loop.py#L1053) |
| `._run_preplanner` | `(t_plan_start, stance_arm, stance_a, stance_b, r_com_0, ...)` | **yes** | [L1268](../../../crawlbot/simulation/sim_loop.py#L1268) |
| `._capture_snapshot` | `(log, t, label)` | **yes** | [L1376](../../../crawlbot/simulation/sim_loop.py#L1376) |
| `.run` | `(verbose=True)` | **yes** | [L1381](../../../crawlbot/simulation/sim_loop.py#L1381) |
| `._swing_query_time` | `(t_raw, phase, ss_end)` | **yes** | [L1918](../../../crawlbot/simulation/sim_loop.py#L1918) |
| `._step` | `(t, phase, step_idx, swing_arm, stance_arm, cc_ss, targe...)` | **yes** | [L1936](../../../crawlbot/simulation/sim_loop.py#L1936) |
| `._get_ee_data` | `(rs, arm)` | **yes** | [L2223](../../../crawlbot/simulation/sim_loop.py#L2223) |
| `._print_summary` | `(log)` | **yes** | [L2230](../../../crawlbot/simulation/sim_loop.py#L2230) |
| `.plot` | `(log, save_path=None, cfg=None)` | not exercised | [L2269](../../../crawlbot/simulation/sim_loop.py#L2269) |

---

---

## 1. Two phases, not three

`DS` (double support) and `SS` (single support). Explicit project rule: *do not
implement a three-phase state machine (DS/SS/EXT) — the architecture is two-phase
per spec 7.1.*

```
setup()                          models, planners, solvers, anchors
  |
  +-- run()                      loop over steps
        +-- _setup_torso_for_step()   docking IK + torso phase
        +-- _run_preplanner()         T_step + feasible CoM trajectory
        +-- _step()                   SS: the swing
        +-- _run_ds_passivity_loop()  DS: passive settle
        +-- _dock_gate() -> plant.activate_weld() + plant.apply_dock_impact()
```

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

`_run_ds_passivity_loop` dissipates residual energy through a passivity
**inequality** in the QP rather than a damping cost. The distinction matters: a
cost trades against the other tasks and can be outvoted; an inequality cannot.
It guarantees the energy budget is non-increasing whatever the task weights do.

## 5. `_step()` — the largest block, and where the next cut goes

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

`run()` is still 600 lines of nesting; its DS loop becomes part of a single tick
loop in extraction 4.

## 6. The `use_m2_stack` trap

`SimConfig.use_m2_stack` **looks dead** — its `WholeBodyQPConfig` twin was
removed in CLEANUP-8 — but it gates two paths unrelated to the task stack:

| site | what it gates |
|---|---|
| `controller.py:421-423` | torso-reference routing (delta-mapping vs raw quintic) |
| `controller.py:567-568` | `passivity_active` — **the DS passivity constraint** |

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
| `class PlannerReferences` | [L97-196](../../../crawlbot/simulation/sim_loop.py#L97-L196) |
| `PlannerReferences.com_at` | [L113-170](../../../crawlbot/simulation/sim_loop.py#L113-L170) |
| `PlannerReferences.L_com_at` | [L172-174](../../../crawlbot/simulation/sim_loop.py#L172-L174) |
| `PlannerReferences.torso_at` | [L176-189](../../../crawlbot/simulation/sim_loop.py#L176-L189) |
| `PlannerReferences.swing_at` | [L191-196](../../../crawlbot/simulation/sim_loop.py#L191-L196) |
| `class SimulationLoop` | [L199-2270](../../../crawlbot/simulation/sim_loop.py#L199-L2270) |
| `SimulationLoop.mj_model` | [L300-301](../../../crawlbot/simulation/sim_loop.py#L300-L301) |
| `SimulationLoop.mj_data` | [L304-305](../../../crawlbot/simulation/sim_loop.py#L304-L305) |
| `SimulationLoop._sat_total_calls` | [L310-311](../../../crawlbot/simulation/sim_loop.py#L310-L311) |
| `SimulationLoop._sat_clipped_calls` | [L314-315](../../../crawlbot/simulation/sim_loop.py#L314-L315) |
| `SimulationLoop._sat_max_clip_mm` | [L318-319](../../../crawlbot/simulation/sim_loop.py#L318-L319) |
| `SimulationLoop._diag_pure_pd` | [L323-324](../../../crawlbot/simulation/sim_loop.py#L323-L324) |
| `SimulationLoop._diag_pure_pd` | [L327-328](../../../crawlbot/simulation/sim_loop.py#L327-L328) |
| `SimulationLoop._diag_freeze_ref` | [L331-332](../../../crawlbot/simulation/sim_loop.py#L331-L332) |
| `SimulationLoop._diag_freeze_ref` | [L335-336](../../../crawlbot/simulation/sim_loop.py#L335-L336) |
| `SimulationLoop._diag_disable_aocs` | [L339-340](../../../crawlbot/simulation/sim_loop.py#L339-L340) |
| `SimulationLoop._diag_disable_aocs` | [L343-344](../../../crawlbot/simulation/sim_loop.py#L343-L344) |
| `SimulationLoop._diag_lock_arm_joints` | [L347-348](../../../crawlbot/simulation/sim_loop.py#L347-L348) |
| `SimulationLoop._diag_lock_arm_joints` | [L351-352](../../../crawlbot/simulation/sim_loop.py#L351-L352) |
| `SimulationLoop.setup` | [L356-671](../../../crawlbot/simulation/sim_loop.py#L356-L671) |
| `SimulationLoop._settle_setup` | [L673-756](../../../crawlbot/simulation/sim_loop.py#L673-L756) |
| `SimulationLoop._run_ds_passivity_loop` | [L758-925](../../../crawlbot/simulation/sim_loop.py#L758-L925) |
| `SimulationLoop._build_qp` | [L927-980](../../../crawlbot/simulation/sim_loop.py#L927-L980) |
| `SimulationLoop._gripper_distance` | [L982-984](../../../crawlbot/simulation/sim_loop.py#L982-L984) |
| `SimulationLoop._gripper_speed` | [L986-999](../../../crawlbot/simulation/sim_loop.py#L986-L999) |
| `SimulationLoop._gripper_ori_err_deg` | [L1001-1014](../../../crawlbot/simulation/sim_loop.py#L1001-L1014) |
| `SimulationLoop._dock_gate` | [L1016-1048](../../../crawlbot/simulation/sim_loop.py#L1016-L1048) |
| `SimulationLoop._setup_torso_for_step` | [L1053-1266](../../../crawlbot/simulation/sim_loop.py#L1053-L1266) |
| `SimulationLoop._run_preplanner` | [L1268-1372](../../../crawlbot/simulation/sim_loop.py#L1268-L1372) |
| `SimulationLoop._capture_snapshot` | [L1376-1379](../../../crawlbot/simulation/sim_loop.py#L1376-L1379) |
| `SimulationLoop.run` | [L1381-1914](../../../crawlbot/simulation/sim_loop.py#L1381-L1914) |
| `SimulationLoop._swing_query_time` | [L1918-1934](../../../crawlbot/simulation/sim_loop.py#L1918-L1934) |
| `SimulationLoop._step` | [L1936-2220](../../../crawlbot/simulation/sim_loop.py#L1936-L2220) |
| `SimulationLoop._get_ee_data` | [L2223-2226](../../../crawlbot/simulation/sim_loop.py#L2223-L2226) |
| `SimulationLoop._print_summary` | [L2230-2261](../../../crawlbot/simulation/sim_loop.py#L2230-L2261) |
| `SimulationLoop.plot` | [L2269-2270](../../../crawlbot/simulation/sim_loop.py#L2269-L2270) |

---

## See also

- package overview: [`simulation.md`](simulation.md)
