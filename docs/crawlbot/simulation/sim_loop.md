# `crawlbot.simulation.sim_loop`

**File**: [`crawlbot/simulation/sim_loop.py`](../../../crawlbot/simulation/sim_loop.py) — **3489 lines** — canonical coverage **83 %**

> Module docstring: *"SimulationLoop — Closed-loop MuJoCo simulation with two-stage controller."*

**The closed loop.** DS/SS state machine, orchestration of planners and
solvers, weld activation, AOCS, logging. The largest file in the repository and
the one carrying the most architectural history.

---

## Public API

| symbol | signature | canonical? | code |
|---|---|---|---|
| **`TickState`** *(dataclass)* |  |  | [L63](../../../crawlbot/simulation/sim_loop.py#L63) |
|   `t` | `` | _field_ | [L90](../../../crawlbot/simulation/sim_loop.py#L90) |
|   `phase` | `` | _field_ | [L91](../../../crawlbot/simulation/sim_loop.py#L91) |
|   `step_idx` | `` | _field_ | [L92](../../../crawlbot/simulation/sim_loop.py#L92) |
|   `ss_end` | `` | _field_ | [L93](../../../crawlbot/simulation/sim_loop.py#L93) |
|   `settle_mode` | `` | _field_ | [L94](../../../crawlbot/simulation/sim_loop.py#L94) |
|   `swing_arm` | `` | _field_ | [L95](../../../crawlbot/simulation/sim_loop.py#L95) |
|   `stance_arm` | `` | _field_ | [L96](../../../crawlbot/simulation/sim_loop.py#L96) |
|   `stance_a` | `` | _field_ | [L97](../../../crawlbot/simulation/sim_loop.py#L97) |
|   `stance_b` | `` | _field_ | [L98](../../../crawlbot/simulation/sim_loop.py#L98) |
|   `target_anchor` | `` | _field_ | [L99](../../../crawlbot/simulation/sim_loop.py#L99) |
|   `hw` | `` | _field_ | [L100](../../../crawlbot/simulation/sim_loop.py#L100) |
|   `L_com_prev` | `` | _field_ | [L101](../../../crawlbot/simulation/sim_loop.py#L101) |
|   `nmpc_ok` | `` | _field_ | [L105](../../../crawlbot/simulation/sim_loop.py#L105) |
|   `nmpc_status_code` | `` | _field_ | [L106](../../../crawlbot/simulation/sim_loop.py#L106) |
|   `nmpc_cost_val` | `` | _field_ | [L107](../../../crawlbot/simulation/sim_loop.py#L107) |
|   `t_nmpc_ms` | `` | _field_ | [L108](../../../crawlbot/simulation/sim_loop.py#L108) |
|   `nmpc_info` | `` | _field_ | [L109](../../../crawlbot/simulation/sim_loop.py#L109) |
|   `lambda_ref` | `` | _field_ | [L110](../../../crawlbot/simulation/sim_loop.py#L110) |
|   `v_com_ref` | `` | _field_ | [L111](../../../crawlbot/simulation/sim_loop.py#L111) |
|   `r_com_ref` | `` | _field_ | [L112](../../../crawlbot/simulation/sim_loop.py#L112) |
|   `qp_ok` | `` | _field_ | [L116](../../../crawlbot/simulation/sim_loop.py#L116) |
|   `t_qp_ms` | `` | _field_ | [L117](../../../crawlbot/simulation/sim_loop.py#L117) |
|   `lambda_qp` | `` | _field_ | [L118](../../../crawlbot/simulation/sim_loop.py#L118) |
|   `tau_joints` | `` | _field_ | [L119](../../../crawlbot/simulation/sim_loop.py#L119) |
|   `tau_wheels` | `` | _field_ | [L120](../../../crawlbot/simulation/sim_loop.py#L120) |
|   `transport_term_mag` | `` | _field_ | [L121](../../../crawlbot/simulation/sim_loop.py#L121) |
|   `p_torso_ref_used` | `None` | _field_ | [L127](../../../crawlbot/simulation/sim_loop.py#L127) |
| **`SimulationLoop`** |  |  | [L132](../../../crawlbot/simulation/sim_loop.py#L132) |
| `.setup` | `(n_steps=3, start_a=2, start_b=2, sequence_path=None)` | **yes** | [L264](../../../crawlbot/simulation/sim_loop.py#L264) |
| `._settle_setup` | `(start_a, start_b)` | **yes** | [L573](../../../crawlbot/simulation/sim_loop.py#L573) |
| `._run_ds_passivity_loop` | `(contact_config, max_steps, epsilon_v, plateau_window, p...)` | **yes** | [L660](../../../crawlbot/simulation/sim_loop.py#L660) |
| `._interstep_aocs_command` | `(rs, cc_ds, lambda_qp_sol, omega_s_prev)` | **yes** | [L897](../../../crawlbot/simulation/sim_loop.py#L897) |
| `._log_ds_tick` | `(log, t_abs, step_idx, just_landed_arm, anchor_a_idx, an...)` | **yes** | [L969](../../../crawlbot/simulation/sim_loop.py#L969) |
| `._build_qp` | `(ae, ap, aw, kpc, kdc, kpt, kdt, kpe, kde, kpe_ang=5.0, ...)` | **yes** | [L1180](../../../crawlbot/simulation/sim_loop.py#L1180) |
| `._build_weld_map` | `()` | **yes** | [L1237](../../../crawlbot/simulation/sim_loop.py#L1237) |
| `._deactivate_all_welds` | `()` | **yes** | [L1248](../../../crawlbot/simulation/sim_loop.py#L1248) |
| `._activate_weld` | `(arm, anchor_idx)` | **yes** | [L1252](../../../crawlbot/simulation/sim_loop.py#L1252) |
| `._deactivate_weld` | `(arm, anchor_idx)` | **yes** | [L1257](../../../crawlbot/simulation/sim_loop.py#L1257) |
| `._cache_site_ids` | `()` | **yes** | [L1262](../../../crawlbot/simulation/sim_loop.py#L1262) |
| `._gripper_distance` | `(arm, anchor_idx)` | **yes** | [L1283](../../../crawlbot/simulation/sim_loop.py#L1283) |
| `._gripper_speed` | `(arm)` | not exercised | [L1291](../../../crawlbot/simulation/sim_loop.py#L1291) |
| `._gripper_ori_err_deg` | `(arm, anchor_idx)` | **yes** | [L1306](../../../crawlbot/simulation/sim_loop.py#L1306) |
| `._weld_relative_twist` | `(arm, anchor_idx)` | **yes** | [L1321](../../../crawlbot/simulation/sim_loop.py#L1321) |
| `._dock_gate` | `(swing_arm, target_idx, log, t, step_idx)` | **yes** | [L1349](../../../crawlbot/simulation/sim_loop.py#L1349) |
| `._planned_arm_config` | `(t, rs)` | not exercised | [L1385](../../../crawlbot/simulation/sim_loop.py#L1385) |
| `._setup_torso_for_step` | `(t_ss_start, swing_arm, stance_a, stance_b, target_arm, ...)` | **yes** | [L1423](../../../crawlbot/simulation/sim_loop.py#L1423) |
| `._run_preplanner` | `(t_plan_start, stance_arm, stance_a, stance_b, r_com_0, ...)` | **yes** | [L1645](../../../crawlbot/simulation/sim_loop.py#L1645) |
| `._capture_snapshot` | `(log, t, label)` | **yes** | [L1754](../../../crawlbot/simulation/sim_loop.py#L1754) |
| `.run` | `(verbose=True)` | **yes** | [L1762](../../../crawlbot/simulation/sim_loop.py#L1762) |
| `._swing_query_time` | `(t_raw, phase, ss_end)` | **yes** | [L2365](../../../crawlbot/simulation/sim_loop.py#L2365) |
| `._step` | `(t, phase, step_idx, swing_arm, stance_arm, cc_ss, targe...)` | **yes** | [L2383](../../../crawlbot/simulation/sim_loop.py#L2383) |
| `._log_ss_tick` | `(log, ts)` | **yes** | [L3235](../../../crawlbot/simulation/sim_loop.py#L3235) |
| `._get_ee_data` | `(rs, arm)` | **yes** | [L3439](../../../crawlbot/simulation/sim_loop.py#L3439) |
| `._print_summary` | `(log)` | **yes** | [L3448](../../../crawlbot/simulation/sim_loop.py#L3448) |
| `.plot` | `(log, save_path=None, cfg=None)` | not exercised | [L3487](../../../crawlbot/simulation/sim_loop.py#L3487) |

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
        +-- _dock_gate() -> _activate_weld()
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
`ori < 5 deg`.* `_dock_gate` applies both; `_activate_weld` runs only after.

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

## 5. `_step()` — 851 lines, and where the next cut goes

Still the largest block, but no longer 1014: CLEANUP-31 lifted the logging tail
into `_log_ss_tick`, the single-support counterpart of the long-standing
`_log_ds_tick` (203 lines against its 210 — the asymmetry was drift, not design).

The cut was chosen by measurement, not by eye. For every statement boundary in
`_step`, count the locals assigned before it and read after it; that number is
the signature any helper extracted there would need:

```
:2333    0
:2437   17  #################
:2507   25  #########################   <- NMPC + QP + integration core
:3181   27  ###########################
:3247   14  ##############
:3318    3  ###                          <- the tail decays to nothing
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

### What is left

The 667-line core (coupling plateau ~25) is the remaining block, and it needs the
same treatment: a state object rather than a parameter list. `run()` is 600 lines
in only **28 top-level statements** — its problem is nesting depth, not sequence,
so there is no cheap top-level seam and extraction has to come from inside the
loop body.

Related debt: `WholeBodyQP.solve()` takes **40 parameters**, 30 of which are read
in exactly one block. Restructuring the signature touches both call sites and was
deliberately deferred (A1). `TickState` is the pattern that would fix it.

## 6. The `use_m2_stack` trap

`SimConfig.use_m2_stack` **looks dead** — its `WholeBodyQPConfig` twin was
removed in CLEANUP-8 — but it gates two paths unrelated to the task stack:

| site | what it gates |
|---|---|
| `sim_loop.py:2581-2584` | torso-reference routing (delta-mapping vs raw quintic) |
| `sim_loop.py:2728-2729` | `passivity_active` — **the DS passivity constraint** |

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
| `class TickState` | [L63-127](../../../crawlbot/simulation/sim_loop.py#L63-L127) |
| `class SimulationLoop` | [L132-3488](../../../crawlbot/simulation/sim_loop.py#L132-L3488) |
| `SimulationLoop.setup` | [L264-571](../../../crawlbot/simulation/sim_loop.py#L264-L571) |
| `SimulationLoop._settle_setup` | [L573-658](../../../crawlbot/simulation/sim_loop.py#L573-L658) |
| `SimulationLoop._run_ds_passivity_loop` | [L660-895](../../../crawlbot/simulation/sim_loop.py#L660-L895) |
| `SimulationLoop._interstep_aocs_command` | [L897-967](../../../crawlbot/simulation/sim_loop.py#L897-L967) |
| `SimulationLoop._log_ds_tick` | [L969-1178](../../../crawlbot/simulation/sim_loop.py#L969-L1178) |
| `SimulationLoop._build_qp` | [L1180-1233](../../../crawlbot/simulation/sim_loop.py#L1180-L1233) |
| `SimulationLoop._build_weld_map` | [L1237-1246](../../../crawlbot/simulation/sim_loop.py#L1237-L1246) |
| `SimulationLoop._deactivate_all_welds` | [L1248-1250](../../../crawlbot/simulation/sim_loop.py#L1248-L1250) |
| `SimulationLoop._activate_weld` | [L1252-1255](../../../crawlbot/simulation/sim_loop.py#L1252-L1255) |
| `SimulationLoop._deactivate_weld` | [L1257-1260](../../../crawlbot/simulation/sim_loop.py#L1257-L1260) |
| `SimulationLoop._cache_site_ids` | [L1262-1281](../../../crawlbot/simulation/sim_loop.py#L1262-L1281) |
| `SimulationLoop._gripper_distance` | [L1283-1289](../../../crawlbot/simulation/sim_loop.py#L1283-L1289) |
| `SimulationLoop._gripper_speed` | [L1291-1304](../../../crawlbot/simulation/sim_loop.py#L1291-L1304) |
| `SimulationLoop._gripper_ori_err_deg` | [L1306-1319](../../../crawlbot/simulation/sim_loop.py#L1306-L1319) |
| `SimulationLoop._weld_relative_twist` | [L1321-1347](../../../crawlbot/simulation/sim_loop.py#L1321-L1347) |
| `SimulationLoop._dock_gate` | [L1349-1381](../../../crawlbot/simulation/sim_loop.py#L1349-L1381) |
| `SimulationLoop._planned_arm_config` | [L1385-1421](../../../crawlbot/simulation/sim_loop.py#L1385-L1421) |
| `SimulationLoop._setup_torso_for_step` | [L1423-1643](../../../crawlbot/simulation/sim_loop.py#L1423-L1643) |
| `SimulationLoop._run_preplanner` | [L1645-1750](../../../crawlbot/simulation/sim_loop.py#L1645-L1750) |
| `SimulationLoop._capture_snapshot` | [L1754-1760](../../../crawlbot/simulation/sim_loop.py#L1754-L1760) |
| `SimulationLoop.run` | [L1762-2361](../../../crawlbot/simulation/sim_loop.py#L1762-L2361) |
| `SimulationLoop._swing_query_time` | [L2365-2381](../../../crawlbot/simulation/sim_loop.py#L2365-L2381) |
| `SimulationLoop._step` | [L2383-3233](../../../crawlbot/simulation/sim_loop.py#L2383-L3233) |
| `SimulationLoop._log_ss_tick` | [L3235-3437](../../../crawlbot/simulation/sim_loop.py#L3235-L3437) |
| `SimulationLoop._get_ee_data` | [L3439-3444](../../../crawlbot/simulation/sim_loop.py#L3439-L3444) |
| `SimulationLoop._print_summary` | [L3448-3479](../../../crawlbot/simulation/sim_loop.py#L3448-L3479) |
| `SimulationLoop.plot` | [L3487-3488](../../../crawlbot/simulation/sim_loop.py#L3487-L3488) |

---

## See also

- package overview: [`simulation.md`](simulation.md)
