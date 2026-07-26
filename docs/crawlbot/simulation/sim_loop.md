# `crawlbot.simulation.sim_loop`

**File**: [`crawlbot/simulation/sim_loop.py`](../../../crawlbot/simulation/sim_loop.py) — **3015 lines** — canonical coverage **63 %**

> Module docstring: *"SimulationLoop — Closed-loop MuJoCo simulation with two-stage controller."*

**The closed loop.** DS/SS state machine, orchestration of planners and
solvers, weld activation, AOCS, logging. The largest file in the repository and
the one carrying the most architectural history.

---

## Public API

| symbol | signature | canonical? | code |
|---|---|---|---|
| **`SimulationLoop`** *(dataclass)* |  |  | [L71](../../../crawlbot/simulation/sim_loop.py#L71) |
| `.setup` | `(n_steps=3, start_a=2, start_b=2, sequence_path=None)` | **yes** | [L203](../../../crawlbot/simulation/sim_loop.py#L203) |
| `._settle_setup` | `(start_a, start_b)` | **yes** | [L512](../../../crawlbot/simulation/sim_loop.py#L512) |
| `._run_ds_passivity_loop` | `(contact_config, max_steps, epsilon_v, plateau_window, p...)` | **yes** | [L599](../../../crawlbot/simulation/sim_loop.py#L599) |
| `._interstep_aocs_command` | `(rs, cc_ds, lambda_qp_sol, omega_s_prev)` | **yes** | [L836](../../../crawlbot/simulation/sim_loop.py#L836) |
| `._build_qp` | `(ae, ap, aw, kpc, kdc, kpt, kdt, kpe, kde, kpe_ang=5.0, ...)` | **yes** | [L909](../../../crawlbot/simulation/sim_loop.py#L909) |
| `._build_weld_map` | `()` | **yes** | [L966](../../../crawlbot/simulation/sim_loop.py#L966) |
| `._deactivate_all_welds` | `()` | **yes** | [L977](../../../crawlbot/simulation/sim_loop.py#L977) |
| `._activate_weld` | `(arm, anchor_idx)` | **yes** | [L981](../../../crawlbot/simulation/sim_loop.py#L981) |
| `._deactivate_weld` | `(arm, anchor_idx)` | **yes** | [L986](../../../crawlbot/simulation/sim_loop.py#L986) |
| `._cache_site_ids` | `()` | **yes** | [L991](../../../crawlbot/simulation/sim_loop.py#L991) |
| `._gripper_distance` | `(arm, anchor_idx)` | **yes** | [L1012](../../../crawlbot/simulation/sim_loop.py#L1012) |
| `._gripper_speed` | `(arm)` | not exercised | [L1020](../../../crawlbot/simulation/sim_loop.py#L1020) |
| `._gripper_ori_err_deg` | `(arm, anchor_idx)` | not exercised | [L1035](../../../crawlbot/simulation/sim_loop.py#L1035) |
| `._weld_relative_twist` | `(arm, anchor_idx)` | not exercised | [L1050](../../../crawlbot/simulation/sim_loop.py#L1050) |
| `._dock_gate` | `(swing_arm, target_idx, log, t, step_idx)` | not exercised | [L1078](../../../crawlbot/simulation/sim_loop.py#L1078) |
| `._planned_arm_config` | `(t, rs)` | not exercised | [L1114](../../../crawlbot/simulation/sim_loop.py#L1114) |
| `._setup_torso_for_step` | `(t_ss_start, swing_arm, stance_a, stance_b, target_arm, ...)` | **yes** | [L1152](../../../crawlbot/simulation/sim_loop.py#L1152) |
| `._run_preplanner` | `(t_plan_start, stance_arm, stance_a, stance_b, r_com_0, ...)` | **yes** | [L1374](../../../crawlbot/simulation/sim_loop.py#L1374) |
| `._capture_snapshot` | `(log, t, label)` | **yes** | [L1483](../../../crawlbot/simulation/sim_loop.py#L1483) |
| `.run` | `(verbose=True)` | **yes** | [L1491](../../../crawlbot/simulation/sim_loop.py#L1491) |
| `._swing_query_time` | `(t_raw, phase, ss_end)` | **yes** | [L2094](../../../crawlbot/simulation/sim_loop.py#L2094) |
| `._step` | `(t, phase, step_idx, swing_arm, stance_arm, cc_ss, targe...)` | **yes** | [L2112](../../../crawlbot/simulation/sim_loop.py#L2112) |
| `._get_ee_data` | `(rs, arm)` | **yes** | [L2965](../../../crawlbot/simulation/sim_loop.py#L2965) |
| `._print_summary` | `(log)` | not exercised | [L2974](../../../crawlbot/simulation/sim_loop.py#L2974) |
| `.plot` | `(log, save_path=None, cfg=None)` | not exercised | [L3013](../../../crawlbot/simulation/sim_loop.py#L3013) |

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
| `class SimulationLoop` | [L71-3014](../../../crawlbot/simulation/sim_loop.py#L71-L3014) |
| `SimulationLoop.setup` | [L203-510](../../../crawlbot/simulation/sim_loop.py#L203-L510) |
| `SimulationLoop._settle_setup` | [L512-597](../../../crawlbot/simulation/sim_loop.py#L512-L597) |
| `SimulationLoop._run_ds_passivity_loop` | [L599-834](../../../crawlbot/simulation/sim_loop.py#L599-L834) |
| `SimulationLoop._interstep_aocs_command` | [L836-906](../../../crawlbot/simulation/sim_loop.py#L836-L906) |
| `SimulationLoop._build_qp` | [L909-962](../../../crawlbot/simulation/sim_loop.py#L909-L962) |
| `SimulationLoop._build_weld_map` | [L966-975](../../../crawlbot/simulation/sim_loop.py#L966-L975) |
| `SimulationLoop._deactivate_all_welds` | [L977-979](../../../crawlbot/simulation/sim_loop.py#L977-L979) |
| `SimulationLoop._activate_weld` | [L981-984](../../../crawlbot/simulation/sim_loop.py#L981-L984) |
| `SimulationLoop._deactivate_weld` | [L986-989](../../../crawlbot/simulation/sim_loop.py#L986-L989) |
| `SimulationLoop._cache_site_ids` | [L991-1010](../../../crawlbot/simulation/sim_loop.py#L991-L1010) |
| `SimulationLoop._gripper_distance` | [L1012-1018](../../../crawlbot/simulation/sim_loop.py#L1012-L1018) |
| `SimulationLoop._gripper_speed` | [L1020-1033](../../../crawlbot/simulation/sim_loop.py#L1020-L1033) |
| `SimulationLoop._gripper_ori_err_deg` | [L1035-1048](../../../crawlbot/simulation/sim_loop.py#L1035-L1048) |
| `SimulationLoop._weld_relative_twist` | [L1050-1076](../../../crawlbot/simulation/sim_loop.py#L1050-L1076) |
| `SimulationLoop._dock_gate` | [L1078-1110](../../../crawlbot/simulation/sim_loop.py#L1078-L1110) |
| `SimulationLoop._planned_arm_config` | [L1114-1150](../../../crawlbot/simulation/sim_loop.py#L1114-L1150) |
| `SimulationLoop._setup_torso_for_step` | [L1152-1372](../../../crawlbot/simulation/sim_loop.py#L1152-L1372) |
| `SimulationLoop._run_preplanner` | [L1374-1479](../../../crawlbot/simulation/sim_loop.py#L1374-L1479) |
| `SimulationLoop._capture_snapshot` | [L1483-1489](../../../crawlbot/simulation/sim_loop.py#L1483-L1489) |
| `SimulationLoop.run` | [L1491-2090](../../../crawlbot/simulation/sim_loop.py#L1491-L2090) |
| `SimulationLoop._swing_query_time` | [L2094-2110](../../../crawlbot/simulation/sim_loop.py#L2094-L2110) |
| `SimulationLoop._step` | [L2112-2962](../../../crawlbot/simulation/sim_loop.py#L2112-L2962) |
| `SimulationLoop._get_ee_data` | [L2965-2970](../../../crawlbot/simulation/sim_loop.py#L2965-L2970) |
| `SimulationLoop._print_summary` | [L2974-3005](../../../crawlbot/simulation/sim_loop.py#L2974-L3005) |
| `SimulationLoop.plot` | [L3013-3014](../../../crawlbot/simulation/sim_loop.py#L3013-L3014) |

---

## See also

- package overview: [`simulation.md`](simulation.md)
