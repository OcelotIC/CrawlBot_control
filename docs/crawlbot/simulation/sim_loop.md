# `crawlbot.simulation.sim_loop`

**File**: [`crawlbot/simulation/sim_loop.py`](../../../crawlbot/simulation/sim_loop.py) — **3387 lines** — canonical coverage **83 %**

> Module docstring: *"SimulationLoop — Closed-loop MuJoCo simulation with two-stage controller."*

**The closed loop.** DS/SS state machine, orchestration of planners and
solvers, weld activation, AOCS, logging. The largest file in the repository and
the one carrying the most architectural history.

---

## Public API

| symbol | signature | canonical? | code |
|---|---|---|---|
| **`SimulationLoop`** |  |  | [L63](../../../crawlbot/simulation/sim_loop.py#L63) |
| `.setup` | `(n_steps=3, start_a=2, start_b=2, sequence_path=None)` | **yes** | [L195](../../../crawlbot/simulation/sim_loop.py#L195) |
| `._settle_setup` | `(start_a, start_b)` | **yes** | [L504](../../../crawlbot/simulation/sim_loop.py#L504) |
| `._run_ds_passivity_loop` | `(contact_config, max_steps, epsilon_v, plateau_window, p...)` | **yes** | [L591](../../../crawlbot/simulation/sim_loop.py#L591) |
| `._interstep_aocs_command` | `(rs, cc_ds, lambda_qp_sol, omega_s_prev)` | **yes** | [L828](../../../crawlbot/simulation/sim_loop.py#L828) |
| `._log_ds_tick` | `(log, t_abs, step_idx, just_landed_arm, anchor_a_idx, an...)` | **yes** | [L900](../../../crawlbot/simulation/sim_loop.py#L900) |
| `._build_qp` | `(ae, ap, aw, kpc, kdc, kpt, kdt, kpe, kde, kpe_ang=5.0, ...)` | **yes** | [L1111](../../../crawlbot/simulation/sim_loop.py#L1111) |
| `._build_weld_map` | `()` | **yes** | [L1168](../../../crawlbot/simulation/sim_loop.py#L1168) |
| `._deactivate_all_welds` | `()` | **yes** | [L1179](../../../crawlbot/simulation/sim_loop.py#L1179) |
| `._activate_weld` | `(arm, anchor_idx)` | **yes** | [L1183](../../../crawlbot/simulation/sim_loop.py#L1183) |
| `._deactivate_weld` | `(arm, anchor_idx)` | **yes** | [L1188](../../../crawlbot/simulation/sim_loop.py#L1188) |
| `._cache_site_ids` | `()` | **yes** | [L1193](../../../crawlbot/simulation/sim_loop.py#L1193) |
| `._gripper_distance` | `(arm, anchor_idx)` | **yes** | [L1214](../../../crawlbot/simulation/sim_loop.py#L1214) |
| `._gripper_speed` | `(arm)` | not exercised | [L1222](../../../crawlbot/simulation/sim_loop.py#L1222) |
| `._gripper_ori_err_deg` | `(arm, anchor_idx)` | **yes** | [L1237](../../../crawlbot/simulation/sim_loop.py#L1237) |
| `._weld_relative_twist` | `(arm, anchor_idx)` | **yes** | [L1252](../../../crawlbot/simulation/sim_loop.py#L1252) |
| `._dock_gate` | `(swing_arm, target_idx, log, t, step_idx)` | **yes** | [L1280](../../../crawlbot/simulation/sim_loop.py#L1280) |
| `._planned_arm_config` | `(t, rs)` | not exercised | [L1316](../../../crawlbot/simulation/sim_loop.py#L1316) |
| `._setup_torso_for_step` | `(t_ss_start, swing_arm, stance_a, stance_b, target_arm, ...)` | **yes** | [L1354](../../../crawlbot/simulation/sim_loop.py#L1354) |
| `._run_preplanner` | `(t_plan_start, stance_arm, stance_a, stance_b, r_com_0, ...)` | **yes** | [L1584](../../../crawlbot/simulation/sim_loop.py#L1584) |
| `._capture_snapshot` | `(log, t, label)` | **yes** | [L1693](../../../crawlbot/simulation/sim_loop.py#L1693) |
| `.run` | `(verbose=True)` | **yes** | [L1701](../../../crawlbot/simulation/sim_loop.py#L1701) |
| `._swing_query_time` | `(t_raw, phase, ss_end)` | **yes** | [L2304](../../../crawlbot/simulation/sim_loop.py#L2304) |
| `._step` | `(t, phase, step_idx, swing_arm, stance_arm, cc_ss, targe...)` | **yes** | [L2322](../../../crawlbot/simulation/sim_loop.py#L2322) |
| `._get_ee_data` | `(rs, arm)` | **yes** | [L3337](../../../crawlbot/simulation/sim_loop.py#L3337) |
| `._print_summary` | `(log)` | **yes** | [L3346](../../../crawlbot/simulation/sim_loop.py#L3346) |
| `.plot` | `(log, save_path=None, cfg=None)` | not exercised | [L3385](../../../crawlbot/simulation/sim_loop.py#L3385) |

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

## 5. `_step()` — 1013 lines, the main debt

The largest remaining block. Decomposition is identified but **not done**: it
needs its own coupling measurement first (`CLEANUP_CARRYOVER` A).

Related debt: `WholeBodyQP.solve()` takes **40 parameters**, 30 of which are read
in exactly one block. Restructuring the signature touches both call sites and was
deliberately deferred (A1).

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
| `class SimulationLoop` | [L63-3386](../../../crawlbot/simulation/sim_loop.py#L63-L3386) |
| `SimulationLoop.setup` | [L195-502](../../../crawlbot/simulation/sim_loop.py#L195-L502) |
| `SimulationLoop._settle_setup` | [L504-589](../../../crawlbot/simulation/sim_loop.py#L504-L589) |
| `SimulationLoop._run_ds_passivity_loop` | [L591-826](../../../crawlbot/simulation/sim_loop.py#L591-L826) |
| `SimulationLoop._interstep_aocs_command` | [L828-898](../../../crawlbot/simulation/sim_loop.py#L828-L898) |
| `SimulationLoop._log_ds_tick` | [L900-1109](../../../crawlbot/simulation/sim_loop.py#L900-L1109) |
| `SimulationLoop._build_qp` | [L1111-1164](../../../crawlbot/simulation/sim_loop.py#L1111-L1164) |
| `SimulationLoop._build_weld_map` | [L1168-1177](../../../crawlbot/simulation/sim_loop.py#L1168-L1177) |
| `SimulationLoop._deactivate_all_welds` | [L1179-1181](../../../crawlbot/simulation/sim_loop.py#L1179-L1181) |
| `SimulationLoop._activate_weld` | [L1183-1186](../../../crawlbot/simulation/sim_loop.py#L1183-L1186) |
| `SimulationLoop._deactivate_weld` | [L1188-1191](../../../crawlbot/simulation/sim_loop.py#L1188-L1191) |
| `SimulationLoop._cache_site_ids` | [L1193-1212](../../../crawlbot/simulation/sim_loop.py#L1193-L1212) |
| `SimulationLoop._gripper_distance` | [L1214-1220](../../../crawlbot/simulation/sim_loop.py#L1214-L1220) |
| `SimulationLoop._gripper_speed` | [L1222-1235](../../../crawlbot/simulation/sim_loop.py#L1222-L1235) |
| `SimulationLoop._gripper_ori_err_deg` | [L1237-1250](../../../crawlbot/simulation/sim_loop.py#L1237-L1250) |
| `SimulationLoop._weld_relative_twist` | [L1252-1278](../../../crawlbot/simulation/sim_loop.py#L1252-L1278) |
| `SimulationLoop._dock_gate` | [L1280-1312](../../../crawlbot/simulation/sim_loop.py#L1280-L1312) |
| `SimulationLoop._planned_arm_config` | [L1316-1352](../../../crawlbot/simulation/sim_loop.py#L1316-L1352) |
| `SimulationLoop._setup_torso_for_step` | [L1354-1582](../../../crawlbot/simulation/sim_loop.py#L1354-L1582) |
| `SimulationLoop._run_preplanner` | [L1584-1689](../../../crawlbot/simulation/sim_loop.py#L1584-L1689) |
| `SimulationLoop._capture_snapshot` | [L1693-1699](../../../crawlbot/simulation/sim_loop.py#L1693-L1699) |
| `SimulationLoop.run` | [L1701-2300](../../../crawlbot/simulation/sim_loop.py#L1701-L2300) |
| `SimulationLoop._swing_query_time` | [L2304-2320](../../../crawlbot/simulation/sim_loop.py#L2304-L2320) |
| `SimulationLoop._step` | [L2322-3335](../../../crawlbot/simulation/sim_loop.py#L2322-L3335) |
| `SimulationLoop._get_ee_data` | [L3337-3342](../../../crawlbot/simulation/sim_loop.py#L3337-L3342) |
| `SimulationLoop._print_summary` | [L3346-3377](../../../crawlbot/simulation/sim_loop.py#L3346-L3377) |
| `SimulationLoop.plot` | [L3385-3386](../../../crawlbot/simulation/sim_loop.py#L3385-L3386) |

---

## See also

- package overview: [`simulation.md`](simulation.md)
