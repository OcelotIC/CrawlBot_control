# `crawlbot.simulation.sim_loop`

**File**: `crawlbot/simulation/sim_loop.py` — **3387 lines** — canonical coverage **83 %**

> Module docstring: *"SimulationLoop — Closed-loop MuJoCo simulation with two-stage controller."*

**The closed loop.** DS/SS state machine, orchestration of planners and
solvers, weld activation, AOCS, logging. The largest file in the repository and
the one carrying the most architectural history.

---

## Public API

| symbol | signature | canonical? |
|---|---|---|
| **`SimulationLoop`** |  |  |
| `.setup` | `(n_steps=3, start_a=2, start_b=2, sequence_path=None)` | **yes** |
| `._settle_setup` | `(start_a, start_b)` | **yes** |
| `._run_ds_passivity_loop` | `(contact_config, max_steps, epsilon_v, plateau_window, p...)` | **yes** |
| `._interstep_aocs_command` | `(rs, cc_ds, lambda_qp_sol, omega_s_prev)` | **yes** |
| `._log_ds_tick` | `(log, t_abs, step_idx, just_landed_arm, anchor_a_idx, an...)` | **yes** |
| `._build_qp` | `(ae, ap, aw, kpc, kdc, kpt, kdt, kpe, kde, kpe_ang=5.0, ...)` | **yes** |
| `._build_weld_map` | `()` | **yes** |
| `._deactivate_all_welds` | `()` | **yes** |
| `._activate_weld` | `(arm, anchor_idx)` | **yes** |
| `._deactivate_weld` | `(arm, anchor_idx)` | **yes** |
| `._cache_site_ids` | `()` | **yes** |
| `._gripper_distance` | `(arm, anchor_idx)` | **yes** |
| `._gripper_speed` | `(arm)` | not exercised |
| `._gripper_ori_err_deg` | `(arm, anchor_idx)` | **yes** |
| `._weld_relative_twist` | `(arm, anchor_idx)` | **yes** |
| `._dock_gate` | `(swing_arm, target_idx, log, t, step_idx)` | **yes** |
| `._planned_arm_config` | `(t, rs)` | not exercised |
| `._setup_torso_for_step` | `(t_ss_start, swing_arm, stance_a, stance_b, target_arm, ...)` | **yes** |
| `._run_preplanner` | `(t_plan_start, stance_arm, stance_a, stance_b, r_com_0, ...)` | **yes** |
| `._capture_snapshot` | `(log, t, label)` | **yes** |
| `.run` | `(verbose=True)` | **yes** |
| `._swing_query_time` | `(t_raw, phase, ss_end)` | **yes** |
| `._step` | `(t, phase, step_idx, swing_arm, stance_arm, cc_ss, targe...)` | **yes** |
| `._get_ee_data` | `(rs, arm)` | **yes** |
| `._print_summary` | `(log)` | **yes** |
| `.plot` | `(log, save_path=None, cfg=None)` | not exercised |

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

## See also

- package overview: [`simulation.md`](simulation.md)
