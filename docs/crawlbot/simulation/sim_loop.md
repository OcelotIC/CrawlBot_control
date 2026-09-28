# `crawlbot.simulation.sim_loop`

**File**: [`crawlbot/simulation/sim_loop.py`](../../../crawlbot/simulation/sim_loop.py) — **2917 lines** — canonical coverage **82 %**

> Module docstring: *"SimulationLoop — Closed-loop MuJoCo simulation with two-stage controller."*

**The closed loop.** DS/SS state machine, orchestration of planners and
solvers, AOCS, logging. Every MuJoCo **write** (step, welds, dock impact,
actuators, diagnostic lock) now goes through [`plant.py`](plant.md) — extraction 1
of the `refactor/sim-loop-split` chantier. The largest file in the repository and
the one carrying the most architectural history.

---

## Public API

| symbol | signature | canonical? | code |
|---|---|---|---|
| **`SimulationLoop`** |  |  | [L93](../../../crawlbot/simulation/sim_loop.py#L93) |
| `.mj_model` | `()` | **yes** | [L228](../../../crawlbot/simulation/sim_loop.py#L228) |
| `.mj_data` | `()` | **yes** | [L232](../../../crawlbot/simulation/sim_loop.py#L232) |
| `.setup` | `(n_steps=3, start_a=2, start_b=2, sequence_path=None)` | **yes** | [L237](../../../crawlbot/simulation/sim_loop.py#L237) |
| `._settle_setup` | `(start_a, start_b)` | **yes** | [L541](../../../crawlbot/simulation/sim_loop.py#L541) |
| `._run_ds_passivity_loop` | `(contact_config, max_steps, epsilon_v, plateau_window, p...)` | **yes** | [L628](../../../crawlbot/simulation/sim_loop.py#L628) |
| `._interstep_aocs_command` | `(rs, cc_ds, lambda_qp_sol, omega_s_prev)` | **yes** | [L866](../../../crawlbot/simulation/sim_loop.py#L866) |
| `._build_qp` | `(ae, ap, aw, kpc, kdc, kpt, kdt, kpe, kde, kpe_ang=5.0, ...)` | **yes** | [L939](../../../crawlbot/simulation/sim_loop.py#L939) |
| `._gripper_distance` | `(arm, anchor_idx)` | **yes** | [L994](../../../crawlbot/simulation/sim_loop.py#L994) |
| `._gripper_speed` | `(arm)` | not exercised | [L1002](../../../crawlbot/simulation/sim_loop.py#L1002) |
| `._gripper_ori_err_deg` | `(arm, anchor_idx)` | **yes** | [L1017](../../../crawlbot/simulation/sim_loop.py#L1017) |
| `._weld_relative_twist` | `(arm, anchor_idx)` | **yes** | [L1032](../../../crawlbot/simulation/sim_loop.py#L1032) |
| `._dock_gate` | `(swing_arm, target_idx, log, t, step_idx)` | **yes** | [L1060](../../../crawlbot/simulation/sim_loop.py#L1060) |
| `._setup_torso_for_step` | `(t_ss_start, swing_arm, stance_a, stance_b, target_arm, ...)` | **yes** | [L1097](../../../crawlbot/simulation/sim_loop.py#L1097) |
| `._run_preplanner` | `(t_plan_start, stance_arm, stance_a, stance_b, r_com_0, ...)` | **yes** | [L1316](../../../crawlbot/simulation/sim_loop.py#L1316) |
| `._capture_snapshot` | `(log, t, label)` | **yes** | [L1425](../../../crawlbot/simulation/sim_loop.py#L1425) |
| `.run` | `(verbose=True)` | **yes** | [L1433](../../../crawlbot/simulation/sim_loop.py#L1433) |
| `._swing_query_time` | `(t_raw, phase, ss_end)` | **yes** | [L1977](../../../crawlbot/simulation/sim_loop.py#L1977) |
| `._step` | `(t, phase, step_idx, swing_arm, stance_arm, cc_ss, targe...)` | **yes** | [L1995](../../../crawlbot/simulation/sim_loop.py#L1995) |
| `._get_ee_data` | `(rs, arm)` | **yes** | [L2867](../../../crawlbot/simulation/sim_loop.py#L2867) |
| `._print_summary` | `(log)` | **yes** | [L2876](../../../crawlbot/simulation/sim_loop.py#L2876) |
| `.plot` | `(log, save_path=None, cfg=None)` | not exercised | [L2915](../../../crawlbot/simulation/sim_loop.py#L2915) |

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
**STAGE 2, whole-body QP sub-loop** (the 615-line `for qs`, dt 0.01 s) → **hand
off to telemetry**.

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
| `class SimulationLoop` | [L93-2916](../../../crawlbot/simulation/sim_loop.py#L93-L2916) |
| `SimulationLoop.mj_model` | [L228-229](../../../crawlbot/simulation/sim_loop.py#L228-L229) |
| `SimulationLoop.mj_data` | [L232-233](../../../crawlbot/simulation/sim_loop.py#L232-L233) |
| `SimulationLoop.setup` | [L237-539](../../../crawlbot/simulation/sim_loop.py#L237-L539) |
| `SimulationLoop._settle_setup` | [L541-626](../../../crawlbot/simulation/sim_loop.py#L541-L626) |
| `SimulationLoop._run_ds_passivity_loop` | [L628-864](../../../crawlbot/simulation/sim_loop.py#L628-L864) |
| `SimulationLoop._interstep_aocs_command` | [L866-936](../../../crawlbot/simulation/sim_loop.py#L866-L936) |
| `SimulationLoop._build_qp` | [L939-992](../../../crawlbot/simulation/sim_loop.py#L939-L992) |
| `SimulationLoop._gripper_distance` | [L994-1000](../../../crawlbot/simulation/sim_loop.py#L994-L1000) |
| `SimulationLoop._gripper_speed` | [L1002-1015](../../../crawlbot/simulation/sim_loop.py#L1002-L1015) |
| `SimulationLoop._gripper_ori_err_deg` | [L1017-1030](../../../crawlbot/simulation/sim_loop.py#L1017-L1030) |
| `SimulationLoop._weld_relative_twist` | [L1032-1058](../../../crawlbot/simulation/sim_loop.py#L1032-L1058) |
| `SimulationLoop._dock_gate` | [L1060-1092](../../../crawlbot/simulation/sim_loop.py#L1060-L1092) |
| `SimulationLoop._setup_torso_for_step` | [L1097-1314](../../../crawlbot/simulation/sim_loop.py#L1097-L1314) |
| `SimulationLoop._run_preplanner` | [L1316-1421](../../../crawlbot/simulation/sim_loop.py#L1316-L1421) |
| `SimulationLoop._capture_snapshot` | [L1425-1431](../../../crawlbot/simulation/sim_loop.py#L1425-L1431) |
| `SimulationLoop.run` | [L1433-1973](../../../crawlbot/simulation/sim_loop.py#L1433-L1973) |
| `SimulationLoop._swing_query_time` | [L1977-1993](../../../crawlbot/simulation/sim_loop.py#L1977-L1993) |
| `SimulationLoop._step` | [L1995-2864](../../../crawlbot/simulation/sim_loop.py#L1995-L2864) |
| `SimulationLoop._get_ee_data` | [L2867-2872](../../../crawlbot/simulation/sim_loop.py#L2867-L2872) |
| `SimulationLoop._print_summary` | [L2876-2907](../../../crawlbot/simulation/sim_loop.py#L2876-L2907) |
| `SimulationLoop.plot` | [L2915-2916](../../../crawlbot/simulation/sim_loop.py#L2915-L2916) |

---

## See also

- package overview: [`simulation.md`](simulation.md)
