# `crawlbot.simulation.sim_loop`

**File**: [`crawlbot/simulation/sim_loop.py`](../../../crawlbot/simulation/sim_loop.py) — **3053 lines** — canonical coverage **83 %**

> Module docstring: *"SimulationLoop — Closed-loop MuJoCo simulation with two-stage controller."*

**The closed loop.** DS/SS state machine, orchestration of planners and
solvers, weld activation, AOCS, logging. The largest file in the repository and
the one carrying the most architectural history.

---

## Public API

| symbol | signature | canonical? | code |
|---|---|---|---|
| **`SimulationLoop`** |  |  | [L92](../../../crawlbot/simulation/sim_loop.py#L92) |
| `.setup` | `(n_steps=3, start_a=2, start_b=2, sequence_path=None)` | **yes** | [L228](../../../crawlbot/simulation/sim_loop.py#L228) |
| `._settle_setup` | `(start_a, start_b)` | **yes** | [L537](../../../crawlbot/simulation/sim_loop.py#L537) |
| `._run_ds_passivity_loop` | `(contact_config, max_steps, epsilon_v, plateau_window, p...)` | **yes** | [L624](../../../crawlbot/simulation/sim_loop.py#L624) |
| `._interstep_aocs_command` | `(rs, cc_ds, lambda_qp_sol, omega_s_prev)` | **yes** | [L867](../../../crawlbot/simulation/sim_loop.py#L867) |
| `._build_qp` | `(ae, ap, aw, kpc, kdc, kpt, kdt, kpe, kde, kpe_ang=5.0, ...)` | **yes** | [L940](../../../crawlbot/simulation/sim_loop.py#L940) |
| `._build_weld_map` | `()` | **yes** | [L997](../../../crawlbot/simulation/sim_loop.py#L997) |
| `._deactivate_all_welds` | `()` | **yes** | [L1008](../../../crawlbot/simulation/sim_loop.py#L1008) |
| `._activate_weld` | `(arm, anchor_idx)` | **yes** | [L1012](../../../crawlbot/simulation/sim_loop.py#L1012) |
| `._deactivate_weld` | `(arm, anchor_idx)` | **yes** | [L1017](../../../crawlbot/simulation/sim_loop.py#L1017) |
| `._cache_site_ids` | `()` | **yes** | [L1022](../../../crawlbot/simulation/sim_loop.py#L1022) |
| `._gripper_distance` | `(arm, anchor_idx)` | **yes** | [L1043](../../../crawlbot/simulation/sim_loop.py#L1043) |
| `._gripper_speed` | `(arm)` | not exercised | [L1051](../../../crawlbot/simulation/sim_loop.py#L1051) |
| `._gripper_ori_err_deg` | `(arm, anchor_idx)` | **yes** | [L1066](../../../crawlbot/simulation/sim_loop.py#L1066) |
| `._weld_relative_twist` | `(arm, anchor_idx)` | **yes** | [L1081](../../../crawlbot/simulation/sim_loop.py#L1081) |
| `._dock_gate` | `(swing_arm, target_idx, log, t, step_idx)` | **yes** | [L1109](../../../crawlbot/simulation/sim_loop.py#L1109) |
| `._setup_torso_for_step` | `(t_ss_start, swing_arm, stance_a, stance_b, target_arm, ...)` | **yes** | [L1146](../../../crawlbot/simulation/sim_loop.py#L1146) |
| `._run_preplanner` | `(t_plan_start, stance_arm, stance_a, stance_b, r_com_0, ...)` | **yes** | [L1365](../../../crawlbot/simulation/sim_loop.py#L1365) |
| `._capture_snapshot` | `(log, t, label)` | **yes** | [L1474](../../../crawlbot/simulation/sim_loop.py#L1474) |
| `.run` | `(verbose=True)` | **yes** | [L1482](../../../crawlbot/simulation/sim_loop.py#L1482) |
| `._swing_query_time` | `(t_raw, phase, ss_end)` | **yes** | [L2091](../../../crawlbot/simulation/sim_loop.py#L2091) |
| `._step` | `(t, phase, step_idx, swing_arm, stance_arm, cc_ss, targe...)` | **yes** | [L2109](../../../crawlbot/simulation/sim_loop.py#L2109) |
| `._get_ee_data` | `(rs, arm)` | **yes** | [L3003](../../../crawlbot/simulation/sim_loop.py#L3003) |
| `._print_summary` | `(log)` | **yes** | [L3012](../../../crawlbot/simulation/sim_loop.py#L3012) |
| `.plot` | `(log, save_path=None, cfg=None)` | not exercised | [L3051](../../../crawlbot/simulation/sim_loop.py#L3051) |

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
- **`qp_time_ms` times the WBC block, not a QP solve.** The timer opens before
  the `for qs in range(self.n_qp_per_nmpc)` loop and closes after it, so it
  contains ten QP solves *plus* ten Pinocchio updates, ten AOCS evaluations,
  ten `mj_step` calls and the tick logging. The QP is ~71 % of it (median,
  canonical). `qp_solve_ms_*` are the QP.
- **`qp_ok` is not a measurement of the run.** Where `_step` computes it, it
  comes from a `try/except` around a backend that does not raise on failure
  (`solvers/hierarchical_qp.md` §4); on the 1368 inter-step ticks the recorder
  hardcodes it True. Use `qp_status_worst` / `qp_n_failed`.

Both loops now feed a `QPStatAccumulator` (`tick_logging.py`): `_step` adds one
entry per WBC sub-step, `_run_ds_passivity_loop` one per tick, and the aggregate
crosses into the log through `TickState`. Telemetry only — the accumulator is
never read back, and the canonical replay stays byte-identical on all 66 frozen
columns with it in place.

`run()` also persists `_preplanner_stats` into `log.preplanner_stats` on the way
out. The six coarse-pre-planner IPOPT solves were previously collected, printed,
and dropped on the floor; they gate every step and appeared in no artifact.

Unexercised: `_gripper_speed`, `_planned_arm_config`, `plot`.

## Code map

| unit | source |
|---|---|
| `class SimulationLoop` | [L92-3052](../../../crawlbot/simulation/sim_loop.py#L92-L3052) |
| `SimulationLoop.setup` | [L228-535](../../../crawlbot/simulation/sim_loop.py#L228-L535) |
| `SimulationLoop._settle_setup` | [L537-622](../../../crawlbot/simulation/sim_loop.py#L537-L622) |
| `SimulationLoop._run_ds_passivity_loop` | [L624-865](../../../crawlbot/simulation/sim_loop.py#L624-L865) |
| `SimulationLoop._interstep_aocs_command` | [L867-937](../../../crawlbot/simulation/sim_loop.py#L867-L937) |
| `SimulationLoop._build_qp` | [L940-993](../../../crawlbot/simulation/sim_loop.py#L940-L993) |
| `SimulationLoop._build_weld_map` | [L997-1006](../../../crawlbot/simulation/sim_loop.py#L997-L1006) |
| `SimulationLoop._deactivate_all_welds` | [L1008-1010](../../../crawlbot/simulation/sim_loop.py#L1008-L1010) |
| `SimulationLoop._activate_weld` | [L1012-1015](../../../crawlbot/simulation/sim_loop.py#L1012-L1015) |
| `SimulationLoop._deactivate_weld` | [L1017-1020](../../../crawlbot/simulation/sim_loop.py#L1017-L1020) |
| `SimulationLoop._cache_site_ids` | [L1022-1041](../../../crawlbot/simulation/sim_loop.py#L1022-L1041) |
| `SimulationLoop._gripper_distance` | [L1043-1049](../../../crawlbot/simulation/sim_loop.py#L1043-L1049) |
| `SimulationLoop._gripper_speed` | [L1051-1064](../../../crawlbot/simulation/sim_loop.py#L1051-L1064) |
| `SimulationLoop._gripper_ori_err_deg` | [L1066-1079](../../../crawlbot/simulation/sim_loop.py#L1066-L1079) |
| `SimulationLoop._weld_relative_twist` | [L1081-1107](../../../crawlbot/simulation/sim_loop.py#L1081-L1107) |
| `SimulationLoop._dock_gate` | [L1109-1141](../../../crawlbot/simulation/sim_loop.py#L1109-L1141) |
| `SimulationLoop._setup_torso_for_step` | [L1146-1363](../../../crawlbot/simulation/sim_loop.py#L1146-L1363) |
| `SimulationLoop._run_preplanner` | [L1365-1470](../../../crawlbot/simulation/sim_loop.py#L1365-L1470) |
| `SimulationLoop._capture_snapshot` | [L1474-1480](../../../crawlbot/simulation/sim_loop.py#L1474-L1480) |
| `SimulationLoop.run` | [L1482-2087](../../../crawlbot/simulation/sim_loop.py#L1482-L2087) |
| `SimulationLoop._swing_query_time` | [L2091-2107](../../../crawlbot/simulation/sim_loop.py#L2091-L2107) |
| `SimulationLoop._step` | [L2109-3000](../../../crawlbot/simulation/sim_loop.py#L2109-L3000) |
| `SimulationLoop._get_ee_data` | [L3003-3008](../../../crawlbot/simulation/sim_loop.py#L3003-L3008) |
| `SimulationLoop._print_summary` | [L3012-3043](../../../crawlbot/simulation/sim_loop.py#L3012-L3043) |
| `SimulationLoop.plot` | [L3051-3052](../../../crawlbot/simulation/sim_loop.py#L3051-L3052) |

---

## See also

- package overview: [`simulation.md`](simulation.md)
