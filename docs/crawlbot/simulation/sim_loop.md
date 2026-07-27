# `crawlbot.simulation.sim_loop`

**File**: [`crawlbot/simulation/sim_loop.py`](../../../crawlbot/simulation/sim_loop.py) — **3144 lines** — canonical coverage **83 %**

> Module docstring: *"SimulationLoop — Closed-loop MuJoCo simulation with two-stage controller."*

**The closed loop.** DS/SS state machine, orchestration of planners and
solvers, weld activation, AOCS, logging. The largest file in the repository and
the one carrying the most architectural history.

---

## Public API

| symbol | signature | canonical? | code |
|---|---|---|---|
| **`SimulationLoop`** |  |  | [L92](../../../crawlbot/simulation/sim_loop.py#L92) |
| `.setup` | `(n_steps=3, start_a=2, start_b=2, sequence_path=None)` | **yes** | [L233](../../../crawlbot/simulation/sim_loop.py#L233) |
| `._settle_setup` | `(start_a, start_b)` | **yes** | [L542](../../../crawlbot/simulation/sim_loop.py#L542) |
| `._run_ds_passivity_loop` | `(contact_config, max_steps, epsilon_v, plateau_window, p...)` | **yes** | [L629](../../../crawlbot/simulation/sim_loop.py#L629) |
| `._interstep_aocs_command` | `(rs, cc_ds, lambda_qp_sol, omega_s_prev, decomposition=None)` | **yes** | [L879](../../../crawlbot/simulation/sim_loop.py#L879) |
| `._build_qp` | `(ae, ap, aw, kpc, kdc, kpt, kdt, kpe, kde, kpe_ang=5.0, ...)` | **yes** | [L954](../../../crawlbot/simulation/sim_loop.py#L954) |
| `._build_weld_map` | `()` | **yes** | [L1011](../../../crawlbot/simulation/sim_loop.py#L1011) |
| `._deactivate_all_welds` | `()` | **yes** | [L1022](../../../crawlbot/simulation/sim_loop.py#L1022) |
| `._activate_weld` | `(arm, anchor_idx)` | **yes** | [L1026](../../../crawlbot/simulation/sim_loop.py#L1026) |
| `._deactivate_weld` | `(arm, anchor_idx)` | **yes** | [L1031](../../../crawlbot/simulation/sim_loop.py#L1031) |
| `._cache_site_ids` | `()` | **yes** | [L1036](../../../crawlbot/simulation/sim_loop.py#L1036) |
| `._gripper_distance` | `(arm, anchor_idx)` | **yes** | [L1057](../../../crawlbot/simulation/sim_loop.py#L1057) |
| `._gripper_speed` | `(arm)` | not exercised | [L1065](../../../crawlbot/simulation/sim_loop.py#L1065) |
| `._gripper_ori_err_deg` | `(arm, anchor_idx)` | **yes** | [L1080](../../../crawlbot/simulation/sim_loop.py#L1080) |
| `._weld_relative_twist` | `(arm, anchor_idx)` | **yes** | [L1095](../../../crawlbot/simulation/sim_loop.py#L1095) |
| `._dock_gate` | `(swing_arm, target_idx, log, t, step_idx)` | **yes** | [L1123](../../../crawlbot/simulation/sim_loop.py#L1123) |
| `._dock_thresholds` | `()` | **yes** | [L1164](../../../crawlbot/simulation/sim_loop.py#L1164) |
| `._twist_components` | `(twist_vec)` | **yes** | [L1183](../../../crawlbot/simulation/sim_loop.py#L1183) |
| `._setup_torso_for_step` | `(t_ss_start, swing_arm, stance_a, stance_b, target_arm, ...)` | **yes** | [L1213](../../../crawlbot/simulation/sim_loop.py#L1213) |
| `._run_preplanner` | `(t_plan_start, stance_arm, stance_a, stance_b, r_com_0, ...)` | **yes** | [L1432](../../../crawlbot/simulation/sim_loop.py#L1432) |
| `._capture_snapshot` | `(log, t, label)` | **yes** | [L1541](../../../crawlbot/simulation/sim_loop.py#L1541) |
| `.run` | `(verbose=True)` | **yes** | [L1549](../../../crawlbot/simulation/sim_loop.py#L1549) |
| `._swing_query_time` | `(t_raw, phase, ss_end)` | **yes** | [L2174](../../../crawlbot/simulation/sim_loop.py#L2174) |
| `._step` | `(t, phase, step_idx, swing_arm, stance_arm, cc_ss, targe...)` | **yes** | [L2192](../../../crawlbot/simulation/sim_loop.py#L2192) |
| `._get_ee_data` | `(rs, arm)` | **yes** | [L3094](../../../crawlbot/simulation/sim_loop.py#L3094) |
| `._print_summary` | `(log)` | **yes** | [L3103](../../../crawlbot/simulation/sim_loop.py#L3103) |
| `.plot` | `(log, save_path=None, cfg=None)` | not exercised | [L3142](../../../crawlbot/simulation/sim_loop.py#L3142) |

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

**The dock gate now records what it saw, not just its verdict (C2.1).**
`_weld_relative_twist` returns a 6-vector; `_dock_gate` used to collapse it to a
norm and discard the components, so a refused capture could say "twist too high"
without saying whether the excess was linear or angular. `_twist_components`
rotates it into the **structure frame** (the raw `mj_jacSite` difference is in
MuJoCo's global frame) and splits it, and `_dock_thresholds` stamps the criteria
in force onto every `dock_events` row. The norm — the quantity the gate actually
thresholds — is **rotation-invariant**, which is precisely why none of this
touches control. This matters because C1.6 found that `dock_twist_max`, a bound
documented in-source as untuned, is what sets the worst reported dock and ~70 %
of the managed-vs-unmanaged traversal difference; a dock should never again be
readable without the gate that produced it.

**The AOCS decomposition (C2.3)** rides on the same pattern: `aocs_decomp` is
filled by the last WBC sub-step of each tick — the same sub-step whose sum is
logged as `tau_wheels`, so parts and total describe one instant. It is bound
*before* the `has_rwa` / `aocs_active_in_interstep` branches in the inter-step
loop, so the not-measured paths write the sentinel instead of raising.

Unexercised: `_gripper_speed`, `_planned_arm_config`, `plot`.

## Code map

| unit | source |
|---|---|
| `class SimulationLoop` | [L92-3143](../../../crawlbot/simulation/sim_loop.py#L92-L3143) |
| `SimulationLoop.setup` | [L233-540](../../../crawlbot/simulation/sim_loop.py#L233-L540) |
| `SimulationLoop._settle_setup` | [L542-627](../../../crawlbot/simulation/sim_loop.py#L542-L627) |
| `SimulationLoop._run_ds_passivity_loop` | [L629-877](../../../crawlbot/simulation/sim_loop.py#L629-L877) |
| `SimulationLoop._interstep_aocs_command` | [L879-951](../../../crawlbot/simulation/sim_loop.py#L879-L951) |
| `SimulationLoop._build_qp` | [L954-1007](../../../crawlbot/simulation/sim_loop.py#L954-L1007) |
| `SimulationLoop._build_weld_map` | [L1011-1020](../../../crawlbot/simulation/sim_loop.py#L1011-L1020) |
| `SimulationLoop._deactivate_all_welds` | [L1022-1024](../../../crawlbot/simulation/sim_loop.py#L1022-L1024) |
| `SimulationLoop._activate_weld` | [L1026-1029](../../../crawlbot/simulation/sim_loop.py#L1026-L1029) |
| `SimulationLoop._deactivate_weld` | [L1031-1034](../../../crawlbot/simulation/sim_loop.py#L1031-L1034) |
| `SimulationLoop._cache_site_ids` | [L1036-1055](../../../crawlbot/simulation/sim_loop.py#L1036-L1055) |
| `SimulationLoop._gripper_distance` | [L1057-1063](../../../crawlbot/simulation/sim_loop.py#L1057-L1063) |
| `SimulationLoop._gripper_speed` | [L1065-1078](../../../crawlbot/simulation/sim_loop.py#L1065-L1078) |
| `SimulationLoop._gripper_ori_err_deg` | [L1080-1093](../../../crawlbot/simulation/sim_loop.py#L1080-L1093) |
| `SimulationLoop._weld_relative_twist` | [L1095-1121](../../../crawlbot/simulation/sim_loop.py#L1095-L1121) |
| `SimulationLoop._dock_gate` | [L1123-1162](../../../crawlbot/simulation/sim_loop.py#L1123-L1162) |
| `SimulationLoop._dock_thresholds` | [L1164-1181](../../../crawlbot/simulation/sim_loop.py#L1164-L1181) |
| `SimulationLoop._twist_components` | [L1183-1208](../../../crawlbot/simulation/sim_loop.py#L1183-L1208) |
| `SimulationLoop._setup_torso_for_step` | [L1213-1430](../../../crawlbot/simulation/sim_loop.py#L1213-L1430) |
| `SimulationLoop._run_preplanner` | [L1432-1537](../../../crawlbot/simulation/sim_loop.py#L1432-L1537) |
| `SimulationLoop._capture_snapshot` | [L1541-1547](../../../crawlbot/simulation/sim_loop.py#L1541-L1547) |
| `SimulationLoop.run` | [L1549-2170](../../../crawlbot/simulation/sim_loop.py#L1549-L2170) |
| `SimulationLoop._swing_query_time` | [L2174-2190](../../../crawlbot/simulation/sim_loop.py#L2174-L2190) |
| `SimulationLoop._step` | [L2192-3091](../../../crawlbot/simulation/sim_loop.py#L2192-L3091) |
| `SimulationLoop._get_ee_data` | [L3094-3099](../../../crawlbot/simulation/sim_loop.py#L3094-L3099) |
| `SimulationLoop._print_summary` | [L3103-3134](../../../crawlbot/simulation/sim_loop.py#L3103-L3134) |
| `SimulationLoop.plot` | [L3142-3143](../../../crawlbot/simulation/sim_loop.py#L3142-L3143) |

---

## See also

- package overview: [`simulation.md`](simulation.md)
