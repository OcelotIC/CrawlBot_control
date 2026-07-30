# `crawlbot.simulation.sim_loop`

**File**: [`crawlbot/simulation/sim_loop.py`](../../../crawlbot/simulation/sim_loop.py) — **3101 lines** — canonical coverage **83 %**

> Module docstring: *"SimulationLoop — Closed-loop MuJoCo simulation with two-stage controller."*

**The closed loop.** DS/SS state machine, orchestration of planners and
solvers, weld activation, AOCS, logging. The largest file in the repository and
the one carrying the most architectural history.

---

## Public API

| symbol | signature | canonical? | code |
|---|---|---|---|
| **`SimulationLoop`** |  |  | [L92](../../../crawlbot/simulation/sim_loop.py#L92) |
| `.setup` | `(n_steps=3, start_a=2, start_b=2, sequence_path=None)` | **yes** | [L244](../../../crawlbot/simulation/sim_loop.py#L244) |
| `._settle_setup` | `(start_a, start_b)` | **yes** | [L559](../../../crawlbot/simulation/sim_loop.py#L559) |
| `._run_ds_passivity_loop` | `(contact_config, max_steps, epsilon_v, plateau_window, p...)` | **yes** | [L646](../../../crawlbot/simulation/sim_loop.py#L646) |
| `._interstep_aocs_command` | `(rs, cc_ds, lambda_qp_sol, omega_s_prev)` | **yes** | [L883](../../../crawlbot/simulation/sim_loop.py#L883) |
| `._build_qp` | `(ae, ap, aw, kpc, kdc, kpt, kdt, kpe, kde, kpe_ang=5.0, ...)` | **yes** | [L956](../../../crawlbot/simulation/sim_loop.py#L956) |
| `._build_weld_map` | `()` | **yes** | [L1013](../../../crawlbot/simulation/sim_loop.py#L1013) |
| `._deactivate_all_welds` | `()` | **yes** | [L1024](../../../crawlbot/simulation/sim_loop.py#L1024) |
| `._activate_weld` | `(arm, anchor_idx)` | **yes** | [L1028](../../../crawlbot/simulation/sim_loop.py#L1028) |
| `._deactivate_weld` | `(arm, anchor_idx)` | **yes** | [L1033](../../../crawlbot/simulation/sim_loop.py#L1033) |
| `._cache_site_ids` | `()` | **yes** | [L1038](../../../crawlbot/simulation/sim_loop.py#L1038) |
| `._gripper_distance` | `(arm, anchor_idx)` | **yes** | [L1059](../../../crawlbot/simulation/sim_loop.py#L1059) |
| `._gripper_speed` | `(arm)` | not exercised | [L1067](../../../crawlbot/simulation/sim_loop.py#L1067) |
| `._gripper_ori_err_deg` | `(arm, anchor_idx)` | **yes** | [L1082](../../../crawlbot/simulation/sim_loop.py#L1082) |
| `._weld_relative_twist` | `(arm, anchor_idx)` | **yes** | [L1097](../../../crawlbot/simulation/sim_loop.py#L1097) |
| `._dock_gate` | `(swing_arm, target_idx, log, t, step_idx)` | **yes** | [L1125](../../../crawlbot/simulation/sim_loop.py#L1125) |
| `._setup_torso_for_step` | `(t_ss_start, swing_arm, stance_a, stance_b, target_arm, ...)` | **yes** | [L1162](../../../crawlbot/simulation/sim_loop.py#L1162) |
| `._run_preplanner` | `(t_plan_start, stance_arm, stance_a, stance_b, r_com_0, ...)` | **yes** | [L1381](../../../crawlbot/simulation/sim_loop.py#L1381) |
| `._capture_snapshot` | `(log, t, label)` | **yes** | [L1490](../../../crawlbot/simulation/sim_loop.py#L1490) |
| `.run` | `(verbose=True)` | **yes** | [L1498](../../../crawlbot/simulation/sim_loop.py#L1498) |
| `._swing_query_time` | `(t_raw, phase, ss_end)` | **yes** | [L2101](../../../crawlbot/simulation/sim_loop.py#L2101) |
| `._com_ref_at` | `(t_query, settle_mode)` | **yes** | [L2119](../../../crawlbot/simulation/sim_loop.py#L2119) |
| `._step` | `(t, phase, step_idx, swing_arm, stance_arm, cc_ss, targe...)` | **yes** | [L2150](../../../crawlbot/simulation/sim_loop.py#L2150) |
| `._get_ee_data` | `(rs, arm)` | **yes** | [L3051](../../../crawlbot/simulation/sim_loop.py#L3051) |
| `._print_summary` | `(log)` | **yes** | [L3060](../../../crawlbot/simulation/sim_loop.py#L3060) |
| `.plot` | `(log, save_path=None, cfg=None)` | not exercised | [L3099](../../../crawlbot/simulation/sim_loop.py#L3099) |

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

### 2.1 Three timescales

| symbol | value | meaning |
|---|---|---|
| `dt_qp` | 0.01 s | QP / MuJoCo tick — the inner loop |
| **`nmpc_period`** | 0.1 s | **control period** — how often the NMPC re-solves and the plan is applied |
| **`nmpc_pred_dt`** | 0.1 s | **prediction step** — the RK4 step inside the NLP, i.e. horizon knot spacing |

The last two are different quantities: a controller may legitimately re-solve at
10 Hz while predicting on a finer grid. They have simply always been equal here.

⚠ **They used to be called `dt_nmpc` and `nmpc_dt`** — near-anagrams for
different things. That is how three code paths came to conflate them
(`NMPC_AUDIT` F3): code using the wrong one read perfectly *and* computed the
right answer, because the two values were equal. The rename is the actual fix
for the class of bug; the F3 arithmetic fix only repaired the instances.
`tests/…::TestTimescaleNaming` fails if either old name returns.

Two derived counts keep them apart:

```python
n_qp_per_nmpc = round(nmpc_period / dt_qp)   # QP ticks per CONTROL period
_qp_per_knot  = round(nmpc_pred_dt / dt_qp)   # QP ticks per PREDICTION knot
```

`_step` interpolates the NMPC plan with `u = qs / _qp_per_knot`, taking
`floor(u)` and `floor(u)+1` as the bracketing knots — so a sub-step at 0.06 s
into the control period reads the plan at 0.06 s of plan time regardless of the
knot spacing. The previous form walked knots 0 → 1 across the whole control
period, which dilated the reference by `nmpc_period / nmpc_pred_dt` whenever the two
differed.

`_qp_per_knot` is an **integer** on purpose: `(qs·dt_qp)/nmpc_pred_dt` is
algebraically the same and 1 ULP different, which is enough to break the
canonical's bit-identity. `__init__` rejects an `nmpc_pred_dt` that is not an integer
multiple of `dt_qp` rather than letting the indexing drift.

The same ratio drives `CentroidalNMPCConfig.control_period`, which tells the
warm-start shift and the infeasibility fallback how many knots one control
period spans.

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

Unexercised: `_gripper_speed`, `_planned_arm_config`, `plot`.

## Code map

| unit | source |
|---|---|
| `class SimulationLoop` | [L92-3100](../../../crawlbot/simulation/sim_loop.py#L92-L3100) |
| `SimulationLoop.setup` | [L244-557](../../../crawlbot/simulation/sim_loop.py#L244-L557) |
| `SimulationLoop._settle_setup` | [L559-644](../../../crawlbot/simulation/sim_loop.py#L559-L644) |
| `SimulationLoop._run_ds_passivity_loop` | [L646-881](../../../crawlbot/simulation/sim_loop.py#L646-L881) |
| `SimulationLoop._interstep_aocs_command` | [L883-953](../../../crawlbot/simulation/sim_loop.py#L883-L953) |
| `SimulationLoop._build_qp` | [L956-1009](../../../crawlbot/simulation/sim_loop.py#L956-L1009) |
| `SimulationLoop._build_weld_map` | [L1013-1022](../../../crawlbot/simulation/sim_loop.py#L1013-L1022) |
| `SimulationLoop._deactivate_all_welds` | [L1024-1026](../../../crawlbot/simulation/sim_loop.py#L1024-L1026) |
| `SimulationLoop._activate_weld` | [L1028-1031](../../../crawlbot/simulation/sim_loop.py#L1028-L1031) |
| `SimulationLoop._deactivate_weld` | [L1033-1036](../../../crawlbot/simulation/sim_loop.py#L1033-L1036) |
| `SimulationLoop._cache_site_ids` | [L1038-1057](../../../crawlbot/simulation/sim_loop.py#L1038-L1057) |
| `SimulationLoop._gripper_distance` | [L1059-1065](../../../crawlbot/simulation/sim_loop.py#L1059-L1065) |
| `SimulationLoop._gripper_speed` | [L1067-1080](../../../crawlbot/simulation/sim_loop.py#L1067-L1080) |
| `SimulationLoop._gripper_ori_err_deg` | [L1082-1095](../../../crawlbot/simulation/sim_loop.py#L1082-L1095) |
| `SimulationLoop._weld_relative_twist` | [L1097-1123](../../../crawlbot/simulation/sim_loop.py#L1097-L1123) |
| `SimulationLoop._dock_gate` | [L1125-1157](../../../crawlbot/simulation/sim_loop.py#L1125-L1157) |
| `SimulationLoop._setup_torso_for_step` | [L1162-1379](../../../crawlbot/simulation/sim_loop.py#L1162-L1379) |
| `SimulationLoop._run_preplanner` | [L1381-1486](../../../crawlbot/simulation/sim_loop.py#L1381-L1486) |
| `SimulationLoop._capture_snapshot` | [L1490-1496](../../../crawlbot/simulation/sim_loop.py#L1490-L1496) |
| `SimulationLoop.run` | [L1498-2097](../../../crawlbot/simulation/sim_loop.py#L1498-L2097) |
| `SimulationLoop._swing_query_time` | [L2101-2117](../../../crawlbot/simulation/sim_loop.py#L2101-L2117) |
| `SimulationLoop._com_ref_at` | [L2119-2148](../../../crawlbot/simulation/sim_loop.py#L2119-L2148) |
| `SimulationLoop._step` | [L2150-3048](../../../crawlbot/simulation/sim_loop.py#L2150-L3048) |
| `SimulationLoop._get_ee_data` | [L3051-3056](../../../crawlbot/simulation/sim_loop.py#L3051-L3056) |
| `SimulationLoop._print_summary` | [L3060-3091](../../../crawlbot/simulation/sim_loop.py#L3060-L3091) |
| `SimulationLoop.plot` | [L3099-3100](../../../crawlbot/simulation/sim_loop.py#L3099-L3100) |

---

## See also

- package overview: [`simulation.md`](simulation.md)
