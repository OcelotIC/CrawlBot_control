# `crawlbot.simulation.sim_loop`

**File**: [`crawlbot/simulation/sim_loop.py`](../../../crawlbot/simulation/sim_loop.py) — **3010 lines** — canonical coverage **81 %**

> Module docstring: *"SimulationLoop — Closed-loop MuJoCo simulation with two-stage controller."*

**The closed loop.** DS/SS state machine, orchestration of planners and
solvers, weld activation, AOCS, logging. The largest file in the repository and
the one carrying the most architectural history.

---

## Public API

| symbol | signature | canonical? | code |
|---|---|---|---|
| **`SimulationLoop`** |  |  | [L66](../../../crawlbot/simulation/sim_loop.py#L66) |
| `.setup` | `(n_steps=3, start_a=2, start_b=2, sequence_path=None)` | **yes** | [L198](../../../crawlbot/simulation/sim_loop.py#L198) |
| `._settle_setup` | `(start_a, start_b)` | **yes** | [L507](../../../crawlbot/simulation/sim_loop.py#L507) |
| `._run_ds_passivity_loop` | `(contact_config, max_steps, epsilon_v, plateau_window, p...)` | **yes** | [L594](../../../crawlbot/simulation/sim_loop.py#L594) |
| `._interstep_aocs_command` | `(rs, cc_ds, lambda_qp_sol, omega_s_prev)` | **yes** | [L831](../../../crawlbot/simulation/sim_loop.py#L831) |
| `._build_qp` | `(ae, ap, aw, kpc, kdc, kpt, kdt, kpe, kde, kpe_ang=5.0, ...)` | **yes** | [L904](../../../crawlbot/simulation/sim_loop.py#L904) |
| `._build_weld_map` | `()` | **yes** | [L961](../../../crawlbot/simulation/sim_loop.py#L961) |
| `._deactivate_all_welds` | `()` | **yes** | [L972](../../../crawlbot/simulation/sim_loop.py#L972) |
| `._activate_weld` | `(arm, anchor_idx)` | **yes** | [L976](../../../crawlbot/simulation/sim_loop.py#L976) |
| `._deactivate_weld` | `(arm, anchor_idx)` | **yes** | [L981](../../../crawlbot/simulation/sim_loop.py#L981) |
| `._cache_site_ids` | `()` | **yes** | [L986](../../../crawlbot/simulation/sim_loop.py#L986) |
| `._gripper_distance` | `(arm, anchor_idx)` | **yes** | [L1007](../../../crawlbot/simulation/sim_loop.py#L1007) |
| `._gripper_speed` | `(arm)` | not exercised | [L1015](../../../crawlbot/simulation/sim_loop.py#L1015) |
| `._gripper_ori_err_deg` | `(arm, anchor_idx)` | **yes** | [L1030](../../../crawlbot/simulation/sim_loop.py#L1030) |
| `._weld_relative_twist` | `(arm, anchor_idx)` | **yes** | [L1045](../../../crawlbot/simulation/sim_loop.py#L1045) |
| `._dock_gate` | `(swing_arm, target_idx, log, t, step_idx)` | **yes** | [L1073](../../../crawlbot/simulation/sim_loop.py#L1073) |
| `._planned_arm_config` | `(t, rs)` | not exercised | [L1109](../../../crawlbot/simulation/sim_loop.py#L1109) |
| `._setup_torso_for_step` | `(t_ss_start, swing_arm, stance_a, stance_b, target_arm, ...)` | **yes** | [L1147](../../../crawlbot/simulation/sim_loop.py#L1147) |
| `._run_preplanner` | `(t_plan_start, stance_arm, stance_a, stance_b, r_com_0, ...)` | **yes** | [L1369](../../../crawlbot/simulation/sim_loop.py#L1369) |
| `._capture_snapshot` | `(log, t, label)` | **yes** | [L1478](../../../crawlbot/simulation/sim_loop.py#L1478) |
| `.run` | `(verbose=True)` | **yes** | [L1486](../../../crawlbot/simulation/sim_loop.py#L1486) |
| `._swing_query_time` | `(t_raw, phase, ss_end)` | **yes** | [L2089](../../../crawlbot/simulation/sim_loop.py#L2089) |
| `._step` | `(t, phase, step_idx, swing_arm, stance_arm, cc_ss, targe...)` | **yes** | [L2107](../../../crawlbot/simulation/sim_loop.py#L2107) |
| `._get_ee_data` | `(rs, arm)` | **yes** | [L2960](../../../crawlbot/simulation/sim_loop.py#L2960) |
| `._print_summary` | `(log)` | **yes** | [L2969](../../../crawlbot/simulation/sim_loop.py#L2969) |
| `.plot` | `(log, save_path=None, cfg=None)` | not exercised | [L3008](../../../crawlbot/simulation/sim_loop.py#L3008) |

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
| `class SimulationLoop` | [L66-3009](../../../crawlbot/simulation/sim_loop.py#L66-L3009) |
| `SimulationLoop.setup` | [L198-505](../../../crawlbot/simulation/sim_loop.py#L198-L505) |
| `SimulationLoop._settle_setup` | [L507-592](../../../crawlbot/simulation/sim_loop.py#L507-L592) |
| `SimulationLoop._run_ds_passivity_loop` | [L594-829](../../../crawlbot/simulation/sim_loop.py#L594-L829) |
| `SimulationLoop._interstep_aocs_command` | [L831-901](../../../crawlbot/simulation/sim_loop.py#L831-L901) |
| `SimulationLoop._build_qp` | [L904-957](../../../crawlbot/simulation/sim_loop.py#L904-L957) |
| `SimulationLoop._build_weld_map` | [L961-970](../../../crawlbot/simulation/sim_loop.py#L961-L970) |
| `SimulationLoop._deactivate_all_welds` | [L972-974](../../../crawlbot/simulation/sim_loop.py#L972-L974) |
| `SimulationLoop._activate_weld` | [L976-979](../../../crawlbot/simulation/sim_loop.py#L976-L979) |
| `SimulationLoop._deactivate_weld` | [L981-984](../../../crawlbot/simulation/sim_loop.py#L981-L984) |
| `SimulationLoop._cache_site_ids` | [L986-1005](../../../crawlbot/simulation/sim_loop.py#L986-L1005) |
| `SimulationLoop._gripper_distance` | [L1007-1013](../../../crawlbot/simulation/sim_loop.py#L1007-L1013) |
| `SimulationLoop._gripper_speed` | [L1015-1028](../../../crawlbot/simulation/sim_loop.py#L1015-L1028) |
| `SimulationLoop._gripper_ori_err_deg` | [L1030-1043](../../../crawlbot/simulation/sim_loop.py#L1030-L1043) |
| `SimulationLoop._weld_relative_twist` | [L1045-1071](../../../crawlbot/simulation/sim_loop.py#L1045-L1071) |
| `SimulationLoop._dock_gate` | [L1073-1105](../../../crawlbot/simulation/sim_loop.py#L1073-L1105) |
| `SimulationLoop._planned_arm_config` | [L1109-1145](../../../crawlbot/simulation/sim_loop.py#L1109-L1145) |
| `SimulationLoop._setup_torso_for_step` | [L1147-1367](../../../crawlbot/simulation/sim_loop.py#L1147-L1367) |
| `SimulationLoop._run_preplanner` | [L1369-1474](../../../crawlbot/simulation/sim_loop.py#L1369-L1474) |
| `SimulationLoop._capture_snapshot` | [L1478-1484](../../../crawlbot/simulation/sim_loop.py#L1478-L1484) |
| `SimulationLoop.run` | [L1486-2085](../../../crawlbot/simulation/sim_loop.py#L1486-L2085) |
| `SimulationLoop._swing_query_time` | [L2089-2105](../../../crawlbot/simulation/sim_loop.py#L2089-L2105) |
| `SimulationLoop._step` | [L2107-2957](../../../crawlbot/simulation/sim_loop.py#L2107-L2957) |
| `SimulationLoop._get_ee_data` | [L2960-2965](../../../crawlbot/simulation/sim_loop.py#L2960-L2965) |
| `SimulationLoop._print_summary` | [L2969-3000](../../../crawlbot/simulation/sim_loop.py#L2969-L3000) |
| `SimulationLoop.plot` | [L3008-3009](../../../crawlbot/simulation/sim_loop.py#L3008-L3009) |

---

## See also

- package overview: [`simulation.md`](simulation.md)
