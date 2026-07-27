# `crawlbot.simulation.logging`

**File**: [`crawlbot/simulation/logging.py`](../../../crawlbot/simulation/logging.py) — **343 lines** — canonical coverage **93 %**

> Module docstring: *"Simulation data logger."*

`SimLog`: one array per quantity, one entry per tick, plus a capture of the
execution environment. This is the file every downstream analysis reads.

---

## Public API

| symbol | signature | canonical? | code |
|---|---|---|---|
| `capture_environment` | `()` | **yes** | [L21](../../../crawlbot/simulation/logging.py#L21) |
| **`SimLog`** *(dataclass)* |  |  | [L109](../../../crawlbot/simulation/logging.py#L109) |
|   `t` | `field(default_factory=list)` | _field_ | [L112](../../../crawlbot/simulation/logging.py#L112) |
|   `phase` | `field(default_factory=list)` | _field_ | [L113](../../../crawlbot/simulation/logging.py#L113) |
|   `step_idx` | `field(default_factory=list)` | _field_ | [L114](../../../crawlbot/simulation/logging.py#L114) |
|   `p_torso` | `field(default_factory=list)` | _field_ | [L117](../../../crawlbot/simulation/logging.py#L117) |
|   `p_torso_ref` | `field(default_factory=list)` | _field_ | [L118](../../../crawlbot/simulation/logging.py#L118) |
|   `e_torso_pos` | `field(default_factory=list)` | _field_ | [L119](../../../crawlbot/simulation/logging.py#L119) |
|   `e_torso_ori` | `field(default_factory=list)` | _field_ | [L120](../../../crawlbot/simulation/logging.py#L120) |
|   `q_torso` | `field(default_factory=list)` | _field_ | [L121](../../../crawlbot/simulation/logging.py#L121) |
|   `q_torso_ref` | `field(default_factory=list)` | _field_ | [L122](../../../crawlbot/simulation/logging.py#L122) |
|   `d_grip_swing` | `field(default_factory=list)` | _field_ | [L125](../../../crawlbot/simulation/logging.py#L125) |
|   `d_grip_stance` | `field(default_factory=list)` | _field_ | [L126](../../../crawlbot/simulation/logging.py#L126) |
|   `swing_arm` | `field(default_factory=list)` | _field_ | [L127](../../../crawlbot/simulation/logging.py#L127) |
|   `p_ee` | `field(default_factory=list)` | _field_ | [L128](../../../crawlbot/simulation/logging.py#L128) |
|   `p_ee_ref` | `field(default_factory=list)` | _field_ | [L129](../../../crawlbot/simulation/logging.py#L129) |
|   `q_ee` | `field(default_factory=list)` | _field_ | [L130](../../../crawlbot/simulation/logging.py#L130) |
|   `q_ee_ref` | `field(default_factory=list)` | _field_ | [L131](../../../crawlbot/simulation/logging.py#L131) |
|   `qvel_joints_a` | `field(default_factory=list)` | _field_ | [L134](../../../crawlbot/simulation/logging.py#L134) |
|   `qvel_joints_b` | `field(default_factory=list)` | _field_ | [L135](../../../crawlbot/simulation/logging.py#L135) |
|   `v_ee_a` | `field(default_factory=list)` | _field_ | [L136](../../../crawlbot/simulation/logging.py#L136) |
|   `v_ee_b` | `field(default_factory=list)` | _field_ | [L137](../../../crawlbot/simulation/logging.py#L137) |
|   `omega_ee_a` | `field(default_factory=list)` | _field_ | [L138](../../../crawlbot/simulation/logging.py#L138) |
|   `omega_ee_b` | `field(default_factory=list)` | _field_ | [L139](../../../crawlbot/simulation/logging.py#L139) |
|   `v_torso` | `field(default_factory=list)` | _field_ | [L140](../../../crawlbot/simulation/logging.py#L140) |
|   `omega_torso` | `field(default_factory=list)` | _field_ | [L141](../../../crawlbot/simulation/logging.py#L141) |
|   `r_com` | `field(default_factory=list)` | _field_ | [L144](../../../crawlbot/simulation/logging.py#L144) |
|   `r_com_ref` | `field(default_factory=list)` | _field_ | [L145](../../../crawlbot/simulation/logging.py#L145) |
|   `e_com` | `field(default_factory=list)` | _field_ | [L146](../../../crawlbot/simulation/logging.py#L146) |
|   `v_com` | `field(default_factory=list)` | _field_ | [L147](../../../crawlbot/simulation/logging.py#L147) |
|   `v_com_ref` | `field(default_factory=list)` | _field_ | [L148](../../../crawlbot/simulation/logging.py#L148) |
|   `L_com` | `field(default_factory=list)` | _field_ | [L151](../../../crawlbot/simulation/logging.py#L151) |
|   `L_com_norm` | `field(default_factory=list)` | _field_ | [L152](../../../crawlbot/simulation/logging.py#L152) |
|   `L_com_ref` | `field(default_factory=list)` | _field_ | [L153](../../../crawlbot/simulation/logging.py#L153) |
|   `L_dot` | `field(default_factory=list)` | _field_ | [L154](../../../crawlbot/simulation/logging.py#L154) |
|   `L_dot_norm` | `field(default_factory=list)` | _field_ | [L155](../../../crawlbot/simulation/logging.py#L155) |
|   `hw` | `field(default_factory=list)` | _field_ | [L156](../../../crawlbot/simulation/logging.py#L156) |
|   `hw_physical` | `field(default_factory=list)` | _field_ | [L159](../../../crawlbot/simulation/logging.py#L159) |
|   `tau_w` | `field(default_factory=list)` | _field_ | [L160](../../../crawlbot/simulation/logging.py#L160) |
|   `rw_speed` | `field(default_factory=list)` | _field_ | [L161](../../../crawlbot/simulation/logging.py#L161) |
|   `t_ss_hifreq` | `field(default_factory=list)` | _field_ | [L167](../../../crawlbot/simulation/logging.py#L167) |
|   `tau_w_ss_hifreq` | `field(default_factory=list)` | _field_ | [L168](../../../crawlbot/simulation/logging.py#L168) |
|   `hw_ss_hifreq` | `field(default_factory=list)` | _field_ | [L169](../../../crawlbot/simulation/logging.py#L169) |
|   `e_ee_pos` | `field(default_factory=list)` | _field_ | [L172](../../../crawlbot/simulation/logging.py#L172) |
|   `e_ee_ori` | `field(default_factory=list)` | _field_ | [L173](../../../crawlbot/simulation/logging.py#L173) |
|   `gmo_residual_norm` | `field(default_factory=list)` | _field_ | [L176](../../../crawlbot/simulation/logging.py#L176) |
|   `gmo_swing_residual` | `field(default_factory=list)` | _field_ | [L177](../../../crawlbot/simulation/logging.py#L177) |
|   `gmo_contact_state` | `field(default_factory=list)` | _field_ | [L178](../../../crawlbot/simulation/logging.py#L178) |
|   `H_rO` | `field(default_factory=list)` | _field_ | [L181](../../../crawlbot/simulation/logging.py#L181) |
|   `H_dot_est` | `field(default_factory=list)` | _field_ | [L182](../../../crawlbot/simulation/logging.py#L182) |
|   `omega_struct` | `field(default_factory=list)` | _field_ | [L183](../../../crawlbot/simulation/logging.py#L183) |
|   `qfrc_constraint_torque` | `field(default_factory=list)` | _field_ | [L184](../../../crawlbot/simulation/logging.py#L184) |
|   `tau` | `field(default_factory=list)` | _field_ | [L187](../../../crawlbot/simulation/logging.py#L187) |
|   `tau_max_joint` | `field(default_factory=list)` | _field_ | [L188](../../../crawlbot/simulation/logging.py#L188) |
|   `struct_pos` | `field(default_factory=list)` | _field_ | [L191](../../../crawlbot/simulation/logging.py#L191) |
|   `struct_quat` | `field(default_factory=list)` | _field_ | [L192](../../../crawlbot/simulation/logging.py#L192) |
|   `struct_euler_deg` | `field(default_factory=list)` | _field_ | [L193](../../../crawlbot/simulation/logging.py#L193) |
|   `omega_s` | `field(default_factory=list)` | _field_ | [L194](../../../crawlbot/simulation/logging.py#L194) |
|   `nmpc_ok` | `field(default_factory=list)` | _field_ | [L197](../../../crawlbot/simulation/logging.py#L197) |
|   `qp_ok` | `field(default_factory=list)` | _field_ | [L198](../../../crawlbot/simulation/logging.py#L198) |
|   `lambda_ref_norm` | `field(default_factory=list)` | _field_ | [L199](../../../crawlbot/simulation/logging.py#L199) |
|   `lambda_qp` | `field(default_factory=list)` | _field_ | [L202](../../../crawlbot/simulation/logging.py#L202) |
|   `lambda_qp_norm` | `field(default_factory=list)` | _field_ | [L203](../../../crawlbot/simulation/logging.py#L203) |
|   `nmpc_time_ms` | `field(default_factory=list)` | _field_ | [L204](../../../crawlbot/simulation/logging.py#L204) |
|   `qp_time_ms` | `field(default_factory=list)` | _field_ | [L205](../../../crawlbot/simulation/logging.py#L205) |
|   `nmpc_status` | `field(default_factory=list)` | _field_ | [L206](../../../crawlbot/simulation/logging.py#L206) |
|   `nmpc_cost` | `field(default_factory=list)` | _field_ | [L207](../../../crawlbot/simulation/logging.py#L207) |
|   `nmpc_status_str` | `field(default_factory=list)` | _field_ | [L208](../../../crawlbot/simulation/logging.py#L208) |
|   `nmpc_iterations` | `field(default_factory=list)` | _field_ | [L209](../../../crawlbot/simulation/logging.py#L209) |
|   `qp_solve_ms_sum` | `field(default_factory=list)` | _field_ | [L228](../../../crawlbot/simulation/logging.py#L228) |
|   `qp_solve_ms_max` | `field(default_factory=list)` | _field_ | [L229](../../../crawlbot/simulation/logging.py#L229) |
|   `qp_iter_sum` | `field(default_factory=list)` | _field_ | [L230](../../../crawlbot/simulation/logging.py#L230) |
|   `qp_n_solves` | `field(default_factory=list)` | _field_ | [L231](../../../crawlbot/simulation/logging.py#L231) |
|   `qp_n_failed` | `field(default_factory=list)` | _field_ | [L232](../../../crawlbot/simulation/logging.py#L232) |
|   `qp_status_worst` | `field(default_factory=list)` | _field_ | [L239](../../../crawlbot/simulation/logging.py#L239) |
|   `transport_term_mag` | `field(default_factory=list)` | _field_ | [L240](../../../crawlbot/simulation/logging.py#L240) |
|   `lambda_ref` | `field(default_factory=list)` | _field_ | [L245](../../../crawlbot/simulation/logging.py#L245) |
|   `lambda_qp` | `field(default_factory=list)` | _field_ | [L246](../../../crawlbot/simulation/logging.py#L246) |
|   `T_kinetic` | `field(default_factory=list)` | _field_ | [L249](../../../crawlbot/simulation/logging.py#L249) |
|   `settling_t` | `field(default_factory=list)` | _field_ | [L252](../../../crawlbot/simulation/logging.py#L252) |
|   `settling_T` | `field(default_factory=list)` | _field_ | [L253](../../../crawlbot/simulation/logging.py#L253) |
|   `settling_T_target` | `0.0` | _field_ | [L254](../../../crawlbot/simulation/logging.py#L254) |
|   `settling_stage1_steps` | `0` | _field_ | [L255](../../../crawlbot/simulation/logging.py#L255) |
|   `settling_stage2_steps` | `0` | _field_ | [L256](../../../crawlbot/simulation/logging.py#L256) |
|   `settling_exit_reason` | `''` | _field_ | [L257](../../../crawlbot/simulation/logging.py#L257) |
|   `inter_step_settles` | `field(default_factory=list)` | _field_ | [L263](../../../crawlbot/simulation/logging.py#L263) |
|   `dock_events` | `field(default_factory=list)` | _field_ | [L266](../../../crawlbot/simulation/logging.py#L266) |
|   `dock_gate_trace` | `field(default_factory=list)` | _field_ | [L274](../../../crawlbot/simulation/logging.py#L274) |
|   `ds_mobile_trace` | `field(default_factory=list)` | _field_ | [L281](../../../crawlbot/simulation/logging.py#L281) |
|   `dock_work_trace` | `field(default_factory=list)` | _field_ | [L287](../../../crawlbot/simulation/logging.py#L287) |
|   `aborted_steps` | `field(default_factory=list)` | _field_ | [L292](../../../crawlbot/simulation/logging.py#L292) |
|   `preplanner_T_steps` | `field(default_factory=list)` | _field_ | [L296](../../../crawlbot/simulation/logging.py#L296) |
|   `preplanner_stats` | `field(default_factory=list)` | _field_ | [L304](../../../crawlbot/simulation/logging.py#L304) |
|   `snapshots` | `field(default_factory=list)` | _field_ | [L307](../../../crawlbot/simulation/logging.py#L307) |
|   `environment` | `field(default_factory=dict)` | _field_ | [L313](../../../crawlbot/simulation/logging.py#L313) |
| `.to_dict` | `()` | **yes** | [L315](../../../crawlbot/simulation/logging.py#L315) |
| `.save` | `(path)` | **yes** | [L330](../../../crawlbot/simulation/logging.py#L330) |
| `.load` | `(path)` | not exercised | [L335](../../../crawlbot/simulation/logging.py#L335) |

### Module constants

| name | value |
|---|---|
| `_ENV_VAR_NAMES` | `('MUJOCO_GL', 'OMP_NUM_THREADS', 'OPENBLAS_N` |

---

---

## 1. Structure

A dataclass of channels (`t`, `phase`, `p_torso`, `hw`, `tau_w`, `dock_events`,
`gmo_residual_norm`, ...), serialised by `to_dict()` / `save()`. The canonical
run produces **2077 ticks** across 86 channels.

`capture_environment()` freezes the stack versions (python, numpy, mujoco,
pinocchio, casadi, scipy) into the log. This is not bookkeeping: byte-identical
reproduction is meaningless on an unpinned stack, so the log has to say which
stack produced it. `gate/environment.lock` is the counterpart on the gate side.

`load` is unexercised.

## 2. ⚠ Three exported channels carry no signal

Measured on the canonical log:

| channel | actual content | cause |
|---|---|---|
| `H_rO` | **0 everywhere** | `MomentumDisturbanceEstimator.update()` never called |
| `H_dot_est` | **0 everywhere** | same |
| `gmo_contact_state` | **constant 0** | `ContactStateMachine.update()` never called |

The objects are constructed and their state is read every tick for the log — but
nothing advances them. Details in `aocs/force_estimator.md` and
`estimation/contact_estimator.md`.

Worth knowing before plotting or analysing any of the three.

## 3. Conventions that will mislead a reader who does not know them

**`nmpc_ok = 0` means "not called", not "failed."** The NMPC runs only in SS and
the terminal settle; DS interstep ticks are exported as 0. On the canonical that
is **1368 of 2077 ticks**, so reading the column naively gives a **false 34.1 %**
success rate against a true **100 % (709/709)**.

The encoding was left untouched on purpose: any fix — a different sentinel, or an
`nmpc_called` column — changes the fulldiag CSV and would require regenerating
the frozen paper baseline under a Tier-1 gate exception. Documented instead;
revisit after submission.

**The CoM reference snaps to the measured CoM** at SS->DS entry (`_log_ds_tick`
writes `e_com = 0` with `ref := measured`). Logging convention, decision pending.

**The torso reference is continuous** across SS->DS->SS since the terminal-hold
fix — logging only, control proven byte-identical by full re-run.

**`qp_time_ms` is not a QP solve time.** Its timer spans the whole WBC block
(`sim_loop.py`, `t_qp_start` … `t_qp_ms`), which contains
`for qs in range(n_qp_per_nmpc)` — **ten** QP solves plus ten Pinocchio
`computeAllTerms`, ten AOCS evaluations, ten `mj_step` calls and this logging.
Measured on the canonical, the QP itself is **~71 %** of that block (median),
so `qp_time_ms / 10` overstates a solve by roughly 1.4×. Use `qp_solve_ms_*`
below for the QP; keep `qp_time_ms` for the block. The name is wrong and stays
wrong: it is one of the 66 frozen columns.

**`qp_ok` is not a measurement.** Two independent reasons, both structural:

1. `_log_ds_tick` hardcodes `log.qp_ok.append(True)` on inter-step ticks even
   though that loop solves a QP every tick — **1368 of 2077 rows** on the
   canonical.
2. Where it *is* computed, it comes from a `try/except RuntimeError` around a
   backend configured with `error_on_fail: False`, which returns instead of
   raising on failure. See `solvers/hierarchical_qp.md` §4.

So "0 QP failures" was never a claim about the run. `qp_status_worst` /
`qp_n_failed` below are the measurement; they cover all 2077 ticks and all
8458 solves.

## 3b. The `qp_*` channels (C2.2.1) and their sentinel convention

Six per-tick aggregates over the QP solves inside one logged tick, reduced by
`tick_logging.QPStatAccumulator` (one definition, shared by both recorders):
`qp_solve_ms_sum`, `qp_solve_ms_max`, `qp_iter_sum`, `qp_n_solves`,
`qp_n_failed`, `qp_status_worst`.

They aggregate because the CSV is one row per logged tick while an SS or
DS-terminal tick contains ten solves; an inter-step tick contains one.

**Sentinel convention — read this before computing any statistic.** A row where
no solve was offered to the accumulator carries
`sum = max = 0.0`, `iter_sum = n_solves = n_failed = 0`, `status_worst = -1`.
`status_worst = -1` is *not measured*, never an outcome. **Test
`qp_n_solves == 0`** for sentinel rows — do not test the timers against 0.0,
which is how the `nmpc_time_ms` convention traps readers.

On the canonical there are **zero** sentinel rows: 1368 ticks × 1 solve +
709 ticks × 10 solves = 8458 solves, all recorded. That is the difference from
`qp_ok`, which is a measurement on 709 ticks and a constant on the rest.

`qp_status_worst` ordering, ascending severity: `-1` not measured · `0` the
backend reported success · `1` the backend reported **not** success (the case
`qp_ok` cannot see) · `2` the solve raised.

## 3c. `preplanner_stats` and `environment['host']`

`preplanner_stats` (C2.2.3) carries one record per coarse-pre-planner NLP
solve — `{success, solve_ms, iter_count, cost, status, t_plan_start, T_step}`,
six on a six-step traversal. These were collected in
`SimulationLoop._preplanner_stats` and printed, but never persisted, so no
artifact predating this field records the IPOPT solves that gate every step.

`environment['host']` (C2.2.4) records CPU model, logical CPU count, total
memory and platform. Wall-clock solver timings move ~25 % between machines
while iteration counts are byte-identical, so a timing number is only citable
next to the hardware that produced it — and the frozen canonical artifacts,
which predate this field, do not carry one.

## 4. What is authoritative

Reference metrics do not come from here directly but from the gate
(`gate/dock_check.py`) and the exporters (`scripts/diag_full_diag_export.py`,
`scripts/export_figure_data.py`), which re-read this `sim_log.json`.

## Code map

| unit | source |
|---|---|
| `capture_environment()` | [L21-105](../../../crawlbot/simulation/logging.py#L21-L105) |
| `class SimLog` | [L109-342](../../../crawlbot/simulation/logging.py#L109-L342) |
| `SimLog.to_dict` | [L315-328](../../../crawlbot/simulation/logging.py#L315-L328) |
| `SimLog.save` | [L330-332](../../../crawlbot/simulation/logging.py#L330-L332) |
| `SimLog.load` | [L335-342](../../../crawlbot/simulation/logging.py#L335-L342) |

---

## See also

- package overview: [`simulation.md`](simulation.md)
