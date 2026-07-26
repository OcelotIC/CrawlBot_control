# `crawlbot.simulation.logging`

**File**: [`crawlbot/simulation/logging.py`](../../../crawlbot/simulation/logging.py) — **269 lines** — canonical coverage **93 %**

> Module docstring: *"Simulation data logger."*

`SimLog`: one array per quantity, one entry per tick, plus a capture of the
execution environment. This is the file every downstream analysis reads.

---

## Public API

| symbol | signature | canonical? | code |
|---|---|---|---|
| `capture_environment` | `()` | **yes** | [L21](../../../crawlbot/simulation/logging.py#L21) |
| **`SimLog`** *(dataclass)* |  |  | [L73](../../../crawlbot/simulation/logging.py#L73) |
|   `t` | `field(default_factory=list)` | _field_ | [L76](../../../crawlbot/simulation/logging.py#L76) |
|   `phase` | `field(default_factory=list)` | _field_ | [L77](../../../crawlbot/simulation/logging.py#L77) |
|   `step_idx` | `field(default_factory=list)` | _field_ | [L78](../../../crawlbot/simulation/logging.py#L78) |
|   `p_torso` | `field(default_factory=list)` | _field_ | [L81](../../../crawlbot/simulation/logging.py#L81) |
|   `p_torso_ref` | `field(default_factory=list)` | _field_ | [L82](../../../crawlbot/simulation/logging.py#L82) |
|   `e_torso_pos` | `field(default_factory=list)` | _field_ | [L83](../../../crawlbot/simulation/logging.py#L83) |
|   `e_torso_ori` | `field(default_factory=list)` | _field_ | [L84](../../../crawlbot/simulation/logging.py#L84) |
|   `q_torso` | `field(default_factory=list)` | _field_ | [L85](../../../crawlbot/simulation/logging.py#L85) |
|   `q_torso_ref` | `field(default_factory=list)` | _field_ | [L86](../../../crawlbot/simulation/logging.py#L86) |
|   `d_grip_swing` | `field(default_factory=list)` | _field_ | [L89](../../../crawlbot/simulation/logging.py#L89) |
|   `d_grip_stance` | `field(default_factory=list)` | _field_ | [L90](../../../crawlbot/simulation/logging.py#L90) |
|   `swing_arm` | `field(default_factory=list)` | _field_ | [L91](../../../crawlbot/simulation/logging.py#L91) |
|   `p_ee` | `field(default_factory=list)` | _field_ | [L92](../../../crawlbot/simulation/logging.py#L92) |
|   `p_ee_ref` | `field(default_factory=list)` | _field_ | [L93](../../../crawlbot/simulation/logging.py#L93) |
|   `q_ee` | `field(default_factory=list)` | _field_ | [L94](../../../crawlbot/simulation/logging.py#L94) |
|   `q_ee_ref` | `field(default_factory=list)` | _field_ | [L95](../../../crawlbot/simulation/logging.py#L95) |
|   `qvel_joints_a` | `field(default_factory=list)` | _field_ | [L98](../../../crawlbot/simulation/logging.py#L98) |
|   `qvel_joints_b` | `field(default_factory=list)` | _field_ | [L99](../../../crawlbot/simulation/logging.py#L99) |
|   `v_ee_a` | `field(default_factory=list)` | _field_ | [L100](../../../crawlbot/simulation/logging.py#L100) |
|   `v_ee_b` | `field(default_factory=list)` | _field_ | [L101](../../../crawlbot/simulation/logging.py#L101) |
|   `omega_ee_a` | `field(default_factory=list)` | _field_ | [L102](../../../crawlbot/simulation/logging.py#L102) |
|   `omega_ee_b` | `field(default_factory=list)` | _field_ | [L103](../../../crawlbot/simulation/logging.py#L103) |
|   `v_torso` | `field(default_factory=list)` | _field_ | [L104](../../../crawlbot/simulation/logging.py#L104) |
|   `omega_torso` | `field(default_factory=list)` | _field_ | [L105](../../../crawlbot/simulation/logging.py#L105) |
|   `r_com` | `field(default_factory=list)` | _field_ | [L108](../../../crawlbot/simulation/logging.py#L108) |
|   `r_com_ref` | `field(default_factory=list)` | _field_ | [L109](../../../crawlbot/simulation/logging.py#L109) |
|   `e_com` | `field(default_factory=list)` | _field_ | [L110](../../../crawlbot/simulation/logging.py#L110) |
|   `v_com` | `field(default_factory=list)` | _field_ | [L111](../../../crawlbot/simulation/logging.py#L111) |
|   `v_com_ref` | `field(default_factory=list)` | _field_ | [L112](../../../crawlbot/simulation/logging.py#L112) |
|   `L_com` | `field(default_factory=list)` | _field_ | [L115](../../../crawlbot/simulation/logging.py#L115) |
|   `L_com_norm` | `field(default_factory=list)` | _field_ | [L116](../../../crawlbot/simulation/logging.py#L116) |
|   `L_com_ref` | `field(default_factory=list)` | _field_ | [L117](../../../crawlbot/simulation/logging.py#L117) |
|   `L_dot` | `field(default_factory=list)` | _field_ | [L118](../../../crawlbot/simulation/logging.py#L118) |
|   `L_dot_norm` | `field(default_factory=list)` | _field_ | [L119](../../../crawlbot/simulation/logging.py#L119) |
|   `hw` | `field(default_factory=list)` | _field_ | [L120](../../../crawlbot/simulation/logging.py#L120) |
|   `hw_physical` | `field(default_factory=list)` | _field_ | [L123](../../../crawlbot/simulation/logging.py#L123) |
|   `tau_w` | `field(default_factory=list)` | _field_ | [L124](../../../crawlbot/simulation/logging.py#L124) |
|   `rw_speed` | `field(default_factory=list)` | _field_ | [L125](../../../crawlbot/simulation/logging.py#L125) |
|   `t_ss_hifreq` | `field(default_factory=list)` | _field_ | [L131](../../../crawlbot/simulation/logging.py#L131) |
|   `tau_w_ss_hifreq` | `field(default_factory=list)` | _field_ | [L132](../../../crawlbot/simulation/logging.py#L132) |
|   `hw_ss_hifreq` | `field(default_factory=list)` | _field_ | [L133](../../../crawlbot/simulation/logging.py#L133) |
|   `e_ee_pos` | `field(default_factory=list)` | _field_ | [L136](../../../crawlbot/simulation/logging.py#L136) |
|   `e_ee_ori` | `field(default_factory=list)` | _field_ | [L137](../../../crawlbot/simulation/logging.py#L137) |
|   `gmo_residual_norm` | `field(default_factory=list)` | _field_ | [L140](../../../crawlbot/simulation/logging.py#L140) |
|   `gmo_swing_residual` | `field(default_factory=list)` | _field_ | [L141](../../../crawlbot/simulation/logging.py#L141) |
|   `gmo_contact_state` | `field(default_factory=list)` | _field_ | [L142](../../../crawlbot/simulation/logging.py#L142) |
|   `H_rO` | `field(default_factory=list)` | _field_ | [L145](../../../crawlbot/simulation/logging.py#L145) |
|   `H_dot_est` | `field(default_factory=list)` | _field_ | [L146](../../../crawlbot/simulation/logging.py#L146) |
|   `omega_struct` | `field(default_factory=list)` | _field_ | [L147](../../../crawlbot/simulation/logging.py#L147) |
|   `qfrc_constraint_torque` | `field(default_factory=list)` | _field_ | [L148](../../../crawlbot/simulation/logging.py#L148) |
|   `tau` | `field(default_factory=list)` | _field_ | [L151](../../../crawlbot/simulation/logging.py#L151) |
|   `tau_max_joint` | `field(default_factory=list)` | _field_ | [L152](../../../crawlbot/simulation/logging.py#L152) |
|   `struct_pos` | `field(default_factory=list)` | _field_ | [L155](../../../crawlbot/simulation/logging.py#L155) |
|   `struct_quat` | `field(default_factory=list)` | _field_ | [L156](../../../crawlbot/simulation/logging.py#L156) |
|   `struct_euler_deg` | `field(default_factory=list)` | _field_ | [L157](../../../crawlbot/simulation/logging.py#L157) |
|   `omega_s` | `field(default_factory=list)` | _field_ | [L158](../../../crawlbot/simulation/logging.py#L158) |
|   `nmpc_ok` | `field(default_factory=list)` | _field_ | [L161](../../../crawlbot/simulation/logging.py#L161) |
|   `qp_ok` | `field(default_factory=list)` | _field_ | [L162](../../../crawlbot/simulation/logging.py#L162) |
|   `lambda_ref_norm` | `field(default_factory=list)` | _field_ | [L163](../../../crawlbot/simulation/logging.py#L163) |
|   `lambda_qp` | `field(default_factory=list)` | _field_ | [L166](../../../crawlbot/simulation/logging.py#L166) |
|   `lambda_qp_norm` | `field(default_factory=list)` | _field_ | [L167](../../../crawlbot/simulation/logging.py#L167) |
|   `nmpc_time_ms` | `field(default_factory=list)` | _field_ | [L168](../../../crawlbot/simulation/logging.py#L168) |
|   `qp_time_ms` | `field(default_factory=list)` | _field_ | [L169](../../../crawlbot/simulation/logging.py#L169) |
|   `nmpc_status` | `field(default_factory=list)` | _field_ | [L170](../../../crawlbot/simulation/logging.py#L170) |
|   `nmpc_cost` | `field(default_factory=list)` | _field_ | [L171](../../../crawlbot/simulation/logging.py#L171) |
|   `nmpc_status_str` | `field(default_factory=list)` | _field_ | [L172](../../../crawlbot/simulation/logging.py#L172) |
|   `nmpc_iterations` | `field(default_factory=list)` | _field_ | [L173](../../../crawlbot/simulation/logging.py#L173) |
|   `transport_term_mag` | `field(default_factory=list)` | _field_ | [L174](../../../crawlbot/simulation/logging.py#L174) |
|   `lambda_ref` | `field(default_factory=list)` | _field_ | [L179](../../../crawlbot/simulation/logging.py#L179) |
|   `lambda_qp` | `field(default_factory=list)` | _field_ | [L180](../../../crawlbot/simulation/logging.py#L180) |
|   `T_kinetic` | `field(default_factory=list)` | _field_ | [L183](../../../crawlbot/simulation/logging.py#L183) |
|   `settling_t` | `field(default_factory=list)` | _field_ | [L186](../../../crawlbot/simulation/logging.py#L186) |
|   `settling_T` | `field(default_factory=list)` | _field_ | [L187](../../../crawlbot/simulation/logging.py#L187) |
|   `settling_T_target` | `0.0` | _field_ | [L188](../../../crawlbot/simulation/logging.py#L188) |
|   `settling_stage1_steps` | `0` | _field_ | [L189](../../../crawlbot/simulation/logging.py#L189) |
|   `settling_stage2_steps` | `0` | _field_ | [L190](../../../crawlbot/simulation/logging.py#L190) |
|   `settling_exit_reason` | `''` | _field_ | [L191](../../../crawlbot/simulation/logging.py#L191) |
|   `inter_step_settles` | `field(default_factory=list)` | _field_ | [L197](../../../crawlbot/simulation/logging.py#L197) |
|   `dock_events` | `field(default_factory=list)` | _field_ | [L200](../../../crawlbot/simulation/logging.py#L200) |
|   `dock_gate_trace` | `field(default_factory=list)` | _field_ | [L208](../../../crawlbot/simulation/logging.py#L208) |
|   `ds_mobile_trace` | `field(default_factory=list)` | _field_ | [L215](../../../crawlbot/simulation/logging.py#L215) |
|   `dock_work_trace` | `field(default_factory=list)` | _field_ | [L221](../../../crawlbot/simulation/logging.py#L221) |
|   `aborted_steps` | `field(default_factory=list)` | _field_ | [L226](../../../crawlbot/simulation/logging.py#L226) |
|   `preplanner_T_steps` | `field(default_factory=list)` | _field_ | [L230](../../../crawlbot/simulation/logging.py#L230) |
|   `snapshots` | `field(default_factory=list)` | _field_ | [L233](../../../crawlbot/simulation/logging.py#L233) |
|   `environment` | `field(default_factory=dict)` | _field_ | [L239](../../../crawlbot/simulation/logging.py#L239) |
| `.to_dict` | `()` | **yes** | [L241](../../../crawlbot/simulation/logging.py#L241) |
| `.save` | `(path)` | **yes** | [L256](../../../crawlbot/simulation/logging.py#L256) |
| `.load` | `(path)` | not exercised | [L261](../../../crawlbot/simulation/logging.py#L261) |

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

## 4. What is authoritative

Reference metrics do not come from here directly but from the gate
(`gate/dock_check.py`) and the exporters (`scripts/diag_full_diag_export.py`,
`scripts/export_figure_data.py`), which re-read this `sim_log.json`.

## Code map

| unit | source |
|---|---|
| `capture_environment()` | [L21-69](../../../crawlbot/simulation/logging.py#L21-L69) |
| `class SimLog` | [L73-268](../../../crawlbot/simulation/logging.py#L73-L268) |
| `SimLog.to_dict` | [L241-254](../../../crawlbot/simulation/logging.py#L241-L254) |
| `SimLog.save` | [L256-258](../../../crawlbot/simulation/logging.py#L256-L258) |
| `SimLog.load` | [L261-268](../../../crawlbot/simulation/logging.py#L261-L268) |

---

## See also

- package overview: [`simulation.md`](simulation.md)
