# `crawlbot.simulation.logging`

**File**: `crawlbot/simulation/logging.py` — **269 lines** — canonical coverage **93 %**

> Module docstring: *"Simulation data logger."*

`SimLog`: one array per quantity, one entry per tick, plus a capture of the
execution environment. This is the file every downstream analysis reads.

---

## Public API

| symbol | signature | canonical? |
|---|---|---|
| `capture_environment` | `()` | **yes** |
| **`SimLog`** *(dataclass)* |  |  |
|   `t` | `field(default_factory=list)` | _field_ |
|   `phase` | `field(default_factory=list)` | _field_ |
|   `step_idx` | `field(default_factory=list)` | _field_ |
|   `p_torso` | `field(default_factory=list)` | _field_ |
|   `p_torso_ref` | `field(default_factory=list)` | _field_ |
|   `e_torso_pos` | `field(default_factory=list)` | _field_ |
|   `e_torso_ori` | `field(default_factory=list)` | _field_ |
|   `q_torso` | `field(default_factory=list)` | _field_ |
|   `q_torso_ref` | `field(default_factory=list)` | _field_ |
|   `d_grip_swing` | `field(default_factory=list)` | _field_ |
|   `d_grip_stance` | `field(default_factory=list)` | _field_ |
|   `swing_arm` | `field(default_factory=list)` | _field_ |
|   `p_ee` | `field(default_factory=list)` | _field_ |
|   `p_ee_ref` | `field(default_factory=list)` | _field_ |
|   `q_ee` | `field(default_factory=list)` | _field_ |
|   `q_ee_ref` | `field(default_factory=list)` | _field_ |
|   `qvel_joints_a` | `field(default_factory=list)` | _field_ |
|   `qvel_joints_b` | `field(default_factory=list)` | _field_ |
|   `v_ee_a` | `field(default_factory=list)` | _field_ |
|   `v_ee_b` | `field(default_factory=list)` | _field_ |
|   `omega_ee_a` | `field(default_factory=list)` | _field_ |
|   `omega_ee_b` | `field(default_factory=list)` | _field_ |
|   `v_torso` | `field(default_factory=list)` | _field_ |
|   `omega_torso` | `field(default_factory=list)` | _field_ |
|   `r_com` | `field(default_factory=list)` | _field_ |
|   `r_com_ref` | `field(default_factory=list)` | _field_ |
|   `e_com` | `field(default_factory=list)` | _field_ |
|   `v_com` | `field(default_factory=list)` | _field_ |
|   `v_com_ref` | `field(default_factory=list)` | _field_ |
|   `L_com` | `field(default_factory=list)` | _field_ |
|   `L_com_norm` | `field(default_factory=list)` | _field_ |
|   `L_com_ref` | `field(default_factory=list)` | _field_ |
|   `L_dot` | `field(default_factory=list)` | _field_ |
|   `L_dot_norm` | `field(default_factory=list)` | _field_ |
|   `hw` | `field(default_factory=list)` | _field_ |
|   `hw_physical` | `field(default_factory=list)` | _field_ |
|   `tau_w` | `field(default_factory=list)` | _field_ |
|   `rw_speed` | `field(default_factory=list)` | _field_ |
|   `t_ss_hifreq` | `field(default_factory=list)` | _field_ |
|   `tau_w_ss_hifreq` | `field(default_factory=list)` | _field_ |
|   `hw_ss_hifreq` | `field(default_factory=list)` | _field_ |
|   `e_ee_pos` | `field(default_factory=list)` | _field_ |
|   `e_ee_ori` | `field(default_factory=list)` | _field_ |
|   `gmo_residual_norm` | `field(default_factory=list)` | _field_ |
|   `gmo_swing_residual` | `field(default_factory=list)` | _field_ |
|   `gmo_contact_state` | `field(default_factory=list)` | _field_ |
|   `H_rO` | `field(default_factory=list)` | _field_ |
|   `H_dot_est` | `field(default_factory=list)` | _field_ |
|   `omega_struct` | `field(default_factory=list)` | _field_ |
|   `qfrc_constraint_torque` | `field(default_factory=list)` | _field_ |
|   `tau` | `field(default_factory=list)` | _field_ |
|   `tau_max_joint` | `field(default_factory=list)` | _field_ |
|   `struct_pos` | `field(default_factory=list)` | _field_ |
|   `struct_quat` | `field(default_factory=list)` | _field_ |
|   `struct_euler_deg` | `field(default_factory=list)` | _field_ |
|   `omega_s` | `field(default_factory=list)` | _field_ |
|   `nmpc_ok` | `field(default_factory=list)` | _field_ |
|   `qp_ok` | `field(default_factory=list)` | _field_ |
|   `lambda_ref_norm` | `field(default_factory=list)` | _field_ |
|   `lambda_qp` | `field(default_factory=list)` | _field_ |
|   `lambda_qp_norm` | `field(default_factory=list)` | _field_ |
|   `nmpc_time_ms` | `field(default_factory=list)` | _field_ |
|   `qp_time_ms` | `field(default_factory=list)` | _field_ |
|   `nmpc_status` | `field(default_factory=list)` | _field_ |
|   `nmpc_cost` | `field(default_factory=list)` | _field_ |
|   `nmpc_status_str` | `field(default_factory=list)` | _field_ |
|   `nmpc_iterations` | `field(default_factory=list)` | _field_ |
|   `transport_term_mag` | `field(default_factory=list)` | _field_ |
|   `lambda_ref` | `field(default_factory=list)` | _field_ |
|   `lambda_qp` | `field(default_factory=list)` | _field_ |
|   `T_kinetic` | `field(default_factory=list)` | _field_ |
|   `settling_t` | `field(default_factory=list)` | _field_ |
|   `settling_T` | `field(default_factory=list)` | _field_ |
|   `settling_T_target` | `0.0` | _field_ |
|   `settling_stage1_steps` | `0` | _field_ |
|   `settling_stage2_steps` | `0` | _field_ |
|   `settling_exit_reason` | `''` | _field_ |
|   `inter_step_settles` | `field(default_factory=list)` | _field_ |
|   `dock_events` | `field(default_factory=list)` | _field_ |
|   `dock_gate_trace` | `field(default_factory=list)` | _field_ |
|   `ds_mobile_trace` | `field(default_factory=list)` | _field_ |
|   `dock_work_trace` | `field(default_factory=list)` | _field_ |
|   `aborted_steps` | `field(default_factory=list)` | _field_ |
|   `preplanner_T_steps` | `field(default_factory=list)` | _field_ |
|   `snapshots` | `field(default_factory=list)` | _field_ |
|   `environment` | `field(default_factory=dict)` | _field_ |
| `.to_dict` | `()` | **yes** |
| `.save` | `(path)` | **yes** |
| `.load` | `(path)` | not exercised |

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

## See also

- package overview: [`simulation.md`](simulation.md)
