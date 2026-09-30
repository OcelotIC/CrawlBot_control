# `crawlbot.simulation.config`

**File**: [`crawlbot/simulation/config.py`](../../../crawlbot/simulation/config.py) — **451 lines** — canonical coverage **100 %**

> Module docstring: *"Simulation configuration dataclass."*

`SimConfig` — the single tuning surface. Every adjustable parameter in the
controller lives here, with its unit and justification. **100 % coverage.**

---

## Public API

| symbol | signature | canonical? | code |
|---|---|---|---|
| **`SimConfig`** *(dataclass)* |  |  | [L12](../../../crawlbot/simulation/config.py#L12) |
|   `dt_nmpc` | `0.1` | _field_ | [L24](../../../crawlbot/simulation/config.py#L24) |
|   `dt_qp` | `0.01` | _field_ | [L25](../../../crawlbot/simulation/config.py#L25) |
|   `t_ss_margin` | `1.0` | _field_ | [L26](../../../crawlbot/simulation/config.py#L26) |
|   `t_hold_max` | `3.0` | _field_ | [L27](../../../crawlbot/simulation/config.py#L27) |
|   `dock_check_delay` | `0.5` | _field_ | [L28](../../../crawlbot/simulation/config.py#L28) |
|   `n_ds_max_steps` | `1000` | _field_ | [L29](../../../crawlbot/simulation/config.py#L29) |
|   `tau_max` | `20.0` | _field_ | [L32](../../../crawlbot/simulation/config.py#L32) |
|   `weld_radius` | `0.005` | _field_ | [L35](../../../crawlbot/simulation/config.py#L35) |
|   `dock_vel_max` | `0.01` | _field_ | [L36](../../../crawlbot/simulation/config.py#L36) |
|   `dock_ori_threshold_deg` | `5.0` | _field_ | [L42](../../../crawlbot/simulation/config.py#L42) |
|   `dock_use_6d_twist` | `True` | _field_ | [L57](../../../crawlbot/simulation/config.py#L57) |
|   `dock_twist_max` | `0.05` | _field_ | [L58](../../../crawlbot/simulation/config.py#L58) |
|   `gmo_K_O` | `80.0` | _field_ | [L61](../../../crawlbot/simulation/config.py#L61) |
|   `gmo_F_threshold` | `5.0` | _field_ | [L62](../../../crawlbot/simulation/config.py#L62) |
|   `gmo_d_proximity` | `0.02` | _field_ | [L63](../../../crawlbot/simulation/config.py#L63) |
|   `gmo_d_contact` | `0.005` | _field_ | [L64](../../../crawlbot/simulation/config.py#L64) |
|   `gmo_d_reset` | `0.03` | _field_ | [L65](../../../crawlbot/simulation/config.py#L65) |
|   `gmo_debounce_count` | `3` | _field_ | [L66](../../../crawlbot/simulation/config.py#L66) |
|   `hw_init` | `np.zeros(3)` | _field_ | [L69](../../../crawlbot/simulation/config.py#L69) |
|   `hw_min` | `np.full(3, -5.0)` | _field_ | [L70](../../../crawlbot/simulation/config.py#L70) |
|   `hw_max` | `np.full(3, 5.0)` | _field_ | [L71](../../../crawlbot/simulation/config.py#L71) |
|   `hw_qp_tight` | `np.full(3, 3.0)` | _field_ | [L77](../../../crawlbot/simulation/config.py#L77) |
|   `L_max` | `10.0` | _field_ | [L78](../../../crawlbot/simulation/config.py#L78) |
|   `tau_w_max` | `2.5` | _field_ | [L79](../../../crawlbot/simulation/config.py#L79) |
|   `nmpc_tau_w_max` | `None` | _field_ | [L90](../../../crawlbot/simulation/config.py#L90) |
|   `aocs_K_hw` | `2.0` | _field_ | [L93](../../../crawlbot/simulation/config.py#L93) |
|   `aocs_tau_w_max` | `2.5` | _field_ | [L94](../../../crawlbot/simulation/config.py#L94) |
|   `rwa_I_w` | `0.01` | _field_ | [L95](../../../crawlbot/simulation/config.py#L95) |
|   `aocs_mode` | `'legacy_pid_numerical'` | _field_ | [L101](../../../crawlbot/simulation/config.py#L101) |
|   `aocs_K_omega` | `50.0` | _field_ | [L102](../../../crawlbot/simulation/config.py#L102) |
|   `aocs_K_d` | `25.0` | _field_ | [L103](../../../crawlbot/simulation/config.py#L103) |
|   `aocs_K_theta` | `1.0` | _field_ | [L104](../../../crawlbot/simulation/config.py#L104) |
|   `aocs_use_wrench_ff_in_ds` | `False` | _field_ | [L112](../../../crawlbot/simulation/config.py#L112) |
|   `ds_torso_ref_from_state` | `False` | _field_ | [L121](../../../crawlbot/simulation/config.py#L121) |
|   `aocs_active_in_interstep` | `True` | _field_ | [L136](../../../crawlbot/simulation/config.py#L136) |
|   `interstep_hw_refresh` | `True` | _field_ | [L150](../../../crawlbot/simulation/config.py#L150) |
|   `interstep_settle_alpha_wrench` | `0.0` | _field_ | [L160](../../../crawlbot/simulation/config.py#L160) |
|   `stop_on_failed_step` | `True` | _field_ | [L170](../../../crawlbot/simulation/config.py#L170) |
|   `frames_per_step` | `0` | _field_ | [L176](../../../crawlbot/simulation/config.py#L176) |
|   `alpha_passivity` | `1.0` | _field_ | [L182](../../../crawlbot/simulation/config.py#L182) |
|   `enforce_hw_conservation` | `False` | _field_ | [L185](../../../crawlbot/simulation/config.py#L185) |
|   `h_max_tight` | `np.full(3, 5.0)` | _field_ | [L186](../../../crawlbot/simulation/config.py#L186) |
|   `w_L_nmpc` | `1.0` | _field_ | [L187](../../../crawlbot/simulation/config.py#L187) |
|   `kappa_terminal` | `1.0` | _field_ | [L188](../../../crawlbot/simulation/config.py#L188) |
|   `preplanner_M` | `15` | _field_ | [L200](../../../crawlbot/simulation/config.py#L200) |
|   `preplanner_kappa` | `0.7` | _field_ | [L201](../../../crawlbot/simulation/config.py#L201) |
|   `preplanner_f_max` | `25.0` | _field_ | [L202](../../../crawlbot/simulation/config.py#L202) |
|   `preplanner_tau_max` | `8.0` | _field_ | [L203](../../../crawlbot/simulation/config.py#L203) |
|   `preplanner_w_L` | `1.0` | _field_ | [L204](../../../crawlbot/simulation/config.py#L204) |
|   `preplanner_w_u` | `0.01` | _field_ | [L205](../../../crawlbot/simulation/config.py#L205) |
|   `preplanner_max_iter` | `300` | _field_ | [L206](../../../crawlbot/simulation/config.py#L206) |
|   `preplanner_a_cruise_max` | `0.0` | _field_ | [L207](../../../crawlbot/simulation/config.py#L207) |
|   `preplanner_cruise_ramp_frac` | `0.2` | _field_ | [L208](../../../crawlbot/simulation/config.py#L208) |
|   `preplanner_tstep_standoff_gain` | `0.0` | _field_ | [L216](../../../crawlbot/simulation/config.py#L216) |
|   `preplanner_tstep_standoff_knee` | `1000000000.0` | _field_ | [L217](../../../crawlbot/simulation/config.py#L217) |
|   `preplanner_tstep_scale_step` | `-1` | _field_ | [L221](../../../crawlbot/simulation/config.py#L221) |
|   `preplanner_tstep_scale_factor` | `1.0` | _field_ | [L222](../../../crawlbot/simulation/config.py#L222) |
|   `nmpc_N` | `8` | _field_ | [L225](../../../crawlbot/simulation/config.py#L225) |
|   `nmpc_dt` | `0.1` | _field_ | [L226](../../../crawlbot/simulation/config.py#L226) |
|   `nmpc_f_max` | `300.0` | _field_ | [L227](../../../crawlbot/simulation/config.py#L227) |
|   `nmpc_tau_max` | `8.0` | _field_ | [L228](../../../crawlbot/simulation/config.py#L228) |
|   `nmpc_Wv` | `10.0` | _field_ | [L229](../../../crawlbot/simulation/config.py#L229) |
|   `nmpc_p_max` | `50.0` | _field_ | [L230](../../../crawlbot/simulation/config.py#L230) |
|   `nmpc_Wr` | `100.0` | _field_ | [L236](../../../crawlbot/simulation/config.py#L236) |
|   `nmpc_Wu_f` | `0.01` | _field_ | [L237](../../../crawlbot/simulation/config.py#L237) |
|   `nmpc_Wu_tau` | `0.001` | _field_ | [L238](../../../crawlbot/simulation/config.py#L238) |
|   `nmpc_Qf_r` | `1000.0` | _field_ | [L239](../../../crawlbot/simulation/config.py#L239) |
|   `nmpc_Qf_v` | `100.0` | _field_ | [L240](../../../crawlbot/simulation/config.py#L240) |
|   `nmpc_Qf_L` | `10.0` | _field_ | [L241](../../../crawlbot/simulation/config.py#L241) |
|   `t_settle_final` | `20.0` | _field_ | [L242](../../../crawlbot/simulation/config.py#L242) |
|   `t_settle_inter` | `0.0` | _field_ | [L249](../../../crawlbot/simulation/config.py#L249) |
|   `use_energy_settle_inter` | `True` | _field_ | [L250](../../../crawlbot/simulation/config.py#L250) |
|   `settle_inter_epsilon_v` | `0.001` | _field_ | [L251](../../../crawlbot/simulation/config.py#L251) |
|   `interstep_settle_epsilon_v` | `0.0` | _field_ | [L259](../../../crawlbot/simulation/config.py#L259) |
|   `n_settle_inter_max_steps` | `500` | _field_ | [L260](../../../crawlbot/simulation/config.py#L260) |
|   `t_settle_inter_min` | `0.1` | _field_ | [L261](../../../crawlbot/simulation/config.py#L261) |
|   `ss_alpha_ee` | `1000.0` | _field_ | [L264](../../../crawlbot/simulation/config.py#L264) |
|   `ss_alpha_posture` | `20.0` | _field_ | [L265](../../../crawlbot/simulation/config.py#L265) |
|   `ss_alpha_wrench` | `1.0` | _field_ | [L266](../../../crawlbot/simulation/config.py#L266) |
|   `ss_alpha_lambda_int` | `0.0` | _field_ | [L267](../../../crawlbot/simulation/config.py#L267) |
|   `ss_alpha_mom` | `400.0` | _field_ | [L272](../../../crawlbot/simulation/config.py#L272) |
|   `log_hifreq_ss` | `False` | _field_ | [L276](../../../crawlbot/simulation/config.py#L276) |
|   `log_hifreq_all` | `False` | _field_ | [L283](../../../crawlbot/simulation/config.py#L283) |
|   `alpha_torso_pose` | `2000.0` | _field_ | [L290](../../../crawlbot/simulation/config.py#L290) |
|   `dt_ds` | `0.5` | _field_ | [L299](../../../crawlbot/simulation/config.py#L299) |
|   `dock_hold_passivity_on` | `False` | _field_ | [L313](../../../crawlbot/simulation/config.py#L313) |
|   `passivity_W_budget` | `0.0` | _field_ | [L314](../../../crawlbot/simulation/config.py#L314) |
|   `log_dock_work` | `False` | _field_ | [L315](../../../crawlbot/simulation/config.py#L315) |
|   `qp_envelope_exact` | `False` | _field_ | [L325](../../../crawlbot/simulation/config.py#L325) |
|   `ds_alpha_com` | `100.0` | _field_ | [L332](../../../crawlbot/simulation/config.py#L332) |
|   `ds_alpha_torso_ori` | `200.0` | _field_ | [L333](../../../crawlbot/simulation/config.py#L333) |
|   `ds_alpha_posture` | `50.0` | _field_ | [L334](../../../crawlbot/simulation/config.py#L334) |
|   `ss_Kp_com` | `3.0` | _field_ | [L337](../../../crawlbot/simulation/config.py#L337) |
|   `ss_Kd_com` | `3.0` | _field_ | [L338](../../../crawlbot/simulation/config.py#L338) |
|   `ss_Kp_torso` | `6.0` | _field_ | [L339](../../../crawlbot/simulation/config.py#L339) |
|   `ss_Kd_torso` | `5.0` | _field_ | [L340](../../../crawlbot/simulation/config.py#L340) |
|   `ss_Kp_ee` | `10.0` | _field_ | [L341](../../../crawlbot/simulation/config.py#L341) |
|   `ss_Kd_ee` | `12.0` | _field_ | [L342](../../../crawlbot/simulation/config.py#L342) |
|   `ss_Kp_ee_ang` | `6.0` | _field_ | [L343](../../../crawlbot/simulation/config.py#L343) |
|   `ss_Kd_ee_ang` | `4.5` | _field_ | [L344](../../../crawlbot/simulation/config.py#L344) |
|   `swing_clearance` | `0.03` | _field_ | [L347](../../../crawlbot/simulation/config.py#L347) |
|   `swing_bump_peak_tau` | `0.5` | _field_ | [L353](../../../crawlbot/simulation/config.py#L353) |
|   `ik_fixed_rotation` | `True` | _field_ | [L364](../../../crawlbot/simulation/config.py#L364) |
|   `ik_fixed_rotation_w_min` | `0.0001` | _field_ | [L365](../../../crawlbot/simulation/config.py#L365) |
|   `ik_level_axis` | `None` | _field_ | [L380](../../../crawlbot/simulation/config.py#L380) |
|   `ik_q_nominal` | `None` | _field_ | [L381](../../../crawlbot/simulation/config.py#L381) |
|   `ik_w_posture` | `0.0` | _field_ | [L382](../../../crawlbot/simulation/config.py#L382) |
|   `use_com_z_standoff` | `False` | _field_ | [L395](../../../crawlbot/simulation/config.py#L395) |
|   `com_z_standoff` | `-0.35` | _field_ | [L396](../../../crawlbot/simulation/config.py#L396) |
|   `torso_early_finish_fraction` | `1.0` | _field_ | [L421](../../../crawlbot/simulation/config.py#L421) |
|   `swing_early_finish_fraction` | `1.0` | _field_ | [L430](../../../crawlbot/simulation/config.py#L430) |
|   `n_settle_steps` | `500` | _field_ | [L433](../../../crawlbot/simulation/config.py#L433) |
|   `Kd_settle_damping` | `20.0` | _field_ | [L444](../../../crawlbot/simulation/config.py#L444) |
|   `n_settle_max_steps` | `1000` | _field_ | [L445](../../../crawlbot/simulation/config.py#L445) |
|   `settle_epsilon_v` | `0.001` | _field_ | [L446](../../../crawlbot/simulation/config.py#L446) |
|   `settle_plateau_ratio` | `0.999` | _field_ | [L447](../../../crawlbot/simulation/config.py#L447) |
|   `gait_anchor_dx` | `0.8` | _field_ | [L450](../../../crawlbot/simulation/config.py#L450) |

---

---

## 1. Rule 5

> *No silent parameter changes. All tunable parameters live in `SimConfig` with
> units and justification.*

One dataclass, ~500 lines, from which `CentroidalNMPCConfig`,
`WholeBodyQPConfig`, `CoarsePrePlannerConfig` and `ContactObserverConfig` are all
constructed. No magic constant buried in the loop.

The payoff is that a run is fully described by one object, which is what makes
byte-identical reproduction possible at all.

## 2. ⚠ But a `SimConfig` default is not the canonical value either

The canonical run is built in two stages:

```
Misc/scripts/run_m7_single_step._make_m7_config()   ->  base SimConfig
scripts/diag_cooperative_arms.main(**kwargs)        ->  per-run overrides
```

To learn a canonical value: the "Key Parameters" table in CLAUDE.md, or
instrument the run. **Never read a default** — that is exactly the error the
chantier retracted (F1), where `enforce_hw_conservation=False` in a dataclass was
taken for the canonical setting while the run sets it `True`.

## 3. `use_m2_stack` — removed (R2c)

It gated **the DS passivity constraint** (and, until R2b, the torso-reference
routing). Frozen at its canonical value `True` and removed: DS passivity is
always on. A bare `SimConfig()` defaulted to `False`, so default-config callers
now get DS passivity. See `sim_loop.md` section 6.

## 4. Parameters not to touch without reading CLAUDE.md

| parameter | frozen value | why |
|---|---|---|
| `tau_w_max` | **2.5** Nm | enforced at 3 points: NMPC, QP, MJCF actuator |
| `hw_max` | +/-5 Nms | unchanged by design |
| `weight_ratio` | 1.0 | the alphas *are* the hierarchy |
| `alpha_wrench` | 1.0 | above 1 it starves the torso/EE tasks |
| `preplanner_a_cruise_max` | 0.0 | CoM shaping disabled |

`tau_w_max` is worth expanding: the cap is enforced in the NMPC constraint, in
the QP box, in the AOCS clip **and** in the MuJoCo actuator `ctrlrange`. The last
one is the plant, so it holds even if a controller-side bug lets a larger demand
through — that redundancy is what let the unmanaged comparison run be measured
honestly (controller demanding 26.9 Nm, actuator delivering 2.5).

The `preplanner_tstep_*` knobs are diagnostics exposed by `dca`, all neutral by
default.

## Code map

| unit | source |
|---|---|
| `class SimConfig` | [L12-450](../../../crawlbot/simulation/config.py#L12-L450) |

---

## See also

- package overview: [`simulation.md`](simulation.md)
