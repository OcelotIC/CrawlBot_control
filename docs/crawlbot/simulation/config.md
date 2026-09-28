# `crawlbot.simulation.config`

**File**: [`crawlbot/simulation/config.py`](../../../crawlbot/simulation/config.py) — **521 lines** — canonical coverage **100 %**

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
|   `aocs_K_hw` | `2.0` | _field_ | [L82](../../../crawlbot/simulation/config.py#L82) |
|   `aocs_tau_w_max` | `2.5` | _field_ | [L83](../../../crawlbot/simulation/config.py#L83) |
|   `rwa_I_w` | `0.01` | _field_ | [L84](../../../crawlbot/simulation/config.py#L84) |
|   `aocs_mode` | `'legacy'` | _field_ | [L98](../../../crawlbot/simulation/config.py#L98) |
|   `aocs_use_H_estimator` | `False` | _field_ | [L99](../../../crawlbot/simulation/config.py#L99) |
|   `aocs_use_legacy_corrected` | `False` | _field_ | [L100](../../../crawlbot/simulation/config.py#L100) |
|   `aocs_filter_tau` | `0.016` | _field_ | [L101](../../../crawlbot/simulation/config.py#L101) |
|   `aocs_K_omega` | `50.0` | _field_ | [L102](../../../crawlbot/simulation/config.py#L102) |
|   `aocs_K_d` | `25.0` | _field_ | [L103](../../../crawlbot/simulation/config.py#L103) |
|   `aocs_K_theta` | `1.0` | _field_ | [L104](../../../crawlbot/simulation/config.py#L104) |
|   `aocs_K_h` | `0.5` | _field_ | [L105](../../../crawlbot/simulation/config.py#L105) |
|   `aocs_hw_target` | `np.zeros(3)` | _field_ | [L106](../../../crawlbot/simulation/config.py#L106) |
|   `aocs_carry_across_nmpc_ticks` | `False` | _field_ | [L120](../../../crawlbot/simulation/config.py#L120) |
|   `aocs_use_wrench_ff_in_ds` | `False` | _field_ | [L128](../../../crawlbot/simulation/config.py#L128) |
|   `ds_torso_ref_from_state` | `False` | _field_ | [L137](../../../crawlbot/simulation/config.py#L137) |
|   `aocs_off_in_ds` | `False` | _field_ | [L140](../../../crawlbot/simulation/config.py#L140) |
|   `aocs_active_in_interstep` | `True` | _field_ | [L155](../../../crawlbot/simulation/config.py#L155) |
|   `interstep_hw_refresh` | `True` | _field_ | [L169](../../../crawlbot/simulation/config.py#L169) |
|   `interstep_settle_alpha_wrench` | `0.0` | _field_ | [L179](../../../crawlbot/simulation/config.py#L179) |
|   `stop_on_failed_step` | `True` | _field_ | [L189](../../../crawlbot/simulation/config.py#L189) |
|   `frames_per_step` | `0` | _field_ | [L195](../../../crawlbot/simulation/config.py#L195) |
|   `use_m2_stack` | `False` | _field_ | [L203](../../../crawlbot/simulation/config.py#L203) |
|   `alpha_passivity` | `1.0` | _field_ | [L204](../../../crawlbot/simulation/config.py#L204) |
|   `enforce_hw_conservation` | `False` | _field_ | [L207](../../../crawlbot/simulation/config.py#L207) |
|   `h_max_tight` | `np.full(3, 5.0)` | _field_ | [L208](../../../crawlbot/simulation/config.py#L208) |
|   `w_L_nmpc` | `1.0` | _field_ | [L209](../../../crawlbot/simulation/config.py#L209) |
|   `kappa_terminal` | `1.0` | _field_ | [L210](../../../crawlbot/simulation/config.py#L210) |
|   `preplanner_M` | `15` | _field_ | [L222](../../../crawlbot/simulation/config.py#L222) |
|   `preplanner_kappa` | `0.7` | _field_ | [L223](../../../crawlbot/simulation/config.py#L223) |
|   `preplanner_f_max` | `25.0` | _field_ | [L224](../../../crawlbot/simulation/config.py#L224) |
|   `preplanner_tau_max` | `8.0` | _field_ | [L225](../../../crawlbot/simulation/config.py#L225) |
|   `preplanner_w_L` | `1.0` | _field_ | [L226](../../../crawlbot/simulation/config.py#L226) |
|   `preplanner_w_u` | `0.01` | _field_ | [L227](../../../crawlbot/simulation/config.py#L227) |
|   `preplanner_max_iter` | `300` | _field_ | [L228](../../../crawlbot/simulation/config.py#L228) |
|   `preplanner_a_cruise_max` | `0.0` | _field_ | [L229](../../../crawlbot/simulation/config.py#L229) |
|   `preplanner_cruise_ramp_frac` | `0.2` | _field_ | [L230](../../../crawlbot/simulation/config.py#L230) |
|   `preplanner_tstep_standoff_gain` | `0.0` | _field_ | [L238](../../../crawlbot/simulation/config.py#L238) |
|   `preplanner_tstep_standoff_knee` | `1000000000.0` | _field_ | [L239](../../../crawlbot/simulation/config.py#L239) |
|   `preplanner_tstep_scale_step` | `-1` | _field_ | [L243](../../../crawlbot/simulation/config.py#L243) |
|   `preplanner_tstep_scale_factor` | `1.0` | _field_ | [L244](../../../crawlbot/simulation/config.py#L244) |
|   `fsat_jitter_margin` | `0.05` | _field_ | [L254](../../../crawlbot/simulation/config.py#L254) |
|   `nmpc_N` | `8` | _field_ | [L257](../../../crawlbot/simulation/config.py#L257) |
|   `nmpc_dt` | `0.1` | _field_ | [L258](../../../crawlbot/simulation/config.py#L258) |
|   `nmpc_f_max` | `300.0` | _field_ | [L259](../../../crawlbot/simulation/config.py#L259) |
|   `nmpc_tau_max` | `8.0` | _field_ | [L260](../../../crawlbot/simulation/config.py#L260) |
|   `nmpc_Wv` | `10.0` | _field_ | [L261](../../../crawlbot/simulation/config.py#L261) |
|   `nmpc_p_max` | `50.0` | _field_ | [L262](../../../crawlbot/simulation/config.py#L262) |
|   `nmpc_Wr` | `100.0` | _field_ | [L268](../../../crawlbot/simulation/config.py#L268) |
|   `nmpc_Wu_f` | `0.01` | _field_ | [L269](../../../crawlbot/simulation/config.py#L269) |
|   `nmpc_Wu_tau` | `0.001` | _field_ | [L270](../../../crawlbot/simulation/config.py#L270) |
|   `nmpc_Qf_r` | `1000.0` | _field_ | [L271](../../../crawlbot/simulation/config.py#L271) |
|   `nmpc_Qf_v` | `100.0` | _field_ | [L272](../../../crawlbot/simulation/config.py#L272) |
|   `nmpc_Qf_L` | `10.0` | _field_ | [L273](../../../crawlbot/simulation/config.py#L273) |
|   `t_settle_final` | `20.0` | _field_ | [L274](../../../crawlbot/simulation/config.py#L274) |
|   `t_settle_inter` | `0.0` | _field_ | [L281](../../../crawlbot/simulation/config.py#L281) |
|   `use_energy_settle_inter` | `True` | _field_ | [L282](../../../crawlbot/simulation/config.py#L282) |
|   `settle_inter_epsilon_v` | `0.001` | _field_ | [L283](../../../crawlbot/simulation/config.py#L283) |
|   `interstep_settle_epsilon_v` | `0.0` | _field_ | [L291](../../../crawlbot/simulation/config.py#L291) |
|   `n_settle_inter_max_steps` | `500` | _field_ | [L292](../../../crawlbot/simulation/config.py#L292) |
|   `t_settle_inter_min` | `0.1` | _field_ | [L293](../../../crawlbot/simulation/config.py#L293) |
|   `ss_alpha_ee` | `1000.0` | _field_ | [L296](../../../crawlbot/simulation/config.py#L296) |
|   `ss_alpha_posture` | `20.0` | _field_ | [L297](../../../crawlbot/simulation/config.py#L297) |
|   `ss_alpha_wrench` | `1.0` | _field_ | [L298](../../../crawlbot/simulation/config.py#L298) |
|   `ss_alpha_lambda_int` | `0.0` | _field_ | [L299](../../../crawlbot/simulation/config.py#L299) |
|   `ss_alpha_mom` | `400.0` | _field_ | [L304](../../../crawlbot/simulation/config.py#L304) |
|   `log_hifreq_ss` | `False` | _field_ | [L308](../../../crawlbot/simulation/config.py#L308) |
|   `ss_two_task_mode` | `False` | _field_ | [L316](../../../crawlbot/simulation/config.py#L316) |
|   `alpha_torso_pose` | `2000.0` | _field_ | [L317](../../../crawlbot/simulation/config.py#L317) |
|   `dt_ds` | `0.5` | _field_ | [L326](../../../crawlbot/simulation/config.py#L326) |
|   `dock_hold_passivity_on` | `False` | _field_ | [L340](../../../crawlbot/simulation/config.py#L340) |
|   `passivity_W_budget` | `0.0` | _field_ | [L341](../../../crawlbot/simulation/config.py#L341) |
|   `log_dock_work` | `False` | _field_ | [L342](../../../crawlbot/simulation/config.py#L342) |
|   `qp_envelope_exact` | `False` | _field_ | [L352](../../../crawlbot/simulation/config.py#L352) |
|   `ds_centroidal_mode` | `False` | _field_ | [L357](../../../crawlbot/simulation/config.py#L357) |
|   `ds_alpha_com` | `100.0` | _field_ | [L358](../../../crawlbot/simulation/config.py#L358) |
|   `ds_alpha_torso_ori` | `200.0` | _field_ | [L359](../../../crawlbot/simulation/config.py#L359) |
|   `ds_alpha_posture` | `50.0` | _field_ | [L360](../../../crawlbot/simulation/config.py#L360) |
|   `ss_Kp_com` | `3.0` | _field_ | [L363](../../../crawlbot/simulation/config.py#L363) |
|   `ss_Kd_com` | `3.0` | _field_ | [L364](../../../crawlbot/simulation/config.py#L364) |
|   `ss_Kp_torso` | `6.0` | _field_ | [L365](../../../crawlbot/simulation/config.py#L365) |
|   `ss_Kd_torso` | `5.0` | _field_ | [L366](../../../crawlbot/simulation/config.py#L366) |
|   `ss_Kp_ee` | `10.0` | _field_ | [L367](../../../crawlbot/simulation/config.py#L367) |
|   `ss_Kd_ee` | `12.0` | _field_ | [L368](../../../crawlbot/simulation/config.py#L368) |
|   `ss_Kp_ee_ang` | `6.0` | _field_ | [L369](../../../crawlbot/simulation/config.py#L369) |
|   `ss_Kd_ee_ang` | `4.5` | _field_ | [L370](../../../crawlbot/simulation/config.py#L370) |
|   `swing_clearance` | `0.03` | _field_ | [L373](../../../crawlbot/simulation/config.py#L373) |
|   `swing_bump_peak_tau` | `0.5` | _field_ | [L379](../../../crawlbot/simulation/config.py#L379) |
|   `ik_fixed_rotation` | `True` | _field_ | [L390](../../../crawlbot/simulation/config.py#L390) |
|   `ik_fixed_rotation_w_min` | `0.0001` | _field_ | [L391](../../../crawlbot/simulation/config.py#L391) |
|   `ik_level_axis` | `None` | _field_ | [L406](../../../crawlbot/simulation/config.py#L406) |
|   `ik_q_nominal` | `None` | _field_ | [L407](../../../crawlbot/simulation/config.py#L407) |
|   `ik_w_posture` | `0.0` | _field_ | [L408](../../../crawlbot/simulation/config.py#L408) |
|   `use_com_z_standoff` | `False` | _field_ | [L421](../../../crawlbot/simulation/config.py#L421) |
|   `com_z_standoff` | `-0.35` | _field_ | [L422](../../../crawlbot/simulation/config.py#L422) |
|   `torso_early_finish_fraction` | `1.0` | _field_ | [L447](../../../crawlbot/simulation/config.py#L447) |
|   `swing_early_finish_fraction` | `1.0` | _field_ | [L456](../../../crawlbot/simulation/config.py#L456) |
|   `n_settle_steps` | `500` | _field_ | [L459](../../../crawlbot/simulation/config.py#L459) |
|   `Kd_settle_damping` | `20.0` | _field_ | [L470](../../../crawlbot/simulation/config.py#L470) |
|   `n_settle_max_steps` | `1000` | _field_ | [L471](../../../crawlbot/simulation/config.py#L471) |
|   `settle_epsilon_v` | `0.001` | _field_ | [L472](../../../crawlbot/simulation/config.py#L472) |
|   `settle_plateau_ratio` | `0.999` | _field_ | [L473](../../../crawlbot/simulation/config.py#L473) |
|   `diag_freeze_torso_ref_on_abort` | `False` | _field_ | [L480](../../../crawlbot/simulation/config.py#L480) |
|   `diag_force_single_contact_on_abort` | `False` | _field_ | [L486](../../../crawlbot/simulation/config.py#L486) |
|   `diag_disable_passivity_on_abort` | `False` | _field_ | [L492](../../../crawlbot/simulation/config.py#L492) |
|   `mapping_bypass_in_ss` | `False` | _field_ | [L498](../../../crawlbot/simulation/config.py#L498) |
|   `ds_ramp_duration_s` | `2.0` | _field_ | [L508](../../../crawlbot/simulation/config.py#L508) |
|   `gait_anchor_dx` | `0.8` | _field_ | [L520](../../../crawlbot/simulation/config.py#L520) |

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

## 3. The `use_m2_stack` trap

It *looks* dead and in fact gates the torso-reference routing **and the DS
passivity constraint**. Its declaration now carries a note saying so. See
`sim_loop.md` section 6.

## 4. Parameters not to touch without reading CLAUDE.md

| parameter | frozen value | why |
|---|---|---|
| `tau_w_max` | **2.5** Nm | enforced at 3 points: NMPC, QP, MJCF actuator |
| `hw_max` | +/-5 Nms | unchanged by design |
| `weight_ratio` | 1.0 | the alphas *are* the hierarchy |
| `alpha_wrench` | 1.0 | above 1 it starves the torso/EE tasks |
| `preplanner_a_cruise_max` | 0.0 | CoM shaping disabled |
| `aocs_carry_across_nmpc_ticks` | False | True removes a 10 Hz AOCS kick but θ_s peak 0.54° → 0.87° and dock margin 0.01 → 0.003 mm: the AOCS gains were tuned with the kick ([`control/attitude.md`](../control/attitude.md) §3) |

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
| `class SimConfig` | [L12-520](../../../crawlbot/simulation/config.py#L12-L520) |

---

## See also

- package overview: [`simulation.md`](simulation.md)
