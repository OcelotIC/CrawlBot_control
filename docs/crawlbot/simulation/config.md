# `crawlbot.simulation.config`

**File**: [`crawlbot/simulation/config.py`](../../../crawlbot/simulation/config.py) — **507 lines** — canonical coverage **100 %**

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
|   `aocs_use_wrench_ff_in_ds` | `False` | _field_ | [L114](../../../crawlbot/simulation/config.py#L114) |
|   `ds_torso_ref_from_state` | `False` | _field_ | [L123](../../../crawlbot/simulation/config.py#L123) |
|   `aocs_off_in_ds` | `False` | _field_ | [L126](../../../crawlbot/simulation/config.py#L126) |
|   `aocs_active_in_interstep` | `True` | _field_ | [L141](../../../crawlbot/simulation/config.py#L141) |
|   `interstep_hw_refresh` | `True` | _field_ | [L155](../../../crawlbot/simulation/config.py#L155) |
|   `interstep_settle_alpha_wrench` | `0.0` | _field_ | [L165](../../../crawlbot/simulation/config.py#L165) |
|   `stop_on_failed_step` | `True` | _field_ | [L175](../../../crawlbot/simulation/config.py#L175) |
|   `frames_per_step` | `0` | _field_ | [L181](../../../crawlbot/simulation/config.py#L181) |
|   `use_m2_stack` | `False` | _field_ | [L189](../../../crawlbot/simulation/config.py#L189) |
|   `alpha_passivity` | `1.0` | _field_ | [L190](../../../crawlbot/simulation/config.py#L190) |
|   `enforce_hw_conservation` | `False` | _field_ | [L193](../../../crawlbot/simulation/config.py#L193) |
|   `h_max_tight` | `np.full(3, 5.0)` | _field_ | [L194](../../../crawlbot/simulation/config.py#L194) |
|   `w_L_nmpc` | `1.0` | _field_ | [L195](../../../crawlbot/simulation/config.py#L195) |
|   `kappa_terminal` | `1.0` | _field_ | [L196](../../../crawlbot/simulation/config.py#L196) |
|   `preplanner_M` | `15` | _field_ | [L208](../../../crawlbot/simulation/config.py#L208) |
|   `preplanner_kappa` | `0.7` | _field_ | [L209](../../../crawlbot/simulation/config.py#L209) |
|   `preplanner_f_max` | `25.0` | _field_ | [L210](../../../crawlbot/simulation/config.py#L210) |
|   `preplanner_tau_max` | `8.0` | _field_ | [L211](../../../crawlbot/simulation/config.py#L211) |
|   `preplanner_w_L` | `1.0` | _field_ | [L212](../../../crawlbot/simulation/config.py#L212) |
|   `preplanner_w_u` | `0.01` | _field_ | [L213](../../../crawlbot/simulation/config.py#L213) |
|   `preplanner_max_iter` | `300` | _field_ | [L214](../../../crawlbot/simulation/config.py#L214) |
|   `preplanner_a_cruise_max` | `0.0` | _field_ | [L215](../../../crawlbot/simulation/config.py#L215) |
|   `preplanner_cruise_ramp_frac` | `0.2` | _field_ | [L216](../../../crawlbot/simulation/config.py#L216) |
|   `preplanner_tstep_standoff_gain` | `0.0` | _field_ | [L224](../../../crawlbot/simulation/config.py#L224) |
|   `preplanner_tstep_standoff_knee` | `1000000000.0` | _field_ | [L225](../../../crawlbot/simulation/config.py#L225) |
|   `preplanner_tstep_scale_step` | `-1` | _field_ | [L229](../../../crawlbot/simulation/config.py#L229) |
|   `preplanner_tstep_scale_factor` | `1.0` | _field_ | [L230](../../../crawlbot/simulation/config.py#L230) |
|   `fsat_jitter_margin` | `0.05` | _field_ | [L240](../../../crawlbot/simulation/config.py#L240) |
|   `nmpc_N` | `8` | _field_ | [L243](../../../crawlbot/simulation/config.py#L243) |
|   `nmpc_dt` | `0.1` | _field_ | [L244](../../../crawlbot/simulation/config.py#L244) |
|   `nmpc_f_max` | `300.0` | _field_ | [L245](../../../crawlbot/simulation/config.py#L245) |
|   `nmpc_tau_max` | `8.0` | _field_ | [L246](../../../crawlbot/simulation/config.py#L246) |
|   `nmpc_Wv` | `10.0` | _field_ | [L247](../../../crawlbot/simulation/config.py#L247) |
|   `nmpc_p_max` | `50.0` | _field_ | [L248](../../../crawlbot/simulation/config.py#L248) |
|   `nmpc_Wr` | `100.0` | _field_ | [L254](../../../crawlbot/simulation/config.py#L254) |
|   `nmpc_Wu_f` | `0.01` | _field_ | [L255](../../../crawlbot/simulation/config.py#L255) |
|   `nmpc_Wu_tau` | `0.001` | _field_ | [L256](../../../crawlbot/simulation/config.py#L256) |
|   `nmpc_Qf_r` | `1000.0` | _field_ | [L257](../../../crawlbot/simulation/config.py#L257) |
|   `nmpc_Qf_v` | `100.0` | _field_ | [L258](../../../crawlbot/simulation/config.py#L258) |
|   `nmpc_Qf_L` | `10.0` | _field_ | [L259](../../../crawlbot/simulation/config.py#L259) |
|   `t_settle_final` | `20.0` | _field_ | [L260](../../../crawlbot/simulation/config.py#L260) |
|   `t_settle_inter` | `0.0` | _field_ | [L267](../../../crawlbot/simulation/config.py#L267) |
|   `use_energy_settle_inter` | `True` | _field_ | [L268](../../../crawlbot/simulation/config.py#L268) |
|   `settle_inter_epsilon_v` | `0.001` | _field_ | [L269](../../../crawlbot/simulation/config.py#L269) |
|   `interstep_settle_epsilon_v` | `0.0` | _field_ | [L277](../../../crawlbot/simulation/config.py#L277) |
|   `n_settle_inter_max_steps` | `500` | _field_ | [L278](../../../crawlbot/simulation/config.py#L278) |
|   `t_settle_inter_min` | `0.1` | _field_ | [L279](../../../crawlbot/simulation/config.py#L279) |
|   `ss_alpha_ee` | `1000.0` | _field_ | [L282](../../../crawlbot/simulation/config.py#L282) |
|   `ss_alpha_posture` | `20.0` | _field_ | [L283](../../../crawlbot/simulation/config.py#L283) |
|   `ss_alpha_wrench` | `1.0` | _field_ | [L284](../../../crawlbot/simulation/config.py#L284) |
|   `ss_alpha_lambda_int` | `0.0` | _field_ | [L285](../../../crawlbot/simulation/config.py#L285) |
|   `ss_alpha_mom` | `400.0` | _field_ | [L290](../../../crawlbot/simulation/config.py#L290) |
|   `log_hifreq_ss` | `False` | _field_ | [L294](../../../crawlbot/simulation/config.py#L294) |
|   `ss_two_task_mode` | `False` | _field_ | [L302](../../../crawlbot/simulation/config.py#L302) |
|   `alpha_torso_pose` | `2000.0` | _field_ | [L303](../../../crawlbot/simulation/config.py#L303) |
|   `dt_ds` | `0.5` | _field_ | [L312](../../../crawlbot/simulation/config.py#L312) |
|   `dock_hold_passivity_on` | `False` | _field_ | [L326](../../../crawlbot/simulation/config.py#L326) |
|   `passivity_W_budget` | `0.0` | _field_ | [L327](../../../crawlbot/simulation/config.py#L327) |
|   `log_dock_work` | `False` | _field_ | [L328](../../../crawlbot/simulation/config.py#L328) |
|   `qp_envelope_exact` | `False` | _field_ | [L338](../../../crawlbot/simulation/config.py#L338) |
|   `ds_centroidal_mode` | `False` | _field_ | [L343](../../../crawlbot/simulation/config.py#L343) |
|   `ds_alpha_com` | `100.0` | _field_ | [L344](../../../crawlbot/simulation/config.py#L344) |
|   `ds_alpha_torso_ori` | `200.0` | _field_ | [L345](../../../crawlbot/simulation/config.py#L345) |
|   `ds_alpha_posture` | `50.0` | _field_ | [L346](../../../crawlbot/simulation/config.py#L346) |
|   `ss_Kp_com` | `3.0` | _field_ | [L349](../../../crawlbot/simulation/config.py#L349) |
|   `ss_Kd_com` | `3.0` | _field_ | [L350](../../../crawlbot/simulation/config.py#L350) |
|   `ss_Kp_torso` | `6.0` | _field_ | [L351](../../../crawlbot/simulation/config.py#L351) |
|   `ss_Kd_torso` | `5.0` | _field_ | [L352](../../../crawlbot/simulation/config.py#L352) |
|   `ss_Kp_ee` | `10.0` | _field_ | [L353](../../../crawlbot/simulation/config.py#L353) |
|   `ss_Kd_ee` | `12.0` | _field_ | [L354](../../../crawlbot/simulation/config.py#L354) |
|   `ss_Kp_ee_ang` | `6.0` | _field_ | [L355](../../../crawlbot/simulation/config.py#L355) |
|   `ss_Kd_ee_ang` | `4.5` | _field_ | [L356](../../../crawlbot/simulation/config.py#L356) |
|   `swing_clearance` | `0.03` | _field_ | [L359](../../../crawlbot/simulation/config.py#L359) |
|   `swing_bump_peak_tau` | `0.5` | _field_ | [L365](../../../crawlbot/simulation/config.py#L365) |
|   `ik_fixed_rotation` | `True` | _field_ | [L376](../../../crawlbot/simulation/config.py#L376) |
|   `ik_fixed_rotation_w_min` | `0.0001` | _field_ | [L377](../../../crawlbot/simulation/config.py#L377) |
|   `ik_level_axis` | `None` | _field_ | [L392](../../../crawlbot/simulation/config.py#L392) |
|   `ik_q_nominal` | `None` | _field_ | [L393](../../../crawlbot/simulation/config.py#L393) |
|   `ik_w_posture` | `0.0` | _field_ | [L394](../../../crawlbot/simulation/config.py#L394) |
|   `use_com_z_standoff` | `False` | _field_ | [L407](../../../crawlbot/simulation/config.py#L407) |
|   `com_z_standoff` | `-0.35` | _field_ | [L408](../../../crawlbot/simulation/config.py#L408) |
|   `torso_early_finish_fraction` | `1.0` | _field_ | [L433](../../../crawlbot/simulation/config.py#L433) |
|   `swing_early_finish_fraction` | `1.0` | _field_ | [L442](../../../crawlbot/simulation/config.py#L442) |
|   `n_settle_steps` | `500` | _field_ | [L445](../../../crawlbot/simulation/config.py#L445) |
|   `Kd_settle_damping` | `20.0` | _field_ | [L456](../../../crawlbot/simulation/config.py#L456) |
|   `n_settle_max_steps` | `1000` | _field_ | [L457](../../../crawlbot/simulation/config.py#L457) |
|   `settle_epsilon_v` | `0.001` | _field_ | [L458](../../../crawlbot/simulation/config.py#L458) |
|   `settle_plateau_ratio` | `0.999` | _field_ | [L459](../../../crawlbot/simulation/config.py#L459) |
|   `diag_freeze_torso_ref_on_abort` | `False` | _field_ | [L466](../../../crawlbot/simulation/config.py#L466) |
|   `diag_force_single_contact_on_abort` | `False` | _field_ | [L472](../../../crawlbot/simulation/config.py#L472) |
|   `diag_disable_passivity_on_abort` | `False` | _field_ | [L478](../../../crawlbot/simulation/config.py#L478) |
|   `mapping_bypass_in_ss` | `False` | _field_ | [L484](../../../crawlbot/simulation/config.py#L484) |
|   `ds_ramp_duration_s` | `2.0` | _field_ | [L494](../../../crawlbot/simulation/config.py#L494) |
|   `gait_anchor_dx` | `0.8` | _field_ | [L506](../../../crawlbot/simulation/config.py#L506) |

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
| `class SimConfig` | [L12-506](../../../crawlbot/simulation/config.py#L12-L506) |

---

## See also

- package overview: [`simulation.md`](simulation.md)
