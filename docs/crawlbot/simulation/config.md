# `crawlbot.simulation.config`

**File**: [`crawlbot/simulation/config.py`](../../../crawlbot/simulation/config.py) — **561 lines** — canonical coverage **100 %**

> Module docstring: *"Simulation configuration dataclass."*

`SimConfig` — the single tuning surface. Every adjustable parameter in the
controller lives here, with its unit and justification. **100 % coverage.**

---

## Public API

| symbol | signature | canonical? | code |
|---|---|---|---|
| **`SimConfig`** *(dataclass)* |  |  | [L12](../../../crawlbot/simulation/config.py#L12) |
|   `nmpc_period` | `0.1` | _field_ | [L37](../../../crawlbot/simulation/config.py#L37) |
|   `dt_qp` | `0.01` | _field_ | [L38](../../../crawlbot/simulation/config.py#L38) |
|   `t_ss_margin` | `1.0` | _field_ | [L39](../../../crawlbot/simulation/config.py#L39) |
|   `t_hold_max` | `3.0` | _field_ | [L40](../../../crawlbot/simulation/config.py#L40) |
|   `dock_check_delay` | `0.5` | _field_ | [L41](../../../crawlbot/simulation/config.py#L41) |
|   `n_ds_max_steps` | `1000` | _field_ | [L42](../../../crawlbot/simulation/config.py#L42) |
|   `tau_max` | `20.0` | _field_ | [L45](../../../crawlbot/simulation/config.py#L45) |
|   `weld_radius` | `0.005` | _field_ | [L48](../../../crawlbot/simulation/config.py#L48) |
|   `dock_vel_max` | `0.01` | _field_ | [L49](../../../crawlbot/simulation/config.py#L49) |
|   `dock_ori_threshold_deg` | `5.0` | _field_ | [L55](../../../crawlbot/simulation/config.py#L55) |
|   `dock_use_6d_twist` | `True` | _field_ | [L70](../../../crawlbot/simulation/config.py#L70) |
|   `dock_twist_max` | `0.05` | _field_ | [L71](../../../crawlbot/simulation/config.py#L71) |
|   `gmo_K_O` | `80.0` | _field_ | [L74](../../../crawlbot/simulation/config.py#L74) |
|   `gmo_F_threshold` | `5.0` | _field_ | [L75](../../../crawlbot/simulation/config.py#L75) |
|   `gmo_d_proximity` | `0.02` | _field_ | [L76](../../../crawlbot/simulation/config.py#L76) |
|   `gmo_d_contact` | `0.005` | _field_ | [L77](../../../crawlbot/simulation/config.py#L77) |
|   `gmo_d_reset` | `0.03` | _field_ | [L78](../../../crawlbot/simulation/config.py#L78) |
|   `gmo_debounce_count` | `3` | _field_ | [L79](../../../crawlbot/simulation/config.py#L79) |
|   `hw_init` | `np.zeros(3)` | _field_ | [L82](../../../crawlbot/simulation/config.py#L82) |
|   `hw_min` | `np.full(3, -5.0)` | _field_ | [L83](../../../crawlbot/simulation/config.py#L83) |
|   `hw_max` | `np.full(3, 5.0)` | _field_ | [L84](../../../crawlbot/simulation/config.py#L84) |
|   `hw_qp_tight` | `np.full(3, 3.0)` | _field_ | [L90](../../../crawlbot/simulation/config.py#L90) |
|   `L_max` | `10.0` | _field_ | [L91](../../../crawlbot/simulation/config.py#L91) |
|   `tau_w_max` | `2.5` | _field_ | [L92](../../../crawlbot/simulation/config.py#L92) |
|   `aocs_K_hw` | `2.0` | _field_ | [L95](../../../crawlbot/simulation/config.py#L95) |
|   `aocs_tau_w_max` | `2.5` | _field_ | [L96](../../../crawlbot/simulation/config.py#L96) |
|   `rwa_I_w` | `0.01` | _field_ | [L97](../../../crawlbot/simulation/config.py#L97) |
|   `aocs_mode` | `'legacy'` | _field_ | [L111](../../../crawlbot/simulation/config.py#L111) |
|   `aocs_use_H_estimator` | `False` | _field_ | [L112](../../../crawlbot/simulation/config.py#L112) |
|   `aocs_use_legacy_corrected` | `False` | _field_ | [L113](../../../crawlbot/simulation/config.py#L113) |
|   `aocs_filter_tau` | `0.016` | _field_ | [L114](../../../crawlbot/simulation/config.py#L114) |
|   `aocs_K_omega` | `50.0` | _field_ | [L115](../../../crawlbot/simulation/config.py#L115) |
|   `aocs_K_d` | `25.0` | _field_ | [L116](../../../crawlbot/simulation/config.py#L116) |
|   `aocs_K_theta` | `1.0` | _field_ | [L117](../../../crawlbot/simulation/config.py#L117) |
|   `aocs_K_h` | `0.5` | _field_ | [L118](../../../crawlbot/simulation/config.py#L118) |
|   `aocs_hw_target` | `np.zeros(3)` | _field_ | [L119](../../../crawlbot/simulation/config.py#L119) |
|   `aocs_use_wrench_ff_in_ds` | `False` | _field_ | [L127](../../../crawlbot/simulation/config.py#L127) |
|   `ds_torso_ref_from_state` | `False` | _field_ | [L136](../../../crawlbot/simulation/config.py#L136) |
|   `aocs_off_in_ds` | `False` | _field_ | [L139](../../../crawlbot/simulation/config.py#L139) |
|   `aocs_active_in_interstep` | `True` | _field_ | [L154](../../../crawlbot/simulation/config.py#L154) |
|   `interstep_hw_refresh` | `True` | _field_ | [L168](../../../crawlbot/simulation/config.py#L168) |
|   `interstep_settle_alpha_wrench` | `0.0` | _field_ | [L178](../../../crawlbot/simulation/config.py#L178) |
|   `stop_on_failed_step` | `True` | _field_ | [L188](../../../crawlbot/simulation/config.py#L188) |
|   `frames_per_step` | `0` | _field_ | [L194](../../../crawlbot/simulation/config.py#L194) |
|   `use_m2_stack` | `False` | _field_ | [L202](../../../crawlbot/simulation/config.py#L202) |
|   `alpha_passivity` | `1.0` | _field_ | [L203](../../../crawlbot/simulation/config.py#L203) |
|   `enforce_hw_conservation` | `False` | _field_ | [L206](../../../crawlbot/simulation/config.py#L206) |
|   `h_max_tight` | `np.full(3, 5.0)` | _field_ | [L207](../../../crawlbot/simulation/config.py#L207) |
|   `w_L_nmpc` | `1.0` | _field_ | [L208](../../../crawlbot/simulation/config.py#L208) |
|   `kappa_terminal` | `1.0` | _field_ | [L209](../../../crawlbot/simulation/config.py#L209) |
|   `preplanner_M` | `15` | _field_ | [L221](../../../crawlbot/simulation/config.py#L221) |
|   `preplanner_kappa` | `0.7` | _field_ | [L222](../../../crawlbot/simulation/config.py#L222) |
|   `preplanner_f_max` | `25.0` | _field_ | [L223](../../../crawlbot/simulation/config.py#L223) |
|   `preplanner_tau_max` | `8.0` | _field_ | [L224](../../../crawlbot/simulation/config.py#L224) |
|   `preplanner_w_L` | `1.0` | _field_ | [L225](../../../crawlbot/simulation/config.py#L225) |
|   `preplanner_w_u` | `0.01` | _field_ | [L226](../../../crawlbot/simulation/config.py#L226) |
|   `preplanner_max_iter` | `300` | _field_ | [L227](../../../crawlbot/simulation/config.py#L227) |
|   `preplanner_a_cruise_max` | `0.0` | _field_ | [L228](../../../crawlbot/simulation/config.py#L228) |
|   `preplanner_cruise_ramp_frac` | `0.2` | _field_ | [L229](../../../crawlbot/simulation/config.py#L229) |
|   `preplanner_tstep_standoff_gain` | `0.0` | _field_ | [L237](../../../crawlbot/simulation/config.py#L237) |
|   `preplanner_tstep_standoff_knee` | `1000000000.0` | _field_ | [L238](../../../crawlbot/simulation/config.py#L238) |
|   `preplanner_tstep_scale_step` | `-1` | _field_ | [L242](../../../crawlbot/simulation/config.py#L242) |
|   `preplanner_tstep_scale_factor` | `1.0` | _field_ | [L243](../../../crawlbot/simulation/config.py#L243) |
|   `fsat_jitter_margin` | `0.05` | _field_ | [L253](../../../crawlbot/simulation/config.py#L253) |
|   `nmpc_N` | `20` | _field_ | [L280](../../../crawlbot/simulation/config.py#L280) |
|   `nmpc_pred_dt` | `0.1` | _field_ | [L286](../../../crawlbot/simulation/config.py#L286) |
|   `nmpc_per_stage_refs` | `True` | _field_ | [L298](../../../crawlbot/simulation/config.py#L298) |
|   `nmpc_f_max` | `300.0` | _field_ | [L299](../../../crawlbot/simulation/config.py#L299) |
|   `nmpc_tau_max` | `8.0` | _field_ | [L300](../../../crawlbot/simulation/config.py#L300) |
|   `nmpc_Wv` | `10.0` | _field_ | [L301](../../../crawlbot/simulation/config.py#L301) |
|   `nmpc_p_max` | `50.0` | _field_ | [L302](../../../crawlbot/simulation/config.py#L302) |
|   `nmpc_Wr` | `100.0` | _field_ | [L308](../../../crawlbot/simulation/config.py#L308) |
|   `nmpc_Wu_f` | `0.01` | _field_ | [L309](../../../crawlbot/simulation/config.py#L309) |
|   `nmpc_Wu_tau` | `0.001` | _field_ | [L310](../../../crawlbot/simulation/config.py#L310) |
|   `nmpc_Qf_r` | `1000.0` | _field_ | [L311](../../../crawlbot/simulation/config.py#L311) |
|   `nmpc_Qf_v` | `100.0` | _field_ | [L312](../../../crawlbot/simulation/config.py#L312) |
|   `nmpc_Qf_L` | `10.0` | _field_ | [L313](../../../crawlbot/simulation/config.py#L313) |
|   `t_settle_final` | `20.0` | _field_ | [L314](../../../crawlbot/simulation/config.py#L314) |
|   `t_settle_inter` | `0.0` | _field_ | [L321](../../../crawlbot/simulation/config.py#L321) |
|   `use_energy_settle_inter` | `True` | _field_ | [L322](../../../crawlbot/simulation/config.py#L322) |
|   `settle_inter_epsilon_v` | `0.001` | _field_ | [L323](../../../crawlbot/simulation/config.py#L323) |
|   `interstep_settle_epsilon_v` | `0.0` | _field_ | [L331](../../../crawlbot/simulation/config.py#L331) |
|   `n_settle_inter_max_steps` | `500` | _field_ | [L332](../../../crawlbot/simulation/config.py#L332) |
|   `t_settle_inter_min` | `0.1` | _field_ | [L333](../../../crawlbot/simulation/config.py#L333) |
|   `ss_alpha_ee` | `1000.0` | _field_ | [L336](../../../crawlbot/simulation/config.py#L336) |
|   `ss_alpha_posture` | `20.0` | _field_ | [L337](../../../crawlbot/simulation/config.py#L337) |
|   `ss_alpha_wrench` | `1.0` | _field_ | [L338](../../../crawlbot/simulation/config.py#L338) |
|   `ss_alpha_lambda_int` | `0.0` | _field_ | [L339](../../../crawlbot/simulation/config.py#L339) |
|   `ss_alpha_mom` | `400.0` | _field_ | [L344](../../../crawlbot/simulation/config.py#L344) |
|   `log_hifreq_ss` | `False` | _field_ | [L348](../../../crawlbot/simulation/config.py#L348) |
|   `ss_two_task_mode` | `False` | _field_ | [L356](../../../crawlbot/simulation/config.py#L356) |
|   `alpha_torso_pose` | `2000.0` | _field_ | [L357](../../../crawlbot/simulation/config.py#L357) |
|   `dt_ds` | `0.5` | _field_ | [L366](../../../crawlbot/simulation/config.py#L366) |
|   `dock_hold_passivity_on` | `False` | _field_ | [L380](../../../crawlbot/simulation/config.py#L380) |
|   `passivity_W_budget` | `0.0` | _field_ | [L381](../../../crawlbot/simulation/config.py#L381) |
|   `log_dock_work` | `False` | _field_ | [L382](../../../crawlbot/simulation/config.py#L382) |
|   `qp_envelope_exact` | `False` | _field_ | [L392](../../../crawlbot/simulation/config.py#L392) |
|   `ds_centroidal_mode` | `False` | _field_ | [L397](../../../crawlbot/simulation/config.py#L397) |
|   `ds_alpha_com` | `100.0` | _field_ | [L398](../../../crawlbot/simulation/config.py#L398) |
|   `ds_alpha_torso_ori` | `200.0` | _field_ | [L399](../../../crawlbot/simulation/config.py#L399) |
|   `ds_alpha_posture` | `50.0` | _field_ | [L400](../../../crawlbot/simulation/config.py#L400) |
|   `ss_Kp_com` | `3.0` | _field_ | [L403](../../../crawlbot/simulation/config.py#L403) |
|   `ss_Kd_com` | `3.0` | _field_ | [L404](../../../crawlbot/simulation/config.py#L404) |
|   `ss_Kp_torso` | `6.0` | _field_ | [L405](../../../crawlbot/simulation/config.py#L405) |
|   `ss_Kd_torso` | `5.0` | _field_ | [L406](../../../crawlbot/simulation/config.py#L406) |
|   `ss_Kp_ee` | `10.0` | _field_ | [L407](../../../crawlbot/simulation/config.py#L407) |
|   `ss_Kd_ee` | `12.0` | _field_ | [L408](../../../crawlbot/simulation/config.py#L408) |
|   `ss_Kp_ee_ang` | `6.0` | _field_ | [L409](../../../crawlbot/simulation/config.py#L409) |
|   `ss_Kd_ee_ang` | `4.5` | _field_ | [L410](../../../crawlbot/simulation/config.py#L410) |
|   `swing_clearance` | `0.03` | _field_ | [L413](../../../crawlbot/simulation/config.py#L413) |
|   `swing_bump_peak_tau` | `0.5` | _field_ | [L419](../../../crawlbot/simulation/config.py#L419) |
|   `ik_fixed_rotation` | `True` | _field_ | [L430](../../../crawlbot/simulation/config.py#L430) |
|   `ik_fixed_rotation_w_min` | `0.0001` | _field_ | [L431](../../../crawlbot/simulation/config.py#L431) |
|   `ik_level_axis` | `None` | _field_ | [L446](../../../crawlbot/simulation/config.py#L446) |
|   `ik_q_nominal` | `None` | _field_ | [L447](../../../crawlbot/simulation/config.py#L447) |
|   `ik_w_posture` | `0.0` | _field_ | [L448](../../../crawlbot/simulation/config.py#L448) |
|   `use_com_z_standoff` | `False` | _field_ | [L461](../../../crawlbot/simulation/config.py#L461) |
|   `com_z_standoff` | `-0.35` | _field_ | [L462](../../../crawlbot/simulation/config.py#L462) |
|   `torso_early_finish_fraction` | `1.0` | _field_ | [L487](../../../crawlbot/simulation/config.py#L487) |
|   `swing_early_finish_fraction` | `1.0` | _field_ | [L496](../../../crawlbot/simulation/config.py#L496) |
|   `n_settle_steps` | `500` | _field_ | [L499](../../../crawlbot/simulation/config.py#L499) |
|   `Kd_settle_damping` | `20.0` | _field_ | [L510](../../../crawlbot/simulation/config.py#L510) |
|   `n_settle_max_steps` | `1000` | _field_ | [L511](../../../crawlbot/simulation/config.py#L511) |
|   `settle_epsilon_v` | `0.001` | _field_ | [L512](../../../crawlbot/simulation/config.py#L512) |
|   `settle_plateau_ratio` | `0.999` | _field_ | [L513](../../../crawlbot/simulation/config.py#L513) |
|   `diag_freeze_torso_ref_on_abort` | `False` | _field_ | [L520](../../../crawlbot/simulation/config.py#L520) |
|   `diag_force_single_contact_on_abort` | `False` | _field_ | [L526](../../../crawlbot/simulation/config.py#L526) |
|   `diag_disable_passivity_on_abort` | `False` | _field_ | [L532](../../../crawlbot/simulation/config.py#L532) |
|   `mapping_bypass_in_ss` | `False` | _field_ | [L538](../../../crawlbot/simulation/config.py#L538) |
|   `ds_ramp_duration_s` | `2.0` | _field_ | [L548](../../../crawlbot/simulation/config.py#L548) |
|   `gait_anchor_dx` | `0.8` | _field_ | [L560](../../../crawlbot/simulation/config.py#L560) |

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

## 5. `nmpc_N` is two knobs wearing one name

`nmpc_N` (currently **15**, raised from 8) sets the NMPC prediction horizon —
and, because the NMPC holds a **constant** reference across the whole horizon,
it also sets how far ahead the references are sampled. Three sites key on it:

| site | expression | at N=8 | at N=15 |
|---|---|---|---|
| `sim_loop.py:2131` | `t_horizon = t + N·dt` → CoM reference | +0.8 s | **+1.5 s** |
| `sim_loop.py:2148` | `tau_rel = t_horizon − t0` → coarse-preplanner query | +0.8 s | **+1.5 s** |
| `sim_loop.py:2210` | `t_mid = t + N·dt/2` → `L_com` reference | +0.4 s | **+0.75 s** |

So raising `N` lengthens the prediction *and* pushes the target the NMPC chases
further into the future. **They are not separable through this field.** If you
need the horizon without the reference lead (or vice versa), the sampling
expressions have to be decoupled from `nmpc_N` first — otherwise any A/B on
"horizon length" is confounded.

The measured effect of 8 → 15 is in
`results/j2_adjconv/NMPC_HORIZON_N15.md`: docks improve (worst margin 0.02 →
0.07 mm), `h_w` peak falls, but `e_com` peak rises 0.154 → 0.190 m — consistent
with the reference lead, not with the horizon. Solve time roughly doubles and
**one solve in 634 exceeds the 100 ms NMPC period**, which is a real-time
concern for deployment even though the offline sim is unaffected.

## Code map

| unit | source |
|---|---|
| `class SimConfig` | [L12-560](../../../crawlbot/simulation/config.py#L12-L560) |

---

## See also

- package overview: [`simulation.md`](simulation.md)
