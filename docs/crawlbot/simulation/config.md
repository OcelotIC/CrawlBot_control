# `crawlbot.simulation.config`

**File**: [`crawlbot/simulation/config.py`](../../../crawlbot/simulation/config.py) — **528 lines** — canonical coverage **100 %**

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
|   `nmpc_N` | `20` | _field_ | [L252](../../../crawlbot/simulation/config.py#L252) |
|   `nmpc_dt` | `0.1` | _field_ | [L253](../../../crawlbot/simulation/config.py#L253) |
|   `nmpc_per_stage_refs` | `True` | _field_ | [L265](../../../crawlbot/simulation/config.py#L265) |
|   `nmpc_f_max` | `300.0` | _field_ | [L266](../../../crawlbot/simulation/config.py#L266) |
|   `nmpc_tau_max` | `8.0` | _field_ | [L267](../../../crawlbot/simulation/config.py#L267) |
|   `nmpc_Wv` | `10.0` | _field_ | [L268](../../../crawlbot/simulation/config.py#L268) |
|   `nmpc_p_max` | `50.0` | _field_ | [L269](../../../crawlbot/simulation/config.py#L269) |
|   `nmpc_Wr` | `100.0` | _field_ | [L275](../../../crawlbot/simulation/config.py#L275) |
|   `nmpc_Wu_f` | `0.01` | _field_ | [L276](../../../crawlbot/simulation/config.py#L276) |
|   `nmpc_Wu_tau` | `0.001` | _field_ | [L277](../../../crawlbot/simulation/config.py#L277) |
|   `nmpc_Qf_r` | `1000.0` | _field_ | [L278](../../../crawlbot/simulation/config.py#L278) |
|   `nmpc_Qf_v` | `100.0` | _field_ | [L279](../../../crawlbot/simulation/config.py#L279) |
|   `nmpc_Qf_L` | `10.0` | _field_ | [L280](../../../crawlbot/simulation/config.py#L280) |
|   `t_settle_final` | `20.0` | _field_ | [L281](../../../crawlbot/simulation/config.py#L281) |
|   `t_settle_inter` | `0.0` | _field_ | [L288](../../../crawlbot/simulation/config.py#L288) |
|   `use_energy_settle_inter` | `True` | _field_ | [L289](../../../crawlbot/simulation/config.py#L289) |
|   `settle_inter_epsilon_v` | `0.001` | _field_ | [L290](../../../crawlbot/simulation/config.py#L290) |
|   `interstep_settle_epsilon_v` | `0.0` | _field_ | [L298](../../../crawlbot/simulation/config.py#L298) |
|   `n_settle_inter_max_steps` | `500` | _field_ | [L299](../../../crawlbot/simulation/config.py#L299) |
|   `t_settle_inter_min` | `0.1` | _field_ | [L300](../../../crawlbot/simulation/config.py#L300) |
|   `ss_alpha_ee` | `1000.0` | _field_ | [L303](../../../crawlbot/simulation/config.py#L303) |
|   `ss_alpha_posture` | `20.0` | _field_ | [L304](../../../crawlbot/simulation/config.py#L304) |
|   `ss_alpha_wrench` | `1.0` | _field_ | [L305](../../../crawlbot/simulation/config.py#L305) |
|   `ss_alpha_lambda_int` | `0.0` | _field_ | [L306](../../../crawlbot/simulation/config.py#L306) |
|   `ss_alpha_mom` | `400.0` | _field_ | [L311](../../../crawlbot/simulation/config.py#L311) |
|   `log_hifreq_ss` | `False` | _field_ | [L315](../../../crawlbot/simulation/config.py#L315) |
|   `ss_two_task_mode` | `False` | _field_ | [L323](../../../crawlbot/simulation/config.py#L323) |
|   `alpha_torso_pose` | `2000.0` | _field_ | [L324](../../../crawlbot/simulation/config.py#L324) |
|   `dt_ds` | `0.5` | _field_ | [L333](../../../crawlbot/simulation/config.py#L333) |
|   `dock_hold_passivity_on` | `False` | _field_ | [L347](../../../crawlbot/simulation/config.py#L347) |
|   `passivity_W_budget` | `0.0` | _field_ | [L348](../../../crawlbot/simulation/config.py#L348) |
|   `log_dock_work` | `False` | _field_ | [L349](../../../crawlbot/simulation/config.py#L349) |
|   `qp_envelope_exact` | `False` | _field_ | [L359](../../../crawlbot/simulation/config.py#L359) |
|   `ds_centroidal_mode` | `False` | _field_ | [L364](../../../crawlbot/simulation/config.py#L364) |
|   `ds_alpha_com` | `100.0` | _field_ | [L365](../../../crawlbot/simulation/config.py#L365) |
|   `ds_alpha_torso_ori` | `200.0` | _field_ | [L366](../../../crawlbot/simulation/config.py#L366) |
|   `ds_alpha_posture` | `50.0` | _field_ | [L367](../../../crawlbot/simulation/config.py#L367) |
|   `ss_Kp_com` | `3.0` | _field_ | [L370](../../../crawlbot/simulation/config.py#L370) |
|   `ss_Kd_com` | `3.0` | _field_ | [L371](../../../crawlbot/simulation/config.py#L371) |
|   `ss_Kp_torso` | `6.0` | _field_ | [L372](../../../crawlbot/simulation/config.py#L372) |
|   `ss_Kd_torso` | `5.0` | _field_ | [L373](../../../crawlbot/simulation/config.py#L373) |
|   `ss_Kp_ee` | `10.0` | _field_ | [L374](../../../crawlbot/simulation/config.py#L374) |
|   `ss_Kd_ee` | `12.0` | _field_ | [L375](../../../crawlbot/simulation/config.py#L375) |
|   `ss_Kp_ee_ang` | `6.0` | _field_ | [L376](../../../crawlbot/simulation/config.py#L376) |
|   `ss_Kd_ee_ang` | `4.5` | _field_ | [L377](../../../crawlbot/simulation/config.py#L377) |
|   `swing_clearance` | `0.03` | _field_ | [L380](../../../crawlbot/simulation/config.py#L380) |
|   `swing_bump_peak_tau` | `0.5` | _field_ | [L386](../../../crawlbot/simulation/config.py#L386) |
|   `ik_fixed_rotation` | `True` | _field_ | [L397](../../../crawlbot/simulation/config.py#L397) |
|   `ik_fixed_rotation_w_min` | `0.0001` | _field_ | [L398](../../../crawlbot/simulation/config.py#L398) |
|   `ik_level_axis` | `None` | _field_ | [L413](../../../crawlbot/simulation/config.py#L413) |
|   `ik_q_nominal` | `None` | _field_ | [L414](../../../crawlbot/simulation/config.py#L414) |
|   `ik_w_posture` | `0.0` | _field_ | [L415](../../../crawlbot/simulation/config.py#L415) |
|   `use_com_z_standoff` | `False` | _field_ | [L428](../../../crawlbot/simulation/config.py#L428) |
|   `com_z_standoff` | `-0.35` | _field_ | [L429](../../../crawlbot/simulation/config.py#L429) |
|   `torso_early_finish_fraction` | `1.0` | _field_ | [L454](../../../crawlbot/simulation/config.py#L454) |
|   `swing_early_finish_fraction` | `1.0` | _field_ | [L463](../../../crawlbot/simulation/config.py#L463) |
|   `n_settle_steps` | `500` | _field_ | [L466](../../../crawlbot/simulation/config.py#L466) |
|   `Kd_settle_damping` | `20.0` | _field_ | [L477](../../../crawlbot/simulation/config.py#L477) |
|   `n_settle_max_steps` | `1000` | _field_ | [L478](../../../crawlbot/simulation/config.py#L478) |
|   `settle_epsilon_v` | `0.001` | _field_ | [L479](../../../crawlbot/simulation/config.py#L479) |
|   `settle_plateau_ratio` | `0.999` | _field_ | [L480](../../../crawlbot/simulation/config.py#L480) |
|   `diag_freeze_torso_ref_on_abort` | `False` | _field_ | [L487](../../../crawlbot/simulation/config.py#L487) |
|   `diag_force_single_contact_on_abort` | `False` | _field_ | [L493](../../../crawlbot/simulation/config.py#L493) |
|   `diag_disable_passivity_on_abort` | `False` | _field_ | [L499](../../../crawlbot/simulation/config.py#L499) |
|   `mapping_bypass_in_ss` | `False` | _field_ | [L505](../../../crawlbot/simulation/config.py#L505) |
|   `ds_ramp_duration_s` | `2.0` | _field_ | [L515](../../../crawlbot/simulation/config.py#L515) |
|   `gait_anchor_dx` | `0.8` | _field_ | [L527](../../../crawlbot/simulation/config.py#L527) |

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
| `class SimConfig` | [L12-527](../../../crawlbot/simulation/config.py#L12-L527) |

---

## See also

- package overview: [`simulation.md`](simulation.md)
