# `crawlbot.simulation.config`

**File**: [`crawlbot/simulation/config.py`](../../../crawlbot/simulation/config.py) — **566 lines** — canonical coverage **100 %**

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
|   `enforce_hw_terminal` | `None` | _field_ | [L211](../../../crawlbot/simulation/config.py#L211) |
|   `h_max_tight` | `np.full(3, 5.0)` | _field_ | [L212](../../../crawlbot/simulation/config.py#L212) |
|   `w_L_nmpc` | `1.0` | _field_ | [L213](../../../crawlbot/simulation/config.py#L213) |
|   `kappa_terminal` | `1.0` | _field_ | [L214](../../../crawlbot/simulation/config.py#L214) |
|   `preplanner_M` | `15` | _field_ | [L226](../../../crawlbot/simulation/config.py#L226) |
|   `preplanner_kappa` | `0.7` | _field_ | [L227](../../../crawlbot/simulation/config.py#L227) |
|   `preplanner_f_max` | `25.0` | _field_ | [L228](../../../crawlbot/simulation/config.py#L228) |
|   `preplanner_tau_max` | `8.0` | _field_ | [L229](../../../crawlbot/simulation/config.py#L229) |
|   `preplanner_w_L` | `1.0` | _field_ | [L230](../../../crawlbot/simulation/config.py#L230) |
|   `preplanner_w_u` | `0.01` | _field_ | [L231](../../../crawlbot/simulation/config.py#L231) |
|   `preplanner_max_iter` | `300` | _field_ | [L232](../../../crawlbot/simulation/config.py#L232) |
|   `preplanner_a_cruise_max` | `0.0` | _field_ | [L233](../../../crawlbot/simulation/config.py#L233) |
|   `preplanner_cruise_ramp_frac` | `0.2` | _field_ | [L234](../../../crawlbot/simulation/config.py#L234) |
|   `preplanner_tstep_standoff_gain` | `0.0` | _field_ | [L242](../../../crawlbot/simulation/config.py#L242) |
|   `preplanner_tstep_standoff_knee` | `1000000000.0` | _field_ | [L243](../../../crawlbot/simulation/config.py#L243) |
|   `preplanner_tstep_scale_step` | `-1` | _field_ | [L247](../../../crawlbot/simulation/config.py#L247) |
|   `preplanner_tstep_scale_factor` | `1.0` | _field_ | [L248](../../../crawlbot/simulation/config.py#L248) |
|   `fsat_jitter_margin` | `0.05` | _field_ | [L258](../../../crawlbot/simulation/config.py#L258) |
|   `nmpc_N` | `20` | _field_ | [L285](../../../crawlbot/simulation/config.py#L285) |
|   `nmpc_pred_dt` | `0.1` | _field_ | [L291](../../../crawlbot/simulation/config.py#L291) |
|   `nmpc_per_stage_refs` | `True` | _field_ | [L303](../../../crawlbot/simulation/config.py#L303) |
|   `nmpc_f_max` | `300.0` | _field_ | [L304](../../../crawlbot/simulation/config.py#L304) |
|   `nmpc_tau_max` | `8.0` | _field_ | [L305](../../../crawlbot/simulation/config.py#L305) |
|   `nmpc_Wv` | `10.0` | _field_ | [L306](../../../crawlbot/simulation/config.py#L306) |
|   `nmpc_p_max` | `50.0` | _field_ | [L307](../../../crawlbot/simulation/config.py#L307) |
|   `nmpc_Wr` | `100.0` | _field_ | [L313](../../../crawlbot/simulation/config.py#L313) |
|   `nmpc_Wu_f` | `0.01` | _field_ | [L314](../../../crawlbot/simulation/config.py#L314) |
|   `nmpc_Wu_tau` | `0.001` | _field_ | [L315](../../../crawlbot/simulation/config.py#L315) |
|   `nmpc_Qf_r` | `1000.0` | _field_ | [L316](../../../crawlbot/simulation/config.py#L316) |
|   `nmpc_Qf_v` | `100.0` | _field_ | [L317](../../../crawlbot/simulation/config.py#L317) |
|   `nmpc_Qf_L` | `10.0` | _field_ | [L318](../../../crawlbot/simulation/config.py#L318) |
|   `t_settle_final` | `20.0` | _field_ | [L319](../../../crawlbot/simulation/config.py#L319) |
|   `t_settle_inter` | `0.0` | _field_ | [L326](../../../crawlbot/simulation/config.py#L326) |
|   `use_energy_settle_inter` | `True` | _field_ | [L327](../../../crawlbot/simulation/config.py#L327) |
|   `settle_inter_epsilon_v` | `0.001` | _field_ | [L328](../../../crawlbot/simulation/config.py#L328) |
|   `interstep_settle_epsilon_v` | `0.0` | _field_ | [L336](../../../crawlbot/simulation/config.py#L336) |
|   `n_settle_inter_max_steps` | `500` | _field_ | [L337](../../../crawlbot/simulation/config.py#L337) |
|   `t_settle_inter_min` | `0.1` | _field_ | [L338](../../../crawlbot/simulation/config.py#L338) |
|   `ss_alpha_ee` | `1000.0` | _field_ | [L341](../../../crawlbot/simulation/config.py#L341) |
|   `ss_alpha_posture` | `20.0` | _field_ | [L342](../../../crawlbot/simulation/config.py#L342) |
|   `ss_alpha_wrench` | `1.0` | _field_ | [L343](../../../crawlbot/simulation/config.py#L343) |
|   `ss_alpha_lambda_int` | `0.0` | _field_ | [L344](../../../crawlbot/simulation/config.py#L344) |
|   `ss_alpha_mom` | `400.0` | _field_ | [L349](../../../crawlbot/simulation/config.py#L349) |
|   `log_hifreq_ss` | `False` | _field_ | [L353](../../../crawlbot/simulation/config.py#L353) |
|   `ss_two_task_mode` | `False` | _field_ | [L361](../../../crawlbot/simulation/config.py#L361) |
|   `alpha_torso_pose` | `2000.0` | _field_ | [L362](../../../crawlbot/simulation/config.py#L362) |
|   `dt_ds` | `0.5` | _field_ | [L371](../../../crawlbot/simulation/config.py#L371) |
|   `dock_hold_passivity_on` | `False` | _field_ | [L385](../../../crawlbot/simulation/config.py#L385) |
|   `passivity_W_budget` | `0.0` | _field_ | [L386](../../../crawlbot/simulation/config.py#L386) |
|   `log_dock_work` | `False` | _field_ | [L387](../../../crawlbot/simulation/config.py#L387) |
|   `qp_envelope_exact` | `False` | _field_ | [L397](../../../crawlbot/simulation/config.py#L397) |
|   `ds_centroidal_mode` | `False` | _field_ | [L402](../../../crawlbot/simulation/config.py#L402) |
|   `ds_alpha_com` | `100.0` | _field_ | [L403](../../../crawlbot/simulation/config.py#L403) |
|   `ds_alpha_torso_ori` | `200.0` | _field_ | [L404](../../../crawlbot/simulation/config.py#L404) |
|   `ds_alpha_posture` | `50.0` | _field_ | [L405](../../../crawlbot/simulation/config.py#L405) |
|   `ss_Kp_com` | `3.0` | _field_ | [L408](../../../crawlbot/simulation/config.py#L408) |
|   `ss_Kd_com` | `3.0` | _field_ | [L409](../../../crawlbot/simulation/config.py#L409) |
|   `ss_Kp_torso` | `6.0` | _field_ | [L410](../../../crawlbot/simulation/config.py#L410) |
|   `ss_Kd_torso` | `5.0` | _field_ | [L411](../../../crawlbot/simulation/config.py#L411) |
|   `ss_Kp_ee` | `10.0` | _field_ | [L412](../../../crawlbot/simulation/config.py#L412) |
|   `ss_Kd_ee` | `12.0` | _field_ | [L413](../../../crawlbot/simulation/config.py#L413) |
|   `ss_Kp_ee_ang` | `6.0` | _field_ | [L414](../../../crawlbot/simulation/config.py#L414) |
|   `ss_Kd_ee_ang` | `4.5` | _field_ | [L415](../../../crawlbot/simulation/config.py#L415) |
|   `swing_clearance` | `0.03` | _field_ | [L418](../../../crawlbot/simulation/config.py#L418) |
|   `swing_bump_peak_tau` | `0.5` | _field_ | [L424](../../../crawlbot/simulation/config.py#L424) |
|   `ik_fixed_rotation` | `True` | _field_ | [L435](../../../crawlbot/simulation/config.py#L435) |
|   `ik_fixed_rotation_w_min` | `0.0001` | _field_ | [L436](../../../crawlbot/simulation/config.py#L436) |
|   `ik_level_axis` | `None` | _field_ | [L451](../../../crawlbot/simulation/config.py#L451) |
|   `ik_q_nominal` | `None` | _field_ | [L452](../../../crawlbot/simulation/config.py#L452) |
|   `ik_w_posture` | `0.0` | _field_ | [L453](../../../crawlbot/simulation/config.py#L453) |
|   `use_com_z_standoff` | `False` | _field_ | [L466](../../../crawlbot/simulation/config.py#L466) |
|   `com_z_standoff` | `-0.35` | _field_ | [L467](../../../crawlbot/simulation/config.py#L467) |
|   `torso_early_finish_fraction` | `1.0` | _field_ | [L492](../../../crawlbot/simulation/config.py#L492) |
|   `swing_early_finish_fraction` | `1.0` | _field_ | [L501](../../../crawlbot/simulation/config.py#L501) |
|   `n_settle_steps` | `500` | _field_ | [L504](../../../crawlbot/simulation/config.py#L504) |
|   `Kd_settle_damping` | `20.0` | _field_ | [L515](../../../crawlbot/simulation/config.py#L515) |
|   `n_settle_max_steps` | `1000` | _field_ | [L516](../../../crawlbot/simulation/config.py#L516) |
|   `settle_epsilon_v` | `0.001` | _field_ | [L517](../../../crawlbot/simulation/config.py#L517) |
|   `settle_plateau_ratio` | `0.999` | _field_ | [L518](../../../crawlbot/simulation/config.py#L518) |
|   `diag_freeze_torso_ref_on_abort` | `False` | _field_ | [L525](../../../crawlbot/simulation/config.py#L525) |
|   `diag_force_single_contact_on_abort` | `False` | _field_ | [L531](../../../crawlbot/simulation/config.py#L531) |
|   `diag_disable_passivity_on_abort` | `False` | _field_ | [L537](../../../crawlbot/simulation/config.py#L537) |
|   `mapping_bypass_in_ss` | `False` | _field_ | [L543](../../../crawlbot/simulation/config.py#L543) |
|   `ds_ramp_duration_s` | `2.0` | _field_ | [L553](../../../crawlbot/simulation/config.py#L553) |
|   `gait_anchor_dx` | `0.8` | _field_ | [L565](../../../crawlbot/simulation/config.py#L565) |

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
| `class SimConfig` | [L12-565](../../../crawlbot/simulation/config.py#L12-L565) |

---

## See also

- package overview: [`simulation.md`](simulation.md)
