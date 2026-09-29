# `crawlbot.simulation.config`

**File**: [`crawlbot/simulation/config.py`](../../../crawlbot/simulation/config.py) — **491 lines** — canonical coverage **100 %**

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
|   `aocs_mode` | `'legacy_pid_numerical'` | _field_ | [L90](../../../crawlbot/simulation/config.py#L90) |
|   `aocs_K_omega` | `50.0` | _field_ | [L91](../../../crawlbot/simulation/config.py#L91) |
|   `aocs_K_d` | `25.0` | _field_ | [L92](../../../crawlbot/simulation/config.py#L92) |
|   `aocs_K_theta` | `1.0` | _field_ | [L93](../../../crawlbot/simulation/config.py#L93) |
|   `aocs_use_wrench_ff_in_ds` | `False` | _field_ | [L101](../../../crawlbot/simulation/config.py#L101) |
|   `ds_torso_ref_from_state` | `False` | _field_ | [L110](../../../crawlbot/simulation/config.py#L110) |
|   `aocs_active_in_interstep` | `True` | _field_ | [L125](../../../crawlbot/simulation/config.py#L125) |
|   `interstep_hw_refresh` | `True` | _field_ | [L139](../../../crawlbot/simulation/config.py#L139) |
|   `interstep_settle_alpha_wrench` | `0.0` | _field_ | [L149](../../../crawlbot/simulation/config.py#L149) |
|   `stop_on_failed_step` | `True` | _field_ | [L159](../../../crawlbot/simulation/config.py#L159) |
|   `frames_per_step` | `0` | _field_ | [L165](../../../crawlbot/simulation/config.py#L165) |
|   `use_m2_stack` | `False` | _field_ | [L173](../../../crawlbot/simulation/config.py#L173) |
|   `alpha_passivity` | `1.0` | _field_ | [L174](../../../crawlbot/simulation/config.py#L174) |
|   `enforce_hw_conservation` | `False` | _field_ | [L177](../../../crawlbot/simulation/config.py#L177) |
|   `h_max_tight` | `np.full(3, 5.0)` | _field_ | [L178](../../../crawlbot/simulation/config.py#L178) |
|   `w_L_nmpc` | `1.0` | _field_ | [L179](../../../crawlbot/simulation/config.py#L179) |
|   `kappa_terminal` | `1.0` | _field_ | [L180](../../../crawlbot/simulation/config.py#L180) |
|   `preplanner_M` | `15` | _field_ | [L192](../../../crawlbot/simulation/config.py#L192) |
|   `preplanner_kappa` | `0.7` | _field_ | [L193](../../../crawlbot/simulation/config.py#L193) |
|   `preplanner_f_max` | `25.0` | _field_ | [L194](../../../crawlbot/simulation/config.py#L194) |
|   `preplanner_tau_max` | `8.0` | _field_ | [L195](../../../crawlbot/simulation/config.py#L195) |
|   `preplanner_w_L` | `1.0` | _field_ | [L196](../../../crawlbot/simulation/config.py#L196) |
|   `preplanner_w_u` | `0.01` | _field_ | [L197](../../../crawlbot/simulation/config.py#L197) |
|   `preplanner_max_iter` | `300` | _field_ | [L198](../../../crawlbot/simulation/config.py#L198) |
|   `preplanner_a_cruise_max` | `0.0` | _field_ | [L199](../../../crawlbot/simulation/config.py#L199) |
|   `preplanner_cruise_ramp_frac` | `0.2` | _field_ | [L200](../../../crawlbot/simulation/config.py#L200) |
|   `preplanner_tstep_standoff_gain` | `0.0` | _field_ | [L208](../../../crawlbot/simulation/config.py#L208) |
|   `preplanner_tstep_standoff_knee` | `1000000000.0` | _field_ | [L209](../../../crawlbot/simulation/config.py#L209) |
|   `preplanner_tstep_scale_step` | `-1` | _field_ | [L213](../../../crawlbot/simulation/config.py#L213) |
|   `preplanner_tstep_scale_factor` | `1.0` | _field_ | [L214](../../../crawlbot/simulation/config.py#L214) |
|   `fsat_jitter_margin` | `0.05` | _field_ | [L224](../../../crawlbot/simulation/config.py#L224) |
|   `nmpc_N` | `8` | _field_ | [L227](../../../crawlbot/simulation/config.py#L227) |
|   `nmpc_dt` | `0.1` | _field_ | [L228](../../../crawlbot/simulation/config.py#L228) |
|   `nmpc_f_max` | `300.0` | _field_ | [L229](../../../crawlbot/simulation/config.py#L229) |
|   `nmpc_tau_max` | `8.0` | _field_ | [L230](../../../crawlbot/simulation/config.py#L230) |
|   `nmpc_Wv` | `10.0` | _field_ | [L231](../../../crawlbot/simulation/config.py#L231) |
|   `nmpc_p_max` | `50.0` | _field_ | [L232](../../../crawlbot/simulation/config.py#L232) |
|   `nmpc_Wr` | `100.0` | _field_ | [L238](../../../crawlbot/simulation/config.py#L238) |
|   `nmpc_Wu_f` | `0.01` | _field_ | [L239](../../../crawlbot/simulation/config.py#L239) |
|   `nmpc_Wu_tau` | `0.001` | _field_ | [L240](../../../crawlbot/simulation/config.py#L240) |
|   `nmpc_Qf_r` | `1000.0` | _field_ | [L241](../../../crawlbot/simulation/config.py#L241) |
|   `nmpc_Qf_v` | `100.0` | _field_ | [L242](../../../crawlbot/simulation/config.py#L242) |
|   `nmpc_Qf_L` | `10.0` | _field_ | [L243](../../../crawlbot/simulation/config.py#L243) |
|   `t_settle_final` | `20.0` | _field_ | [L244](../../../crawlbot/simulation/config.py#L244) |
|   `t_settle_inter` | `0.0` | _field_ | [L251](../../../crawlbot/simulation/config.py#L251) |
|   `use_energy_settle_inter` | `True` | _field_ | [L252](../../../crawlbot/simulation/config.py#L252) |
|   `settle_inter_epsilon_v` | `0.001` | _field_ | [L253](../../../crawlbot/simulation/config.py#L253) |
|   `interstep_settle_epsilon_v` | `0.0` | _field_ | [L261](../../../crawlbot/simulation/config.py#L261) |
|   `n_settle_inter_max_steps` | `500` | _field_ | [L262](../../../crawlbot/simulation/config.py#L262) |
|   `t_settle_inter_min` | `0.1` | _field_ | [L263](../../../crawlbot/simulation/config.py#L263) |
|   `ss_alpha_ee` | `1000.0` | _field_ | [L266](../../../crawlbot/simulation/config.py#L266) |
|   `ss_alpha_posture` | `20.0` | _field_ | [L267](../../../crawlbot/simulation/config.py#L267) |
|   `ss_alpha_wrench` | `1.0` | _field_ | [L268](../../../crawlbot/simulation/config.py#L268) |
|   `ss_alpha_lambda_int` | `0.0` | _field_ | [L269](../../../crawlbot/simulation/config.py#L269) |
|   `ss_alpha_mom` | `400.0` | _field_ | [L274](../../../crawlbot/simulation/config.py#L274) |
|   `log_hifreq_ss` | `False` | _field_ | [L278](../../../crawlbot/simulation/config.py#L278) |
|   `ss_two_task_mode` | `False` | _field_ | [L286](../../../crawlbot/simulation/config.py#L286) |
|   `alpha_torso_pose` | `2000.0` | _field_ | [L287](../../../crawlbot/simulation/config.py#L287) |
|   `dt_ds` | `0.5` | _field_ | [L296](../../../crawlbot/simulation/config.py#L296) |
|   `dock_hold_passivity_on` | `False` | _field_ | [L310](../../../crawlbot/simulation/config.py#L310) |
|   `passivity_W_budget` | `0.0` | _field_ | [L311](../../../crawlbot/simulation/config.py#L311) |
|   `log_dock_work` | `False` | _field_ | [L312](../../../crawlbot/simulation/config.py#L312) |
|   `qp_envelope_exact` | `False` | _field_ | [L322](../../../crawlbot/simulation/config.py#L322) |
|   `ds_centroidal_mode` | `False` | _field_ | [L327](../../../crawlbot/simulation/config.py#L327) |
|   `ds_alpha_com` | `100.0` | _field_ | [L328](../../../crawlbot/simulation/config.py#L328) |
|   `ds_alpha_torso_ori` | `200.0` | _field_ | [L329](../../../crawlbot/simulation/config.py#L329) |
|   `ds_alpha_posture` | `50.0` | _field_ | [L330](../../../crawlbot/simulation/config.py#L330) |
|   `ss_Kp_com` | `3.0` | _field_ | [L333](../../../crawlbot/simulation/config.py#L333) |
|   `ss_Kd_com` | `3.0` | _field_ | [L334](../../../crawlbot/simulation/config.py#L334) |
|   `ss_Kp_torso` | `6.0` | _field_ | [L335](../../../crawlbot/simulation/config.py#L335) |
|   `ss_Kd_torso` | `5.0` | _field_ | [L336](../../../crawlbot/simulation/config.py#L336) |
|   `ss_Kp_ee` | `10.0` | _field_ | [L337](../../../crawlbot/simulation/config.py#L337) |
|   `ss_Kd_ee` | `12.0` | _field_ | [L338](../../../crawlbot/simulation/config.py#L338) |
|   `ss_Kp_ee_ang` | `6.0` | _field_ | [L339](../../../crawlbot/simulation/config.py#L339) |
|   `ss_Kd_ee_ang` | `4.5` | _field_ | [L340](../../../crawlbot/simulation/config.py#L340) |
|   `swing_clearance` | `0.03` | _field_ | [L343](../../../crawlbot/simulation/config.py#L343) |
|   `swing_bump_peak_tau` | `0.5` | _field_ | [L349](../../../crawlbot/simulation/config.py#L349) |
|   `ik_fixed_rotation` | `True` | _field_ | [L360](../../../crawlbot/simulation/config.py#L360) |
|   `ik_fixed_rotation_w_min` | `0.0001` | _field_ | [L361](../../../crawlbot/simulation/config.py#L361) |
|   `ik_level_axis` | `None` | _field_ | [L376](../../../crawlbot/simulation/config.py#L376) |
|   `ik_q_nominal` | `None` | _field_ | [L377](../../../crawlbot/simulation/config.py#L377) |
|   `ik_w_posture` | `0.0` | _field_ | [L378](../../../crawlbot/simulation/config.py#L378) |
|   `use_com_z_standoff` | `False` | _field_ | [L391](../../../crawlbot/simulation/config.py#L391) |
|   `com_z_standoff` | `-0.35` | _field_ | [L392](../../../crawlbot/simulation/config.py#L392) |
|   `torso_early_finish_fraction` | `1.0` | _field_ | [L417](../../../crawlbot/simulation/config.py#L417) |
|   `swing_early_finish_fraction` | `1.0` | _field_ | [L426](../../../crawlbot/simulation/config.py#L426) |
|   `n_settle_steps` | `500` | _field_ | [L429](../../../crawlbot/simulation/config.py#L429) |
|   `Kd_settle_damping` | `20.0` | _field_ | [L440](../../../crawlbot/simulation/config.py#L440) |
|   `n_settle_max_steps` | `1000` | _field_ | [L441](../../../crawlbot/simulation/config.py#L441) |
|   `settle_epsilon_v` | `0.001` | _field_ | [L442](../../../crawlbot/simulation/config.py#L442) |
|   `settle_plateau_ratio` | `0.999` | _field_ | [L443](../../../crawlbot/simulation/config.py#L443) |
|   `diag_freeze_torso_ref_on_abort` | `False` | _field_ | [L450](../../../crawlbot/simulation/config.py#L450) |
|   `diag_force_single_contact_on_abort` | `False` | _field_ | [L456](../../../crawlbot/simulation/config.py#L456) |
|   `diag_disable_passivity_on_abort` | `False` | _field_ | [L462](../../../crawlbot/simulation/config.py#L462) |
|   `mapping_bypass_in_ss` | `False` | _field_ | [L468](../../../crawlbot/simulation/config.py#L468) |
|   `ds_ramp_duration_s` | `2.0` | _field_ | [L478](../../../crawlbot/simulation/config.py#L478) |
|   `gait_anchor_dx` | `0.8` | _field_ | [L490](../../../crawlbot/simulation/config.py#L490) |

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
| `class SimConfig` | [L12-490](../../../crawlbot/simulation/config.py#L12-L490) |

---

## See also

- package overview: [`simulation.md`](simulation.md)
