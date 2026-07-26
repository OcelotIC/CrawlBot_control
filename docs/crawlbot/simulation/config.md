# `crawlbot.simulation.config`

**File**: `crawlbot/simulation/config.py` — **507 lines** — canonical coverage **100 %**

> Module docstring: *"Simulation configuration dataclass."*

`SimConfig` — the single tuning surface. Every adjustable parameter in the
controller lives here, with its unit and justification. **100 % coverage.**

---

## Public API

| symbol | signature | canonical? |
|---|---|---|
| **`SimConfig`** *(dataclass)* |  |  |
|   `dt_nmpc` | `0.1` | _field_ |
|   `dt_qp` | `0.01` | _field_ |
|   `t_ss_margin` | `1.0` | _field_ |
|   `t_hold_max` | `3.0` | _field_ |
|   `dock_check_delay` | `0.5` | _field_ |
|   `n_ds_max_steps` | `1000` | _field_ |
|   `tau_max` | `20.0` | _field_ |
|   `weld_radius` | `0.005` | _field_ |
|   `dock_vel_max` | `0.01` | _field_ |
|   `dock_ori_threshold_deg` | `5.0` | _field_ |
|   `dock_use_6d_twist` | `True` | _field_ |
|   `dock_twist_max` | `0.05` | _field_ |
|   `gmo_K_O` | `80.0` | _field_ |
|   `gmo_F_threshold` | `5.0` | _field_ |
|   `gmo_d_proximity` | `0.02` | _field_ |
|   `gmo_d_contact` | `0.005` | _field_ |
|   `gmo_d_reset` | `0.03` | _field_ |
|   `gmo_debounce_count` | `3` | _field_ |
|   `hw_init` | `np.zeros(3)` | _field_ |
|   `hw_min` | `np.full(3, -5.0)` | _field_ |
|   `hw_max` | `np.full(3, 5.0)` | _field_ |
|   `hw_qp_tight` | `np.full(3, 3.0)` | _field_ |
|   `L_max` | `10.0` | _field_ |
|   `tau_w_max` | `2.5` | _field_ |
|   `aocs_K_hw` | `2.0` | _field_ |
|   `aocs_tau_w_max` | `2.5` | _field_ |
|   `rwa_I_w` | `0.01` | _field_ |
|   `aocs_mode` | `'legacy'` | _field_ |
|   `aocs_use_H_estimator` | `False` | _field_ |
|   `aocs_use_legacy_corrected` | `False` | _field_ |
|   `aocs_filter_tau` | `0.016` | _field_ |
|   `aocs_K_omega` | `50.0` | _field_ |
|   `aocs_K_d` | `25.0` | _field_ |
|   `aocs_K_theta` | `1.0` | _field_ |
|   `aocs_K_h` | `0.5` | _field_ |
|   `aocs_hw_target` | `np.zeros(3)` | _field_ |
|   `aocs_use_wrench_ff_in_ds` | `False` | _field_ |
|   `ds_torso_ref_from_state` | `False` | _field_ |
|   `aocs_off_in_ds` | `False` | _field_ |
|   `aocs_active_in_interstep` | `True` | _field_ |
|   `interstep_hw_refresh` | `True` | _field_ |
|   `interstep_settle_alpha_wrench` | `0.0` | _field_ |
|   `stop_on_failed_step` | `True` | _field_ |
|   `frames_per_step` | `0` | _field_ |
|   `use_m2_stack` | `False` | _field_ |
|   `alpha_passivity` | `1.0` | _field_ |
|   `enforce_hw_conservation` | `False` | _field_ |
|   `h_max_tight` | `np.full(3, 5.0)` | _field_ |
|   `w_L_nmpc` | `1.0` | _field_ |
|   `kappa_terminal` | `1.0` | _field_ |
|   `preplanner_M` | `15` | _field_ |
|   `preplanner_kappa` | `0.7` | _field_ |
|   `preplanner_f_max` | `25.0` | _field_ |
|   `preplanner_tau_max` | `8.0` | _field_ |
|   `preplanner_w_L` | `1.0` | _field_ |
|   `preplanner_w_u` | `0.01` | _field_ |
|   `preplanner_max_iter` | `300` | _field_ |
|   `preplanner_a_cruise_max` | `0.0` | _field_ |
|   `preplanner_cruise_ramp_frac` | `0.2` | _field_ |
|   `preplanner_tstep_standoff_gain` | `0.0` | _field_ |
|   `preplanner_tstep_standoff_knee` | `1000000000.0` | _field_ |
|   `preplanner_tstep_scale_step` | `-1` | _field_ |
|   `preplanner_tstep_scale_factor` | `1.0` | _field_ |
|   `fsat_jitter_margin` | `0.05` | _field_ |
|   `nmpc_N` | `8` | _field_ |
|   `nmpc_dt` | `0.1` | _field_ |
|   `nmpc_f_max` | `300.0` | _field_ |
|   `nmpc_tau_max` | `8.0` | _field_ |
|   `nmpc_Wv` | `10.0` | _field_ |
|   `nmpc_p_max` | `50.0` | _field_ |
|   `nmpc_Wr` | `100.0` | _field_ |
|   `nmpc_Wu_f` | `0.01` | _field_ |
|   `nmpc_Wu_tau` | `0.001` | _field_ |
|   `nmpc_Qf_r` | `1000.0` | _field_ |
|   `nmpc_Qf_v` | `100.0` | _field_ |
|   `nmpc_Qf_L` | `10.0` | _field_ |
|   `t_settle_final` | `20.0` | _field_ |
|   `t_settle_inter` | `0.0` | _field_ |
|   `use_energy_settle_inter` | `True` | _field_ |
|   `settle_inter_epsilon_v` | `0.001` | _field_ |
|   `interstep_settle_epsilon_v` | `0.0` | _field_ |
|   `n_settle_inter_max_steps` | `500` | _field_ |
|   `t_settle_inter_min` | `0.1` | _field_ |
|   `ss_alpha_ee` | `1000.0` | _field_ |
|   `ss_alpha_posture` | `20.0` | _field_ |
|   `ss_alpha_wrench` | `1.0` | _field_ |
|   `ss_alpha_lambda_int` | `0.0` | _field_ |
|   `ss_alpha_mom` | `400.0` | _field_ |
|   `log_hifreq_ss` | `False` | _field_ |
|   `ss_two_task_mode` | `False` | _field_ |
|   `alpha_torso_pose` | `2000.0` | _field_ |
|   `dt_ds` | `0.5` | _field_ |
|   `dock_hold_passivity_on` | `False` | _field_ |
|   `passivity_W_budget` | `0.0` | _field_ |
|   `log_dock_work` | `False` | _field_ |
|   `qp_envelope_exact` | `False` | _field_ |
|   `ds_centroidal_mode` | `False` | _field_ |
|   `ds_alpha_com` | `100.0` | _field_ |
|   `ds_alpha_torso_ori` | `200.0` | _field_ |
|   `ds_alpha_posture` | `50.0` | _field_ |
|   `ss_Kp_com` | `3.0` | _field_ |
|   `ss_Kd_com` | `3.0` | _field_ |
|   `ss_Kp_torso` | `6.0` | _field_ |
|   `ss_Kd_torso` | `5.0` | _field_ |
|   `ss_Kp_ee` | `10.0` | _field_ |
|   `ss_Kd_ee` | `12.0` | _field_ |
|   `ss_Kp_ee_ang` | `6.0` | _field_ |
|   `ss_Kd_ee_ang` | `4.5` | _field_ |
|   `swing_clearance` | `0.03` | _field_ |
|   `swing_bump_peak_tau` | `0.5` | _field_ |
|   `ik_fixed_rotation` | `True` | _field_ |
|   `ik_fixed_rotation_w_min` | `0.0001` | _field_ |
|   `ik_level_axis` | `None` | _field_ |
|   `ik_q_nominal` | `None` | _field_ |
|   `ik_w_posture` | `0.0` | _field_ |
|   `use_com_z_standoff` | `False` | _field_ |
|   `com_z_standoff` | `-0.35` | _field_ |
|   `torso_early_finish_fraction` | `1.0` | _field_ |
|   `swing_early_finish_fraction` | `1.0` | _field_ |
|   `n_settle_steps` | `500` | _field_ |
|   `Kd_settle_damping` | `20.0` | _field_ |
|   `n_settle_max_steps` | `1000` | _field_ |
|   `settle_epsilon_v` | `0.001` | _field_ |
|   `settle_plateau_ratio` | `0.999` | _field_ |
|   `diag_freeze_torso_ref_on_abort` | `False` | _field_ |
|   `diag_force_single_contact_on_abort` | `False` | _field_ |
|   `diag_disable_passivity_on_abort` | `False` | _field_ |
|   `mapping_bypass_in_ss` | `False` | _field_ |
|   `ds_ramp_duration_s` | `2.0` | _field_ |
|   `gait_anchor_dx` | `0.8` | _field_ |

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

## See also

- package overview: [`simulation.md`](simulation.md)
