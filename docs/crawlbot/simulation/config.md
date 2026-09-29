# `crawlbot.simulation.config`

**File**: [`crawlbot/simulation/config.py`](../../../crawlbot/simulation/config.py) — **502 lines** — canonical coverage **100 %**

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
|   `use_m2_stack` | `False` | _field_ | [L184](../../../crawlbot/simulation/config.py#L184) |
|   `alpha_passivity` | `1.0` | _field_ | [L185](../../../crawlbot/simulation/config.py#L185) |
|   `enforce_hw_conservation` | `False` | _field_ | [L188](../../../crawlbot/simulation/config.py#L188) |
|   `h_max_tight` | `np.full(3, 5.0)` | _field_ | [L189](../../../crawlbot/simulation/config.py#L189) |
|   `w_L_nmpc` | `1.0` | _field_ | [L190](../../../crawlbot/simulation/config.py#L190) |
|   `kappa_terminal` | `1.0` | _field_ | [L191](../../../crawlbot/simulation/config.py#L191) |
|   `preplanner_M` | `15` | _field_ | [L203](../../../crawlbot/simulation/config.py#L203) |
|   `preplanner_kappa` | `0.7` | _field_ | [L204](../../../crawlbot/simulation/config.py#L204) |
|   `preplanner_f_max` | `25.0` | _field_ | [L205](../../../crawlbot/simulation/config.py#L205) |
|   `preplanner_tau_max` | `8.0` | _field_ | [L206](../../../crawlbot/simulation/config.py#L206) |
|   `preplanner_w_L` | `1.0` | _field_ | [L207](../../../crawlbot/simulation/config.py#L207) |
|   `preplanner_w_u` | `0.01` | _field_ | [L208](../../../crawlbot/simulation/config.py#L208) |
|   `preplanner_max_iter` | `300` | _field_ | [L209](../../../crawlbot/simulation/config.py#L209) |
|   `preplanner_a_cruise_max` | `0.0` | _field_ | [L210](../../../crawlbot/simulation/config.py#L210) |
|   `preplanner_cruise_ramp_frac` | `0.2` | _field_ | [L211](../../../crawlbot/simulation/config.py#L211) |
|   `preplanner_tstep_standoff_gain` | `0.0` | _field_ | [L219](../../../crawlbot/simulation/config.py#L219) |
|   `preplanner_tstep_standoff_knee` | `1000000000.0` | _field_ | [L220](../../../crawlbot/simulation/config.py#L220) |
|   `preplanner_tstep_scale_step` | `-1` | _field_ | [L224](../../../crawlbot/simulation/config.py#L224) |
|   `preplanner_tstep_scale_factor` | `1.0` | _field_ | [L225](../../../crawlbot/simulation/config.py#L225) |
|   `fsat_jitter_margin` | `0.05` | _field_ | [L235](../../../crawlbot/simulation/config.py#L235) |
|   `nmpc_N` | `8` | _field_ | [L238](../../../crawlbot/simulation/config.py#L238) |
|   `nmpc_dt` | `0.1` | _field_ | [L239](../../../crawlbot/simulation/config.py#L239) |
|   `nmpc_f_max` | `300.0` | _field_ | [L240](../../../crawlbot/simulation/config.py#L240) |
|   `nmpc_tau_max` | `8.0` | _field_ | [L241](../../../crawlbot/simulation/config.py#L241) |
|   `nmpc_Wv` | `10.0` | _field_ | [L242](../../../crawlbot/simulation/config.py#L242) |
|   `nmpc_p_max` | `50.0` | _field_ | [L243](../../../crawlbot/simulation/config.py#L243) |
|   `nmpc_Wr` | `100.0` | _field_ | [L249](../../../crawlbot/simulation/config.py#L249) |
|   `nmpc_Wu_f` | `0.01` | _field_ | [L250](../../../crawlbot/simulation/config.py#L250) |
|   `nmpc_Wu_tau` | `0.001` | _field_ | [L251](../../../crawlbot/simulation/config.py#L251) |
|   `nmpc_Qf_r` | `1000.0` | _field_ | [L252](../../../crawlbot/simulation/config.py#L252) |
|   `nmpc_Qf_v` | `100.0` | _field_ | [L253](../../../crawlbot/simulation/config.py#L253) |
|   `nmpc_Qf_L` | `10.0` | _field_ | [L254](../../../crawlbot/simulation/config.py#L254) |
|   `t_settle_final` | `20.0` | _field_ | [L255](../../../crawlbot/simulation/config.py#L255) |
|   `t_settle_inter` | `0.0` | _field_ | [L262](../../../crawlbot/simulation/config.py#L262) |
|   `use_energy_settle_inter` | `True` | _field_ | [L263](../../../crawlbot/simulation/config.py#L263) |
|   `settle_inter_epsilon_v` | `0.001` | _field_ | [L264](../../../crawlbot/simulation/config.py#L264) |
|   `interstep_settle_epsilon_v` | `0.0` | _field_ | [L272](../../../crawlbot/simulation/config.py#L272) |
|   `n_settle_inter_max_steps` | `500` | _field_ | [L273](../../../crawlbot/simulation/config.py#L273) |
|   `t_settle_inter_min` | `0.1` | _field_ | [L274](../../../crawlbot/simulation/config.py#L274) |
|   `ss_alpha_ee` | `1000.0` | _field_ | [L277](../../../crawlbot/simulation/config.py#L277) |
|   `ss_alpha_posture` | `20.0` | _field_ | [L278](../../../crawlbot/simulation/config.py#L278) |
|   `ss_alpha_wrench` | `1.0` | _field_ | [L279](../../../crawlbot/simulation/config.py#L279) |
|   `ss_alpha_lambda_int` | `0.0` | _field_ | [L280](../../../crawlbot/simulation/config.py#L280) |
|   `ss_alpha_mom` | `400.0` | _field_ | [L285](../../../crawlbot/simulation/config.py#L285) |
|   `log_hifreq_ss` | `False` | _field_ | [L289](../../../crawlbot/simulation/config.py#L289) |
|   `ss_two_task_mode` | `False` | _field_ | [L297](../../../crawlbot/simulation/config.py#L297) |
|   `alpha_torso_pose` | `2000.0` | _field_ | [L298](../../../crawlbot/simulation/config.py#L298) |
|   `dt_ds` | `0.5` | _field_ | [L307](../../../crawlbot/simulation/config.py#L307) |
|   `dock_hold_passivity_on` | `False` | _field_ | [L321](../../../crawlbot/simulation/config.py#L321) |
|   `passivity_W_budget` | `0.0` | _field_ | [L322](../../../crawlbot/simulation/config.py#L322) |
|   `log_dock_work` | `False` | _field_ | [L323](../../../crawlbot/simulation/config.py#L323) |
|   `qp_envelope_exact` | `False` | _field_ | [L333](../../../crawlbot/simulation/config.py#L333) |
|   `ds_centroidal_mode` | `False` | _field_ | [L338](../../../crawlbot/simulation/config.py#L338) |
|   `ds_alpha_com` | `100.0` | _field_ | [L339](../../../crawlbot/simulation/config.py#L339) |
|   `ds_alpha_torso_ori` | `200.0` | _field_ | [L340](../../../crawlbot/simulation/config.py#L340) |
|   `ds_alpha_posture` | `50.0` | _field_ | [L341](../../../crawlbot/simulation/config.py#L341) |
|   `ss_Kp_com` | `3.0` | _field_ | [L344](../../../crawlbot/simulation/config.py#L344) |
|   `ss_Kd_com` | `3.0` | _field_ | [L345](../../../crawlbot/simulation/config.py#L345) |
|   `ss_Kp_torso` | `6.0` | _field_ | [L346](../../../crawlbot/simulation/config.py#L346) |
|   `ss_Kd_torso` | `5.0` | _field_ | [L347](../../../crawlbot/simulation/config.py#L347) |
|   `ss_Kp_ee` | `10.0` | _field_ | [L348](../../../crawlbot/simulation/config.py#L348) |
|   `ss_Kd_ee` | `12.0` | _field_ | [L349](../../../crawlbot/simulation/config.py#L349) |
|   `ss_Kp_ee_ang` | `6.0` | _field_ | [L350](../../../crawlbot/simulation/config.py#L350) |
|   `ss_Kd_ee_ang` | `4.5` | _field_ | [L351](../../../crawlbot/simulation/config.py#L351) |
|   `swing_clearance` | `0.03` | _field_ | [L354](../../../crawlbot/simulation/config.py#L354) |
|   `swing_bump_peak_tau` | `0.5` | _field_ | [L360](../../../crawlbot/simulation/config.py#L360) |
|   `ik_fixed_rotation` | `True` | _field_ | [L371](../../../crawlbot/simulation/config.py#L371) |
|   `ik_fixed_rotation_w_min` | `0.0001` | _field_ | [L372](../../../crawlbot/simulation/config.py#L372) |
|   `ik_level_axis` | `None` | _field_ | [L387](../../../crawlbot/simulation/config.py#L387) |
|   `ik_q_nominal` | `None` | _field_ | [L388](../../../crawlbot/simulation/config.py#L388) |
|   `ik_w_posture` | `0.0` | _field_ | [L389](../../../crawlbot/simulation/config.py#L389) |
|   `use_com_z_standoff` | `False` | _field_ | [L402](../../../crawlbot/simulation/config.py#L402) |
|   `com_z_standoff` | `-0.35` | _field_ | [L403](../../../crawlbot/simulation/config.py#L403) |
|   `torso_early_finish_fraction` | `1.0` | _field_ | [L428](../../../crawlbot/simulation/config.py#L428) |
|   `swing_early_finish_fraction` | `1.0` | _field_ | [L437](../../../crawlbot/simulation/config.py#L437) |
|   `n_settle_steps` | `500` | _field_ | [L440](../../../crawlbot/simulation/config.py#L440) |
|   `Kd_settle_damping` | `20.0` | _field_ | [L451](../../../crawlbot/simulation/config.py#L451) |
|   `n_settle_max_steps` | `1000` | _field_ | [L452](../../../crawlbot/simulation/config.py#L452) |
|   `settle_epsilon_v` | `0.001` | _field_ | [L453](../../../crawlbot/simulation/config.py#L453) |
|   `settle_plateau_ratio` | `0.999` | _field_ | [L454](../../../crawlbot/simulation/config.py#L454) |
|   `diag_freeze_torso_ref_on_abort` | `False` | _field_ | [L461](../../../crawlbot/simulation/config.py#L461) |
|   `diag_force_single_contact_on_abort` | `False` | _field_ | [L467](../../../crawlbot/simulation/config.py#L467) |
|   `diag_disable_passivity_on_abort` | `False` | _field_ | [L473](../../../crawlbot/simulation/config.py#L473) |
|   `mapping_bypass_in_ss` | `False` | _field_ | [L479](../../../crawlbot/simulation/config.py#L479) |
|   `ds_ramp_duration_s` | `2.0` | _field_ | [L489](../../../crawlbot/simulation/config.py#L489) |
|   `gait_anchor_dx` | `0.8` | _field_ | [L501](../../../crawlbot/simulation/config.py#L501) |

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
| `class SimConfig` | [L12-501](../../../crawlbot/simulation/config.py#L12-L501) |

---

## See also

- package overview: [`simulation.md`](simulation.md)
