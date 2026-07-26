# `crawlbot.simulation.config`

`SimConfig` — le point unique de réglage. **100 % de couverture.**

**Fichier** : `crawlbot/simulation/config.py` — **507 lignes** — couverture canonique **100 %**

> Docstring du module : *« Simulation configuration dataclass. »*

---

## API publique

| symbole | signature | canonique ? |
|---|---|---|
| **`SimConfig`** *(dataclass)* |  |  |
|   `dt_nmpc` | `0.1` | _champ_ |
|   `dt_qp` | `0.01` | _champ_ |
|   `t_ss_margin` | `1.0` | _champ_ |
|   `t_hold_max` | `3.0` | _champ_ |
|   `dock_check_delay` | `0.5` | _champ_ |
|   `n_ds_max_steps` | `1000` | _champ_ |
|   `tau_max` | `20.0` | _champ_ |
|   `weld_radius` | `0.005` | _champ_ |
|   `dock_vel_max` | `0.01` | _champ_ |
|   `dock_ori_threshold_deg` | `5.0` | _champ_ |
|   `dock_use_6d_twist` | `True` | _champ_ |
|   `dock_twist_max` | `0.05` | _champ_ |
|   `gmo_K_O` | `80.0` | _champ_ |
|   `gmo_F_threshold` | `5.0` | _champ_ |
|   `gmo_d_proximity` | `0.02` | _champ_ |
|   `gmo_d_contact` | `0.005` | _champ_ |
|   `gmo_d_reset` | `0.03` | _champ_ |
|   `gmo_debounce_count` | `3` | _champ_ |
|   `hw_init` | `np.zeros(3)` | _champ_ |
|   `hw_min` | `np.full(3, -5.0)` | _champ_ |
|   `hw_max` | `np.full(3, 5.0)` | _champ_ |
|   `hw_qp_tight` | `np.full(3, 3.0)` | _champ_ |
|   `L_max` | `10.0` | _champ_ |
|   `tau_w_max` | `2.5` | _champ_ |
|   `aocs_K_hw` | `2.0` | _champ_ |
|   `aocs_tau_w_max` | `2.5` | _champ_ |
|   `rwa_I_w` | `0.01` | _champ_ |
|   `aocs_mode` | `'legacy'` | _champ_ |
|   `aocs_use_H_estimator` | `False` | _champ_ |
|   `aocs_use_legacy_corrected` | `False` | _champ_ |
|   `aocs_filter_tau` | `0.016` | _champ_ |
|   `aocs_K_omega` | `50.0` | _champ_ |
|   `aocs_K_d` | `25.0` | _champ_ |
|   `aocs_K_theta` | `1.0` | _champ_ |
|   `aocs_K_h` | `0.5` | _champ_ |
|   `aocs_hw_target` | `np.zeros(3)` | _champ_ |
|   `aocs_use_wrench_ff_in_ds` | `False` | _champ_ |
|   `ds_torso_ref_from_state` | `False` | _champ_ |
|   `aocs_off_in_ds` | `False` | _champ_ |
|   `aocs_active_in_interstep` | `True` | _champ_ |
|   `interstep_hw_refresh` | `True` | _champ_ |
|   `interstep_settle_alpha_wrench` | `0.0` | _champ_ |
|   `stop_on_failed_step` | `True` | _champ_ |
|   `frames_per_step` | `0` | _champ_ |
|   `use_m2_stack` | `False` | _champ_ |
|   `alpha_passivity` | `1.0` | _champ_ |
|   `enforce_hw_conservation` | `False` | _champ_ |
|   `h_max_tight` | `np.full(3, 5.0)` | _champ_ |
|   `w_L_nmpc` | `1.0` | _champ_ |
|   `kappa_terminal` | `1.0` | _champ_ |
|   `preplanner_M` | `15` | _champ_ |
|   `preplanner_kappa` | `0.7` | _champ_ |
|   `preplanner_f_max` | `25.0` | _champ_ |
|   `preplanner_tau_max` | `8.0` | _champ_ |
|   `preplanner_w_L` | `1.0` | _champ_ |
|   `preplanner_w_u` | `0.01` | _champ_ |
|   `preplanner_max_iter` | `300` | _champ_ |
|   `preplanner_a_cruise_max` | `0.0` | _champ_ |
|   `preplanner_cruise_ramp_frac` | `0.2` | _champ_ |
|   `preplanner_tstep_standoff_gain` | `0.0` | _champ_ |
|   `preplanner_tstep_standoff_knee` | `1000000000.0` | _champ_ |
|   `preplanner_tstep_scale_step` | `-1` | _champ_ |
|   `preplanner_tstep_scale_factor` | `1.0` | _champ_ |
|   `fsat_jitter_margin` | `0.05` | _champ_ |
|   `nmpc_N` | `8` | _champ_ |
|   `nmpc_dt` | `0.1` | _champ_ |
|   `nmpc_f_max` | `300.0` | _champ_ |
|   `nmpc_tau_max` | `8.0` | _champ_ |
|   `nmpc_Wv` | `10.0` | _champ_ |
|   `nmpc_p_max` | `50.0` | _champ_ |
|   `nmpc_Wr` | `100.0` | _champ_ |
|   `nmpc_Wu_f` | `0.01` | _champ_ |
|   `nmpc_Wu_tau` | `0.001` | _champ_ |
|   `nmpc_Qf_r` | `1000.0` | _champ_ |
|   `nmpc_Qf_v` | `100.0` | _champ_ |
|   `nmpc_Qf_L` | `10.0` | _champ_ |
|   `t_settle_final` | `20.0` | _champ_ |
|   `t_settle_inter` | `0.0` | _champ_ |
|   `use_energy_settle_inter` | `True` | _champ_ |
|   `settle_inter_epsilon_v` | `0.001` | _champ_ |
|   `interstep_settle_epsilon_v` | `0.0` | _champ_ |
|   `n_settle_inter_max_steps` | `500` | _champ_ |
|   `t_settle_inter_min` | `0.1` | _champ_ |
|   `ss_alpha_ee` | `1000.0` | _champ_ |
|   `ss_alpha_posture` | `20.0` | _champ_ |
|   `ss_alpha_wrench` | `1.0` | _champ_ |
|   `ss_alpha_lambda_int` | `0.0` | _champ_ |
|   `ss_alpha_mom` | `400.0` | _champ_ |
|   `log_hifreq_ss` | `False` | _champ_ |
|   `ss_two_task_mode` | `False` | _champ_ |
|   `alpha_torso_pose` | `2000.0` | _champ_ |
|   `dt_ds` | `0.5` | _champ_ |
|   `dock_hold_passivity_on` | `False` | _champ_ |
|   `passivity_W_budget` | `0.0` | _champ_ |
|   `log_dock_work` | `False` | _champ_ |
|   `qp_envelope_exact` | `False` | _champ_ |
|   `ds_centroidal_mode` | `False` | _champ_ |
|   `ds_alpha_com` | `100.0` | _champ_ |
|   `ds_alpha_torso_ori` | `200.0` | _champ_ |
|   `ds_alpha_posture` | `50.0` | _champ_ |
|   `ss_Kp_com` | `3.0` | _champ_ |
|   `ss_Kd_com` | `3.0` | _champ_ |
|   `ss_Kp_torso` | `6.0` | _champ_ |
|   `ss_Kd_torso` | `5.0` | _champ_ |
|   `ss_Kp_ee` | `10.0` | _champ_ |
|   `ss_Kd_ee` | `12.0` | _champ_ |
|   `ss_Kp_ee_ang` | `6.0` | _champ_ |
|   `ss_Kd_ee_ang` | `4.5` | _champ_ |
|   `swing_clearance` | `0.03` | _champ_ |
|   `swing_bump_peak_tau` | `0.5` | _champ_ |
|   `ik_fixed_rotation` | `True` | _champ_ |
|   `ik_fixed_rotation_w_min` | `0.0001` | _champ_ |
|   `ik_level_axis` | `None` | _champ_ |
|   `ik_q_nominal` | `None` | _champ_ |
|   `ik_w_posture` | `0.0` | _champ_ |
|   `use_com_z_standoff` | `False` | _champ_ |
|   `com_z_standoff` | `-0.35` | _champ_ |
|   `torso_early_finish_fraction` | `1.0` | _champ_ |
|   `swing_early_finish_fraction` | `1.0` | _champ_ |
|   `n_settle_steps` | `500` | _champ_ |
|   `Kd_settle_damping` | `20.0` | _champ_ |
|   `n_settle_max_steps` | `1000` | _champ_ |
|   `settle_epsilon_v` | `0.001` | _champ_ |
|   `settle_plateau_ratio` | `0.999` | _champ_ |
|   `diag_freeze_torso_ref_on_abort` | `False` | _champ_ |
|   `diag_force_single_contact_on_abort` | `False` | _champ_ |
|   `diag_disable_passivity_on_abort` | `False` | _champ_ |
|   `mapping_bypass_in_ss` | `False` | _champ_ |
|   `ds_ramp_duration_s` | `2.0` | _champ_ |
|   `gait_anchor_dx` | `0.8` | _champ_ |

---

## Règle 5 du projet

> *No silent parameter changes. All tunable parameters live in `SimConfig` with
> units and justification.*

Une seule dataclass, ~500 lignes, d'où descendent `CentroidalNMPCConfig`,
`WholeBodyQPConfig`, `CoarsePrePlannerConfig` et `ContactObserverConfig`. Pas de
constante magique enfouie dans la boucle.

## ⚠ Mais un défaut de `SimConfig` n'est pas non plus la valeur canonique

Le run canonique est construit en deux temps :

```
Misc/scripts/run_m7_single_step._make_m7_config()   ->  SimConfig de base
scripts/diag_cooperative_arms.main(**kwargs)        ->  surcharges du run
```

Pour connaître une valeur canonique : le tableau « Key Parameters » de
CLAUDE.md, ou l'instrumentation du run. **Jamais la lecture d'un défaut** —
c'est l'erreur rétractée par le chantier (F1).

## Le piège `use_m2_stack`

Il *semble* mort et commande en réalité le routage de la référence de torse
**et la contrainte de passivité en DS**. Sa déclaration porte une note à cet
effet. Voir `sim_loop.md`.

## Paramètres à ne pas toucher sans lire CLAUDE.md

| paramètre | valeur gelée | pourquoi |
|---|---|---|
| `tau_w_max` | **2.5** N·m | appliqué en 3 points (NMPC, QP, MJCF) |
| `hw_max` | ±5 N·m·s | inchangé par conception |
| `weight_ratio` | 1.0 | les α *sont* la hiérarchie |
| `alpha_wrench` | 1.0 | > 1 étouffe les tâches torse/EE |
| `preplanner_a_cruise_max` | 0.0 | mise en forme du CoM désactivée |

Les knobs `preplanner_tstep_*` sont des diagnostics exposés par `dca`, tous
neutres par défaut.

## Voir aussi

- vue d'ensemble du paquet : [`simulation.md`](simulation.md)
