# `crawlbot.simulation.logging`

`SimLog` : un tableau par grandeur, un élément par tick, plus la capture de
l'environnement d'exécution.

**Fichier** : `crawlbot/simulation/logging.py` — **269 lignes** — couverture canonique **93 %**

> Docstring du module : *« Simulation data logger. »*

---

## API publique

| symbole | signature | canonique ? |
|---|---|---|
| `capture_environment` | `()` | **oui** |
| **`SimLog`** *(dataclass)* |  |  |
|   `t` | `field(default_factory=list)` | _champ_ |
|   `phase` | `field(default_factory=list)` | _champ_ |
|   `step_idx` | `field(default_factory=list)` | _champ_ |
|   `p_torso` | `field(default_factory=list)` | _champ_ |
|   `p_torso_ref` | `field(default_factory=list)` | _champ_ |
|   `e_torso_pos` | `field(default_factory=list)` | _champ_ |
|   `e_torso_ori` | `field(default_factory=list)` | _champ_ |
|   `q_torso` | `field(default_factory=list)` | _champ_ |
|   `q_torso_ref` | `field(default_factory=list)` | _champ_ |
|   `d_grip_swing` | `field(default_factory=list)` | _champ_ |
|   `d_grip_stance` | `field(default_factory=list)` | _champ_ |
|   `swing_arm` | `field(default_factory=list)` | _champ_ |
|   `p_ee` | `field(default_factory=list)` | _champ_ |
|   `p_ee_ref` | `field(default_factory=list)` | _champ_ |
|   `q_ee` | `field(default_factory=list)` | _champ_ |
|   `q_ee_ref` | `field(default_factory=list)` | _champ_ |
|   `qvel_joints_a` | `field(default_factory=list)` | _champ_ |
|   `qvel_joints_b` | `field(default_factory=list)` | _champ_ |
|   `v_ee_a` | `field(default_factory=list)` | _champ_ |
|   `v_ee_b` | `field(default_factory=list)` | _champ_ |
|   `omega_ee_a` | `field(default_factory=list)` | _champ_ |
|   `omega_ee_b` | `field(default_factory=list)` | _champ_ |
|   `v_torso` | `field(default_factory=list)` | _champ_ |
|   `omega_torso` | `field(default_factory=list)` | _champ_ |
|   `r_com` | `field(default_factory=list)` | _champ_ |
|   `r_com_ref` | `field(default_factory=list)` | _champ_ |
|   `e_com` | `field(default_factory=list)` | _champ_ |
|   `v_com` | `field(default_factory=list)` | _champ_ |
|   `v_com_ref` | `field(default_factory=list)` | _champ_ |
|   `L_com` | `field(default_factory=list)` | _champ_ |
|   `L_com_norm` | `field(default_factory=list)` | _champ_ |
|   `L_com_ref` | `field(default_factory=list)` | _champ_ |
|   `L_dot` | `field(default_factory=list)` | _champ_ |
|   `L_dot_norm` | `field(default_factory=list)` | _champ_ |
|   `hw` | `field(default_factory=list)` | _champ_ |
|   `hw_physical` | `field(default_factory=list)` | _champ_ |
|   `tau_w` | `field(default_factory=list)` | _champ_ |
|   `rw_speed` | `field(default_factory=list)` | _champ_ |
|   `t_ss_hifreq` | `field(default_factory=list)` | _champ_ |
|   `tau_w_ss_hifreq` | `field(default_factory=list)` | _champ_ |
|   `hw_ss_hifreq` | `field(default_factory=list)` | _champ_ |
|   `e_ee_pos` | `field(default_factory=list)` | _champ_ |
|   `e_ee_ori` | `field(default_factory=list)` | _champ_ |
|   `gmo_residual_norm` | `field(default_factory=list)` | _champ_ |
|   `gmo_swing_residual` | `field(default_factory=list)` | _champ_ |
|   `gmo_contact_state` | `field(default_factory=list)` | _champ_ |
|   `H_rO` | `field(default_factory=list)` | _champ_ |
|   `H_dot_est` | `field(default_factory=list)` | _champ_ |
|   `omega_struct` | `field(default_factory=list)` | _champ_ |
|   `qfrc_constraint_torque` | `field(default_factory=list)` | _champ_ |
|   `tau` | `field(default_factory=list)` | _champ_ |
|   `tau_max_joint` | `field(default_factory=list)` | _champ_ |
|   `struct_pos` | `field(default_factory=list)` | _champ_ |
|   `struct_quat` | `field(default_factory=list)` | _champ_ |
|   `struct_euler_deg` | `field(default_factory=list)` | _champ_ |
|   `omega_s` | `field(default_factory=list)` | _champ_ |
|   `nmpc_ok` | `field(default_factory=list)` | _champ_ |
|   `qp_ok` | `field(default_factory=list)` | _champ_ |
|   `lambda_ref_norm` | `field(default_factory=list)` | _champ_ |
|   `lambda_qp` | `field(default_factory=list)` | _champ_ |
|   `lambda_qp_norm` | `field(default_factory=list)` | _champ_ |
|   `nmpc_time_ms` | `field(default_factory=list)` | _champ_ |
|   `qp_time_ms` | `field(default_factory=list)` | _champ_ |
|   `nmpc_status` | `field(default_factory=list)` | _champ_ |
|   `nmpc_cost` | `field(default_factory=list)` | _champ_ |
|   `nmpc_status_str` | `field(default_factory=list)` | _champ_ |
|   `nmpc_iterations` | `field(default_factory=list)` | _champ_ |
|   `transport_term_mag` | `field(default_factory=list)` | _champ_ |
|   `lambda_ref` | `field(default_factory=list)` | _champ_ |
|   `lambda_qp` | `field(default_factory=list)` | _champ_ |
|   `T_kinetic` | `field(default_factory=list)` | _champ_ |
|   `settling_t` | `field(default_factory=list)` | _champ_ |
|   `settling_T` | `field(default_factory=list)` | _champ_ |
|   `settling_T_target` | `0.0` | _champ_ |
|   `settling_stage1_steps` | `0` | _champ_ |
|   `settling_stage2_steps` | `0` | _champ_ |
|   `settling_exit_reason` | `''` | _champ_ |
|   `inter_step_settles` | `field(default_factory=list)` | _champ_ |
|   `dock_events` | `field(default_factory=list)` | _champ_ |
|   `dock_gate_trace` | `field(default_factory=list)` | _champ_ |
|   `ds_mobile_trace` | `field(default_factory=list)` | _champ_ |
|   `dock_work_trace` | `field(default_factory=list)` | _champ_ |
|   `aborted_steps` | `field(default_factory=list)` | _champ_ |
|   `preplanner_T_steps` | `field(default_factory=list)` | _champ_ |
|   `snapshots` | `field(default_factory=list)` | _champ_ |
|   `environment` | `field(default_factory=dict)` | _champ_ |
| `.to_dict` | `()` | **oui** |
| `.save` | `(path)` | **oui** |
| `.load` | `(path)` | non exerce |

### Constantes de module

| nom | valeur |
|---|---|
| `_ENV_VAR_NAMES` | `('MUJOCO_GL', 'OMP_NUM_THREADS', 'OPENBLAS_N` |

---

## Structure

Dataclass de canaux (`t`, `phase`, `p_torso`, `hw`, `tau_w`, `dock_events`, …),
sérialisée par `to_dict()` / `save()`. Le run canonique produit **2077 ticks**.

`capture_environment()` fige les versions de la pile (python, numpy, mujoco,
pinocchio, casadi, scipy) dans le log — indispensable, car la reproduction à
l'octet n'a de sens que sur une pile épinglée.

`load` n'est pas exercé.

## ⚠ Trois canaux exportés ne portent aucun signal

Mesuré sur le log canonique :

| canal | valeur réelle | cause |
|---|---|---|
| `H_rO` | **0 partout** | `MomentumDisturbanceEstimator.update()` jamais appelé |
| `H_dot_est` | **0 partout** | idem |
| `gmo_contact_state` | **constant 0** | `ContactStateMachine.update()` jamais appelé |

À savoir avant de tracer ou d'analyser ces canaux. Détails dans
`aocs/force_estimator.md` §4 et `estimation/contact_estimator.md`.

## Conventions à connaître

- **`nmpc_ok = 0` = « non appelé »**, pas « échoué ». 1368 des 2077 ticks. Le
  taux réel est 100 % (709/709). Corriger l'encodage changerait le CSV gelé et
  demanderait une exception Tier-1 du gate — reporté après soumission.
- La référence de CoM saute vers le CoM mesuré à l'entrée SS→DS (convention de
  `_log_ds_tick`).
- La référence de torse exportée est **continue** depuis le correctif de maintien
  terminal — journalisation seule, contrôle prouvé identique à l'octet.

## Ce qui fait foi

Les métriques de référence ne viennent pas d'ici mais du gate
(`gate/dock_check.py`) et des exports (`scripts/diag_full_diag_export.py`,
`scripts/export_figure_data.py`), qui relisent ce `sim_log.json`.

## Voir aussi

- vue d'ensemble du paquet : [`simulation.md`](simulation.md)
