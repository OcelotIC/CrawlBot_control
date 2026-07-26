# `crawlbot.simulation.sim_loop`

**La boucle fermée.** Machine d'états DS/SS, orchestration des planificateurs
et solveurs, AOCS, soudures, journalisation. Le plus gros fichier du dépôt.

**Fichier** : `crawlbot/simulation/sim_loop.py` — **3387 lignes** — couverture canonique **83 %**

> Docstring du module : *« SimulationLoop — Closed-loop MuJoCo simulation with two-stage controller. »*

---

## API publique

| symbole | signature | canonique ? |
|---|---|---|
| **`SimulationLoop`** |  |  |
| `.setup` | `(n_steps=3, start_a=2, start_b=2, sequence_path=None)` | **oui** |
| `._settle_setup` | `(start_a, start_b)` | **oui** |
| `._run_ds_passivity_loop` | `(contact_config, max_steps, epsilon_v, plateau_window, p...)` | **oui** |
| `._interstep_aocs_command` | `(rs, cc_ds, lambda_qp_sol, omega_s_prev)` | **oui** |
| `._log_ds_tick` | `(log, t_abs, step_idx, just_landed_arm, anchor_a_idx, an...)` | **oui** |
| `._build_qp` | `(ae, ap, aw, kpc, kdc, kpt, kdt, kpe, kde, kpe_ang=5.0, ...)` | **oui** |
| `._build_weld_map` | `()` | **oui** |
| `._deactivate_all_welds` | `()` | **oui** |
| `._activate_weld` | `(arm, anchor_idx)` | **oui** |
| `._deactivate_weld` | `(arm, anchor_idx)` | **oui** |
| `._cache_site_ids` | `()` | **oui** |
| `._gripper_distance` | `(arm, anchor_idx)` | **oui** |
| `._gripper_speed` | `(arm)` | non exerce |
| `._gripper_ori_err_deg` | `(arm, anchor_idx)` | **oui** |
| `._weld_relative_twist` | `(arm, anchor_idx)` | **oui** |
| `._dock_gate` | `(swing_arm, target_idx, log, t, step_idx)` | **oui** |
| `._planned_arm_config` | `(t, rs)` | non exerce |
| `._setup_torso_for_step` | `(t_ss_start, swing_arm, stance_a, stance_b, target_arm, ...)` | **oui** |
| `._run_preplanner` | `(t_plan_start, stance_arm, stance_a, stance_b, r_com_0, ...)` | **oui** |
| `._capture_snapshot` | `(log, t, label)` | **oui** |
| `.run` | `(verbose=True)` | **oui** |
| `._swing_query_time` | `(t_raw, phase, ss_end)` | **oui** |
| `._step` | `(t, phase, step_idx, swing_arm, stance_arm, cc_ss, targe...)` | **oui** |
| `._get_ee_data` | `(rs, arm)` | **oui** |
| `._print_summary` | `(log)` | **oui** |
| `.plot` | `(log, save_path=None, cfg=None)` | non exerce |

---

## Architecture : deux phases, pas trois

`DS` (double appui) et `SS` (simple appui). Interdit explicite de CLAUDE.md :
*« Do not implement a three-phase state machine (DS/SS/EXT) — the architecture
is two-phase per spec §7.1. »*

```
setup()                        modeles, planificateurs, solveurs, ancrages
  |
  +-- run()                    boucle sur les pas
        +-- _setup_torso_for_step()  IK d'accostage + phase de torse
        +-- _run_preplanner()        T_step + trajectoire de CoM faisable
        +-- _step()                  SS : le vol
        +-- _run_ds_passivity_loop() DS : stabilisation passive
        +-- _dock_gate() -> _activate_weld()
```

## La porte d'accostage

Interdit de CLAUDE.md : *« Do not activate welds on position alone — require
both `d < 5 mm` AND `ori < 5°` »*. `_dock_gate` applique les deux conditions ;
`_activate_weld` n'est appelé qu'ensuite.

**Règle 10 — la métrique est celle au moment de la soudure.** La précision
d'accostage est le `d_mm` des `dock_events`, jamais le minimum sur le vol : un
passage plus proche avant l'accostage est un artefact de survol (leçon du pas 2 :
3.0 mm en survol contre **4.89 mm à la soudure**).

## `_step()` — 1013 lignes, la dette principale

Le plus gros bloc restant. Sa décomposition est identifiée mais **non faite** :
elle demande d'abord sa propre mesure de couplage (`CLEANUP_CARRYOVER` §A).

Autre dette : l'appel `WholeBodyQP.solve()` prend **40 paramètres**, dont 30 ne
sont lus que dans un seul bloc. Restructurer la signature touche les deux sites
d'appel — reporté délibérément (§A1).

## Le piège `use_m2_stack`

`SimConfig.use_m2_stack` **a l'air mort** (son jumeau côté QP a été supprimé en
CLEANUP-8), mais il commande deux chemins sans rapport avec la pile de tâches :

| site | ce qu'il commande |
|---|---|
| `sim_loop.py:2581-2584` | routage de la référence de torse (mapping δ vs quintique brute) |
| `sim_loop.py:2728-2729` | `passivity_active` — **la contrainte de passivité en DS** |

Le supprimer désactiverait silencieusement la passivité DS.

## Crochets de diagnostic — vivants, à conserver

`_diag_freeze_ref`, `_diag_lock_arm_joints`, `_diag_pure_pd` : non exercés par
le canonique mais utilisés par des scripts de `Misc/scripts/`. Troisième classe
de code « non exercé » à ne pas confondre avec du sédiment.

## Conventions de journalisation à connaître

- **`nmpc_ok = 0` signifie « non appelé », pas « échoué »** : le NMPC ne tourne
  qu'en SS et dans la stabilisation terminale. Sur le canonique cela concerne
  **1368 des 2077 ticks**, d'où un taux de succès apparent trompeur de 34 % pour
  un taux réel de **100 % (709/709)**.
- La référence de CoM **saute vers le CoM mesuré** à l'entrée SS→DS :
  `_log_ds_tick` inscrit `e_com = 0` avec `ref := mesuré`
  (`sim_loop.py:1038-1041`). Convention de log, décision en attente.
- `H_rO`, `H_dot_est` et `gmo_contact_state` **ne portent aucun signal** — voir
  `aocs/force_estimator.md` et `estimation/contact_estimator.md`.

Non exercés : `_gripper_speed`, `_planned_arm_config`, `plot`.

## Voir aussi

- vue d'ensemble du paquet : [`simulation.md`](simulation.md)
