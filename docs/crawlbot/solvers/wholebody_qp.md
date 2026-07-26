# `crawlbot.solvers.wholebody_qp`

**Étage 2 du contrôleur — le cœur canonique.** Traduit à 100 Hz les références
(torse, effecteur, moment, posture) en accélérations articulaires et couples.

**Fichier** : `crawlbot/solvers/wholebody_qp.py` — **950 lignes** — couverture canonique **97 %**

> Docstring du module : *« WholeBodyQP - Whole-body Quadratic Program for high-rate tracking. »*

---

## API publique

| symbole | signature | canonique ? |
|---|---|---|
| **`WholeBodyQPConfig`** *(dataclass)* |  |  |
|   `nq` | `14` | _champ_ |
|   `nc_max` | `2` | _champ_ |
|   `method` | `'weighted'` | _champ_ |
|   `solver` | `'qpoases'` | _champ_ |
|   `weight_ratio` | `1.0` | _champ_ |
|   `alpha_ee` | `500.0` | _champ_ |
|   `alpha_posture` | `100.0` | _champ_ |
|   `alpha_wrench` | `10.0` | _champ_ |
|   `alpha_torque` | `1.0` | _champ_ |
|   `alpha_reg` | `0.01` | _champ_ |
|   `alpha_lambda_int` | `0.0` | _champ_ |
|   `ds_centroidal_mode` | `False` | _champ_ |
|   `ds_alpha_com` | `100.0` | _champ_ |
|   `ds_alpha_torso_ori` | `200.0` | _champ_ |
|   `ds_alpha_posture` | `50.0` | _champ_ |
|   `ss_two_task_mode` | `False` | _champ_ |
|   `ss_alpha_mom` | `500.0` | _champ_ |
|   `alpha_torso_pose` | `1000.0` | _champ_ |
|   `alpha_passivity` | `1.0` | _champ_ |
|   `passivity_W_budget` | `0.0` | _champ_ |
|   `qp_envelope_exact` | `False` | _champ_ |
|   `w_hw_slack` | `800.0` | _champ_ |
|   `Kp_com` | `100.0 * np.ones(3)` | _champ_ |
|   `Kd_com` | `20.0 * np.ones(3)` | _champ_ |
|   `Kp_torso` | `np.array([8.0, 8.0, 8.0, 5.0, 5.0, 5.0])` | _champ_ |
|   `Kd_torso` | `np.array([6.0, 6.0, 6.0, 4.0, 4.0, 4.0])` | _champ_ |
|   `Kp_ee` | `80.0 * np.ones(3)` | _champ_ |
|   `Kd_ee` | `15.0 * np.ones(3)` | _champ_ |
|   `Kp_ee_ang` | `5.0 * np.ones(3)` | _champ_ |
|   `Kd_ee_ang` | `3.0 * np.ones(3)` | _champ_ |
|   `Kp_posture` | `25.0` | _champ_ |
|   `Kd_posture` | `10.0` | _champ_ |
|   `Kd_settle` | `10.0` | _champ_ |
|   `alpha_settle` | `1000.0` | _champ_ |
|   `tau_max` | `50.0 * np.ones(14)` | _champ_ |
|   `qdd_max` | `50.0` | _champ_ |
|   `dt_qp` | `0.008` | _champ_ |
|   `f_max` | `3000.0` | _champ_ |
|   `tau_contact_max` | `300.0` | _champ_ |
|   `L_max` | `np.inf` | _champ_ |
|   `tau_w_max` | `np.inf` | _champ_ |
| **`WholeBodyQP`** |  |  |
| `.set_nominal_posture` | `(q_nom)` | **oui** |
| `.solve` | `(dq_t, q, dq, r_com_ref, v_com_ref, lambda_ref, a_com_ff...)` | **oui** |
| `._add_equality_constraints` | `(qp, H_robot, C_robot, J_contacts, Jdot_dq_contacts, con...)` | **oui** |
| `._add_inequality_constraints` | `(qp, H_robot, dq, r_com, hw_current, hw_min, hw_max, L_c...)` | **oui** |
| `._set_variable_bounds` | `(qp, contact_config, hw_constraint_active)` | **oui** |
| `._com_task_rows` | `(J_com, Jdot_dq_com, dq_robot, r_com, r_com_ref, v_com_r...)` | **oui** |
| `._compute_indices` | `()` | **oui** |
| `.n_vars` | `()` | non exerce |
| `.variable_indices` | `()` | non exerce |

---

## La pile à deux tâches — la configuration canonique

Quatre tâches, **toutes pondérées**, résolues en un seul QP :

| tâche | α canonique | rôle |
|---|---:|---|
| pose de torse (6-D) | **2000** | levier fin de précision d'accostage |
| effecteur en vol | **1000** | levier d'allonge — doit rester ≥ ~1000 |
| moment linéaire (T-MOM) | **400** | quasi inerte sur Ḣ_s ; le NMPC tient l'enveloppe |
| posture | **20** | régularisation articulaire |
| minimisation de couple | 5 | ≳ 5 × le plancher de régularisation |
| suivi de torseur | 1.0 | au-delà de 1, bloque l'autorité torse/EE |
| régularisation d'accélération | 1.0 | plancher de conditionnement |
| slack sur `h_w` | 800 | actif seulement si la boîte `h_w` est violée |

## ⚠ Il n'y a pas de projection dans l'espace nul

`weight_ratio = 1.0` ⇒ **les magnitudes d'α _sont_ la hiérarchie.** Les entiers
de priorité nominale sont inertes. Trois interdits en découlent, tous dans
CLAUDE.md :

- pas de `weight_ratio > 1` ;
- pas de `alpha_wrench > 1` — à 100, la régularisation de torseur consommait
  20 % du budget du QP et étouffait les tâches torse et effecteur ;
- **règle 14** : `alpha_torque ≳ 5 × alpha_reg`. À un rapport 1:1, la résolution
  de redondance en SS dégénère en dépassement de délai d'accostage.

## Structure de `solve()` après CLEANUP-11

Le corps est passé de 543 à ~346 lignes par extraction de quatre helpers, tous
sur le chemin canonique :

| helper | rôle |
|---|---|
| `_add_equality_constraints` | dynamique + contacts |
| `_add_inequality_constraints` | boîtes ; renvoie `hw_constraint_active` |
| `_set_variable_bounds` | bornes des variables de décision |
| `_com_task_rows` | lignes de la tâche CoM ; renvoie `(A_com, b_com)` |

⚠ Ce qui a été **explicitement écarté** : fusionner ou réordonner les blocs de
tâches. L'ordre encode la séquence d'assemblage du coût — le changer serait une
modification de comportement déguisée en refactorisation.

## Valeurs canoniques silencieuses (règle 5)

Huit champs ne sont **jamais** écrasés par `sim_loop`, donc leur défaut *est* la
valeur canonique : `method`, `solver`, `weight_ratio`, `w_hw_slack`,
`alpha_settle`, `Kd_settle`, `qdd_max`, `tau_contact_max`. Seul `w_hw_slack` est
cité dans CLAUDE.md (`CLEANUP_CARRYOVER` §C4).

## Dette identifiée

`solve()` prend **40 paramètres**, dont 30 ne sont lus que dans un seul bloc.
Restructurer la signature touche les deux sites d'appel dans `sim_loop` —
reporté délibérément pour ne pas brouiller le diff prouvant que l'extraction
était inerte (`CLEANUP_CARRYOVER` §A1).

Le fichier est passé de 1385 à 950 lignes pendant le chantier, et c'est le
mieux couvert du dépôt (**97 %**).

## Voir aussi

- vue d'ensemble du paquet : [`solvers.md`](solvers.md)
