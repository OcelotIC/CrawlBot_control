# `crawlbot.solvers.nmpc_solver`

Constructeur de NLP par tir multiple, indépendant du problème : dynamique,
coûts, contraintes, bornes. `centroidal_nmpc` l'instancie.

**Fichier** : `crawlbot/solvers/nmpc_solver.py` — **650 lignes** — couverture canonique **95 %**

> Docstring du module : *« NMPCSolver - Generic Nonlinear Model Predictive Control solver with CasADi. »*

---

## API publique

| symbole | signature | canonique ? |
|---|---|---|
| **`NMPCSolveInfo`** *(dataclass)* |  |  |
|   `cost` | `np.inf` | _champ_ |
|   `success` | `False` | _champ_ |
|   `status` | `''` | _champ_ |
|   `iterations` | `0` | _champ_ |
|   `solve_time_ms` | `0.0` | _champ_ |
|   `solver_stats` | `None` | _champ_ |
| **`NMPCSolver`** |  |  |
| `.set_continuous_dynamics` | `(ode_func)` | **oui** |
| `.set_stage_cost` | `(cost_func)` | **oui** |
| `.set_terminal_cost` | `(cost_func)` | **oui** |
| `.set_path_constraints` | `(constraint_func, ng)` | **oui** |
| `.set_terminal_constraints` | `(constraint_func, ng)` | **oui** |
| `.set_state_bounds` | `(x_min, x_max)` | **oui** |
| `.set_control_bounds` | `(u_min, u_max)` | **oui** |
| `.apply_control_bounds_all_stages` | `(u_min, u_max)` | **oui** |
| `.set_parameters` | `(np_)` | **oui** |
| `.build` | `(solver_opts=None)` | **oui** |
| `.solve` | `(x0, params=None, warm_start=True)` | **oui** |
| `.shift_warm_start` | `()` | **oui** |
| `.reset_warm_start` | `()` | **oui** |
| `._build_initial_guess` | `(x0, warm_start)` | **oui** |
| `._build_w0_from_trajectories` | `(x_traj, u_traj)` | **oui** |
| `._parse_solution` | `(w)` | **oui** |
| `._get_default_solver_options` | `()` | **oui** |

---

## Rôle

`NMPCSolver` ne connaît rien de la robotique : on lui donne une dynamique
continue, des coûts d'étage et terminal, des contraintes de chemin et
terminales, des bornes — il transcrit en NLP CasADi et résout.

C'est le module le mieux couvert du paquet (**95 %**) : **toutes** les méthodes
publiques sont sur le chemin canonique.

## Deux apports du chantier CLEANUP

- **`apply_control_bounds_all_stages`** (CLEANUP-1/3) : applique les bornes de
  commande à tous les étages de l'horizon en une fois, au lieu d'une boucle
  dispersée.
- **Correctif F2** : le warm-start n'est repris que si `info.success` est vrai.
  Auparavant une solution ratée pouvait amorcer le pas suivant.

## À savoir

`build()` est coûteux (transcription + génération de code CasADi) et n'est
appelé **qu'une fois**, au `setup()` de la simulation. Les appels suivants à
`solve()` ne font que remettre à jour les paramètres du NLP déjà construit.

## Voir aussi

- vue d'ensemble du paquet : [`solvers.md`](solvers.md)
