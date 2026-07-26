# `crawlbot.solvers.hierarchical_qp`

Backend QP générique : empile des tâches pondérées ou strictement
hiérarchisées, ajoute contraintes et bornes, et résout.

**Fichier** : `crawlbot/solvers/hierarchical_qp.py` — **529 lignes** — couverture canonique **70 %**

> Docstring du module : *« HierarchicalQP - Generic hierarchical quadratic program solver. »*

---

## API publique

| symbole | signature | canonique ? |
|---|---|---|
| **`Task`** *(dataclass)* |  |  |
|   `A` |  | _champ_ |
|   `b` |  | _champ_ |
|   `W` |  | _champ_ |
|   `priority` |  | _champ_ |
| **`QPSolveInfo`** *(dataclass)* |  |  |
|   `method` | `''` | _champ_ |
|   `success` | `False` | _champ_ |
|   `exitflag` | `-1` | _champ_ |
|   `cost` | `np.inf` | _champ_ |
|   `lambda_eq` | `None` | _champ_ |
|   `lambda_ineq` | `None` | _champ_ |
|   `lambda_lb` | `None` | _champ_ |
|   `lambda_ub` | `None` | _champ_ |
|   `failed_priority` | `None` | _champ_ |
|   `solve_time_ms` | `0.0` | _champ_ |
|   `n_iter` | `0` | _champ_ |
| **`HierarchicalQP`** |  |  |
| `.add_task` | `(A, b, W, priority)` | **oui** |
| `.add_equality_constraint` | `(C, d)` | **oui** |
| `.add_inequality_constraint` | `(C, d)` | **oui** |
| `.set_bounds` | `(lb, ub)` | **oui** |
| `.clear_tasks` | `()` | non exerce |
| `.clear_constraints` | `()` | non exerce |
| `.solve` | `(x0=None)` | **oui** |
| `._solve_weighted` | `(sorted_tasks, x0)` | **oui** |
| `._solve_strict` | `(sorted_tasks, x0)` | non exerce |
| `._solve_qp_raw` | `(H, g, C_eq, d_eq, C_ineq, d_ineq, lb, ub, x0=None)` | **oui** |
| `._get_solver_options` | `()` | **oui** |
| `.n_tasks` | `()` | non exerce |

---

## Deux méthodes de résolution, une seule utilisée

| chemin | canonique ? | note |
|---|---|---|
| `_solve_weighted` | **oui** — `method='weighted'` | tâches en somme pondérée |
| `_solve_qp_raw` | **oui** | l'appel solveur proprement dit (qpOASES) |
| `_solve_strict` | **non exercé** (76 lignes) | hiérarchie stricte par projection |

`_solve_strict` est le plus gros bloc non exercé restant dans `crawlbot/`. Il
n'est pas obsolète : 2 tests et 6 scripts l'utilisent. Sa suppression est une
décision — « la voie hiérarchique stricte reste-t-elle reproductible ? » — pas
une évidence (`CLEANUP_CARRYOVER` §B2).

## Conditionnement

Régularisation de Tikhonov `ε = 1e-6`, **inerte en pratique** :
`λ_min(H_LS) = 1 ≫ ε`. Le conditionnement canonique mesuré est
`κ_SS(H) ≈ 7.5e3`, soit 530× mieux qu'avant le gel 2.5.

⚠ Le replay canonique force explicitement `regularization = 1e-6`
(`gate/replay_canonical.py`) pour reproduire à l'octet les artefacts gelés.

## `Task`

`Task(A, b, W, priority)` — en mode pondéré, **`priority` est inerte** : ce sont
les magnitudes de `W` qui font la hiérarchie (voir `wholebody_qp.md`).

## Voir aussi

- vue d'ensemble du paquet : [`solvers.md`](solvers.md)
