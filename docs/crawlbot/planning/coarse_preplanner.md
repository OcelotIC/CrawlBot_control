# `crawlbot.planning.coarse_preplanner`

**Décide la durée du pas.** Un NLP centroïdal résolu une fois par pas, qui
produit `T_step` et une trajectoire de CoM faisable en moment.

**Fichier** : `crawlbot/planning/coarse_preplanner.py` — **540 lignes** — couverture canonique **81 %**

> Docstring du module : *« CoarsePrePlanner — momentum-feasible CoM trajectory optimization (M6). »*

---

## API publique

| symbole | signature | canonique ? |
|---|---|---|
| **`CoarsePrePlannerConfig`** *(dataclass)* |  |  |
|   `M` | `15` | _champ_ |
|   `robot_mass` | `71.0` | _champ_ |
|   `h_max` | `np.full(3, 5.0)` | _champ_ |
|   `kappa_terminal` | `0.7` | _champ_ |
|   `f_max` | `25.0` | _champ_ |
|   `tau_max` | `8.0` | _champ_ |
|   `tau_w_max` | `2.5` | _champ_ |
|   `w_L` | `1.0` | _champ_ |
|   `w_u` | `0.01` | _champ_ |
|   `w_v_terminal` | `100.0` | _champ_ |
|   `w_L_terminal` | `100.0` | _champ_ |
|   `eps_v_terminal` | `0.005` | _champ_ |
|   `eps_L_terminal` | `0.05` | _champ_ |
|   `T_step_default` | `6.0` | _champ_ |
|   `a_cruise_max` | `0.0` | _champ_ |
|   `cruise_ramp_frac` | `0.2` | _champ_ |
|   `ipopt_print_level` | `0` | _champ_ |
|   `ipopt_max_iter` | `300` | _champ_ |
|   `ipopt_tol` | `1e-06` | _champ_ |
| **`CoarsePlanResult`** *(dataclass)* |  |  |
|   `T_step` |  | _champ_ |
|   `t_grid` |  | _champ_ |
|   `r_com` |  | _champ_ |
|   `v_com` |  | _champ_ |
|   `L_com` |  | _champ_ |
|   `f_stance` |  | _champ_ |
|   `tau_stance` |  | _champ_ |
|   `success` |  | _champ_ |
|   `solve_time_ms` |  | _champ_ |
|   `cost` |  | _champ_ |
|   `status` |  | _champ_ |
|   `iter_count` | `0` | _champ_ |
| `.r_com_at` | `(t)` | **oui** |
| `.v_com_at` | `(t)` | **oui** |
| `.L_com_at` | `(t)` | non exerce |
| `._interp` | `(t, traj)` | **oui** |
| `.hw_at_knots` | `(c_const)` | non exerce |
| `.from_heuristic` | `(cls, r_com_0, r_com_goal, h_max, robot_mass, M=15, leve...)` | non exerce |
| **`CoarsePrePlanner`** |  |  |
| `.build` | `()` | **oui** |
| `.solve` | `(r_com_0, v_com_0, L_com_0, r_com_goal, r_C_stance, c_co...)` | **oui** |

---

## Le point essentiel : `T_step` n'est pas choisi, il est calculé

C'est ce module qui déduit la durée d'un pas de l'enveloppe de moment. Le
`ContactScheduler` crée les phases SS avec `duration = 0.0` ; la vraie valeur
n'existe qu'après ce solve, et est installée par `GaitPlan.set_step_duration()`.

## Le problème

M = 15 intervalles de collocation. Variables : `r_com, v_com, L_com` aux M+1
nœuds, `f_stance, tau_stance` aux M intervalles (un seul contact actif en SS).

Contraintes :

1. **boîte de moment à chaque nœud** :
   `c − L_com(k) − r_com(k) × m·v_com(k) ∈ [−h_max', +h_max']`
2. borne de taux `|Ḣ_s|_∞ ≤ τ_w,max = 2.5 N·m`
3. boîte force/couple : `|f|_∞ ≤ 25 N`, `|τ|_∞ ≤ 8 N·m`
4. conditions aux limites + **marge terminale** `κ = 0.7` (la boîte est
   resserrée au dernier nœud)

Intégration RK4, NLP construit une fois par `build()`, puis seuls les paramètres
changent d'un solve à l'autre.

## En cas d'échec : le pas est sauté

Pas de repli heuristique silencieux. `sim_loop` journalise, maintient la
position, saute le pas. C'est un choix de conception explicite.

## ⚠ `from_heuristic` : 83 lignes documentées comme fixture de test, sans test

Quatre commentaires (`sim_loop.py:1369`, `:1599`, `:1633`, `config.py:207`)
affirment que les tests unitaires l'utilisent pour éviter la dépendance IPOPT.
**Aucun test ne l'appelle.** Recherché sur *tout* l'historique du dépôt, le seul
`.from_heuristic(` existant est ce commentaire.

Pire : les commentaires sont porteurs, car `sim_loop.py:1632-1640` **réimplémente
sa formule d'enveloppe en ligne** et cite la fixture pour justifier la
duplication. Suppression en attente (audit CLEANUP-19).

## Valeurs canoniques silencieuses (règle 5)

Cinq champs ne sont jamais écrasés par `sim_loop`, donc leur défaut *est* la
valeur canonique — et **aucun n'est dans CLAUDE.md** :

| champ | valeur | ce qu'il décide |
|---|---|---|
| `eps_v_terminal` | 5e-3 m/s | **boîte dure** sur la vitesse de CoM terminale |
| `eps_L_terminal` | 5e-2 N·m·s | **boîte dure** sur le moment terminal |
| `w_v_terminal`, `w_L_terminal` | 1e2 | pénalités douces sur le même résidu |
| `ipopt_tol` | 1e-6 | tolérance de convergence |

Les deux premières décident **où un pas a le droit de finir**. `T_step_default`
(6.0 s) n'est jamais utilisé : `sim_loop` passe toujours `T_step`.

## Le reste du code non couvert : à conserver

Les 12 lignes mortes de `solve()` sont **entièrement** l'échelle de traitement
d'erreur (`except RuntimeError` d'IPOPT, repli d'extraction de valeurs, repli de
stats) plus deux défauts d'API. Signature d'un système sain — pas du sédiment.

Le bloc d'accélération de croisière (`a_cruise_max`, M7 v21) est désactivé
(`0.0`) et n'est atteignable qu'en éditant `SimConfig` à la main. Sa suppression
attend un arbitrage : c'est un paramètre *documenté* dans CLAUDE.md.

## Voir aussi

- vue d'ensemble du paquet : [`planning.md`](planning.md)
