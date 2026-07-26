# `crawlbot.solvers.centroidal_nmpc`

**Étage 1 du contrôleur.** Génère, sur un horizon glissant, une trajectoire de
CoM et de moment *faisable vis-à-vis de l'enveloppe des roues à réaction*.

**Fichier** : `crawlbot/solvers/centroidal_nmpc.py` — **702 lignes** — couverture canonique **88 %**

> Docstring du module : *« CentroidalNMPC - Centroidal NMPC for momentum-feasible trajectory generation. »*

---

## API publique

| symbole | signature | canonique ? |
|---|---|---|
| **`CentroidalNMPCConfig`** *(dataclass)* |  |  |
|   `robot_mass` | `90.0` | _champ_ |
|   `N` | `20` | _champ_ |
|   `dt` | `0.05` | _champ_ |
|   `Wr` | `100.0 * np.ones(3)` | _champ_ |
|   `Wv` | `10.0 * np.ones(3)` | _champ_ |
|   `Wu_f` | `0.01` | _champ_ |
|   `Wu_tau` | `0.001` | _champ_ |
|   `Qf_r` | `1000.0 * np.ones(3)` | _champ_ |
|   `Qf_v` | `100.0 * np.ones(3)` | _champ_ |
|   `f_max` | `3000.0` | _champ_ |
|   `tau_max` | `300.0` | _champ_ |
|   `L_max` | `np.inf` | _champ_ |
|   `tau_w_max` | `np.inf` | _champ_ |
|   `p_max` | `np.inf` | _champ_ |
|   `enforce_hw_conservation` | `False` | _champ_ |
|   `h_max_tight` | `np.full(3, 5.0)` | _champ_ |
|   `w_L` | `1.0` | _champ_ |
|   `Qf_L` | `10.0` | _champ_ |
|   `kappa_terminal` | `1.0` | _champ_ |
|   `solver_name` | `'ipopt'` | _champ_ |
|   `solver_opts` | `field(default_factory=dict)` | _champ_ |
| **`CentroidalNMPC`** |  |  |
| `.build` | `(solver_opts=None)` | **oui** |
| `.solve` | `(r_com, v_com, L_com, r_com_ref, v_com_ref, contact_conf...)` | **oui** |
| `.get_last_trajectory` | `()` | **oui** |
| `.get_shifted_fallback` | `()` | non exerce |
| `.compute_c_simple` | `(r_com, v_com, L_com, hw_current=None)` | **oui** |
| `.reset_warm_start` | `()` | **oui** |
| `.get_full_trajectory` | `(r_com, v_com, L_com, r_com_ref, v_com_ref, contact_conf...)` | non exerce |
| `.compute_feedforward_acceleration` | `(lambda_ref)` | **oui** |
| `._assemble_params` | `(r_com, v_com, L_com, r_com_ref, v_com_ref, contact_conf...)` | **oui** |
| `._apply_contact_bounds` | `(contact_config)` | **oui** |

---

## Le problème résolu

État `x = [r_com, v_com, L_com]` (**9**), commande `u = [f_A; τ_A; f_B; τ_B]`
(**12**). Horizon **N = 8** à **dt = 0.1 s** — tir multiple, transcrit en NLP
CasADi et résolu par IPOPT.

Dynamique centroïdale, un ou deux contacts actifs :

```
ṙ_com = v_com
v̇_com = Σ f_i / m
L̇_com = Σ (r_Ci − r_com) × f_i + τ_i
```

## La contrainte qui donne son sens au module

Sans elle, ce NMPC ne serait qu'un générateur de trajectoire de CoM. Avec
`enforce_hw_conservation`, le moment des roues est **reconstruit à chaque nœud**
depuis la conservation du moment total, et borné :

```
h_w(k) = c − L_com(k) − r_com(k) × m·v_com(k)  ∈  [−h_max', +h_max']
```

`c` est la constante de conservation, calculée une fois par pas par
`compute_c_simple()`. C'est ce qui rend le plan *faisable* : une trajectoire qui
saturerait les roues est rejetée à l'optimisation, pas découverte en vol.

S'y ajoute la borne de taux `|Ḣ_s|_∞ ≤ τ_w,max = 2.5 N·m` — le même cap qu'au
QP et que dans le modèle MuJoCo (trois points d'application).

## ⚠ Les défauts de `CentroidalNMPCConfig` ne sont PAS les valeurs canoniques

Le tableau ci-dessus affiche `robot_mass=90.0`, `N=20`, `dt=0.05`. **Aucune de
ces valeurs n'est utilisée par le run canonique** : `sim_loop.py:383-398` écrase
chaque champ depuis `SimConfig` (≈ 71 kg, `N=8`, `dt=0.1`).

Le cas le plus coûteux est `enforce_hw_conservation`, dont le défaut est `False`
alors que **le canonique le met à `True`**. C'est exactement l'erreur rétractée
par le chantier (F1) : le chemin `h_w` avait été déclaré mort d'après ce défaut.
Mesuré sur le run réel : `enforce_hw=True`, `ng_path=17`, `ng_term=6`.

> Une valeur par défaut de dataclass n'est pas la valeur canonique.

## Deux méthodes non exercées, deux statuts opposés

| méthode | pourquoi non exercée | verdict |
|---|---|---|
| `get_shifted_fallback` | **aucun solve n'échoue** sur le canonique | **conserver** — c'est le repli |
| `get_full_trajectory` | zéro appelant dans `crawlbot/` | 5 tests + 1 script l'utilisent — décision en attente |

`get_shifted_fallback` illustre la classe de code mort la plus dangereuse à
supprimer : morte *parce que le système est sain*, pas parce qu'elle est
obsolète.

## Correctif F2 (CLEANUP-3)

Le warm-start n'est repris que si `info.success` est vrai. Auparavant une
solution ratée pouvait amorcer le pas suivant et propager sa divergence.

## Voir aussi

- vue d'ensemble du paquet : [`solvers.md`](solvers.md)
