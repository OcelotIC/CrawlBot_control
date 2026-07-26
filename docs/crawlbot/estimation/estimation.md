# `crawlbot.estimation`

Détection de contact sans capteur, par observateur de moment généralisé (GMO).

**Un seul fichier** : `crawlbot/estimation/contact_estimator.py` (261 lignes,
**69 % couvert** sur le canonique).

---

## Théorie — De Luca (2006)

Équations du mouvement (gravité nulle) :

```
M(q)·v̇ + C(q,v)·v = Sᵀ·τ + J_cᵀ·f_ext
```

Le moment généralisé `p = M(q)·v` vérifie `ṗ = Sᵀ·τ + Cᵀ·v + J_cᵀ·f_ext`.
L'observateur intégral s'écrit :

```
β̇ = Sᵀ·τ + Cᵀ·v + r        (intégrateur bouclé)
r  = K_O · (p − β)          (résidu)
```

Convergence : `ṙ = −K_O·r + K_O·τ_ext`, donc `r → τ_ext = J_cᵀ·f_ext`.

L'intérêt : **aucune mesure d'accélération ni capteur d'effort** n'est requis.
Dérivation complète dans `Misc/reports/contact_estimator_derivation.md`.

---

## `ContactObserverConfig`

| Paramètre | Défaut | Unité | Rôle |
|---|---|---|---|
| `K_O` | 80.0 | 1/s | gain de l'observateur |
| `dt` | 0.01 | s | pas de temps (cadence QP) |
| `nv` | 18 | — | dimension de la vitesse généralisée |
| `F_threshold` | 5.0 | N | seuil du résidu de swing → CONTACT |
| `d_proximity` | 0.020 | m | NO_CONTACT → PROXIMITY |
| `d_contact` | 0.010 | m | PROXIMITY → CONTACT |
| `d_reset` | 0.030 | m | tout état → NO_CONTACT |
| `debounce_count` | 5 | cycles | CONTACT → CONFIRMED |

En simulation `nv` est repris de `robot.model.nv` (`sim_loop.py:461`), pas du
défaut. Les autres champs sont alimentés depuis les `cfg.gmo_*` de `SimConfig`.

---

## `GeneralizedMomentumObserver` — actif sur le canonique

| méthode | canonique ? | où |
|---|---|---|
| `update(M, v, C_matrix, tau_applied)` | **OUI** | `sim_loop.py:3123` |
| `reset(M, v)` | **OUI** | `sim_loop.py:1947` |
| `residual` | **OUI** (log) | `sim_loop.py:1055`, `:3217` |
| `swing_residual_norm(slice)` | oui | — |
| `initialized` | non exercé | — |

Un pas d'intégration (Euler explicite) :

```python
CT_v  = C_matrixᵀ @ v
β    += dt · (τ_applied + CT_v + r)
p     = M @ v
r     = K_O · (p − β)
```

`update()` n'est appelé **qu'en simple appui** (`sim_loop.py:3123`). En double
appui le log inscrit `0.0` (`sim_loop.py:1057`), et le commentaire
`sim_loop.py:1058` précise que `gmo_swing_residual` demande une tranche de
vitesse non suivie en DS.

Le résidu est un vrai signal, mesuré sur le log canonique :

```
gmo_residual_norm   max = 8.088   moyenne = 1.017   non nuls = 2067/2077
```

---

## ⚠ `ContactStateMachine` est inerte sur le canonique

L'objet est construit (`sim_loop.py:468`) et remis à zéro (`:1948`), et son état
est lu à chaque tick pour le log (`:1061`) — mais **`update()` n'est jamais
appelé**. Couverture : 0/59 lignes.

Mesuré sur le log canonique :

```
gmo_contact_state   valeurs distinctes = [0]      (NO_CONTACT, constant)
```

Le canal est donc une constante sur toute la traversée, pas une détection.

**Ce n'est pas un bug, c'est l'architecture** : l'accostage est décidé
géométriquement, pas par le GMO. La règle est explicite dans CLAUDE.md —
*« Do not activate welds on position alone — require both `d < 5 mm` AND
`ori < 5°` »*. Le GMO fournit un résidu observable et journalisé ; la machine
d'état qui le transformerait en décision de contact n'est pas branchée sur le
chemin canonique.

À savoir avant d'utiliser `gmo_contact_state` dans une figure ou une analyse.

---

## `ContactState`

`NO_CONTACT` · `PROXIMITY` · `CONTACT` · `CONFIRMED` — les transitions sont
définies dans `ContactStateMachine.update` (`:187`), non exercée (voir ci-dessus).

---

## Fichiers liés

| quoi | où |
|---|---|
| construction GMO + machine d'état | `crawlbot/simulation/sim_loop.py:458-468` |
| pas d'observateur (SS) | `crawlbot/simulation/sim_loop.py:3123` |
| reset en début de pas | `crawlbot/simulation/sim_loop.py:1947-1948` |
| dérivation complète | `Misc/reports/contact_estimator_derivation.md` |
| porte d'accostage (géométrique) | CLAUDE.md, section « Do Not » |

---

## Documentation par module

- [`contact_estimator.md`](contact_estimator.md)
