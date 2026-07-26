# `crawlbot.estimation.contact_estimator`

Observateur de moment généralisé (De Luca 2006) : détecte le contact **sans
capteur d'effort ni mesure d'accélération**.

**Fichier** : `crawlbot/estimation/contact_estimator.py` — **261 lignes** — couverture canonique **69 %**

> Docstring du module : *« Generalized Momentum Observer (GMO) for sensorless contact detection. »*

---

## API publique

| symbole | signature | canonique ? |
|---|---|---|
| **`ContactObserverConfig`** *(dataclass)* |  |  |
|   `K_O` | `80.0` | _champ_ |
|   `dt` | `0.01` | _champ_ |
|   `nv` | `18` | _champ_ |
|   `F_threshold` | `5.0` | _champ_ |
|   `d_proximity` | `0.02` | _champ_ |
|   `d_contact` | `0.01` | _champ_ |
|   `d_reset` | `0.03` | _champ_ |
|   `debounce_count` | `5` | _champ_ |
| **`ContactState`** |  |  |
| **`GeneralizedMomentumObserver`** |  |  |
| `.reset` | `(M, v)` | **oui** |
| `.update` | `(M, v, C_matrix, tau_applied)` | **oui** |
| `.residual` | `()` | **oui** |
| `.initialized` | `()` | non exerce |
| `.swing_residual_norm` | `(swing_v_slice)` | **oui** |
| **`ContactStateMachine`** |  |  |
| `.update` | `(r_swing_norm, d_FK, force_mode=False)` | non exerce |
| `.reset` | `()` | **oui** |
| `.state` | `()` | **oui** |
| `.is_docked` | `()` | non exerce |

---

## Théorie

Équations du mouvement (gravité nulle) :

```
M(q)·v̇ + C(q,v)·v = Sᵀ·τ + J_cᵀ·f_ext
```

Le moment généralisé `p = M(q)·v` vérifie `ṗ = Sᵀ·τ + Cᵀ·v + J_cᵀ·f_ext`.
L'observateur intégral :

```
β̇ = Sᵀ·τ + Cᵀ·v + r
r  = K_O · (p − β)
```

Convergence : `ṙ = −K_O·r + K_O·τ_ext`, donc `r → τ_ext = J_cᵀ·f_ext`.

Un pas (Euler explicite, `K_O = 80`, `dt = 0.01`) :

```python
β += dt · (τ_applied + C_matrixᵀ @ v + r)
r  = K_O · (M @ v − β)
```

## Ce qui tourne

`GeneralizedMomentumObserver.update()` est appelé **en simple appui seulement**
(`sim_loop.py:3123`) ; en double appui le journal inscrit `0.0`. Le résidu est
un vrai signal, mesuré sur le canonique :

```
gmo_residual_norm   max = 8.088   moyenne = 1.017   non nuls = 2067/2077
```

## ⚠ `ContactStateMachine` est inerte

Construite (`sim_loop.py:468`), remise à zéro (`:1948`), son état est lu à chaque
tick pour le journal (`:1061`) — mais **`update()` n'est jamais appelé**
(0/59 lignes).

```
gmo_contact_state   valeurs distinctes = [0]     (NO_CONTACT, constant)
```

**Ce n'est pas un bug, c'est l'architecture** : l'accostage est décidé
géométriquement, jamais par le GMO. Règle de CLAUDE.md : *« require both
`d < 5 mm` AND `ori < 5°` »*. Le GMO fournit un résidu observable et journalisé ;
la machine d'état qui le transformerait en décision n'est pas branchée.

À savoir avant d'utiliser `gmo_contact_state` dans une figure.

## Voir aussi

- vue d'ensemble du paquet : [`estimation.md`](estimation.md)
