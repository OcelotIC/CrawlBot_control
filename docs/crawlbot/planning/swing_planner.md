# `crawlbot.planning.swing_planner`

Trajectoire de l'effecteur en vol : quintique + dégagement, pilotée par le plan
de marche. **95 % de couverture.**

**Fichier** : `crawlbot/planning/swing_planner.py` — **338 lignes** — couverture canonique **95 %**

> Docstring du module : *« Swing arm trajectory planner for crawling locomotion. »*

---

## API publique

| symbole | signature | canonique ? |
|---|---|---|
| **`SwingReference`** *(dataclass)* |  |  |
|   `p_ee` |  | _champ_ |
|   `v_ee` |  | _champ_ |
|   `a_ee` |  | _champ_ |
|   `R_ee` |  | _champ_ |
|   `omega_ee` |  | _champ_ |
|   `alpha_ee` |  | _champ_ |
|   `swing_arm` |  | _champ_ |
|   `is_swinging` |  | _champ_ |
|   `phase_progress` |  | _champ_ |
| **`SwingPlanner`** |  |  |
| `.set_swing_orientation` | `(R_start)` | **oui** |
| `.plan` | `()` | **oui** |
| `._quintic` | `(tau)` | **oui** |
| `._quintic_dot` | `(tau)` | **oui** |
| `._quintic_ddot` | `(tau)` | **oui** |
| `._bump` | `(tau)` | **oui** |
| `._bump_dot` | `(tau)` | **oui** |
| `._bump_ddot` | `(tau)` | **oui** |
| `._delayed_cosine` | `(tau, tau_d)` | **oui** |
| `._delayed_cosine_dot` | `(tau, tau_d)` | **oui** |
| `._delayed_cosine_ddot` | `(tau, tau_d)` | **oui** |
| `.reference_at` | `(t)` | **oui** |
| `._last_swing_position` | `(current_idx)` | **oui** |

### Constantes de module

| nom | valeur |
|---|---|
| `DEFAULT_CLEARANCE` | `0.03` |
| `DEFAULT_AWAY_NORMAL` | `np.array([0.0, 0.0, -1.0])` |

---

## Profils

Trois briques composables, toutes exercées avec leurs dérivées première et
seconde :

| profil | rôle |
|---|---|
| `_quintic` | `s(τ) = 10τ³ − 15τ⁴ + 6τ⁵` — accélération nulle aux extrémités |
| `_bump` | la bosse de dégagement (`clearance` par défaut 0.03 m) |
| `_delayed_cosine` | mise en rotation retardée |

Normale de dégagement : `−z` en repère structure (le robot est *sous* la
structure).

## `reference_at(t)`

Interroge le plan de marche :

- **DS** → dernière position de vol figée, vitesse et accélération nulles ;
- **SS** → interpolation sur `T_eff = T_step × early_finish_fraction`, avec `τ`
  écrêté à 1. Une fois `τ = 1` atteint, la position est à la cible et vitesse et
  accélération sont nulles par construction — les trois profils ont `ṗ(τ=1) = 0`.

C'est le mécanisme d'« arrivée anticipée » : l'effecteur atteint sa cible avant
la fin de la phase et s'y tient, ce qui laisse le temps à la porte d'accostage
de se déclencher proprement.

## Ce que CLEANUP-18 a retiré

Tout le mécanisme de **surcharge de phase** : `add_phase`,
`_override_reference_at`, `clear_phase_overrides`, `_phase_overrides` et sa
boucle de dispatch, plus `adaptive_reference_at` et `swing_trajectory`.

`reference_at()` prend désormais **toujours** le chemin piloté par le plan —
celui que le canonique a toujours emprunté. Couverture passée de **47 % à 95 %**,
fichier de 728 à 337 lignes.

⚠ Ne pas confondre : **`torso_planner.add_phase` est vivante**. Seul le
`add_phase` du *swing* était mort.

## Voir aussi

- vue d'ensemble du paquet : [`planning.md`](planning.md)
