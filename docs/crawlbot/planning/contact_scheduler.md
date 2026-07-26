# `crawlbot.planning.contact_scheduler`

Le plan de marche : séquence DS/SS/DS…, grille d'ancrages, et le timing qui se
recalcule en cascade quand la durée d'un pas est connue.

**Fichier** : `crawlbot/planning/contact_scheduler.py` — **350 lignes** — couverture canonique **87 %**

> Docstring du module : *« ContactScheduler — Gait timing and contact management for VISPA crawling. »*

---

## API publique

| symbole | signature | canonique ? |
|---|---|---|
| `make_anchor_grid` | `(n=DEFAULT_N_ANCHORS, dx=DEFAULT_DX, dy=DEFAULT_DY)` | non exerce |
| `read_anchors_from_mujoco` | `(mj_model, mj_data)` | **oui** |
| **`GaitPhase`** *(dataclass)* |  |  |
|   `phase` |  | _champ_ |
|   `duration` |  | _champ_ |
|   `anchor_a_idx` |  | _champ_ |
|   `anchor_b_idx` |  | _champ_ |
|   `swing_arm` | `''` | _champ_ |
|   `swing_from_idx` | `-1` | _champ_ |
|   `swing_to_idx` | `-1` | _champ_ |
| **`GaitPlan`** *(dataclass)* |  |  |
|   `phases` |  | _champ_ |
|   `t_start` |  | _champ_ |
|   `t_end` |  | _champ_ |
|   `total_duration` |  | _champ_ |
| `.phase_at` | `(t)` | **oui** |
| `.set_step_duration` | `(idx, T_step)` | **oui** |
| **`ContactScheduler`** |  |  |
| `.plan_traversal` | `(start_a=0, start_b=0, n_steps=4)` | **oui** |
| `.plan` | `()` | **oui** |
| `.contact_config_at` | `(t)` | **oui** |
| `.contact_sequence_over_horizon` | `(t, dt, N)` | non exerce |
| `.anchor_se3` | `(arm, idx)` | **oui** |

### Constantes de module

| nom | valeur |
|---|---|
| `DEFAULT_DX` | `0.8` |
| `DEFAULT_DY` | `0.3` |
| `DEFAULT_N_ANCHORS` | `6` |

---

## Production du plan

`plan_traversal(start_a, start_b, n_steps)` construit un `GaitPlan` : suite de
`GaitPhase` (DOUBLE / SINGLE_A / SINGLE_B) portant les indices d'ancrage et le
bras en vol. Les bras avancent en alternance, d'un ancrage à la fois.

⚠ Les phases SS naissent avec **`duration = 0.0`** : la durée réelle vient du
pré-planificateur. `GaitPlan.set_step_duration(idx, T_step)` l'installe et
**reconstruit tout le timing en cascade** (`sim_loop.py:1495`), en préservant
l'invariant `t_end[k] = t_start[k] + duration[k]`, `t_start[k+1] = t_end[k]`.

## Ancrages

Canonique : `read_anchors_from_mujoco` — lus directement du modèle MuJoCo, donc
cohérents avec la scène simulée.

`make_anchor_grid` (grille analytique, `dx = 0.8`, `dy = 0.3`, 6 ancrages) sert
de repli quand aucun ancrage n'est fourni, et reste utilisée par un script de
diagnostic. Non exercée sur le canonique.

## Fichier essentiellement propre

16 lignes non couvertes sur 120 — **15 sont des gardes et des replis** :

- `set_step_duration` : `IndexError` (indice hors bornes), `ValueError`
  (`T_step ≤ 0`)
- propriété `plan` : `RuntimeError("Call plan_traversal() first.")`
- `plan_traversal` : les deux `break` d'épuisement de la grille
- `read_anchors_from_mujoco` : `ImportError`, `except: break`, `RuntimeError`
- `__init__` : le repli `anchors_a is None`

Toutes mortes **parce que le système est sain**. Ne pas les supprimer.

**Une seule méthode réellement morte** : `contact_sequence_over_horizon`
(19 lignes, zéro appelant) — plomberie prévue pour l'horizon NMPC que le NMPC
n'a jamais consommée.

## Voir aussi

- vue d'ensemble du paquet : [`planning.md`](planning.md)
