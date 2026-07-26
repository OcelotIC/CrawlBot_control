# `crawlbot.core.com_to_torso_mapping`

Convertit une référence de CoM en référence de torse : `p_torso = r_com − δ(q)`.
**Chemin DS uniquement.**

**Fichier** : `crawlbot/core/com_to_torso_mapping.py` — **257 lignes** — couverture canonique **52 %**

> Docstring du module : *« CoM-to-torso reference mapping (M1, v1: with delta_dot). »*

---

## API publique

| symbole | signature | canonique ? |
|---|---|---|
| **`CoMToTorsoMapping`** |  |  |
| `.compute_delta` | `(q)` | **oui** |
| `.compute_delta_dot` | `(q, dq)` | **oui** |
| `.compute_delta_local` | `(q)` | non exerce |
| `.compute_delta_local_dot` | `(q, dq)` | non exerce |
| `.compute` | `(r_com_ref, v_com_ref, a_com_ff, q_current, dq_current=None)` | non exerce |
| `.body_com_jacobian` | `(data, joint_idx)` | **oui** |
| `.torso_pos_jacobian_from_com` | `(q)` | non exerce |

### Constantes de module

| nom | valeur |
|---|---|
| `TORSO_JOINT_IDX` | `1` |

---

## Le principe

Le NMPC raisonne sur le CoM ; le QP suit une pose de torse. δ(q) est l'écart
entre les deux dans la configuration courante, et sa dérivée δ̇(q, q̇) donne la
référence de vitesse cohérente.

## ⚠ En SS, ce mapping n'est PAS utilisé

Interdit explicite de CLAUDE.md :

> *Do not route the SS torso reference through the δ-mapping in two-task mode —
> SS uses the raw TorsoPlanner quintic (`sim_loop.py:2581-2584`); the mapping
> (δ(q_current)+F-SAT) remains a **DS-only** path.*

En simple appui, la tâche de pose de torse est alimentée par la quintique+SLERP
brute du `TorsoPlanner`. Le mapping ne sert qu'en double appui.

C'est ce qui explique la couverture : `compute()` — l'entrée « complète » du
mapping — n'est **pas exercée**, tandis que les briques `compute_delta` et
`compute_delta_dot` le sont, appelées depuis le chemin DS.

## Pourquoi δ n'est recalculé qu'à 10 Hz

Recalculer δ(q) à la cadence du QP (100 Hz) referme une boucle de rétroaction
sur le mapping qui fait osciller la référence jusqu'à **237 mm/tick** sur les
grands vols. Le correctif (F-RATE) recalcule δ et δ̇ **une fois par tick NMPC**
(10 Hz) ; le terme interpolé continue de varier à 100 Hz. Voir le commentaire
détaillé à `sim_loop.py:2584-2596`.

Non exercés : `compute`, `compute_delta_local`, `compute_delta_local_dot`,
`torso_pos_jacobian_from_com`.

## Voir aussi

- vue d'ensemble du paquet : [`core.md`](core.md)
