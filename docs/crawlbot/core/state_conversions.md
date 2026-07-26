# `crawlbot.core.state_conversions`

Le pont MuJoCo ↔ Pinocchio. **Seul module du paquet couvert à 100 %** — tout
passe par lui.

**Fichier** : `crawlbot/core/state_conversions.py` — **165 lignes** — couverture canonique **100 %**

> Docstring du module : *« State conversions between MuJoCo (world frame) and Pinocchio (structure frame). »*

---

## API publique

| symbole | signature | canonique ? |
|---|---|---|
| `mujoco_to_pinocchio` | `(mj_qpos, mj_qvel)` | **oui** |
| `pinocchio_to_mujoco` | `(pin_q, pin_v, struct_pos=None, struct_quat=None, rwa=False)` | **oui** |
| `quat_wxyz_to_euler_deg` | `(qw, qx, qy, qz)` | **oui** |

### Constantes de module

| nom | valeur |
|---|---|
| `_MJ_STRUCT_NQ` | `7` |
| `_MJ_STRUCT_NV` | `6` |
| `_MJ_RWA_NQ` | `3` |
| `_MJ_RWA_NV` | `3` |
| `_MJ_TORSO_NQ` | `7` |
| `_MJ_TORSO_NV` | `6` |

---

## ⚠ Conventions de quaternion — ne jamais supposer

```
Pinocchio : (x, y, z, w)          MuJoCo : (w, x, y, z)
```

Règle explicite de CLAUDE.md : *« Do not assume quaternion conventions — verify
in `state_conversions.py` »*. **Ce fichier est la référence**, pas une
commodité : toute conversion doit y passer plutôt que d'être réécrite sur place.

## Les deux repères

| côté | repère | dimension |
|---|---|---|
| MuJoCo | **monde**, structure incluse | `nq=31 / nv=29 / nu=17` (7-DOF + 3 roues) |
| Pinocchio | **repère structure** | `nq=21 / nv=20 / nu=14` |

`mujoco_to_pinocchio` ne fait pas que réordonner un quaternion : il exprime
l'état du robot **dans le repère de la structure**, ce qui est la convention de
tout le contrôleur (spec §0). `pinocchio_to_mujoco` fait le retour pour appliquer
les commandes.

`quat_wxyz_to_euler_deg` sert au journal (`struct_euler_deg`), d'où θ_s est
extrait — le pic canonique **0.540°** est la norme de ce vecteur.

Couverture 100 % : les trois fonctions sont sur le chemin canonique, à chaque
tick.

## Voir aussi

- vue d'ensemble du paquet : [`core.md`](core.md)
