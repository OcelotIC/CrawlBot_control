# `crawlbot.core`

Couche fondamentale : modèle Pinocchio, cinématique inverse d'accostage, pont
MuJoCo ↔ Pinocchio, et mapping CoM → torse.

| fichier | lignes | couverture canonique |
|---|---:|---:|
| `ik.py` | 1468 | 40 % |
| `robot_interface.py` | 460 | 87 % |
| `com_to_torso_mapping.py` | 257 | 52 % |
| `state_conversions.py` | 232 | **100 %** |

---

## 1. `robot_interface.py` — le modèle

`RobotInterface` encapsule Pinocchio et produit, en un appel, **toutes** les
quantités dont le contrôleur a besoin.

```python
robot = RobotInterface('models/VISPA_crawling_fixed.urdf',
                       tau_max=20.0, gravity='zero')
rs = robot.update(q, v, omega_struct=...)      # -> RobotState
```

### `RobotState` — le paquet produit par `update()`

Dataclass, 27 champs. Les principaux :

| champ | contenu |
|---|---|
| `q`, `v` | configuration et vitesse généralisées |
| `q_joints`, `dq_joints` | tranche articulaire (bras) |
| `q_torso`, `dq_torso` | tranche flottante (torse) |
| `H`, `C`, `C_matrix` | matrice de masse, Coriolis (vecteur et matrice) |
| `r_com`, `v_com`, `J_com`, `Jdot_dq_com` | centre de masse |
| `h_centroidal`, `L_com` | moment centroïdal, sa part angulaire |
| `oMf_tool_a/b`, `J_tool_a/b`, `Jdot_dq_tool_a/b` | effecteurs |
| `oMf_torso`, `J_torso`, `Jdot_dq_torso` | torse |
| `q_min`, `q_max`, `tau_max`, `total_mass` | bornes et masse |

### ⚠ Piège : les constantes de module sont mutables

`FRAME_TORSO`, `FRAME_TOOL_A`, `FRAME_TOOL_B`, `JOINT_6A_ID`, `JOINT_6B_ID`,
`N_JOINTS`, `NQ`, `NV` sont déclarées au niveau module **avec des valeurs
héritées du modèle 6-DOF**, puis réécrites par `global` dans `__init__`
(`robot_interface.py:157-158`, `:213-218`) à partir du modèle réellement chargé.

Mesuré :

```
à l'import        : NQ=19  NV=18  N_JOINTS=12      (valeurs 6-DOF périmées)
après construction: NQ=21  NV=20  N_JOINTS=14      (le modèle 7-DOF réel)
```

Conséquence : **`from crawlbot.core.robot_interface import NQ` fige la valeur
périmée** — la ré-affectation `global` ne remonte pas dans le module importateur.

```python
import crawlbot.core.robot_interface as ri
ri.NQ           # correct APRÈS avoir construit une RobotInterface
from crawlbot.core.robot_interface import NQ   # 19, pour toujours
```

Préférer les attributs de l'instance (`robot.model.nq`, `robot.n_joints`,
`robot.frame_torso`) : ils sont toujours justes. Les dimensions canoniques sont
`nq=21 / nv=20 / nu=14` côté Pinocchio (CLAUDE.md).

### Méthodes

| méthode | canonique ? |
|---|---|
| `update(q, v, omega_struct)` | **OUI** — le cœur |
| `state` | **OUI** |
| `get_contact_jacobians(...)` | **OUI** |
| `compute_gjm(swing_arm)` | non exercé |
| `neutral_configuration()` | non exercé |

---

## 2. `state_conversions.py` — le pont MuJoCo ↔ Pinocchio

**Seul fichier du paquet couvert à 100 %.** Les trois fonctions sont sur le
chemin canonique :

| fonction | rôle |
|---|---|
| `mujoco_to_pinocchio(...)` | monde MuJoCo → repère structure Pinocchio |
| `pinocchio_to_mujoco(...)` | le retour |
| `quat_wxyz_to_euler_deg(w,x,y,z)` | angles d'Euler [deg] pour le log |

### ⚠ Conventions de quaternion — ne jamais supposer

```
Pinocchio : (x, y, z, w)        MuJoCo : (w, x, y, z)
```

C'est une règle explicite du projet (CLAUDE.md, « Do not assume quaternion
conventions — verify in `state_conversions.py` »). Ce fichier est la référence ;
toute conversion passe par lui.

---

## 3. `ik.py` — cinématique inverse (1468 lignes, 40 % couvert)

Le fichier le plus gros du paquet, et celui dont la plus grande part n'est **pas**
sur le chemin canonique.

| fonction | canonique ? | rôle |
|---|---|---|
| `solve_ik` | **OUI** | solveur générique multi-cibles (amorti, avec posture) |
| `dock_configuration_fixed_rotation` | **OUI** | pose d'accostage à orientation de torse imposée |
| `manipulability_config` | **OUI** | configuration maximisant la manipulabilité |
| `dock_configuration` | non exercé | variante à orientation libre |
| `precompute_torso_map` | non exercé | cache de poses par paire d'ancrages |
| `manipulability_config_trajectory` | non exercé | IK trajectoire (M7 Phase 4) |
| `manipulability_config_mid_waypoint` | non exercé | reshape mi-parcours (retiré de `sim_loop` en CLEANUP-15) |
| `check_path_feasibility` | non exercé | sonde de faisabilité de chemin |
| `solve_ik_waypoints` | non exercé | IK par points de passage |

Les six non exercées ne sont pas mortes : elles servaient des variantes de
recherche dont le câblage a été retiré de `sim_loop` par CLEANUP-15. Elles
n'ont **aucune couverture par le gate** — les modifier ne déclenchera rien.

Le paramètre `com_z_target` de `dock_configuration_fixed_rotation` porte le
standoff canonique **−0.35 m** (CLAUDE.md).

---

## 4. `com_to_torso_mapping.py` — δ(q), et où il s'applique

Convertit une référence de CoM en référence de torse via
`p_torso = r_com − δ(q)`, où δ est l'écart CoM→torse dans la configuration
courante.

| méthode | canonique ? |
|---|---|
| `compute_delta(q)` | **OUI** |
| `compute_delta_dot(q, dq)` | **OUI** |
| `body_com_jacobian(data, joint_idx)` | **OUI** |
| `compute(...)` | non exercé |
| `compute_delta_local` / `_dot` | non exercé |
| `torso_pos_jacobian_from_com(q)` | non exercé |

### ⚠ Le mapping est un chemin DS uniquement

Point d'architecture facile à se tromper, et explicitement listé dans les
interdits de CLAUDE.md :

> *Do not route the SS torso reference through the δ-mapping in two-task mode —
> SS uses the raw TorsoPlanner quintic (`sim_loop.py:2581-2584`); the mapping
> (δ(q_current)+F-SAT) remains a **DS-only** path.*

En simple appui la référence de torse est la quintique brute du `TorsoPlanner`,
sans passer par δ. C'est ce qui explique que `compute()` — l'entrée « complète »
du mapping — ne soit pas exercée : seules les briques `compute_delta` /
`compute_delta_dot` le sont, appelées depuis le chemin DS.

---

## Fichiers liés

| quoi | où |
|---|---|
| dimensions canoniques nq/nv/nu | CLAUDE.md, tableau des paramètres |
| conventions de quaternion | `crawlbot/core/state_conversions.py` (référence) |
| routage de la référence de torse | `crawlbot/simulation/sim_loop.py:2581-2584` |
| modèle contrôleur (URDF) | `models/VISPA_crawling_fixed.urdf` |
