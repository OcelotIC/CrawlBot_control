# `crawlbot.core.robot_interface`

Enveloppe Pinocchio : en un appel à `update()`, produit **toutes** les
quantités dont le contrôleur a besoin, dans un unique `RobotState`.

**Fichier** : `crawlbot/core/robot_interface.py` — **460 lignes** — couverture canonique **87 %**

> Docstring du module : *« RobotInterface — Pinocchio wrapper for the VISPA crawling controller. »*

---

## API publique

| symbole | signature | canonique ? |
|---|---|---|
| **`RobotState`** *(dataclass)* |  |  |
|   `q` |  | _champ_ |
|   `v` |  | _champ_ |
|   `q_joints` |  | _champ_ |
|   `dq_joints` |  | _champ_ |
|   `q_torso` |  | _champ_ |
|   `dq_torso` |  | _champ_ |
|   `H` |  | _champ_ |
|   `C` |  | _champ_ |
|   `C_matrix` |  | _champ_ |
|   `r_com` |  | _champ_ |
|   `v_com` |  | _champ_ |
|   `J_com` |  | _champ_ |
|   `Jdot_dq_com` |  | _champ_ |
|   `h_centroidal` |  | _champ_ |
|   `L_com` |  | _champ_ |
|   `oMf_tool_a` |  | _champ_ |
|   `oMf_tool_b` |  | _champ_ |
|   `J_tool_a` |  | _champ_ |
|   `J_tool_b` |  | _champ_ |
|   `Jdot_dq_tool_a` |  | _champ_ |
|   `Jdot_dq_tool_b` |  | _champ_ |
|   `oMf_torso` |  | _champ_ |
|   `J_torso` |  | _champ_ |
|   `Jdot_dq_torso` |  | _champ_ |
|   `q_min` |  | _champ_ |
|   `q_max` |  | _champ_ |
|   `tau_max` |  | _champ_ |
|   `total_mass` |  | _champ_ |
| **`RobotInterface`** |  |  |
| `.update` | `(q, v, omega_struct=None)` | **oui** |
| `.state` | `()` | **oui** |
| `.compute_gjm` | `(swing_arm)` | non exerce |
| `.get_contact_jacobians` | `(active_A, active_B)` | **oui** |
| `.neutral_configuration` | `()` | non exerce |

### Constantes de module

| nom | valeur |
|---|---|
| `FRAME_TORSO` | `4` |
| `FRAME_TOOL_A` | `18` |
| `FRAME_TOOL_B` | `32` |
| `JOINT_6A_ID` | `7` |
| `JOINT_6B_ID` | `13` |
| `N_JOINTS` | `12` |
| `NQ` | `19` |
| `NV` | `18` |

---

## Usage

```python
robot = RobotInterface('models/VISPA_crawling_fixed.urdf',
                       tau_max=20.0, gravity='zero')
rs = robot.update(q, v, omega_struct=...)     # -> RobotState
```

`gravity='zero'` est le régime du projet : microgravité.

## ⚠ Le piège : les constantes de module sont réécrites à la construction

`FRAME_TORSO`, `FRAME_TOOL_A`, `FRAME_TOOL_B`, `JOINT_6A_ID`, `JOINT_6B_ID`,
`N_JOINTS`, `NQ`, `NV` sont déclarées au niveau module avec des valeurs héritées
du modèle **6-DOF**, puis réécrites par `global` dans `__init__` (`:157-158`,
`:213-218`) d'après le modèle réellement chargé.

Mesuré :

```
à l'import         : NQ=19  NV=18  N_JOINTS=12     (valeurs 6-DOF perimees)
apres construction : NQ=21  NV=20  N_JOINTS=14     (le modele 7-DOF reel)
```

Conséquence : **`from crawlbot.core.robot_interface import NQ` fige 19 pour
toujours** — la réaffectation `global` ne remonte pas dans le module qui a
importé le nom.

```python
import crawlbot.core.robot_interface as ri
ri.NQ                                          # correct APRES construction
from crawlbot.core.robot_interface import NQ   # 19, definitivement
```

Préférer les attributs de l'instance : `robot.model.nq`, `robot.n_joints`,
`robot.frame_torso`. Ils sont toujours justes.

## `RobotState`

Dataclass de 27 champs, produite à chaque `update()` : configuration et vitesse,
tranches articulaire et flottante, matrice de masse et Coriolis (`H`, `C`,
`C_matrix`), CoM (`r_com`, `v_com`, `J_com`, `Jdot_dq_com`), moment centroïdal
(`h_centroidal`, `L_com`), poses et jacobiennes des effecteurs et du torse,
bornes articulaires et `total_mass`.

Le tout est calculé en un passage Pinocchio : c'est ce qui permet au QP de
tourner à 100 Hz.

## Détection générique du nombre de DDL

Le module ne code pas en dur « 6 DOF par bras » : il détecte la tranche
articulaire depuis le modèle chargé. C'est ce qui a permis le passage à 7 DDL
par bras (`nq=21 / nv=20 / nu=14`) sans réécriture.

Non exercés : `compute_gjm` (carte de moment généralisé) et
`neutral_configuration`.

## Voir aussi

- vue d'ensemble du paquet : [`core.md`](core.md)
