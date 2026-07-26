# `crawlbot.core.ik`

Cinématique inverse d'accostage : pose des deux mains sur leurs ancrages,
orientation de torse imposée, maximisation de manipulabilité.

**Fichier** : `crawlbot/core/ik.py` — **1468 lignes** — couverture canonique **40 %**

> Docstring du module : *« Inverse kinematics for VISPA docking configurations. »*

---

## API publique

| symbole | signature | canonique ? |
|---|---|---|
| `solve_ik` | `(model, q0, targets, max_iter=500, tol=1e-08, base_gain=...)` | **oui** |
| `dock_configuration` | `(model, anchor_a, anchor_b, torso_pos=None, q_init=None,...)` | non exerce |
| `dock_configuration_fixed_rotation` | `(model, anchor_a, anchor_b, R_torso_fixed, torso_pos=Non...)` | **oui** |
| `manipulability_config` | `(model, anchor_a, anchor_b, level_axis, q_nominal, w_pos...)` | **oui** |
| `precompute_torso_map` | `(model, anchors_a, anchors_b, anchor_pair_sequence, q_in...)` | non exerce |
| `manipulability_config_trajectory` | `(model, anchor_a, anchor_b, q_start, n_samples=5, q_gues...)` | non exerce |
| `manipulability_config_mid_waypoint` | `(model, anchor_a_pose, anchor_b_pose, q_start, q_end, sw...)` | non exerce |
| `check_path_feasibility` | `(model, q_start, q_end, anchor_a_pose, anchor_b_pose, sw...)` | non exerce |
| `solve_ik_waypoints` | `(model, q_start, stance_frame, stance_target, swing_fram...)` | non exerce |

---

## Le plus gros fichier du paquet, et le moins exercé

1468 lignes, **40 % de couverture**. Trois fonctions seulement sont sur le
chemin canonique ; six ne le sont pas.

## Les trois fonctions canoniques

| fonction | rôle |
|---|---|
| `solve_ik` | solveur générique multi-cibles, amorti, avec terme de posture |
| `dock_configuration_fixed_rotation` | **la pose d'accostage** — orientation de torse imposée |
| `manipulability_config` | configuration maximisant la manipulabilité |

`dock_configuration_fixed_rotation` porte le **standoff canonique −0.35 m** via
`com_z_target` : la hauteur de rampement à laquelle le CoM est maintenu
(CLAUDE.md).

## Les six non exercées — sédiment de recherche, mais non couvert

`dock_configuration`, `precompute_torso_map`, `manipulability_config_trajectory`,
`manipulability_config_mid_waypoint`, `check_path_feasibility`,
`solve_ik_waypoints`.

Elles servaient des variantes dont le câblage a été retiré de `sim_loop` par
CLEANUP-15 (chemin FK, IK trajectoire, reshape mi-parcours, sonde de
faisabilité). Elles ne sont pas mortes au sens strict — mais elles n'ont
**aucune couverture par le gate** : les modifier ne déclenchera rien.

C'est le principal angle mort de la vérification dans `crawlbot/`.

## Un piège de conception à connaître

Interdit de CLAUDE.md : *« Do not generate trajectory acceleration profiles
without checking actuator feasibility — quintic on 591 mm torso displacement
saturates 20 Nm joints. »* L'IK donne une pose atteignable, pas une trajectoire
réalisable : la faisabilité en couple se vérifie séparément.

## Voir aussi

- vue d'ensemble du paquet : [`core.md`](core.md)
