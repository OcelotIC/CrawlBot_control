# `crawlbot.planning.torso_planner`

Pose de référence du torse : quintique en position, SLERP en orientation, une
phase par pas.

**Fichier** : `crawlbot/planning/torso_planner.py` — **481 lignes** — couverture canonique **81 %**

> Docstring du module : *« TorsoPlanner — Generates 6D torso + CoM reference trajectories. »*

---

## API publique

| symbole | signature | canonique ? |
|---|---|---|
| **`TorsoReference`** *(dataclass)* |  |  |
|   `p` |  | _champ_ |
|   `R` |  | _champ_ |
|   `v` |  | _champ_ |
|   `a` |  | _champ_ |
| **`ComReference`** *(dataclass)* |  |  |
|   `r_com` |  | _champ_ |
|   `v_com` |  | _champ_ |
| **`TorsoPlanner`** |  |  |
| `.set_torso_inertia` | `(I_body)` | **oui** |
| `.set_hold` | `(p, R, r_com=None)` | **oui** |
| `.add_phase` | `(t_start, t_end, p_start, R_start, p_end, R_end, delta_c...)` | **oui** |
| `.clear_phases` | `()` | **oui** |
| `.reference_at` | `(t)` | **oui** |
| `.has_phase_at` | `(t)` | **oui** |
| `.reference_at_clamped` | `(t)` | **oui** |
| `.com_reference_at` | `(t)` | **oui** |
| `.l_com_reference_at` | `(t)` | **oui** |
| `._hold_reference` | `()` | **oui** |
| `._profile_params` | `(t, phase)` | **oui** |
| `._quintic_params` | `(t, phase)` | **oui** |
| `._interpolate_phase` | `(t, phase)` | **oui** |
| `._interpolate_com` | `(t, phase)` | **oui** |

---

## Usage dans la boucle

`add_phase(t_start, t_end, p0, R0, pf, Rf)` installe la phase du pas courant
(`sim_loop.py:1544`), après que le pré-planificateur a fourni `T_step`. La
référence est **ré-ancrée à chaque pas** sur la pose mesurée de début de pas.

`reference_at`, `has_phase_at`, `com_reference_at`, `l_com_reference_at` la
consultent ensuite à la cadence du QP.

## ⚠ En SS, cette quintique est utilisée **brute**

Interdit explicite de CLAUDE.md :

> *SS uses the raw TorsoPlanner quintic (`sim_loop.py:2581-2584`); the mapping
> (δ(q_current)+F-SAT) remains a DS-only path.*

En simple appui, la tâche de pose de torse du QP reçoit directement `tr.p / v /
a`. Le mapping CoM→torse n'intervient qu'en double appui.

## `reference_at_clamped` — pour le journal, pas pour le contrôle

Sur les ticks DS de stabilisation où aucune phase ne couvre `t`, cette méthode
renvoie la pose quintique **terminale figée**. Elle sert uniquement à
l'export : sans elle, la référence de torse journalisée sautait à la transition
SS→DS.

Le correctif de continuité est **strictement de journalisation** — le contrôle a
été prouvé identique à l'octet par re-run complet. Règle 11 du projet : *« une
référence exportée qui saute à une transition de phase est soit un bug de
contrôle, soit un artefact d'export — déterminer lequel avant de tracer. »*

## Suivi réel, et pourquoi l'écart n'est pas un défaut

Résidu de torse à la frontière : **18–27 mm** en régime établi (98.6 mm au pas
initial). L'excursion de ~150 mm à mi-vol est **du vrai recul en vol libre**
contre l'enveloppe de moment, pas une erreur de suivi (audit TORSO-REF-AUDIT).

Supprimés en CLEANUP-18 : `set_from_waypoints` (orphelin depuis CLEANUP-14, son
seul appelant de production étant le bloc `ds_mobile_com_magnitude`) et
`_trapezoidal_params` (zéro appelant, y compris interne).

## Voir aussi

- vue d'ensemble du paquet : [`planning.md`](planning.md)
