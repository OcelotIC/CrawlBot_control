# `crawlbot.simulation`

La boucle fermée : machine d'états DS/SS, activation des soudures, AOCS,
journalisation — et le point unique de réglage.

| fichier | lignes | couverture canonique |
|---|---:|---:|
| `sim_loop.py` | 3387 | 83 % |
| `config.py` | 507 | **100 %** |
| `logging.py` | 269 | 93 % |
| `plotting.py` | 154 | 2 % |

---

## 1. `config.py` — `SimConfig`, le point unique de réglage

Une seule dataclass de ~500 lignes. Règle 5 du projet : **tout paramètre
réglable vit ici, avec son unité et sa justification.** Pas de constante
magique enfouie dans la boucle.

`SimConfig` alimente ensuite `CentroidalNMPCConfig`, `WholeBodyQPConfig`,
`CoarsePrePlannerConfig` et `ContactObserverConfig` — c'est le point de vérité
en amont de tous les sous-modules.

⚠ **Mais un défaut de `SimConfig` n'est pas non plus la valeur canonique** : le
run canonique est construit par `Misc/scripts/run_m7_single_step._make_m7_config()`
puis modifié par `scripts/diag_cooperative_arms.main(**kwargs)`. Pour connaître
une valeur canonique : le tableau « Key Parameters » de CLAUDE.md, ou
l'instrumentation du run — jamais la lecture d'un défaut.

### Le piège `use_m2_stack`

Ce champ **a l'air mort** (son jumeau dans `WholeBodyQPConfig` a été supprimé en
CLEANUP-8), mais il commande deux chemins sans rapport avec la pile de tâches :

| site | ce qu'il commande |
|---|---|
| `sim_loop.py:~2871` | routage de la référence de torse (mapping δ vs quintique brute) |
| `sim_loop.py:~3038` | `passivity_active` — **la contrainte de passivité en DS** |

Le supprimer désactiverait silencieusement la passivité DS. Sa déclaration
porte une note à cet effet.

---

## 2. `sim_loop.py` — la boucle

### Architecture : **deux phases, pas trois**

`DS` (double appui) et `SS` (simple appui). Interdit explicite de CLAUDE.md :
*« Do not implement a three-phase state machine (DS/SS/EXT) — the architecture
is two-phase per spec §7.1. »*

```
setup()                       modèles, planificateurs, solveurs, ancrages
  └─ run()                    boucle sur les pas
       ├─ _setup_torso_for_step()   IK d'accostage + phase de torse
       ├─ _run_preplanner()         T_step + trajectoire de CoM faisable
       ├─ _step()                   SS : le vol (1013 lignes)
       ├─ _run_ds_passivity_loop()  DS : stabilisation passive
       └─ _dock_gate() → _activate_weld()
```

### La porte d'accostage

Interdit explicite : *« Do not activate welds on position alone — require both
`d < 5 mm` AND `ori < 5°` »*. `_dock_gate` (`:12/32` lignes couvertes) applique
les deux conditions ; `_activate_weld` n'est appelé qu'après.

**Règle 10 — la métrique d'accostage est celle au moment de la soudure.** La
précision est le `d_mm` des `dock_events` (distance quand la soudure se
déclenche), jamais le minimum sur le vol : un passage plus proche avant
l'accostage est un artefact de survol (leçon du pas 2 : 3.0 mm en survol contre
4.89 mm à la soudure).

### `_step()` — 1013 lignes, le prochain chantier

Le plus gros bloc restant du dépôt. Sa décomposition est identifiée mais **non
faite**, et demande d'abord sa propre mesure de couplage
(`CLEANUP_CARRYOVER` §A).

Autre dette identifiée : l'appel `WholeBodyQP.solve()` prend **40 paramètres**,
dont 30 ne sont lus que dans un seul bloc de `solve()`. Restructurer la
signature est une modification d'API touchant les deux sites d'appel — reportée
délibérément (§A1).

### Points non exercés sur le canonique

`_gripper_speed`, `_planned_arm_config`, `plot` — plus les branches de repli
internes. Comme ailleurs, la majorité est morte **parce que le système est
sain**.

Sont en revanche **vivants et à conserver** : les crochets de diagnostic
`_diag_freeze_ref`, `_diag_lock_arm_joints`, `_diag_pure_pd`, utilisés par des
scripts de `Misc/scripts/`.

---

## 3. `logging.py` — `SimLog`

Dataclass de canaux (un tableau par grandeur, un élément par tick), plus
`capture_environment()` qui fige les versions de la pile dans le log.

`to_dict` / `save` sont sur le chemin canonique ; `load` ne l'est pas.

### ⚠ Deux canaux exportés sont identiquement nuls

Mesuré sur le log canonique (2077 ticks) :

| canal | valeur |
|---|---|
| `H_rO` | **0 partout** — `MomentumDisturbanceEstimator.update()` n'est jamais appelé |
| `H_dot_est` | **0 partout** — idem |
| `gmo_contact_state` | **constant à 0** (NO_CONTACT) — `ContactStateMachine.update()` n'est jamais appelé |

Voir `docs/crawlbot/aocs.md` §4 et `docs/crawlbot/estimation.md`. À savoir avant
de tracer ces canaux.

### Conventions de journalisation à connaître

- La référence de CoM **saute vers le CoM mesuré à l'entrée SS→DS** :
  `_log_ds_tick` inscrit `e_com = 0` avec `ref := mesuré`
  (`sim_loop.py:1038-1041`). Convention de log, décision en attente.
- La référence de torse exportée est **continue** à travers SS→DS→SS depuis le
  correctif de maintien terminal — journalisation seule, contrôle prouvé
  strictement identique.
- `nmpc_ok = 0` signifie « non appelé », pas « échoué » : le NMPC ne tourne
  qu'en SS et dans la stabilisation terminale. Sur le canonique cela concerne
  1368 des 2077 ticks, d'où un taux de succès apparent trompeur de 34 % pour un
  taux réel de **100 % (709/709)**.

---

## 4. `plotting.py` — 2 % couvert

`plot_simulation(log, save_path, cfg)` n'est pas appelée par le run canonique.
Le tracé se fait par les scripts d'export (`scripts/export_figure_data.py`,
`scripts/diag_full_diag_export.py`).

---

## Fichiers liés

| quoi | où |
|---|---|
| valeurs canoniques | CLAUDE.md, « Key Parameters » |
| config réellement construite | `Misc/scripts/run_m7_single_step._make_m7_config()` |
| point d'entrée du run | `scripts/diag_cooperative_arms.py` (`dca.main`) |
| dette identifiée sur `sim_loop` | `results/j2_adjconv/CLEANUP_CARRYOVER.md` §A |
| reproduction vérifiée | `gate/run_gate.py`, `gate/dock_check.py` |

---

## Documentation par module

- [`config.md`](config.md)
- [`logging.md`](logging.md)
- [`plotting.md`](plotting.md)
- [`sim_loop.md`](sim_loop.md)
