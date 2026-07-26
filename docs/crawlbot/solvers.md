# `crawlbot.solvers`

Le contrôleur proprement dit : NMPC centroïdal (étage 1) + QP corps-complet
(étage 2), et leurs deux backends.

| fichier | lignes | couverture canonique |
|---|---:|---:|
| `wholebody_qp.py` | 950 | **97 %** |
| `centroidal_nmpc.py` | 702 | 88 % |
| `nmpc_solver.py` | 649 | 95 % |
| `hierarchical_qp.py` | 529 | 70 % |
| `contact_phase.py` | 138 | 85 % |

---

## Architecture à deux étages

```
        étage 1                             étage 2
  CentroidalNMPC (dt = 0.1 s)   ───►   WholeBodyQP (dt = 0.01 s)
  N = 8, état 9, commande 12           14 DOF libres en SS
  → trajectoire de CoM/moment          → q̈ et couples articulaires
    faisable en moment                   qui la suivent
```

L'étage 1 décide **ce qui est faisable** vis-à-vis de l'enveloppe de moment des
roues ; l'étage 2 décide **comment le réaliser** avec les bras.

---

## ⚠ Les valeurs par défaut des dataclasses ne sont PAS les valeurs canoniques

`CentroidalNMPCConfig` annonce `robot_mass=90.0`, `N=20`, `dt=0.05`. Le run
canonique n'utilise **aucune** de ces valeurs : `sim_loop.py:383-398` écrase
chaque champ depuis `SimConfig` (`N=8`, `dt=0.1`, masse ≈ 71 kg…).

C'est la leçon centrale de ce dépôt (rétractation F1 du chantier CLEANUP) :

> **Une valeur par défaut de dataclass n'est pas la valeur canonique.**
> Tracer la config que le run construit réellement, ou l'instrumenter.

Les valeurs canoniques font foi dans le tableau « Key Parameters » de CLAUDE.md.

**Exception à connaître** : huit champs de `WholeBodyQPConfig` ne sont *jamais*
écrasés, donc leur défaut **est** la valeur canonique — `method`, `solver`,
`weight_ratio`, `w_hw_slack`, `alpha_settle`, `Kd_settle`, `qdd_max`,
`tau_contact_max` (`CLEANUP_CARRYOVER` §C4).

---

## 1. `contact_phase.py` — la carte de moment

Le socle commun aux deux étages.

- `ContactPhase` : `DOUBLE`, `SINGLE_A`, `SINGLE_B`
- `ContactConfig.from_phase(phase, r_contact_A, r_contact_B)` → nombre de
  contacts actifs et leurs positions
- `compute_momentum_map(r_com, contact_config)` → la matrice reliant le torseur
  de contact à la dérivée du moment
- `skew(v)` → matrice antisymétrique

`active_contact_positions` n'est pas exercée.

---

## 2. `centroidal_nmpc.py` — étage 1

État 9 = `[r_com, v_com, L_com]`, commande 12 = deux torseurs de contact
`[f; τ]`. Horizon `N = 8` à `dt = 0.1 s`.

| méthode | canonique ? | rôle |
|---|---|---|
| `build(...)` | **OUI** | construit le NLP CasADi une fois |
| `solve(...)` | **OUI** | un pas, warm-start inclus |
| `compute_c_simple(...)` | **OUI** | constante de conservation `c` |
| `compute_feedforward_acceleration(λ_ref)` | **OUI** | accélération FF pour l'étage 2 |
| `reset_warm_start()` | **OUI** | à chaque changement de pas |
| `get_last_trajectory()` | **OUI** | lecture de la dernière solution |
| `get_shifted_fallback()` | **non exercé** | repli sur échec — **à conserver** |
| `get_full_trajectory(...)` | non exercé | 5 tests + 1 script l'utilisent |

`get_shifted_fallback` est morte **parce que le système est sain**, pas parce
qu'elle est obsolète : c'est le repli quand un solve échoue, et aucun n'échoue
sur le canonique. Ne pas la supprimer (classe (b) du chantier CLEANUP).

### La contrainte qui structure tout : l'enveloppe de moment

`enforce_hw_conservation` active la contrainte de boîte sur le moment des
roues reconstruit à chaque nœud :

```
h_w(k) = c − L_com(k) − r_com(k) × m·v_com(k)  ∈  [−h_max', +h_max']
```

⚠ **Le défaut de ce champ est `False`, mais le canonique le met à `True`.**
C'est précisément l'erreur rétractée en CLEANUP-2 (F1) : le chemin avait été
déclaré mort d'après le défaut de la dataclass, alors qu'il est actif — mesuré
`enforce_hw=True`, `ng_path=17`, `ng_term=6`.

Le cap de taux `tau_w_max = 2.5 N·m` s'applique ici *et* dans le QP *et* dans
le modèle MuJoCo (trois points d'application, CLAUDE.md).

---

## 3. `nmpc_solver.py` — le backend CasADi générique

`NMPCSolver` est un constructeur de NLP par tir multiple, indépendant du
problème : dynamique continue, coûts d'étage et terminal, contraintes de
chemin/terminales, bornes. `centroidal_nmpc` l'instancie.

**95 % couvert**, toutes les méthodes publiques sont sur le chemin canonique,
y compris `apply_control_bounds_all_stages` (ajoutée par CLEANUP-1/3).

Un correctif à connaître (F2, CLEANUP-3) : le warm-start n'est repris que si
`info.success` est vrai — auparavant une solution ratée pouvait amorcer le pas
suivant.

---

## 4. `wholebody_qp.py` — étage 2, **le contrôleur canonique**

Le fichier le mieux couvert du dépôt (**97 %**), et celui qui a le plus maigri
pendant le chantier : 1385 → 950 lignes, le corps de `solve()` passant de 543
à ~346 lignes (CLEANUP-6 à 11).

### La pile à deux tâches (« two-task SS stack »)

C'est **la** configuration canonique. Quatre tâches, toutes pondérées :

| tâche | α canonique | rôle |
|---|---:|---|
| pose de torse (6-D) | **2000** | levier fin de précision d'accostage |
| effecteur en vol (swing-EE) | **1000** | levier d'allonge — doit rester ≥ ~1000 |
| moment linéaire (T-MOM) | **400** | quasi inerte sur Ḣ_s — le NMPC tient l'enveloppe |
| posture | **20** | régularisation articulaire |
| minimisation de couple | 5 | doit rester ≳ 5× le plancher de régularisation |
| suivi de torseur | 1.0 | > 1 bloquerait l'autorité torse/EE |
| régularisation d'accélération | 1.0 | plancher de conditionnement |
| slack sur `h_w` | 800 | actif seulement si la boîte `h_w` est violée |

### ⚠ Il n'y a **pas** de projection dans l'espace nul

`weight_ratio = 1.0` ⇒ **les magnitudes d'α *sont* la hiérarchie.** Les entiers
de priorité nominale sont inertes. Deux interdits explicites de CLAUDE.md en
découlent :

- ne pas utiliser `weight_ratio > 1` ;
- ne pas utiliser `alpha_wrench > 1` (à 100, la régularisation de torseur
  consommait 20 % du budget du QP et bloquait l'autorité torse/EE).

Et la **règle 14** : `alpha_torque ≳ 5 × alpha_reg`. À un rapport 1:1 la
résolution de redondance en SS dégénère en dépassement de délai d'accostage.

### Structure de `solve()` après CLEANUP-11

Quatre helpers extraits, tous sur le chemin canonique :

| helper | rôle |
|---|---|
| `_add_equality_constraints` | dynamique + contacts |
| `_add_inequality_constraints` | boîtes ; renvoie `hw_constraint_active` |
| `_set_variable_bounds` | bornes des variables |
| `_com_task_rows` | lignes de la tâche CoM ; renvoie `(A_com, b_com)` |

⚠ Ce qui a été **explicitement écarté** : fusionner ou réordonner les blocs de
tâches. L'ordre encode la séquence d'assemblage du coût — le changer serait une
modification de comportement déguisée en refactorisation.

### Piège de configuration

`SimConfig.use_m2_stack` **a l'air mort** (son jumeau côté QP a été supprimé en
CLEANUP-8) mais commande deux chemins sans rapport avec la pile de tâches : le
routage de la référence de torse (`sim_loop:~2871`) et **la contrainte de
passivité en DS** (`sim_loop:~3038`). Le supprimer désactiverait silencieusement
la passivité DS.

---

## 5. `hierarchical_qp.py` — le backend QP

`Task(A, b, W, priority)`, puis `add_task` / `add_equality_constraint` /
`add_inequality_constraint` / `set_bounds` / `solve`.

| chemin | canonique ? |
|---|---|
| `_solve_weighted` | **OUI** — `method='weighted'` |
| `_solve_qp_raw` | **OUI** |
| `_solve_strict` | **non exercé** (76 lignes) — 2 tests + 6 scripts |
| `clear_tasks` / `clear_constraints` / `n_tasks` | non exercés |

Régularisation de Tikhonov ε = 1e-6 — **inerte** en pratique :
`λ_min(H_LS) = 1 ≫ ε`. Conditionnement canonique `κ_SS(H) ≈ 7.5e3`, soit 530×
mieux que la valeur d'avant le gel.

`_solve_strict` est le seul gros bloc non exercé restant ; sa suppression est
une décision (la voie hiérarchique stricte reste-t-elle reproductible ?), pas
une évidence — voir `CLEANUP_CARRYOVER` §B2.

---

## Fichiers liés

| quoi | où |
|---|---|
| valeurs canoniques (α, caps, gains) | CLAUDE.md, « Key Parameters » |
| construction du NMPC | `crawlbot/simulation/sim_loop.py:383-398` |
| construction du QP | `crawlbot/simulation/sim_loop.py` (boucles DS et SS) |
| dette et décisions en attente | `results/j2_adjconv/CLEANUP_CARRYOVER.md` |
| gel canonique 2.5 | `results/j2_adjconv/canonical2p5_result.json` |
