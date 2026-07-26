# `crawlbot.planning`

Génération des références : plan de marche, durée de pas faisable en moment,
trajectoires de torse et d'effecteur.

| fichier | lignes | couverture canonique |
|---|---:|---:|
| `coarse_preplanner.py` | 540 | 81 % |
| `torso_planner.py` | 480 | 81 % |
| `contact_scheduler.py` | 350 | 87 % |
| `swing_planner.py` | 337 | **95 %** |
| `sequence_loader.py` | 254 | 0 % — **conservé**, §6 |
| `locomotion_planner.py` | 205 | 17 % — **conservé**, §6 |

Paquet entièrement audité par le chantier CLEANUP-16 → 19 : 3258 → 2175 lignes
(−33 %), `constrained_geodesic.py` (470 lignes) supprimé.

---

## Chaîne de production d'un pas

```
ContactScheduler.plan_traversal()      → squelette DS/SS/DS… (durées SS = 0)
        │
CoarsePrePlanner.solve()               → T_step + trajectoire de CoM faisable
        │
GaitPlan.set_step_duration(idx, T_step)→ installe la durée réelle, cascade le timing
        │
TorsoPlanner.add_phase()  +  SwingPlanner (piloté par le plan)
        │
                                        → références pour le QP
```

Le point clé : **la durée du pas n'est pas choisie, elle est calculée** — c'est
le pré-planificateur qui la déduit de l'enveloppe de moment.

---

## 1. `contact_scheduler.py` — le plan de marche

`ContactScheduler.plan_traversal(start_a, start_b, n_steps)` produit un
`GaitPlan` : suite de `GaitPhase` (DOUBLE / SINGLE_A / SINGLE_B) avec ancrages
et bras en vol.

Les phases SS naissent avec `duration = 0.0` ; `GaitPlan.set_step_duration(idx,
T_step)` installe la vraie durée et **recalcule tout le timing en cascade**
(`sim_loop.py:1495`).

Ancrages : lus du modèle MuJoCo par `read_anchors_from_mujoco` (chemin
canonique). `make_anchor_grid` (grille analytique, `dx=0.8`, `dy=0.3`) n'est pas
exercée sur le canonique mais reste utilisée par un script de diagnostic.

Non exercée : `contact_sequence_over_horizon` — plomberie prévue pour le NMPC
que le NMPC n'a jamais consommée (zéro appelant).

**Les 16 lignes non couvertes de ce fichier sont à 15/16 des gardes et des
replis** (`IndexError`/`ValueError` de `set_step_duration`, `RuntimeError` du
property `plan`, `break` d'épuisement de la grille, chemins d'erreur de
`read_anchors_from_mujoco`). Elles sont mortes **parce que le système est sain**.
Ne pas les supprimer.

---

## 2. `coarse_preplanner.py` — la durée de pas faisable

Un NLP centroïdal (CasADi/IPOPT) résolu **une fois par pas**, sur M = 15
intervalles de collocation.

Variables : `r_com, v_com, L_com` aux M+1 nœuds, `f_stance, tau_stance` aux M
intervalles. Contraintes :

1. boîte de moment à **chaque** nœud :
   `c − L_com(k) − r_com(k) × m·v_com(k) ∈ [−h_max', +h_max']`
2. borne de taux `|Ḣ_s|_∞ ≤ τ_w,max` (2.5 N·m)
3. boîte force/couple `|f|_∞ ≤ 25 N`, `|τ|_∞ ≤ 8 N·m`
4. conditions aux limites, plus une marge terminale `κ = 0.7`

### En cas d'échec : le pas est sauté, sans repli

Pas de retombée heuristique silencieuse. `sim_loop` journalise l'échec,
maintient la position et saute le pas.

⚠ `CoarsePlanResult.from_heuristic` (83 lignes) est documentée dans quatre
commentaires comme « la fixture des tests unitaires ». **Aucun test ne
l'appelle** — recherché sur tout l'historique du dépôt, le seul `.from_heuristic(`
existant est ce commentaire. C'est une fixture écrite pour des tests jamais
écrits (audit CLEANUP-19). Sa suppression est en attente.

### Valeurs canoniques silencieuses (règle 5)

Cinq champs ne sont **jamais** écrasés par `sim_loop`, donc leur défaut *est* la
valeur canonique — et aucun n'est dans CLAUDE.md :

| champ | valeur | rôle |
|---|---|---|
| `eps_v_terminal` | 5e-3 m/s | **boîte dure** sur la vitesse de CoM terminale |
| `eps_L_terminal` | 5e-2 N·m·s | **boîte dure** sur le moment terminal |
| `w_v_terminal` / `w_L_terminal` | 1e2 | pénalités douces sur le même résidu |
| `ipopt_tol` | 1e-6 | tolérance de convergence |

Les deux premières décident **où un pas a le droit de finir**. `T_step_default`
(6.0 s) n'est jamais utilisé : `sim_loop` passe toujours `T_step` explicitement.

---

## 3. `swing_planner.py` — l'effecteur en vol (95 % couvert)

Trajectoire quintique + dégagement, avec trois profils composables : quintique
`s(τ)`, bosse de dégagement `_bump`, cosinus retardé `_delayed_cosine` (et
leurs dérivées première et seconde — toutes exercées).

`reference_at(t)` interroge le plan de marche : en DS il renvoie la dernière
position de vol figée, en SS il interpole sur `T_eff = T_step ×
early_finish_fraction`, avec `τ` écrêté à 1 (position à la cible, vitesse et
accélération nulles).

Dégagement par défaut 0.03 m, normale sortante `−z` en repère structure.

**Le mécanisme de surcharge de phase a été supprimé** (CLEANUP-18) :
`add_phase`, `_override_reference_at`, `clear_phase_overrides`,
`adaptive_reference_at`, `swing_trajectory`. `reference_at()` prend désormais
toujours le chemin piloté par le plan. Couverture passée de 47 % à 95 %.

⚠ Ne pas confondre : `torso_planner.add_phase` est **vivante**. Seul le
`add_phase` du *swing* était mort.

---

## 4. `torso_planner.py` — la pose du torse

Quintique en position + SLERP en orientation, par phase.

`add_phase(t_start, t_end, p0, R0, pf, Rf)` installe la phase du pas courant
(`sim_loop.py:1544`) ; `reference_at`, `reference_at_clamped`, `has_phase_at`,
`com_reference_at`, `l_com_reference_at` la consultent.

`reference_at_clamped` sert au **log** : sur les ticks DS de stabilisation où
aucune phase ne couvre `t`, il renvoie la pose quintique terminale figée, ce qui
rend la référence exportée continue à travers SS→DS→SS (correctif de continuité,
journalisation seule, contrôle strictement inchangé).

### ⚠ En SS, la référence de torse ne passe PAS par le mapping δ

Interdit explicite de CLAUDE.md :

> *Do not route the SS torso reference through the δ-mapping in two-task mode —
> SS uses the raw TorsoPlanner quintic (`sim_loop.py:2581-2584`); the mapping
> remains a DS-only path.*

En simple appui, le QP suit la quintique brute. Le mapping CoM→torse
(`crawlbot.core.com_to_torso_mapping`) ne sert qu'en double appui.

Supprimés en CLEANUP-18 : `set_from_waypoints` (orphelin depuis CLEANUP-14) et
`_trapezoidal_params` (zéro appelant, y compris interne).

---

## 5. Rappel : le suivi de torse et le recul libre

Résidu de torse à la frontière : **18–27 mm** en régime établi (98.6 mm au pas
initial). L'excursion de ~150 mm à mi-vol est **du vrai recul en vol libre**
contre l'enveloppe de moment — pas un défaut de suivi (audit TORSO-REF-AUDIT).

---

## 6. Deux fichiers conservés malgré 0 % / 17 %

| fichier | pourquoi conservé |
|---|---|
| `sequence_loader.py` | jamais importé sur le canonique, mais il porte une vraie fonctionnalité : `sim.setup(sequence_path=…)`, utilisée dès qu'un scénario `.seq` est fourni. **Inutilisé ≠ abandonné.** |
| `locomotion_planner.py` | mort sur le canonique, mais **vivant dans `lutze_baseline/sim_lutze.py`** — la comparaison M0/Lutze qui sous-tend le tableau §II de l'article. Proposé à la suppression puis conservé sur mesure (CLEANUP-18 §3). |

---

## Fichiers liés

| quoi | où |
|---|---|
| installation de `T_step` | `crawlbot/simulation/sim_loop.py:1495` |
| phase de torse du pas | `crawlbot/simulation/sim_loop.py:1544` |
| routage de la référence SS | `crawlbot/simulation/sim_loop.py:2581-2584` |
| audits du paquet | `results/j2_adjconv/PHASE_CLEANUP_{16,18,19}_*.md` |
| scénarios `.seq` | `scenarios/` |

---

## Documentation par module

- [`coarse_preplanner.md`](coarse_preplanner.md)
- [`contact_scheduler.md`](contact_scheduler.md)
- [`locomotion_planner.md`](locomotion_planner.md)
- [`sequence_loader.md`](sequence_loader.md)
- [`swing_planner.md`](swing_planner.md)
- [`torso_planner.md`](torso_planner.md)
