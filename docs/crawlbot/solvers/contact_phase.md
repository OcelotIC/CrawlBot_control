# `crawlbot.solvers.contact_phase`

Socle commun aux deux étages : quelles mains touchent, où, et comment cela se
traduit en carte de moment.

**Fichier** : `crawlbot/solvers/contact_phase.py` — **138 lignes** — couverture canonique **85 %**

> Docstring du module : *« Contact phase definitions for crawling multi-arm robot locomotion. »*

---

## API publique

| symbole | signature | canonique ? |
|---|---|---|
| **`ContactPhase`** |  |  |
| **`ContactConfig`** *(dataclass)* |  |  |
|   `phase` |  | _champ_ |
|   `nc` |  | _champ_ |
|   `active_contacts` |  | _champ_ |
|   `r_contact_A` |  | _champ_ |
|   `r_contact_B` |  | _champ_ |
| `.from_phase` | `(cls, phase, r_contact_A, r_contact_B)` | **oui** |
| `.active_contact_positions` | `()` | non exerce |
| `skew` | `(v)` | **oui** |
| `compute_momentum_map` | `(r_com, contact_config)` | **oui** |

---

## Les trois phases

`ContactPhase` : `DOUBLE` (deux bras soudés), `SINGLE_A` (A porte, B vole),
`SINGLE_B` (B porte, A vole).

⚠ **L'architecture est à deux phases, pas trois.** `SINGLE_A` et `SINGLE_B` sont
deux instances du même régime SS ; il n'existe pas de troisième état de type
« EXT ». Interdit explicite de CLAUDE.md.

## `ContactConfig`

Construite par `from_phase(phase, r_contact_A, r_contact_B)` : en déduit `nc`
(nombre de contacts actifs) et le masque `active_contacts`. C'est l'objet passé
au NMPC et au QP pour savoir combien de torseurs sont en jeu.

## `compute_momentum_map(r_com, contact_config)`

La matrice qui relie le torseur de contact à la dérivée du moment centroïdal :

```
L̇_com = Σ_i (r_Ci − r_com) × f_i + τ_i
```

C'est le maillon commun : le NMPC l'utilise pour prédire, le QP pour réaliser.

`skew(v)` est l'utilitaire de matrice antisymétrique associé.

Non exercée : `active_contact_positions` — accesseur de confort sans appelant.

## Voir aussi

- vue d'ensemble du paquet : [`solvers.md`](solvers.md)
