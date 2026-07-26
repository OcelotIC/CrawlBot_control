# `crawlbot.planning.sequence_loader`

Chargeur de scénarios `.seq` : permet de décrire une traversée dans un fichier
plutôt qu'en arguments.

**Fichier** : `crawlbot/planning/sequence_loader.py` — **255 lignes** — couverture canonique **0 %**

> Docstring du module : *« Locomotion-sequence file loader. »*

---

## API publique

| symbole | signature | canonique ? |
|---|---|---|
| **`SwingTarget`** *(dataclass)* |  |  |
|   `arm` |  | _champ_ |
|   `anchor_idx` |  | _champ_ |
|   `dwell_after` | `0.0` | _champ_ |
| **`LoadedSequence`** *(dataclass)* |  |  |
|   `start_a` |  | _champ_ |
|   `start_b` |  | _champ_ |
|   `swing_targets` |  | _champ_ |
|   `source_path` |  | _champ_ |
| `load_sequence` | `(path, n_anchors)` | non exerce |
| `plan_from_sequence` | `(sched, seq)` | non exerce |

### Constantes de module

| nom | valeur |
|---|---|
| `_ANCHOR_RE` | `re.compile('^anchor_(\\d+)([ab])$')` |

---

## ⚠ 0 % de couverture, et pourtant **conservé**

Le module n'est jamais exécuté par le run canonique. Il porte néanmoins une
fonctionnalité réelle : `sim.setup(sequence_path=…)`, empruntée dès qu'un
scénario est fourni à `dca`.

C'est la distinction que le chantier CLEANUP applique systématiquement :

> **Inutilisé sur le canonique ≠ recherche abandonnée.**

Comparer avec le sédiment de recherche (modes AOCS alternatifs, chemin FK des
planificateurs) : celui-là est derrière des drapeaux d'opt-in issus
d'expérimentations closes. Ici il s'agit d'une entrée utilisateur documentée.

## Scénarios disponibles

`scenarios/` contient `canonical_3step.seq`, `canonical_5step.seq`,
`multi_traversal_2x.seq`, `multi_traversal_10x.seq`,
`multi_traversal_10x_dwell.seq`.

`dca` route la sortie vers un sous-dossier nommé d'après le fichier de scénario.

## Conséquence pratique

Aucune couverture par le gate : une régression introduite ici ne sera détectée
par rien. À vérifier à la main si le chemin scénario est modifié.

## Voir aussi

- vue d'ensemble du paquet : [`planning.md`](planning.md)
