# Documentation `crawlbot/`

Documentation par paquet, rédigée depuis le code au commit courant et depuis la
**couverture de lignes du replay canonique** (`gate/replay_canonical.py`).

| document | paquet | couverture canonique du paquet |
|---|---|---:|
| [`core.md`](core.md) | modèle Pinocchio, IK, pont MuJoCo, mapping CoM→torse | 40–100 % |
| [`solvers.md`](solvers.md) | NMPC centroïdal + QP corps-complet | 70–97 % |
| [`planning.md`](planning.md) | plan de marche, pré-planificateur, torse, vol | 0–95 % |
| [`simulation.md`](simulation.md) | boucle fermée, `SimConfig`, journalisation | 2–100 % |
| [`aocs.md`](aocs.md) | commande des roues à réaction | 39 % |
| [`estimation.md`](estimation.md) | observateur de contact (GMO) | 69 % |
| [`diagnostics.md`](diagnostics.md) | métriques et figures | 0–15 % |

---

## Méthode

Trois sources, aucune de mémoire :

1. **AST** pour les signatures et l'inventaire des symboles publics
   (`gate/_run/api_inventory.py`).
2. **Couverture de lignes du replay canonique** pour distinguer ce qui tourne de
   ce qui ne tourne pas (`gate/_run/api_live.py`).
3. **Lecture du code** pour le reste.

Les mentions « canonique » / « non exercé » sont donc **mesurées**.

### Pourquoi cette précaution

Le dépôt a déjà eu une documentation par paquet — `docs/api/`, aujourd'hui sous
`Misc/reports/api/`. Elle a rouillé sans que personne le voie, jusqu'à porter un
bandeau *« ⚠ SUPERSEDED »* et décrire un module `dynamics` qui n'existe pas.

Le chantier CLEANUP a trouvé **quatre** documents affirmatifs contredits par la
mesure : un défaut de dataclass pris pour la valeur canonique (F1), quatre
commentaires affirmant que des tests utilisent `from_heuristic` (aucun ne
l'appelle, sur tout l'historique), un `REPO_STATE.md` pointant vers un répertoire
qui n'a jamais existé, et un `STATUS.md` citant `crawlbot/planners/` — un paquet
qui n'existe pas.

D'où la règle appliquée ici : **une valeur par défaut de dataclass n'est pas la
valeur canonique**, et un nom de fonction ne dit pas si elle tourne.

---

## Trois pièges transverses

### 1. Les défauts ne sont pas les valeurs canoniques

`CentroidalNMPCConfig` annonce `robot_mass=90`, `N=20`, `dt=0.05` ; le canonique
utilise ≈ 71 kg, `N=8`, `dt=0.1`. Source de vérité : le tableau
« Key Parameters » de CLAUDE.md.

Exception mesurée : huit champs de `WholeBodyQPConfig` et cinq de
`CoarsePrePlannerConfig` ne sont jamais écrasés — pour ceux-là le défaut **est**
la valeur canonique (`CLEANUP_CARRYOVER` §C4).

### 2. « Non exercé » ne veut pas dire « supprimable »

Trois classes distinctes :

| classe | exemple | verdict |
|---|---|---|
| sédiment de recherche derrière un drapeau | modes AOCS alternatifs | supprimable |
| **repli mort parce que le système est sain** | `get_shifted_fallback`, les gardes de `contact_scheduler` | **à conserver** |
| crochet de diagnostic vivant | `_diag_pure_pd`, `_diag_freeze_ref` | **à conserver** |

### 3. Trois canaux exportés ne portent aucun signal

Mesuré sur le log canonique : `H_rO` et `H_dot_est` sont **identiquement nuls**
(l'estimateur est construit mais jamais mis à jour) et `gmo_contact_state` est
**constant** (la machine d'état de contact n'est jamais avancée). Détails dans
`aocs.md` §4 et `estimation.md`.

---

## Vérification

Ces documents sont **vérifiables**, ce qui est la seule différence de fond avec
le `docs/api/` qui a rouillé :

```bash
PYTHONPATH=. python3 gate/verify_docs.py   # chaque reference fichier:ligne et chaque symbole
PYTHONPATH=. python3 gate/link_audit.py    # chaque chemin cité dans le dépôt
```

`verify_docs` sort en erreur si une référence `fichier:ligne` dépasse la taille
du fichier, ou si un symbole cité n'est plus défini dans `crawlbot/`.

Il a déjà servi : la première rédaction citait `sim_loop.py:2865-2870` pour le
routage de la référence de torse — valeur reprise de CLAUDE.md, devenue fausse
depuis que le chantier a raccourci `sim_loop` de 375 lignes. Le vrai site est
`sim_loop.py:2581-2584`. CLAUDE.md, `CLEANUP_CARRYOVER` et ces documents ont été
corrigés ensemble.

Ce qu'il ne couvre pas encore : les valeurs numériques (α, gains, seuils), qui
restent vérifiées à la main contre CLAUDE.md.

---

## Ce que ces documents ne couvrent pas

- `lutze_baseline/` — l'implémentation de comparaison M0/Lutze.
- `gate/` — voir `gate/README.md`.
- `Misc/` — sédiment de recherche, destiné à disparaître.
- Le *pourquoi* architectural profond : `docs/architecture/brainstorming_reworked_architecture.md`
  (spécification) et `docs/architecture/STACK_OVERVIEW.md` (état du code).
