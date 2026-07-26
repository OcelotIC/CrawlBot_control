# `crawlbot.diagnostics.runner`

Orchestrateur de la suite de diagnostic : métriques, figures, captures.

**Fichier** : `crawlbot/diagnostics/runner.py` — **71 lignes** — couverture canonique **15 %**

> Docstring du module : *« Single entry point for the diagnostic suite. »*

---

## API publique

| symbole | signature | canonique ? |
|---|---|---|
| `run_diagnostics` | `(log, output_dir, cfg=None, thresholds=None, model=None,...)` | non exerce |

---

## Usage

```python
from crawlbot.diagnostics import run_diagnostics
run_diagnostics(log, output_dir, cfg=None, thresholds=None, model=None, data=None)
```

Enchaîne `compute_metrics` → `print_metrics` / `save_metrics_csv` →
`generate_plots`, puis `capture_snapshots` si `model` et `data` sont fournis.

Utilisable à la demande sur un log déjà produit :

```bash
MUJOCO_GL=osmesa PYTHONPATH=. python3 -c "
from crawlbot.diagnostics import run_diagnostics
import json
log = json.load(open('results/<log>.json'))
run_diagnostics(log, 'results/<output_dir>/')
"
```

## ⚠ Rappel valable pour tout le paquet

`crawlbot/diagnostics/` n'est **pas exercé par le run canonique**, alors que la
règle 3 de CLAUDE.md l'exige :

> *Every simulation produces diagnostics. Call `run_diagnostics()` at the end of
> every sim. « It docked » is not a pass criterion.*

Constat signalé, non corrigé : c'est une question de conformité à une règle, à
trancher (CLEANUP-20 §5.3). Ce qui fait foi aujourd'hui est le gate
(`gate/run_gate.py`, `gate/dock_check.py`) et les scripts d'export.

Conséquence pratique : **aucune couverture par le gate**. Une régression
introduite dans ce paquet ne sera détectée par rien.

Nuance mesurée : la fermeture d'imports du canonique tire bien
`crawlbot/diagnostics/__init__.py` (qui ré-exporte `run_diagnostics`), mais
**aucun des quatre modules** — et la fonction n'est jamais appelée (0/56 lignes).

## Voir aussi

- vue d'ensemble du paquet : [`diagnostics.md`](diagnostics.md)
