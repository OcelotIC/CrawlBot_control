# `crawlbot.diagnostics`

Suite de diagnostic : métriques à seuils, planches de figures, captures
d'images.

| fichier | lignes | couverture canonique |
|---|---:|---:|
| `metrics.py` | 424 | 5 % |
| `plots.py` | 689 | 5 % |
| `runner.py` | 71 | 15 % |
| `snapshots.py` | 71 | **0 %** |

---

## ⚠ Le paquet n'est pas exercé par le run canonique

C'est le fait principal à connaître, et il contredit une règle du projet.

CLAUDE.md, règle 3 :

> *Every simulation produces diagnostics. Call `run_diagnostics()` at the end of
> every sim. « It docked » is not a pass criterion.*

Mesuré sur la couverture du replay canonique :

| symbole | lignes exécutées |
|---|---|
| `run_diagnostics` | **0 / 56** |
| `compute_metrics` | **0 / 287** |
| `generate_plots` | **0 / 26** |
| `capture_snapshots` | **0 / 59** |

La fermeture d'imports du run canonique tire bien `crawlbot/diagnostics/__init__.py`
(qui ré-exporte `run_diagnostics`) mais **aucun des quatre modules**, et ni
`dca` ni `sim_loop` n'appellent la fonction.

Autrement dit : le run canonique produit son `sim_log.json` et ses exports
CSV, mais **pas** la suite de diagnostic décrite par la règle 3. Les métriques
qui font foi aujourd'hui passent par le gate (`gate/run_gate.py`,
`gate/dock_check.py`) et par les scripts d'export.

Ce constat est signalé, **non corrigé** : c'est une question de conformité à
une règle, à trancher par Idriss, indépendante de la structure du code
(CLEANUP-20 §5.3).

---

## 1. `runner.py` — l'orchestrateur

```python
from crawlbot.diagnostics import run_diagnostics
run_diagnostics(log, output_dir, cfg=None, thresholds=None, model=None, data=None)
```

Enchaîne `compute_metrics` → `print_metrics` / `save_metrics_csv` →
`generate_plots`, et `capture_snapshots` si `model`/`data` sont fournis.

Utilisable à la demande sur un log déjà produit :

```bash
MUJOCO_GL=osmesa PYTHONPATH=. python3 -c "
from crawlbot.diagnostics import run_diagnostics
import json
log = json.load(open('results/<log>.json'))
run_diagnostics(log, 'results/<output_dir>/')
"
```

---

## 2. `metrics.py` — les métriques à seuils

`compute_metrics(log, cfg, thresholds)` renvoie un dictionnaire
`nom → (valeur, seuil, verdict)`. `print_metrics` le formate, `save_metrics_csv`
l'écrit.

C'est le plus gros bloc non exercé du dépôt (287 lignes). Sa taille ne dit rien
de sa validité : elle n'a simplement pas de couverture par le gate, donc **le
gate ne détectera aucune régression introduite ici**.

---

## 3. `plots.py` et `snapshots.py`

`generate_plots(log, output_dir, cfg, dpi=150)` produit les planches de
diagnostic ; `capture_snapshots(model, data, sim_log, output_dir, …)` rend des
images MuJoCo aux instants du log.

`snapshots.py` demande un contexte de rendu : `MUJOCO_GL=osmesa`, ou `disabled`
si le rendu n'est pas disponible.

---

## Précaution de mesure

Les figures publiées ne viennent **pas** d'ici : elles sont produites par
`scripts/export_figure_data.py` et `scripts/diag_full_diag_export.py`, à partir
du `sim_log.json` du run. Ne pas supposer qu'une planche de `plots.py`
correspond à une figure de l'article.

---

## Fichiers liés

| quoi | où |
|---|---|
| règle 3 (diagnostics obligatoires) | CLAUDE.md |
| constat de non-exécution | `results/j2_adjconv/PHASE_CLEANUP_20_REPO_AUDIT.md` §5.3 |
| ce qui fait foi aujourd'hui | `gate/run_gate.py`, `gate/dock_check.py` |
| export des figures | `scripts/export_figure_data.py` |

---

## Documentation par module

- [`metrics.md`](metrics.md)
- [`plots.md`](plots.md)
- [`runner.md`](runner.md)
- [`snapshots.md`](snapshots.md)
