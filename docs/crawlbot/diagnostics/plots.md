# `crawlbot.diagnostics.plots`

Planches de figures de diagnostic à partir d'un log de simulation.

**Fichier** : `crawlbot/diagnostics/plots.py` — **689 lignes** — couverture canonique **5 %**

> Docstring du module : *« Generate the fixed set of 8 diagnostic figures from SimLog. »*

---

## API publique

| symbole | signature | canonique ? |
|---|---|---|
| `generate_plots` | `(log, output_dir, cfg=None, dpi=150)` | non exerce |

### Constantes de module

| nom | valeur |
|---|---|
| `_PHASE_COLORS` | `{'DS': 'blue', 'SS': 'orange'}` |

---

## Usage

`generate_plots(log, output_dir, cfg=None, dpi=150)` — appelé par `run_diagnostics`.

## ⚠ Ce ne sont pas les figures de l'article

Les figures publiées sont produites par `scripts/export_figure_data.py` et
`scripts/diag_full_diag_export.py` à partir du `sim_log.json`. Chaîne
différente : ne pas supposer qu'une planche d'ici correspond à une figure
publiée.

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


## Voir aussi

- vue d'ensemble du paquet : [`diagnostics.md`](diagnostics.md)
