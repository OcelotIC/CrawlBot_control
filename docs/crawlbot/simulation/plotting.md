# `crawlbot.simulation.plotting`

Tracés de simulation — **non utilisé par le run canonique** (2 % de
couverture).

**Fichier** : `crawlbot/simulation/plotting.py` — **154 lignes** — couverture canonique **2 %**

> Docstring du module : *« 9-panel diagnostic plot for simulation results. »*

---

## API publique

| symbole | signature | canonique ? |
|---|---|---|
| `plot_simulation` | `(log, save_path=None, cfg=None)` | non exerce |

---

## Statut

`plot_simulation(log, save_path, cfg)` n'est appelée ni par `dca` ni par
`sim_loop` sur le chemin canonique. Le seul point d'entrée est
`SimulationLoop.plot`, lui-même non exercé.

## Ce qui produit réellement les figures

| usage | outil |
|---|---|
| figures de l'article | `scripts/export_figure_data.py` |
| export fulldiag 66 colonnes | `scripts/diag_full_diag_export.py` |
| planches de diagnostic | `crawlbot/diagnostics/plots.py` (également non exercé) |

⚠ Ne pas supposer qu'une planche produite ici correspond à une figure publiée :
ce n'est pas la même chaîne.

## Conséquence

Aucune couverture par le gate. Une régression introduite ici ne sera détectée
par rien.

## Voir aussi

- vue d'ensemble du paquet : [`simulation.md`](simulation.md)
