# `crawlbot.diagnostics.metrics`

Métriques à seuils : calcule, formate et exporte un verdict par grandeur.

**Fichier** : `crawlbot/diagnostics/metrics.py` — **424 lignes** — couverture canonique **5 %**

> Docstring du module : *« Compute scalar summary metrics from SimLog time series. »*

---

## API publique

| symbole | signature | canonique ? |
|---|---|---|
| `compute_metrics` | `(log, cfg=None, thresholds=None)` | non exerce |
| `print_metrics` | `(results, file=None)` | non exerce |
| `save_metrics_csv` | `(results, path)` | non exerce |

### Constantes de module

| nom | valeur |
|---|---|
| `DEFAULT_THRESHOLDS` | `{'torso_pos_err_peak_mm': 10.0, 'torso_ori_e` |

---

## Principe

`compute_metrics(log, cfg, thresholds)` renvoie un dictionnaire
`nom -> (valeur, seuil, verdict)`. `print_metrics` le formate pour la console,
`save_metrics_csv` l'écrit sur disque.

L'intention est celle de la règle 3 : un run ne « passe » pas parce qu'il a
accosté, mais parce que chaque grandeur mesurée est sous son seuil.

## Le plus gros bloc non exercé du dépôt

287 lignes sans couverture. Sa taille ne dit rien de sa validité — elle dit
seulement qu'elle n'est pas vérifiée.

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
