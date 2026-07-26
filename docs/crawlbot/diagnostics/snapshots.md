# `crawlbot.diagnostics.snapshots`

Captures d'images MuJoCo aux instants du log.

**Fichier** : `crawlbot/diagnostics/snapshots.py` — **71 lignes** — couverture canonique **0 %**

> Docstring du module : *« Render MuJoCo frames at key simulation instants for visual diagnostics. »*

---

## API publique

| symbole | signature | canonique ? |
|---|---|---|
| `capture_snapshots` | `(model, data, sim_log, output_dir, width=1280, height=72...)` | non exerce |

---

## Usage

`capture_snapshots(model, data, sim_log, output_dir, width=1280, height=720,
camera=None)` — rend les poses enregistrées dans `log.snapshots`.

Nécessite un contexte de rendu : `MUJOCO_GL=osmesa`, ou `disabled` si le rendu
n'est pas disponible (règle du projet : ne jamais lancer sans l'une des deux).

Le canonique capture 44 poses lorsque `cfg.frames_per_step > 0` ; le rendu
lui-même passe par `scripts/render_traversal.py`, pas par ce module.

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
