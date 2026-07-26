# `crawlbot.planning.locomotion_planner`

Planificateur de CoM de la génération précédente. **Mort sur le canonique,
conservé pour la ligne de base M0/Lutze de l'article.**

**Fichier** : `crawlbot/planning/locomotion_planner.py` — **206 lignes** — couverture canonique **17 %**

> Docstring du module : *« LocomotionPlanner — CoM reference trajectory generation for VISPA. »*

---

## API publique

| symbole | signature | canonique ? |
|---|---|---|
| **`LocomotionPlanner`** |  |  |
| `.calibrate_from_config` | `(r_com_init)` | non exerce |
| `._build_waypoints` | `()` | non exerce |
| `._equilibrium_com` | `(phase, r_a, r_b)` | non exerce |
| `.reference_at` | `(t)` | non exerce |
| `.full_trajectory` | `(dt)` | non exerce |

### Constantes de module

| nom | valeur |
|---|---|
| `DEFAULT_COM_HEIGHT` | `-0.47` |

---

## Statut : conservé sur mesure

`sim_loop.py:46` porte le commentaire *« LocomotionPlanner removed — CoM
reference comes from TorsoPlanner »*, et de fait `sim_loop` ne le construit
jamais. Couverture **17 %**, `full_trajectory` sans appelant.

CLEANUP-16 l'avait classé « supprimer — 205 lignes, risque faible ». **Révisé sur
mesure** (CLEANUP-18 §3) : il a trois consommateurs, tous dont les imports
résolvent, et le décisif est **`lutze_baseline/sim_lutze.py`** — un *paquet*, pas
un script de recherche, qui porte la comparaison M0/Lutze sous-tendant le tableau
§II de l'article.

`LocomotionPlanner` y est porteur : construit à `sim_lutze.py:175`, calibré à
`:176`, évalué à `:231` et `:266`.

Supprimer aurait échangé 205 lignes contre une ligne de base d'article cassée.

## La leçon de méthode

L'audit avait affirmé que ses consommateurs étaient « déjà non fonctionnels ».
Vérification par résolution d'imports : **faux pour les trois**. D'où la règle
adoptée : ne pas supposer qu'un script est déjà cassé — le tester.

Revisiter cette décision n'est pas une question de code, mais de projet : « la
ligne de base Lutze doit-elle encore être rejouée ? » (`CLEANUP_CARRYOVER` §C5).

## Voir aussi

- vue d'ensemble du paquet : [`planning.md`](planning.md)
