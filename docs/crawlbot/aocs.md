# `crawlbot.aocs`

Contrôle d'attitude de la structure par roues à réaction : estimation du couple
perturbateur et commande des roues.

**Un seul fichier** : `crawlbot/aocs/force_estimator.py` (657 lignes).

> Rédigé depuis le code et depuis la couverture de lignes du replay canonique
> (`gate/replay_canonical.py`). Les mentions « canonique » / « non exercé »
> sont mesurées, pas supposées.

---

## Contexte physique

Le robot rampe le long d'une structure en microgravité. Son mouvement génère du
moment cinétique que les roues à réaction doivent absorber pour maintenir le
pointage de la structure.

Moment cinétique total du robot autour de O (CoM structure) :

```
H_{r/O} = L_com + r_com × (m_r · v_com)
          ─────   ────────────────────────
          spin           orbital
```

Le couple perturbateur subi par la structure est `τ_dist = −Ḣ_{r/O}|_inertiel`,
avec `Ḣ|_inertiel = Ḣ|_struct + ω_s × H_{r/O}`.

---

## Ce qui tourne réellement sur le canonique

`crawlbot/aocs/force_estimator.py` est **couvert à 39 %**. Le module expose six
lois de commande AOCS ; **une seule est exercée** par le run canonique
(`aocs_mode='legacy_pid_numerical'`, fixé dans les kwargs C du replay) :

| symbole | canonique ? |
|---|---|
| `compute_aocs_command_legacy_pid_numerical` | **OUI — la loi canonique** |
| `compute_aocs_command` | non exercé |
| `compute_aocs_command_legacy_corrected` | non exercé |
| `compute_aocs_command_legacy_pd_numerical` | non exercé |
| `compute_aocs_command_legacy_pd_model` | non exercé |
| `compute_aocs_command_legacy_pid_model` | non exercé |
| `MomentumDisturbanceEstimator.update` / `.update_analytical` | **jamais appelé** (§4) |

Les cinq variantes non exercées sont des modes alternatifs sélectionnables par
`aocs_mode` ; elles ne sont pas du code mort mais elles ne sont **pas** validées
par le gate.

---

## 1. La loi canonique — `..._legacy_pid_numerical`

`force_estimator.py:514`. Appelée depuis `sim_loop.py:887`.

```
τ_w = ff_term + K_hw·(clip(h_w) − h_w) + K_θ·θ_s + K_ω·ω_s + K_d·ω̇_s
τ_w ← clip(τ_w, −τ_w,max, +τ_w,max)
```

### Paramètres (valeurs canoniques)

| Paramètre | Valeur | Rôle |
|---|---|---|
| `K_theta` | **1.0** Nm/rad | terme d'attitude — ramène θ_s vers 0 |
| `K_omega` | **50.0** | amortissement sur ω_s |
| `K_d` | 25.0 (défaut) | amortissement sur ω̇_s numérique |
| `K_hw` | 2.0 (défaut) | rappel de désaturation des roues |
| `tau_w_max` | **2.5** Nm | saturation — cap gelé (CLAUDE.md) |

`K_theta`, `K_omega` et `tau_w_max` sont passés explicitement par le run
canonique ; `K_d` et `K_hw` prennent leur valeur par défaut.

### Le signe de K_θ

Positif, même dérivation que `K_ω` et `K_d` : Newton-Euler autour du CoM
structure, avec `τ_w` sur les roues donnant une réaction `−τ_w` sur la
structure. Pour que θ_s > 0 décroisse il faut une accélération angulaire
négative, donc `τ_w > −Ḣ_s`, donc une contribution K_θ positive.

### Borne réelle du terme d'attitude

Il est limité par la **capacité en moment** des roues, pas par leur couple :
ramener la structure de Δθ impose aux roues de porter transitoirement
`|h_w| = I_s·ω_max ≤ h_w,max`, d'où `ω_max = h_w,max / I_s`. Avec
`h_w,max = 5 Nms` et `I_s ≈ 1500 kg·m²` : ≈ 3.3 mrad/s. Une rotation typique
par traversée (~2° = 35 mrad) demande donc ≈ 10 s au minimum.

---

## 2. Les deux feedforwards — et pourquoi il en faut deux

`ff_term` a deux branches, **toutes deux exercées** sur le canonique :

### SS — feedforward cinématique (différences finies), `:585-588`

```
ff_term = −L̇_com − r_com × (m_r · v̇_com)      (dérivées par DF)
```

Valable quand le robot est cinématiquement libre au contact, c.-à-d. en
simple appui.

### DS — feedforward par torseur de contact, `:592`

En double appui la boucle fermée (deux bras soudés) porte une contrainte
interne qui exerce sur la structure un couple `(r_CA − r_CB) × f`
**invisible dans `L_com`**. Le feedforward cinématique est donc incomplet.

`sim_loop.py:877-883` calcule alors directement, depuis `λ_qp` :

```
tau_struct_ff = −Σ_i ( r_Ci × f_i + τ_i )
```

et le passe en argument, ce qui court-circuite la branche DF.

C'est la seule différence de traitement AOCS entre DS et SS.

---

## 3. `EstimatorConfig`

| Paramètre | Défaut | Description |
|---|---|---|
| `robot_mass` | 71.0 kg | masse robot (hors structure et roues) |
| `dt` | 0.01 s | pas de temps (cadence QP) |
| `filter_tau` | 0.016 s | constante EMA (~10 Hz de coupure à 100 Hz) |
| `include_transport` | True | inclure `ω_s × H` (dérivée inertielle) |

---

## 4. ⚠ `MomentumDisturbanceEstimator` n'est pas dans la boucle

L'objet est **construit** (`sim_loop.py:445`) et ses propriétés `H_rO` /
`H_dot` sont **lues à chaque tick pour le log** (`sim_loop.py:1067-1068`),
mais `update()` n'est **jamais appelé** sur le canonique — la couverture donne
0/54 lignes exécutées.

Conséquence mesurée sur le log canonique :

```
H_rO       shape=(2077, 3)   max|.| = 0   all-zero = True
H_dot_est  shape=(2077, 3)   max|.| = 0   all-zero = True
```

**Les deux canaux exportés sont identiquement nuls sur toute la traversée.**
Le feedforward réellement utilisé est calculé en ligne dans
`sim_loop._aocs_command` (§2), pas par cet objet.

À savoir avant de tracer ou d'analyser `H_dot_est` : ce n'est pas un signal,
c'est un zéro. La théorie des variantes A (différences finies filtrées EMA) et
B (analytique via `a_com`) décrite dans le docstring du module reste valide,
mais aucune n'est branchée.

---

## 5. Fichiers liés

| quoi | où |
|---|---|
| appel AOCS dans la boucle | `crawlbot/simulation/sim_loop.py:887` |
| calcul du `tau_struct_ff` DS | `crawlbot/simulation/sim_loop.py:877-883` |
| construction de l'estimateur | `crawlbot/simulation/sim_loop.py:445` |
| note théorique d'origine | `Misc/reports/force_estimator_note.md` |
| cap `tau_w_max` (3 points d'application) | CLAUDE.md, tableau des paramètres |
