# `crawlbot.aocs.force_estimator`

Commande des roues à réaction, et estimation du couple perturbateur que le
mouvement du robot exerce sur la structure.

**Fichier** : `crawlbot/aocs/force_estimator.py` — **657 lignes** — couverture canonique **39 %**

> Docstring du module : *« MomentumDisturbanceEstimator — Estimate the disturbance torque applied by »*

---

## API publique

| symbole | signature | canonique ? |
|---|---|---|
| **`EstimatorConfig`** *(dataclass)* |  |  |
|   `robot_mass` | `71.0` | _champ_ |
|   `dt` | `0.01` | _champ_ |
|   `filter_tau` | `0.016` | _champ_ |
|   `include_transport` | `True` | _champ_ |
| **`MomentumDisturbanceEstimator`** |  |  |
| `.reset` | `()` | non exerce |
| `.update` | `(r_com, v_com, L_com, omega_s)` | non exerce |
| `.update_analytical` | `(r_com, v_com, L_com, L_com_prev, a_com, omega_s)` | non exerce |
| `.H_rO` | `()` | **oui** |
| `.H_dot` | `()` | **oui** |
| `.initialized` | `()` | non exerce |
| `compute_aocs_command` | `(H_dot_est, omega_s, hw_current, hw_target=None, K_omega...)` | non exerce |
| `compute_aocs_command_legacy_corrected` | `(L_com, L_com_prev, r_com, v_com, v_com_prev, hw_current...)` | non exerce |
| `compute_aocs_command_legacy_pd_numerical` | `(L_com, L_com_prev, r_com, v_com, v_com_prev, omega_s, o...)` | non exerce |
| `compute_aocs_command_legacy_pd_model` | `(L_com, L_com_prev, r_com, v_com, v_com_prev, omega_s, t...)` | non exerce |
| `compute_aocs_command_legacy_pid_numerical` | `(L_com, L_com_prev, r_com, v_com, v_com_prev, omega_s, o...)` | **oui** |
| `compute_aocs_command_legacy_pid_model` | `(L_com, L_com_prev, r_com, v_com, v_com_prev, omega_s, t...)` | non exerce |

---

## Contexte physique

Moment cinétique total du robot autour de O (CoM structure) :

```
H_{r/O} = L_com + r_com × (m_r · v_com)
          -----   ------------------------
          spin           orbital
```

Couple perturbateur subi par la structure : `τ_dist = −Ḣ_{r/O}|_inertiel`, avec
`Ḣ|_inertiel = Ḣ|_struct + ω_s × H_{r/O}`.

## Six lois de commande, une seule canonique

Le canonique fixe `aocs_mode='legacy_pid_numerical'`. Les cinq autres sont des
modes alternatifs sélectionnables — pas du code mort, mais **non validés par le
gate**.

### La loi canonique

```
τ_w = ff_term + K_hw·(clip(h_w) − h_w) + K_θ·θ_s + K_ω·ω_s + K_d·ω̇_s
τ_w <- clip(τ_w, ±τ_w,max)
```

| paramètre | valeur canonique | source |
|---|---|---|
| `K_theta` | **1.0** Nm/rad | passé explicitement |
| `K_omega` | **50.0** | passé explicitement |
| `tau_w_max` | **2.5** Nm | passé explicitement (cap gelé) |
| `K_d` | 25.0 | défaut |
| `K_hw` | 2.0 | défaut |

**Signe de K_θ** : positif, même dérivation que K_ω et K_d — Newton-Euler autour
du CoM structure, `τ_w` sur les roues donnant `−τ_w` de réaction sur la
structure.

**Borne réelle du terme d'attitude** : la capacité en *moment*, pas en couple.
Ramener la structure de Δθ impose `|h_w| = I_s·ω_max ≤ h_w,max`, soit
`ω_max ≈ 3.3 mrad/s` pour `h_w,max = 5 Nms` et `I_s ≈ 1500 kg·m²`. Une rotation
typique par traversée (~2°) demande donc ≈ 10 s au minimum.

## Deux feedforwards, tous deux exercés

| phase | branche | pourquoi |
|---|---|---|
| **SS** | `ff = −L̇_com − r_com × m·v̇_com` (différences finies) | le robot est cinématiquement libre au contact |
| **DS** | `ff = tau_struct_ff` passé par l'appelant | la boucle soudée porte une contrainte interne dont le couple `(r_CA − r_CB) × f` est **invisible dans `L_com`** |

`sim_loop.py:877-883` calcule le second directement depuis `λ_qp` :
`tau_struct_ff = −Σ_i (r_Ci × f_i + τ_i)`.

C'est la seule différence de traitement AOCS entre DS et SS.

## ⚠ `MomentumDisturbanceEstimator` n'est pas dans la boucle

L'objet est construit (`sim_loop.py:445`) et ses propriétés `H_rO` / `H_dot`
sont lues à chaque tick pour le journal (`:1067-1068`), mais **`update()` n'est
jamais appelé** (0/54 lignes couvertes).

Mesuré sur le log canonique :

```
H_rO       shape=(2077, 3)   max|.| = 0   all-zero = True
H_dot_est  shape=(2077, 3)   max|.| = 0   all-zero = True
```

**Les deux canaux exportés sont identiquement nuls.** Le feedforward réellement
utilisé est calculé en ligne dans `sim_loop._aocs_command`. La théorie des
variantes A (différences finies filtrées EMA) et B (analytique via `a_com`)
décrite dans le docstring reste valide, mais aucune n'est branchée.

## Voir aussi

- vue d'ensemble du paquet : [`aocs.md`](aocs.md)
