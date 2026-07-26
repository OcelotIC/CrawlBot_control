> **⚠ SUPERSEDED (2026-05-27).** This file is stale — it predates or does not
> track the reworked controller. For what the code actually does,
> `docs/architecture/STACK_OVERVIEW.md` is the code-ground-truth reference and
> supersedes any current-state claim here (e.g. the NMPC is 9-state
> `[r_com,v_com,L_com]`, not 12; module APIs and parameters have changed).

# crawlbot.aocs

Contrôle d'attitude par roues à réaction : estimation du couple perturbateur
et commande des roues.

---

## Contexte physique

Le robot crawle le long d'une structure en microgravité. Son mouvement
génère du moment cinétique que les roues à réaction doivent absorber pour
maintenir le pointage de la structure.

Deux composantes du moment cinétique du robot autour de O (CoM structure) :

```
H_{r/O} = L_com + r_com × (m_r · v_com)
           ─────   ────────────────────────
           spin          orbital
```

- **Spin** (~5 Nms) : dû au mouvement des articulations
- **Orbital** (~20 Nms sur 3 pas) : dû à la translation du CoM robot

Le couple perturbateur sur la structure est `τ_dist = -Ḣ_{r/O}`.

---

## MomentumDisturbanceEstimator

**Fichier** : `crawlbot/aocs/force_estimator.py`

Estime `Ḣ_{r/O}` à partir des mesures Pinocchio pour fournir un
feedforward aux roues.

### Construction

```python
est = MomentumDisturbanceEstimator(robot_mass=71.0, dt=0.01)
# ou avec config complète :
est = MomentumDisturbanceEstimator(config=EstimatorConfig(
    robot_mass=71.0, dt=0.01, filter_tau=0.016, include_transport=True))
```

### EstimatorConfig

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `robot_mass` | 71.0 kg | Masse robot (sans structure ni roues) |
| `dt` | 0.01 s | Pas de temps (fréquence QP) |
| `filter_tau` | 0.016 s | Constante de temps EMA (~10 Hz cutoff à 100 Hz) |
| `include_transport` | True | Inclure `ω_s × H` (dérivée inertielle) |

### Algorithme (Variante A — différences finies, recommandée)

```
1. H_k = L_com + r_com × (m · v_com)           # moment total autour de O

2. H̃_k = α · H_k + (1-α) · H̃_{k-1}           # filtre EMA pré-dérivation
   avec α = dt / (τ_f + dt)

3. Ḣ_struct = (H̃_k - H̃_{k-1}) / dt            # FD en repère structure

4. Ḣ_inertiel = Ḣ_struct + ω_s × H_k           # terme de transport
```

Le filtre EMA **avant** la dérivation est essentiel : il lisse le signal
avant l'amplification du bruit par la différence finie.

### Variante B (analytique, si `a_com` disponible)

```
Ḣ = L̇_com + r_com × (m · a_com) + ω_s × H
```

Nécessite `a_com = J_com · q̈ + J̇_com · q̇`, donc `q̈` (accélération
articulaire) — non directement mesurable sans dynamique inverse.

### Méthodes

| Méthode | Entrées | Sortie |
|---------|---------|--------|
| `update(r_com, v_com, L_com, omega_s)` | Mesures Pinocchio + gyro | `Ḣ_{r/O}` (3,) |
| `update_analytical(r_com, v_com, L_com, L_com_prev, a_com, omega_s)` | + accélération | `Ḣ_{r/O}` (3,) |
| `reset()` | — | Réinitialise l'état interne |
| `H_rO` (property) | — | Dernier `H_{r/O}` calculé |
| `H_dot` (property) | — | Dernier `Ḣ` estimé |

### Précision

Validée dans `tests/test_force_estimator.py` :
- **T2** : signe correct vs `qfrc_constraint` MuJoCo
- **T3** : erreur médiane ~1.0 (FD lag inhérent)
- **T4** : conservation `H + hw + L_struct ≈ 0` (résidu ~0.9 Nms)

---

## compute_aocs_command

Fonction helper calculant la commande roue complète :

```python
tau_w = compute_aocs_command(
    H_dot_est,       # feedforward
    omega_s,         # feedback attitude
    hw_current,      # état courant roues
    hw_target=0,     # cible désaturation
    K_omega=50.0,    # gain amortissement [Nm·s/rad]
    K_h=0.5,         # gain désaturation [1/s]
    tau_w_max=5.0    # saturation [Nm]
)
```

### Loi de commande

```
τ_w = -Ḣ_est - K_ω · ω_s - K_h · (h_w - h_w*)
       ──────   ─────────   ──────────────────
       FF dist   FB attitude   FB désaturation
```

Puis clip à `±tau_w_max`.

### Convention de signe

```
τ_w > 0  →  la roue accélère (ḣ_w = τ_w > 0)
             la structure reçoit -τ_w < 0 (réaction)
```

Pour rejeter une perturbation `Ḣ > 0` (robot gagne du moment) :
`τ_w = -Ḣ < 0` → la roue décélère, absorbant le moment.

---

## Modes AOCS disponibles (dans SimConfig)

| Mode | Feedforward | Feedback | Résultat à 8% |
|------|-------------|----------|----------------|
| `'legacy'` | `-L̇_com` (spin seul) | `K_hw · hw_error` | **3/3 docks, 10°** |
| `'H_est'` | `-Ḣ_{r/O}` (spin+orbital) | `K_ω·ω_s + K_h·hw_err` | Double compensation → 18° |
| `'nmpc_plan'` | `hw_dot_plan` du NMPC | `K_hw · hw_error` | λ_plan ≠ réalité → 169° |

Le mode `'legacy'` est le plus robuste car le NMPC gère déjà l'orbital
dans sa planification (via la hw box + ODE corrigée). Le feedforward
n'a besoin que de compenser le résidu d'exécution (spin).
