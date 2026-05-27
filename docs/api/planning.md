> **⚠ SUPERSEDED (2026-05-27).** This file is stale — it predates or does not
> track the reworked controller. For what the code actually does,
> `docs/architecture/STACK_OVERVIEW.md` is the code-ground-truth reference and
> supersedes any current-state claim here (e.g. the NMPC is 9-state
> `[r_com,v_com,L_com]`, not 12; module APIs and parameters have changed).

# crawlbot.planning

Planificateurs de trajectoire : gait schedule, torso 6D, swing EE, CoM.

Tous les planificateurs travaillent en **repère structure**. Aucune
conversion world/structure n'est nécessaire.

---

## ContactScheduler

**Fichier** : `crawlbot/planning/contact_scheduler.py`

Gestion du timing de marche et des configurations de contact.

### Construction

```python
sched = ContactScheduler(anchors_a=list_of_3d, anchors_b=list_of_3d,
                          dt_ds=0.5, dt_ss=6.0)
plan = sched.plan_traversal(start_a=2, start_b=2, n_steps=3)
```

### Cycle de marche

```
DS (0.5s) → SS_A (6.0s) → DS (0.5s) → SS_B (6.0s) → ...
```

- **DS** : double support, les deux bras accrochés (settling)
- **SS_A** : appui simple sur A, bras B en swing
- **SS_B** : appui simple sur B, bras A en swing

### Méthodes principales

| Méthode | Retour | Usage |
|---------|--------|-------|
| `plan_traversal(start_a, start_b, n_steps)` | `GaitPlan` | Séquence de phases avec timing |
| `anchor_se3(arm, idx)` | `pin.SE3` | Pose SE3 d'une ancre (orientation = identité) |

### Grille d'ancres

```
anchor_ia = [i·0.8,  +0.3, 0.0]    # rangée A (y > 0)
anchor_ib = [i·0.8,  -0.3, 0.0]    # rangée B (y < 0)
```

6 paires (i=0..5), espacement 0.8 m en X.

---

## TorsoPlanner

**Fichier** : `crawlbot/planning/torso_planner.py`

Génère la trajectoire 6D du torso (position + orientation) et en dérive
la référence CoM pour le NMPC.

### Interpolation quintic

Pour une phase de `t_start` à `t_end`, le paramètre normalisé `τ = (t - t_start) / T` :

```
s(τ)  = 10τ³ - 15τ⁴ + 6τ⁵           (position)
ṡ(τ)  = (30τ² - 60τ³ + 30τ⁴) / T    (vitesse)
s̈(τ) = (60τ - 180τ² + 120τ³) / T²  (accélération)
```

Propriétés : départ et arrivée au repos (`ṡ(0) = ṡ(1) = 0`,
`s̈(0) = s̈(1) = 0`), profil de vitesse en cloche.

### Orientation : SLERP via Pinocchio

```python
ω_total = log3(R_start^T · R_end)     # vecteur rotation total
R(t) = R_start · exp3(s(τ) · ω_total)  # interpolation
```

### Dérivation CoM

Le CoM est reconstruit depuis la pose torso via un offset body-frame :

```
r_com(t) = p_torso(t) + R_torso(t) · δ_com(s(t))
v_com = v_lin + ω × (R · δ) + R · δ̇
```

### Méthodes

| Méthode | Retour | Description |
|---------|--------|-------------|
| `set_hold(p, R, r_com)` | — | Référence statique (DS settling) |
| `add_phase(t_start, t_end, p0, R0, p1, R1)` | — | Phase de mouvement |
| `reference_at(t)` | `TorsoReference` | p, R, v(6D), a(6D) à l'instant t |
| `com_reference_at(t)` | `ComReference` | r_com, v_com à l'instant t |
| `clear_phases()` | — | Réinitialise |

---

## SwingPlanner

**Fichier** : `crawlbot/planning/swing_planner.py`

Trajectoire cartésienne du bras en swing avec bump de dégagement.

### Trajectoire

```
p(τ) = p_start + Δp · s(τ) + clearance · n̂ · bump(τ)
```

- `s(τ)` : quintic (même que TorsoPlanner)
- `bump(τ) = sin²(πτ)` : cloche C¹, max à τ=0.5
- `n̂ = [0, 0, -1]` : normale à la structure (pointe vers l'espace libre)
- `clearance = 0.03 m` par défaut

### Vitesse et accélération

```
v(τ) = Δp · ṡ + clearance · n̂ · ḃump
a(τ) = Δp · s̈ + clearance · n̂ · b̈ump
```

avec `ḃump = 2π sin(πτ) cos(πτ) / T` et `b̈ump` par dérivation.

### Méthodes

| Méthode | Retour | Description |
|---------|--------|-------------|
| `reference_at(t)` | `SwingReference` | p_ee, v_ee, a_ee + bras + progression τ |
| `adaptive_reference_at(t, p_ee_current)` | `SwingReference` | Re-planification depuis la position actuelle |

---

## LocomotionPlanner

**Fichier** : `crawlbot/planning/locomotion_planner.py`

Référence CoM par interpolation quintic entre des waypoints d'équilibre
statique. **Deprecated** : remplacé par `TorsoPlanner.com_reference_at()`.

### Waypoints par phase

- **DS** : milieu des deux ancres `(r_a + r_b) / 2`
- **SS_A** : au-dessus de l'ancre A, décalé vers le centre
- **SS_B** : au-dessus de l'ancre B, décalé vers le centre
- **Z** : hauteur fixe `com_height = -0.47 m`

### Pré-compensation masse du bras

```
r_com_ref += (m_arm / m_total) · s(τ) · Δp_arm
```

Compense le déplacement du CoM dû au mouvement du bras en swing.
