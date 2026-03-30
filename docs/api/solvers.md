# crawlbot.solvers

Contrôleur hiérarchique à deux étages : NMPC centroïdal (Stage 1) +
QP corps-complet (Stage 2).

---

## CentroidalNMPC (Stage 1)

**Fichier** : `crawlbot/solvers/centroidal_nmpc.py`

Planification de trajectoire CoM respectant l'enveloppe de capacité des
roues à réaction sur un horizon de prédiction.

### Dimensions

| Vecteur | Taille | Contenu |
|---------|--------|---------|
| État `x` | 12 | `[r_com(3), v_com(3), L_com(3), h_w(3)]` |
| Contrôle `u` | 12 | `[f_1(3), τ_1(3), f_2(3), τ_2(3)]` |
| Paramètres `p` | 12 | `[r_ref(3), v_ref(3), r_C1(3), r_C2(3)]` |

### ODE (intégrée RK4)

```
ṙ_com = v_com
v̇_com = (f₁ + f₂) / m                                          (Newton, µg)
L̇_com = Σⱼ [(r_Cj - r_com) × fⱼ + τⱼ]                        (centroïdal)
ḣ_w   = -L̇_com - (r_com - r_mid) × (f₁ + f₂)                  (conservation)
```

`r_mid = (r_C1 + r_C2) / 2` — proxy pour le CoM structure. Le terme
`(r_com - r_mid) × Σf` est la **correction orbitale** qui comptabilise
le moment dû à la translation du robot.

> **Note** : le point de référence correct est O = origine du repère
> structure (CoM structure, `inertial pos="0 0 0"`), ce qui simplifie
> en `r_com × Σf`. Cette correction est implémentée dans les modèles 8%
> (`VISPA_crawling_rwa3_8pct.xml`). Le modèle par défaut utilise encore
> `r_mid` pour compatibilité avec les tests de baseline.

### Coût

```
Étage :   ‖r_com - r_ref‖²_100 + ‖v_com - v_ref‖²_10 + ‖u‖²_0.01
Terminal : ‖r_com - r_ref‖²_1000 + ‖v_com - v_ref‖²_100 [+ W_hw·‖h_w‖²]
```

### Contraintes

| Type | Expression | Défaut |
|------|-----------|--------|
| hw box (state) | `hw_min ≤ h_w ≤ hw_max` | ±5 Nms (marge 10%) |
| L_com box (state) | `‖L_com‖ ≤ L_max` | 10 Nms |
| Force SOC (path) | `‖fⱼ‖² ≤ f_max²` | 25 N |
| Couple SOC (path) | `‖τⱼ‖² ≤ τ_max²` | 8 Nm |
| L̇ rate (path) | `‖L̇_com‖ ≤ τ_w_max` | 5 Nm |
| Contact inactif | `fⱼ = 0, τⱼ = 0` | via bornes |

### Sorties (pour Stage 2)

```python
r_com_plan, v_com_plan, L_com_plan, lambda_plan, hw_dot_plan, info = nmpc.solve(...)
a_com_ff = nmpc.compute_feedforward_acceleration(lambda_plan)
```

- `r_com_plan, v_com_plan` : état prédit à t+dt (référence QP)
- `lambda_plan` : wrenches de contact optimaux à t=0
- `hw_dot_plan` : `(hw[1] - hw[0]) / dt` — feedforward AOCS planifié
- `a_com_ff` : `(f₁ + f₂) / m` — accélération feedforward

---

## WholeBodyQP (Stage 2)

**Fichier** : `crawlbot/solvers/wholebody_qp.py`
**⚠ NE PAS MODIFIER** (contrainte du cahier des charges)

Tracking haute fréquence des références NMPC avec dynamique corps-complet.

### Variables de décision

```
z = [q̈_t(6), q̈(12), λ(12), τ_q(12)]    total: 42
```

### Contraintes d'égalité

```
1. Dynamique : H · [q̈_t; q̈] + C = [0; I] · τ_q + J_c^T · λ
2. Contact :   J_c · [q̈_t; q̈] = -J̇_c · [dq_t; dq]
```

### Contraintes d'inégalité

```
1. hw box :    hw_min ≤ hw - dt · M_λ · λ ≤ hw_max
2. L_com box : ‖L_com + dt · M_λ · λ‖ ≤ L_max
3. L̇ rate :   ‖M_λ · λ‖ ≤ τ_w_max
```

où `M_λ` est la matrice de momentum centroïdal :
```
M_λ = [ [r_C1 - r_com]× I₃  [r_C2 - r_com]× I₃ ]     (3×12)
```

> **Incohérence connue** : M_λ utilise `r_Cj - r_com` (moment autour du
> CoM mobile), pas `r_Cj - O` (moment autour du CoM structure). L'erreur
> est ~0.05 Nms/pas et non cumulative car le QP se re-linéarise à chaque pas.

### Hiérarchie de tâches (pondérée)

| Priorité | Tâche | α (SS) | Expression |
|----------|-------|--------|------------|
| 1 | CoM tracking | 200 | `J_com · q̈ = a_des - J̇_com · q̇` |
| 1 | Torso 6D | 500 | `J_torso · q̈ = a_6d_des - J̇_torso · q̇` |
| 2 | EE swing | 3000 | `J_ee · q̈ = a_ee_des - J̇_ee · q̇` |
| 3 | Posture | 20 | `q̈ = Kp(q_nom - q) - Kd · dq` |
| 4 | Wrench tracking | 100 | `λ = λ_ref` |
| 5 | Couple min | 1 | `τ_q → 0` |
| 6 | Régul. accél. | 0.01 | `[q̈_t; q̈] → 0` |

PD pour les accélérations désirées :
```
a_des = a_ff + Kp · (r_ref - r) + Kd · (v_ref - v)
```

---

## HierarchicalQP (backend générique)

**Fichier** : `crawlbot/solvers/hierarchical_qp.py`

Solveur QP hiérarchique supportant deux modes :
- **weighted** : QP unique, priorités via ratio de poids (défaut × 1000)
- **strict** : cascade avec projection sur le noyau (exact mais plus lent)

### Problème résolu par tâche

```
min  ½ ‖A · x - b‖²_W
s.t. C_eq · x = d_eq
     C_ineq · x ≤ d_ineq
     lb ≤ x ≤ ub
```

Backend : CasADi conic (qpOASES par défaut, fallback osqp).

---

## ContactPhase / ContactConfig

**Fichier** : `crawlbot/solvers/contact_phase.py`

### ContactPhase (enum)

```python
SINGLE_A   # bras A accroché, B en swing
SINGLE_B   # bras B accroché, A en swing
DOUBLE     # les deux accrochés
```

### ContactConfig (dataclass)

```python
ContactConfig(phase, nc, active_contacts, r_contact_A, r_contact_B)
```

Construit via `ContactConfig.from_phase(phase, r_A, r_B)`.

### `compute_momentum_map(r_com, contact_config) -> M_λ (3×12)`

Matrice projetant les wrenches de contact sur le taux de moment centroïdal :

```
L̇_com = M_λ · λ

M_λ = [ [r_C1 - r_com]×  I₃  [r_C2 - r_com]×  I₃ ]
```

Les colonnes des contacts inactifs sont nulles.

### `skew(v) -> (3×3)`

Matrice antisymétrique : `skew(v) · w = v × w`.
