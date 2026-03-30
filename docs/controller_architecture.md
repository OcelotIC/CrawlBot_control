# Architecture du contrôleur hiérarchique — Vue d'ensemble

## 1. Vue globale

Contrôleur à deux étages en cascade :

```
TorsoPlanner ──→ CoM ref ──→ CentroidalNMPC (Stage 1, 10 Hz)
                                    │
                          r_com_plan, v_com_plan
                          λ_ref, a_com_ff
                                    │
                                    ▼
                             WholeBodyQP (Stage 2, 100 Hz)
                                    │
                              τ_q (12 couples)
                                    │
                                    ▼
                              MuJoCo (1 kHz)
                                    │
                           mj_data.ctrl[:12] = τ_q
                           mj_data.ctrl[12:15] = τ_w (AOCS)
```

**Repère commun :** toutes les quantités internes (positions, vitesses,
forces, moments) sont exprimées dans le **repère body de la structure**
(frame solidaire de la poutre/satellite). `mujoco_to_pinocchio()` transforme
la pose et le twist du torso depuis le repère monde MuJoCo vers ce repère
structure. MuJoCo reste en repère monde ; la conversion se fait aux
interfaces.

Les ancres de contact, les trajectoires torso/swing, les références NMPC
et QP sont toutes dans ce même repère structure. Les avantages :
- Les ancres sont des **constantes géométriques** (positions CAD)
- Aucune correction n'est nécessaire pour la dérive de la structure
- Les Fix 3 / Fix R4 / live anchors sont supprimés

---

## 2. Stage 1 — CentroidalNMPC

### Rôle
Planifier une trajectoire CoM et des wrenches de contact **faisables en
momentum** sur un horizon de prédiction. Garantir que le momentum des
roues `hw` reste dans l'enveloppe `[-h_max, h_max]`.

### Fréquence
10 Hz (dt_nmpc = 0.1 s). Un appel NMPC par boucle externe.

### Vecteur d'état (nx = 12)

```
x = [r_com(3), v_com(3), L_com(3), hw(3)]
```

| Variable | Dim | Description | Repère |
|----------|-----|-------------|--------|
| `r_com` | 3 | Position CoM du robot | Structure body |
| `v_com` | 3 | Vitesse CoM du robot | Structure body |
| `L_com` | 3 | Moment centroïdal du robot (autour du CoM robot) | Structure body |
| `hw` | 3 | Moment cinétique des roues à réaction | Structure body (≈ monde si R_struct ≈ I) |

**Source des états initiaux (à chaque appel) :**
- `r_com, v_com, L_com` : calculés par Pinocchio (`rs.r_com`, `rs.v_com`,
  `rs.L_com`) à partir de l'état MuJoCo converti
- `hw` : lu depuis les vitesses de roues MuJoCo (`I_w * mj_data.qvel[6:9]`)
  si RWA physique, sinon intégré manuellement

### Vecteur de contrôle (nu = 12)

```
u = [f1(3), τ1(3), f2(3), τ2(3)]
```

| Variable | Dim | Description |
|----------|-----|-------------|
| `f1` | 3 | Force de contact au point C1 (gripper A) |
| `τ1` | 3 | Couple de contact au point C1 |
| `f2` | 3 | Force de contact au point C2 (gripper B) |
| `τ2` | 3 | Couple de contact au point C2 |

Les contacts inactifs sont mis à zéro via les bornes de contrôle
(`_apply_contact_bounds`).

### Vecteur de paramètres (np = 12)

```
p = [r_ref(3), v_ref(3), r_C1(3), r_C2(3)]
```

| Paramètre | Dim | Source |
|-----------|-----|--------|
| `r_ref` | 3 | Référence CoM position (TorsoPlanner → com_reference_at) [struct frame] |
| `v_ref` | 3 | Référence CoM vitesse (TorsoPlanner) [struct frame] |
| `r_C1` | 3 | Position du contact A (ancre stance, constante en repère structure) |
| `r_C2` | 3 | Position du contact B (ancre stance, constante en repère structure) |

**Paramètres constants gelés à l'horizon :** r_ref, v_ref, r_C1, r_C2 sont
les mêmes pour tous les pas de l'horizon (pas de variation temporelle).

### Dynamique (ODE, intégrée RK4)

```
ṙ_com = v_com

v̇_com = (f1 + f2) / m                                         [Newton, μg]

L̇_com = Σ_j [(r_Cj - r_com) × f_j + τ_j]                     [centroïdal]

ḣ_w  = -L̇_com - (r_com - r_mid) × (f1 + f2)                  [conservation]
      où r_mid = (r_C1 + r_C2) / 2
```

**Note sur ḣ_w :** la dynamique corrigée calcule le moment des forces de
contact autour du milieu des contacts (proxy du CoM structure), et non
autour du CoM mobile du robot. Le terme `(r_com - r_mid) × Σf_j` est la
**correction orbitale** qui comptabilise le moment cinétique dû au
déplacement du robot le long de la structure.

### Fonction de coût

**Coût d'étage (par pas k) :**
```
ℓ(x,u,p) = ‖r_com - r_ref‖²_Wr + ‖v_com - v_ref‖²_Wv + ‖u‖²_Wu
```

Poids par défaut : Wr = 100·I₃, Wv = 10·I₃, Wu_f = 0.01, Wu_τ = 0.001.

**Coût terminal :**
```
ℓ_f(x,p) = ‖r_com - r_ref‖²_Qf_r + ‖v_com - v_ref‖²_Qf_v
```

Qf_r = 1000·I₃, Qf_v = 100·I₃.

### Contraintes

| Contrainte | Type | Expression | Valeurs par défaut |
|------------|------|------------|-------------------|
| hw box | State bounds | hw_min ≤ hw ≤ hw_max | [-5, 5] Nms (avec marge 10%) |
| L_com box | State bounds | -L_max ≤ L_com ≤ L_max | L_max = 8 Nms |
| Force SOC | Path ineq. | ‖f_j‖² ≤ f_max² | f_max = 3000 N |
| Couple SOC | Path ineq. | ‖τ_j‖² ≤ τ_max² | τ_max = 300 Nm |
| L̇ bilatérale | Path ineq. | -τ_w_max ≤ L̇_com ≤ τ_w_max | τ_w_max = 1.0 Nm |
| Contact inactif | Control bounds | f_j = 0, τ_j = 0 | Dynamique |

**Marge de sécurité (ε = 0.1) :**
- hw_min_safe = (1 + ε) × hw_min = -5.5 Nms
- hw_max_safe = (1 - ε) × hw_max = 4.5 Nms

### Sorties (pour Stage 2)

| Sortie | Description |
|--------|-------------|
| `r_com_plan` | Position CoM planifiée à t+dt (x_opt[0:3, 1]) |
| `v_com_plan` | Vitesse CoM planifiée à t+dt (x_opt[3:6, 1]) |
| `λ_ref` | Wrenches de contact optimaux à t=0 (u_opt[:, 0]) |
| `a_com_ff` | Accélération feedforward = (f1+f2)/m |

---

## 3. Stage 2 — WholeBodyQP

### Rôle
Tracker les références du NMPC en résolvant la dynamique corps-complet du
robot. Produit les couples articulaires envoyés à MuJoCo.

### Fréquence
100 Hz (dt_qp = 0.01 s). 10 appels QP par appel NMPC.

### Variables de décision (n = 6 + 12 + 12 + 12 = 42)

```
z = [q̈_t(6), q̈(12), λ(12), τ_q(12)]
```

| Variable | Dim | Description |
|----------|-----|-------------|
| `q̈_t` | 6 | Accélération torso [linéaire(3), angulaire(3)] |
| `q̈` | 12 | Accélérations articulaires (6 par bras) |
| `λ` | 12 | Wrenches de contact [f1(3), τ1(3), f2(3), τ2(3)] |
| `τ_q` | 12 | Couples articulaires (sortie vers MuJoCo) |

### Entrées

**Depuis Pinocchio (état robot, repère structure) :**

| Entrée | Dim | Description |
|--------|-----|-------------|
| `q_t` | 7 | Pose torso [pos(3), quat(4)] |
| `dq_t` | 6 | Twist torso [lin(3), ang(3)] |
| `q` | 12 | Positions articulaires |
| `dq` | 12 | Vitesses articulaires |
| `H_robot` | 18×18 | Matrice de masse (CRBA) |
| `C_robot` | 18 | Termes Coriolis/centrifuges (RNEA, gravité=0) |
| `J_com` | 3×18 | Jacobienne du CoM |
| `Jdot_dq_com` | 3 | J̇_com · q̇ |
| `r_com` | 3 | Position CoM courante |
| `L_com_current` | 3 | Moment centroïdal courant |

**Depuis le NMPC (Stage 1) :**

| Entrée | Dim | Description |
|--------|-----|-------------|
| `r_com_ref` | 3 | Position CoM référence (= r_com_plan du NMPC) |
| `v_com_ref` | 3 | Vitesse CoM référence (= v_com_plan du NMPC) |
| `λ_ref` | 12 | Wrenches référence (= λ_plan du NMPC) |
| `a_com_ff` | 3 | Accélération feedforward (= (f1+f2)/m du NMPC) |

**Depuis le TorsoPlanner (trajectoire 6D, repère structure) :**

| Entrée | Dim | Description |
|--------|-----|-------------|
| `p_torso_ref` | 3 | Position torso référence |
| `R_torso_ref` | 3×3 | Orientation torso référence |
| `v_torso_ref` | 6 | Twist torso référence [lin, ang] |
| `a_torso_ff` | 6 | Accélération torso feedforward |

**Depuis le SwingPlanner (effecteur en mouvement, repère structure) :**

| Entrée | Dim | Description |
|--------|-----|-------------|
| `p_ee_ref` | 3 | Position EE référence |
| `v_ee_ref` | 3 | Vitesse EE référence |
| `a_ee_ff` | 3 | Accélération EE feedforward |

**Depuis l'AOCS / MuJoCo :**

| Entrée | Dim | Description |
|--------|-----|-------------|
| `hw_current` | 3 | Momentum roues courant |
| `hw_min/max` | 3 | Bornes momentum roues |

**Contacts :**

| Entrée | Dim | Description |
|--------|-----|-------------|
| `contact_config` | — | Phase (SS/DS), positions contacts, contacts actifs |
| `J_contacts` | 6nc×18 | Jacobiennes de contact empilées |
| `Jdot_dq_contacts` | 6nc | J̇_contact · q̇ |

### Contraintes d'égalité

**1. Dynamique corps-complet :**
```
H_robot · q̈_robot + C_robot = B_u · τ_q + J_contacts^T · λ
```

où `q̈_robot = [q̈_t; q̈]` (18D), `B_u = [0_{6×12}; I_{12}]` (les couples
n'agissent que sur les joints, pas sur la base flottante).

**2. Contrainte de contact (accélération nulle au contact) :**
```
J_contact · q̈_robot = -J̇_contact · q̇_robot
```

### Contraintes d'inégalité

**1. Momentum safety (hw box) :**
```
hw_min ≤ hw_current - dt · M_λ · λ ≤ hw_max
```

où `M_λ` est la **matrice de momentum** (3×12) qui projette les wrenches
de contact sur le taux de variation du moment centroïdal :

```
L̇_com = M_λ · λ
M_λ = [S(r_C1 - r_com), I₃, S(r_C2 - r_com), I₃]
```

S(·) = matrice skew-symmetric.

**POINT IMPORTANT :** cette matrice `M_λ` utilise `r_Cj - r_com` (bras de
levier depuis le CoM du robot), pas depuis le CoM structure. C'est le
moment centroïdal L̇_com, **pas** le moment autour du point fixe. Cette
incohérence entre NMPC et QP est un point à examiner (voir section 5).

**2. L_com box :**
```
|L_com_current + dt · M_λ · λ| ≤ L_max (composante par composante)
```

**3. L̇ rate box :**
```
|M_λ · λ| ≤ τ_w_max (composante par composante)
```

### Bornes sur les variables

| Variable | Borne inf | Borne sup |
|----------|-----------|-----------|
| `q̈_t` | -∞ | +∞ |
| `q̈` | -qdd_max | +qdd_max (50 rad/s²) |
| `λ` (actif) | -f_max / -τ_max | +f_max / +τ_max |
| `λ` (inactif) | 0 | 0 |
| `τ_q` | -τ_max | +τ_max (10 Nm) |

### Tâches (hiérarchie pondérée)

Toutes les tâches sont de la forme : minimiser `‖A·z - b‖²` pondéré par α.

| Priorité | Tâche | α (SS) | α (EXT) | Expression |
|----------|-------|--------|---------|------------|
| 1 | CoM tracking | 200 | 100 | J_com·q̈ = a_com_des - J̇_com·q̇ |
| 1 | Torso 6D tracking | 500 | 50 | J_torso·q̈ = a_torso_des - J̇_torso·q̇ |
| 2 | EE tracking (swing) | 3000 | 10000 | J_ee·q̈ = a_ee_des - J̇_ee·q̇ |
| 3 | Posture regulation | 20 | 5 | q̈ = Kp(q_nom - q) - Kd·dq |
| 4 | Wrench tracking | 100 | 100 | λ = λ_ref |
| 5 | Torque minimization | 1 | 1 | τ_q = 0 |
| 6 | Accel. regularization | 0.01 | 0.01 | q̈_robot = 0 |

Les accélérations désirées utilisent un PD :
```
a_com_des = a_com_ff + Kp·(r_ref - r_com) + Kd·(v_ref - v_com)
a_torso_des = a_ff + Kp·e_6d + Kd·(v_ref - v)
a_ee_des = a_ff + Kp·(p_ref - p) + Kd·(v_ref - v)
```

### Sorties

| Sortie | Dim | Description | Destination |
|--------|-----|-------------|-------------|
| `τ_q` | 12 | Couples articulaires | mj_data.ctrl[0:12] (clippé à ±τ_max) |

---

## 4. AOCS (contrôleur des roues à réaction)

### Rôle
Appliquer des couples sur les 3 roues physiques MuJoCo pour absorber le
momentum du robot et maintenir `hw` dans les bornes.

### Fréquence
100 Hz (même boucle que le QP interne).

### Loi de commande

```
hw_phys = I_w · ω_roues                    [lecture depuis MuJoCo qvel[6:9]]
L̇_est = (L_com - L_com_prev) / dt_qp      [différence finie]
hw_error = clip(hw_phys, hw_min, hw_max) - hw_phys
τ_w = -L̇_est - K_hw · hw_error            [feedforward + feedback]
τ_w = clip(τ_w, -τ_w_max_aocs, τ_w_max_aocs)
```

Paramètres : K_hw = 2.0 [1/s], τ_w_max_aocs = 0.5 Nm, I_w = 0.01 kg·m².

### Sortie
`mj_data.ctrl[12:15] = τ_w` (3 couples roues).

---

## 5. Incohérences identifiées

### 5.1 M_λ dans le QP vs correction orbitale dans le NMPC

Le NMPC utilise la correction orbitale pour `ḣ_w` :
```
ḣ_w = -L̇_com - (r_com - r_mid) × Σf_j
```

Le QP utilise `M_λ` pour la contrainte momentum :
```
hw_new ≈ hw - dt · M_λ · λ
M_λ = matrice de moment centroïdal (autour de r_com)
```

**Le QP prédit `Δhw = -dt · L̇_com` (centroïdal) sans la correction
orbitale.** La contrainte hw dans le QP est donc incohérente avec la
dynamique hw du NMPC.

Pour corriger : `M_λ` devrait utiliser `r_Cj - r_mid` au lieu de
`r_Cj - r_com`. Cependant, le cahier des charges interdit de modifier
`wholebody_qp.py`. L'impact est atténué par le fait que le QP opère
sur un seul pas (dt=0.01s), donc l'erreur est petite par pas.

### 5.2 L̇_com dans la path constraint NMPC

La contrainte `|L̇_com| ≤ τ_w_max` dans le NMPC (lignes 224-234 de
centroidal_nmpc.py) utilise L̇_com centroïdal (autour de r_com), pas
le moment autour du point fixe. Cela contraint le taux de variation
du moment centroïdal, mais pas directement le couple sur la structure.

Le couple réel sur la structure est L̇_com + correction orbitale
(= moment autour du point fixe). La contrainte devrait porter sur cette
quantité pour effectivement limiter la rotation de la structure.

### 5.3 Repère de hw

`hw` dans le NMPC et le QP est traité comme un vecteur en repère monde.
Physiquement, `hw = R_struct · I_w · ω_roues` avec les vitesses de roue
en repère structure. Tant que `R_struct ≈ I` (hypothèse de la commande),
c'est correct. Si la structure tourne significativement, l'approximation
se dégrade.

---

## 6. Flux de données par pas NMPC

Toutes les grandeurs ci-dessous sont en **repère structure** sauf indication
contraire.

```
1. Lire MuJoCo: qpos, qvel (repère monde)
2. Convertir: mujoco_to_pinocchio(qpos, qvel) → pin_q, pin_v
   Transforme le torso en repère structure:
     p_local = R_s^T · (p_torso_monde - p_struct)
     R_local = R_s^T · R_torso_monde
     v_local = R_s^T · (v_torso - v_struct - ω_struct × Δp)
     ω_local = R_s^T · (ω_torso - ω_struct)
3. Pinocchio: rs = robot.update(pin_q, pin_v)
   → r_com, v_com, L_com, H, C, J_com, J_torso, J_ee, ...
   (tout en repère structure)
4. Contact config: r_C1, r_C2 = sched.anchors_a/b (constantes locales)
5. TorsoPlanner: reference_at(t) → p_ref, R_ref, v_ref, a_ff
6. TorsoPlanner: com_reference_at(t) → r_com_ref, v_com_ref
7. NMPC: solve(r_com, v_com, L_com, hw, r_com_ref, v_com_ref, contacts)
   → r_com_plan, v_com_plan, λ_ref, a_com_ff
8. Boucle QP interne (×10):
   a. Relire état MuJoCo → mujoco_to_pinocchio (repère structure)
   b. TorsoPlanner.reference_at(tq) (repère structure direct)
   c. QP: solve(...) → τ_q
   d. clip τ_q à ±τ_max
   e. AOCS: calculer τ_w (hw lu depuis roues physiques)
   f. Appliquer: ctrl[0:12] = τ_q, ctrl[12:15] = τ_w
   g. mj_step()
   h. Mettre à jour hw depuis qvel roues
10. Logging
```
