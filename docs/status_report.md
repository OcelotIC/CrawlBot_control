# Status Report — VISPA Crawling Controller
**Date :** 2026-03-30
**Branche :** `claude/addwheels-vispa-7mmNU`

---

## 1. Architecture actuelle

```
                    Structure frame
                         │
TorsoPlanner ──→ CoM ref ──→ CentroidalNMPC (10 Hz)
                                    │
                          r_com_plan, v_com_plan
                          λ_ref, a_com_ff
                                    │
                                    ▼
                             WholeBodyQP (100 Hz)
                                    │
                              τ_q (12 joints)
                                    │
                      ┌─────────────┼──────────────┐
                      ▼             ▼              ▼
                MuJoCo ctrl   AOCS (τ_w)    Weld constraints
                 [0:12]        [12:15]       (contact forces)
```

**Repère commun :** toutes les grandeurs internes (CoM, L_com, ancres,
trajectoires, références) sont en **repère body de la structure**. La
conversion monde↔structure est faite uniquement aux interfaces MuJoCo
(`mujoco_to_pinocchio`, `pinocchio_to_mujoco`).

---

## 2. Ce qui fonctionne

### Refactoring repère structure (cette session)

| Ratio masse | Structure | Docks | Rotation | L_com max | hw peak | NMPC fails |
|---|---|---|---|---|---|---|
| 0.1% | 71 000 kg | **3/3** | 0.23° | 9.5 Nms | 1.0 Nms | 0/406 |
| 1% | 7 110 kg | **3/3** | 1.71° | 7.9 Nms | 1.5 Nms | 0/416 |
| 14% | 500 kg | **3/3** | 23.9° | 8.1 Nms | 0.7 Nms | 3/447 |

- Docking réussi aux 3 paliers de masse, y compris le ratio réaliste 14%
- Le settling DS est beaucoup plus propre qu'avant le refactoring (plus de
  forces fantômes dues à la dérive structure)
- 0 QP failures sur tous les tests

### Corrections AOCS (cette session)

- **Bug L_com_prev** : la dérivée L_dot utilisait L_com du pas NMPC
  précédent (Δt=0.1s) divisé par dt_qp (0.01s) → 10x trop grand.
  Corrigé : suivi au taux QP sub-step.
- **Bug ctrlrange** : `aocs_tau_w_max` dans le contrôleur (5 Nm) n'avait
  aucun effet car MuJoCo clippait à `ctrlrange=±0.5 Nm` en amont.
  Identifié et documenté.

### Modèle physique (session précédente)

- 3 roues à réaction orthogonales dans le MJCF (`VISPA_crawling_rwa3.xml`)
- Correction orbitale dans la dynamique hw du NMPC :
  `hw_dot = -L_dot - (r_com - r_mid) × Σf_j`
- AOCS feedforward+feedback sur les roues physiques MuJoCo

---

## 3. Problèmes ouverts

### 3.1 Moment orbital non compensé par les roues

**Constat :** à 14%, la structure tourne de 24° malgré les roues. Le
moment cinétique du robot a deux composantes :

```
L_robot^O = L_com (spin, centroïdal) + m·(r_com - O) × v_com (orbital)
```

- **Spin (~5 Nms)** : bien compensé par le feedforward AOCS `-L_dot_com`
- **Orbital (~20 Nms)** : transite par les forces de contact (3ème loi de
  Newton), pas par les roues → produit la rotation structure

Le terme orbital est proportionnel à `v_com` (vitesse de crawl) et au bras
de levier `r_com - O` (distance à l'axe de rotation).

**Expériences tentées :**

| Approche | Résultat | Pourquoi ça échoue |
|---|---|---|
| Augmenter hw_max et aocs_tau_w_max | Identique (24°) | Le moment orbital ne passe pas par les roues |
| Augmenter I_w physique (100x) | Identique (24°) | Idem — canal physique différent |
| Feedforward avec lambda_qp | 168°, 2/3 docks | Gap QP-MuJoCo : λ_qp ≠ forces réelles (33x) |
| Feedforward avec m·a_com | 171°, 1/3 docks | Dérivée numérique de v_com trop bruitée |
| Passivité NMPC (W_hw > 0) | 180°, 1/3 docks | Modèle quasi-inertiel incorrect à 14% |
| Mouvement lent (t_swing=18s) | 46°, 3/3 docks | Plus de temps = plus d'accumulation orbitale |

### 3.2 Gap QP-MuJoCo

Le QP optimise des wrenches virtuels `λ` mais n'envoie que `τ_q` à MuJoCo.
Les forces de contact réelles sont déterminées par le constraint solver de
MuJoCo, pas par `λ`. Écart mesuré : `L_dot_réel` atteint 188 Nm vs
contrainte QP de 5 Nm (33×).

**Conséquence :** les wrenches QP ne peuvent pas servir de feedforward
fiable pour l'AOCS (testé et vérifié).

### 3.3 Settling DS — 0 DOF articulaire

En double support avec 2 welds × 6D, les 12 DOF articulaires sont
entièrement contraints. Le QP ne peut pas tracker le torso indépendamment.
Le hold capturé en fin d'EXT ne correspond pas à l'équilibre statique
post-dock → erreur constante.

---

## 4. Piste : estimateur de forces de contact

Pour que l'AOCS compense le moment orbital, il faut connaître la **force
de contact réelle** Σf_j appliquée au robot. Deux sources échouent :
- `lambda_qp` : décision du QP, pas la réalité MuJoCo
- `m·a_com` via dérivée numérique : trop bruité à 100 Hz

**Piste proposée :** un **estimateur de forces de contact** (momentum-based
observer), analogue à ce qui existe en robotique terrestre (De Luca 2006,
Haddadin 2017). Le principe :

```
F_ext_est = K_O · ∫ (H·q̈ + C - τ - J_c^T · F_ext_est) dt
         = K_O · (p_robot - ∫ (τ + J_c^T·F_ext_est - g) dt)
```

En microgravité (g=0) et avec la dynamique corps-complet Pinocchio :

```
Σf_est = (1/dt) · (H·v_new - H·v_old - τ_applied·dt)
```

Avantages :
- Utilise des grandeurs **mesurées** (q, v, τ appliqué) et non prédites
- Filtre naturellement le bruit (intégration vs dérivation)
- Indépendant du constraint solver MuJoCo
- Peut fournir les 6 composantes du wrench (force + couple) par contact

Le moment orbital serait alors :
```
tau_w_orbital = -(r_com - r_mid) × Σf_est
tau_w_total = -L_dot_centroidal + tau_w_orbital
```

**Risques :**
- La matrice H dépend de la configuration → erreur si le modèle Pinocchio
  ne match pas exactement MuJoCo (inerties, masses)
- Le terme `H·q̈` nécessite q̈ (accélération) ou la formulation intégrale
  (momentum p = H·v)
- Latence d'un pas sur l'estimation

---

## 5. Structure du repo

```
CrawlBot_control/
├── simulation_loop.py      # Boucle principale NMPC+QP+AOCS
├── robot_interface.py      # Wrapper Pinocchio (CRBA, RNEA, Jacobiens)
├── contact_scheduler.py    # Gait plan et timing des contacts
├── torso_planner.py        # Trajectoire 6D torso (repère structure)
├── swing_planner.py        # Trajectoire EE swing (repère structure)
├── locomotion_planner.py   # Référence CoM (deprecated, remplacé par TorsoPlanner)
├── dynamics.py             # Dynamique contrainte (SHAKE/RATTLE)
├── ik.py                   # IK pour configurations de dock
│
├── solvers/
│   ├── centroidal_nmpc.py  # Stage 1 — NMPC centroïdal (CasADi/IPOPT)
│   ├── wholebody_qp.py     # Stage 2 — QP corps-complet (qpOASES) [NE PAS MODIFIER]
│   ├── nmpc_solver.py      # Wrapper générique NMPC
│   ├── hierarchical_qp.py  # QP hiérarchique (non utilisé)
│   └── contact_phase.py    # ContactConfig, momentum map
│
├── models/                 # Modèles robot/environnement
│   ├── VISPA_crawling_rwa3.xml      # MJCF avec 3 roues (modèle principal)
│   ├── VISPA_crawling.xml           # MJCF sans roues
│   ├── VISPA_crawling_fixed.urdf    # URDF pour Pinocchio
│   └── VISPA_crawling_rwa4_pyramid.xml  # 4 roues pyramide (futur)
│
├── scripts/                # Expériences et visualisation
├── tests/                  # Suite de tests
├── lutze_baseline/         # Implémentation de référence (Lütze et al.)
├── docs/                   # Documentation technique
└── results/                # Logs JSON et figures PNG
    ├── logs/
    └── figures/
```

---

## 6. Fichiers de documentation clés

| Fichier | Contenu |
|---|---|
| `docs/controller_architecture.md` | Architecture NMPC+QP, flux de données, contraintes |
| `docs/momentum_conservation_analysis.md` | Dérivation du terme orbital, choix du point de référence |
| `docs/refactoring_plan_structure_frame.md` | Plan du refactoring repère structure (complété) |
| `docs/R5fix_R6new.md` | Cahier des charges R5-fix + R6 (session précédente) |
| `docs/status_report.md` | Ce document |
