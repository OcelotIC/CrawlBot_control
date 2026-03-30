# Cahier des charges — R5-fix + R6-new
## VISPA Crawling Robot — RWA Physical Modelling + Full Simulation

**Date :** 2026-03-29  
**Repo :** `https://github.com/OcelotIC/CrawlBot_control.git`  
**Clonage local :** `/home/claude/CrawlBot_control/`  
**Objectif papier :** démontrer que le NMPC maintient `‖hw‖ ≤ h_max = 5 Nm·s` sur une séquence
multi-pas, contrairement à une baseline réactive (Lutze-like).

---

## Contexte et problèmes identifiés

### Bug #1 — Contrainte `hw` 10× trop lâche *(bloquant pour le papier)*

```python
# simulation_loop.py SimConfig (lignes 128-129)
hw_min = np.full(3, -50.)   # ← devrait être -5.0
hw_max = np.full(3,  50.)   # ← devrait être +5.0

# solvers/centroidal_nmpc.py CentroidalNMPCConfig (lignes 75-76)
hw_min = -50.0 * np.ones(3)  # ← devrait être -5.0
hw_max =  50.0 * np.ones(3)  # ← devrait être +5.0
```

**Conséquence :** `hw` monte jusqu'à 17.4 Nm·s sur 3 pas (log `r6_multistep_log.json`).
La contrainte (V-B.5) du papier n'est pas imposée — le mécanisme central du papier n'est pas
démontré.

### Bug #2 — Roues à réaction absentes du modèle physique

Le MJCF `VISPA_crawling.xml` ne contient aucune roue à réaction. La structure (`body name="structure"`)
a un `freejoint` libre mais **aucun actuateur** : elle dérive passivement sans jamais absorber le
momentum du robot. `hw` dans le code est un simple compteur logiciel, jamais relié à une force
physique appliquée sur la structure.

**Layout DOF actuel (nq=26, nv=24, nu=12) :**
```
qpos[0:7]   → structure freejoint (pos + quat wxyz)
qpos[7:14]  → torso freejoint (pos + quat wxyz)
qpos[14:26] → 12 joints robot (Joint_1_a … Joint_6_b)
```

**Conséquence :** la rotation de la structure en simulation n'est pas due aux roues mais à la
réaction directe des contacts — le claim "les roues absorbent le momentum" n'est pas physiquement
simulé.

---

## Scope de R5-fix + R6-new

### R5-fix — Modèle RWA physique (Option B : 3 roues orthogonales, Hsw=I₃)

Ajouter 3 roues à réaction orthogonales dans le corps `structure` du MJCF.
Créer un contrôleur AOCS qui applique des couples sur ces roues pour absorber le momentum robot.
Corriger `hw_min/max = ±5 Nm·s` partout. Valider sur `n_steps=3`.

### R6-new — Simulation complète avec figures papier

Avec le modèle R5-fix validé : lancer la simulation complète NMPC vs Lutze-baseline, générer
les figures publiables (trajectoire `hw(t)`, comparaison saturation, snapshots).

---

## R5-fix — Spécifications détaillées

### Tâche 1 : MJCF — Ajouter 3 roues orthogonales à la structure

**Fichier :** `models/VISPA_crawling.xml`  
**Fichier parallèle à créer :** `models/VISPA_crawling_rwa3.xml` (copie modifiée, ne pas écraser
l'original)

Ajouter dans `<body name="structure">`, après les géoms existants, 3 corps de roue :

```xml
<!-- ============ REACTION WHEEL ASSEMBLY (3 orthogonal axes, Hsw=I3) ============ -->
<!-- Wheel X: spin axis along structure X -->
<body name="rwa_x" pos="0 0 0.05">
  <inertial pos="0 0 0" mass="0.5"
            fullinertia="0.01 5e-4 5e-4 0 0 0"/>
  <joint name="rw_x" type="hinge" axis="1 0 0" damping="1e-4" armature="0.01"/>
  <geom type="cylinder" size="0.05 0.02" rgba="0.8 0.2 0.2 0.7"/>
</body>
<!-- Wheel Y: spin axis along structure Y -->
<body name="rwa_y" pos="0 0 0.10">
  <inertial pos="0 0 0" mass="0.5"
            fullinertia="5e-4 0.01 5e-4 0 0 0"/>
  <joint name="rw_y" type="hinge" axis="0 1 0" damping="1e-4" armature="0.01"/>
  <geom type="cylinder" size="0.05 0.02" rgba="0.2 0.8 0.2 0.7"/>
</body>
<!-- Wheel Z: spin axis along structure Z -->
<body name="rwa_z" pos="0 0 0.15">
  <inertial pos="0 0 0" mass="0.5"
            fullinertia="5e-4 5e-4 0.01 0 0 0"/>
  <joint name="rw_z" type="hinge" axis="0 0 1" damping="1e-4" armature="0.01"/>
  <geom type="cylinder" size="0.05 0.02" rgba="0.2 0.2 0.8 0.7"/>
</body>
```

**Paramètres des roues :**
- Masse par roue : 0.5 kg (3 roues → 1.5 kg sur 500 kg structure = négligeable)
- Inertie spin : `I_w = 0.01 kg·m²` (axiale), transverse `5e-4 kg·m²`
- Damping mécanique : `1e-4 N·m·s/rad` (palier quasi-idéal)
- Ce choix donne `h_w_max = I_w * ω_max`. Pour `ω_max = 500 rad/s` :
  `h_w_max = 0.01 × 500 = 5 Nm·s` — cohérent avec `h_max` du papier.

**Ajouter dans `<actuator>` :**
```xml
<!-- RWA actuators (ctrl indices 12, 13, 14) -->
<motor name="act_rw_x" joint="rw_x" gear="1" ctrlrange="-0.5 0.5"/>
<motor name="act_rw_y" joint="rw_y" gear="1" ctrlrange="-0.5 0.5"/>
<motor name="act_rw_z" joint="rw_z" gear="1" ctrlrange="-0.5 0.5"/>
```

`ctrlrange="-0.5 0.5"` N·m correspond au couple max de roue (`τ_w_max = 0.5 Nm`, cohérent avec
`τ_w_max` dans `SimConfig`).

**Nouveau layout DOF (nq=29, nv=27, nu=15) après modification :**
```
qpos[0:7]   → structure freejoint (pos + quat wxyz)
qpos[7]     → rw_x angle [rad]
qpos[8]     → rw_y angle [rad]
qpos[9]     → rw_z angle [rad]
qpos[10:17] → torso freejoint (pos + quat wxyz)
qpos[17:29] → 12 joints robot
```

---

### Tâche 2 : Mettre à jour `mujoco_to_pinocchio` pour le nouveau layout

**Fichier :** `simulation_loop.py`  
**Fonction :** `mujoco_to_pinocchio(mj_qpos, mj_qvel)` (ligne 68)

Le décalage de 3 DOF (roues) doit être pris en compte. **Pinocchio ne modélise que le robot**
(torso + joints), pas les roues.

```python
def mujoco_to_pinocchio(mj_qpos, mj_qvel):
    """
    Layout MuJoCo (avec RWA) :
      qpos[0:7]   structure freejoint (pos + quat wxyz)
      qpos[7:10]  rw_x, rw_y, rw_z angles (SKIP pour Pinocchio)
      qpos[10:17] torso freejoint (pos + quat wxyz)
      qpos[17:29] 12 joints robot
    Layout Pinocchio :
      pin_q[0:3]  torso pos
      pin_q[3:7]  torso quat (xyzw)
      pin_q[7:19] 12 joints robot
    """
    pin_q = np.zeros(19)
    pin_q[0:3] = mj_qpos[10:13]         # torso pos (offset +3 vs ancien)
    w, x, y, z = mj_qpos[13:17]
    pin_q[3:7] = [x, y, z, w]           # xyzw pour Pinocchio
    pin_q[7:19] = mj_qpos[17:29]        # joints (offset +3 vs ancien)

    pin_v = np.zeros(18)
    pin_v[0:3] = mj_qvel[9:12]          # torso linvel (offset +3)
    pin_v[3:6] = mj_qvel[12:15]         # torso angvel
    pin_v[6:18] = mj_qvel[15:27]        # joints vel

    return pin_q, pin_v
```

Vérifier et mettre à jour aussi `pinocchio_to_mujoco` (fonction inverse) si elle existe.

**Mettre à jour également toutes les lectures directes de `mj_data.qpos` dans
`simulation_loop.py` qui utilisent des indices hardcodés :**

| Ancien index | Nouveau index | Contenu |
|---|---|---|
| `qpos[0:3]` | `qpos[0:3]` | structure pos ✓ inchangé |
| `qpos[3:7]` | `qpos[3:7]` | structure quat ✓ inchangé |
| `qpos[7:14]` | `qpos[10:17]` | torso freejoint |
| `qpos[14:26]` | `qpos[17:29]` | joints robot |

**Nouveaux indices utiles à exposer :**
```python
# Constantes à définir en tête de simulation_loop.py
MJ_IDX_STRUCT_POS  = slice(0, 3)
MJ_IDX_STRUCT_QUAT = slice(3, 7)
MJ_IDX_RW_ANGLES   = slice(7, 10)   # rw_x, rw_y, rw_z
MJ_IDX_TORSO_POS   = slice(10, 13)
MJ_IDX_TORSO_QUAT  = slice(13, 17)
MJ_IDX_JOINTS      = slice(17, 29)
```

---

### Tâche 3 : Contrôleur AOCS dans `simulation_loop._step()`

**Principe :** conservation du momentum total. Les roues doivent absorber le momentum du robot :

```
dh_w/dt = τ_w   (dynamique des roues)
dh_w/dt = -dL_robot/dt   (conservation)
→ τ_w = -dL_robot/dt   (feedforward)
```

On ajoute un terme de feedback pour garantir que `hw` reste dans `[hw_min, hw_max]` :

```
τ_w = -dL_robot/dt - K_hw * (hw - hw_clamp)
```

où `hw_clamp = clip(hw, hw_min, hw_max)` est l'erreur de saturation.

**Paramètres AOCS :**
```python
# Dans SimConfig
aocs_K_hw: float = 2.0   # gain feedback [Nm / (Nm·s)] = [1/s]
aocs_tau_w_max: float = 0.5  # couple max par roue [Nm]
```

**Implémentation dans `_step()`, dans la boucle QP :**

```python
# Lecture vitesses angulaires roues
rw_vel = self.mj_data.qvel[6:9]   # dofs 6,7,8 → rw_x, rw_y, rw_z

# Momentum roues physique (Hsw = I3 * I_w = 0.01 * I3)
I_w = 0.01  # kg·m² — doit correspondre au MJCF
hw_physical = I_w * rw_vel   # [Nm·s], shape (3,)

# Taux de variation L_robot (différence finie)
L_dot_est = (rs.L_com - L_com_prev) / cfg.dt_qp

# Feedforward + feedback
hw_error = np.clip(hw_physical, cfg.hw_min, cfg.hw_max) - hw_physical
tau_w = -L_dot_est - cfg.aocs_K_hw * hw_error
tau_w = np.clip(tau_w, -cfg.aocs_tau_w_max, cfg.aocs_tau_w_max)

# Appliquer aux roues (ctrl indices 12, 13, 14)
self.mj_data.ctrl[12:15] = tau_w
```

**La variable `hw` de la boucle devient `hw_physical` :** lire depuis `mj_data.qvel` (état
physique) plutôt qu'intégrer manuellement `Δhw = -ΔL_com`. L'intégration manuelle reste
pour warm start NMPC uniquement.

---

### Tâche 4 : Corriger `hw_min/max = ±5 Nm·s` partout

**Fichier `simulation_loop.py` `SimConfig` :**
```python
hw_min: np.ndarray = field(default_factory=lambda: np.full(3, -5.0))  # ← corrigé
hw_max: np.ndarray = field(default_factory=lambda: np.full(3,  5.0))  # ← corrigé
hw_init: np.ndarray = field(default_factory=lambda: np.zeros(3))      # ← init à 0
```

**Fichier `solvers/centroidal_nmpc.py` `CentroidalNMPCConfig` :**
```python
hw_min: np.ndarray = field(default_factory=lambda: -5.0 * np.ones(3))  # ← corrigé
hw_max: np.ndarray = field(default_factory=lambda:  5.0 * np.ones(3))  # ← corrigé
```

---

### Tâche 5 : Logging enrichi

Dans `SimLog` et la fonction de logging de `_step()`, ajouter :

```python
hw_physical: list = field(default_factory=list)   # hw lu depuis qvel MuJoCo
tau_w: list = field(default_factory=list)          # couples AOCS appliqués
rw_speed: list = field(default_factory=list)       # vitesses angulaires roues [rad/s]
```

---

### Tâche 6 : Préparer le MJCF Option C (4 roues pyramide) — ne pas intégrer

**Fichier à créer :** `models/VISPA_crawling_rwa4_pyramid.xml`

Copie de `VISPA_crawling_rwa3.xml` avec 4 roues en configuration pyramide standard
(inclinaison β = 54.74° ≈ arctan(√2) depuis la verticale, azimuts 0°, 90°, 180°, 270°) :

```xml
<!-- Axes pyramide standard (β=54.74°, cos β = 1/√3 ≈ 0.5774) -->
<!-- rw1: azimut 0°  -->  axis=" 0.8165  0      0.5774"
<!-- rw2: azimut 90° -->  axis=" 0       0.8165 0.5774"
<!-- rw3: azimut 180°-->  axis="-0.8165  0      0.5774"
<!-- rw4: azimut 270°-->  axis=" 0      -0.8165 0.5774"
```

Hsw pour ce modèle (pour information, pas implémenté) :
```
Hsw = I_w * [ 0.8165  0       -0.8165  0      ]
             [ 0       0.8165   0       -0.8165 ]
             [ 0.5774  0.5774   0.5774   0.5774 ]
```

Ce fichier doit compiler sans erreur MuJoCo mais n'est **pas utilisé dans la simulation R5/R6**.
Un commentaire en tête du fichier doit indiquer : `<!-- Option C — 4-wheel pyramid RWA.
Prepared for future work. Controller not implemented. -->`

---

### Tâche 7 : Suite de validation R5-fix

**Fichier :** `tests/test_r5fix_rwa.py`

Tests à passer (en plus de la régression R3/R4 inchangée) :

| Test | Critère | Seuil |
|---|---|---|
| T1 — Layout DOF | `m.nq==29, m.nv==27, m.nu==15` | exact |
| T2 — Joints roues présents | `rw_x, rw_y, rw_z` dans le modèle | exact |
| T3 — `mujoco_to_pinocchio` cohérent | `pin_q[0:3]` = position torso après offset | numérique |
| T4 — AOCS actif | `max(|tau_w|) > 0` sur la simulation | > 0 |
| T5 — `hw_physical` borné | `max(‖hw_physical‖) ≤ h_max = 5 Nm·s + tolérance 20%` | ≤ 6.0 Nm·s |
| T6 — Docking n_steps=3 | 3/3 docks à d < 5 mm | exact |
| T7 — Violation rate | `hw_physical > 5 Nm·s` < 5% des steps | < 5% |
| T8 — Régression R3 | 11/11 PASS | exact |
| T9 — Régression R4 | 6/6 PASS | exact |

**Note T5 :** une tolérance de 20% est acceptée pour les pics transitoires aux transitions de
phase (phénomène documenté dans le rapport R5 §4 — pic géométrique de la contrainte box ≤ √3 · h_max).

**Commande de lancement :**
```bash
cd /home/claude/CrawlBot_control
pip install pin casadi mujoco numpy --break-system-packages -q
PYTHONPATH=. MUJOCO_GL=disabled python3 tests/test_r5fix_rwa.py
```

---

## R6-new — Spécifications simulation complète

*Ces specs s'appliquent une fois R5-fix entièrement validé (T1–T9 PASS).*

### Scénario de simulation

| Paramètre | Valeur |
|---|---|
| Nombre de pas | **5 pas** (suffisant pour montrer accumulation et prévention) |
| Modèle | `VISPA_crawling_rwa3.xml` (3 roues physiques) |
| Contrôleur A | NMPC + QP (présent travail) |
| Contrôleur B | Lutze-baseline (`lutze_baseline/`) — QP Stage 2 uniquement, sans contrainte hw |
| `hw_min/max` | ±5 Nm·s |
| `h_max` papier | 5 Nm·s |

### Résultats attendus (à démontrer dans la figure)

- **Contrôleur A (NMPC) :** `hw_physical(t)` reste dans `[-5, +5]` Nm·s sur 5 pas, sauf pics
  transitoires ≤ √3 · 5 ≈ 8.66 Nm·s d'une durée < 300 ms
- **Contrôleur B (Lutze) :** `hw_physical(t)` sort des bornes et sature (violoation croissante)

### Figures à générer pour le papier

**Figure 1 — `fig_hw_comparison.pdf/.png`** (figure principale, Section VII)  
4 sous-graphes (2×2) :
- `[0,0]` : `hw_physical(t)` composantes x,y,z — NMPC — bandes grises pour les phases DS/SS/EXT
- `[0,1]` : `‖hw_physical‖(t)` — NMPC vs Lutze — ligne rouge h_max=5, ligne tiretée √3·h_max
- `[1,0]` : `‖hw_physical‖(t)` Lutze — montrant la saturation progressive
- `[1,1]` : `τ_w(t)` (couples AOCS) — 3 composantes

**Figure 2 — `fig_locomotion_multistep.pdf/.png`** (figure trajectoire)  
3 sous-graphes verticaux :
- Trajectoire CoM xy (vue de dessus) avec les 5 ancrages successifs marqués
- Erreur de tracking torso `‖e_torso‖(t)` — NMPC
- Distance gripper→ancre `d_grip(t)` — indication docking events

**Figure 3 — `fig_solver_performance.pdf/.png`** (optionnel, Table 1 du papier)  
- Temps de résolution NMPC et QP par step (boxplot ou scatter)
- Taux de succès solver

### Métriques à reporter dans le texte du papier (table)

```
Metric                  | NMPC (this work) | Lutze baseline
Docking success rate    |  x/5             | x/5
Peak ‖hw‖ [Nm·s]       |  x.xx            | x.xx (saturation)
Mean ‖hw‖ [Nm·s]       |  x.xx            | x.xx
hw violation rate [%]   |  x.x             | x.x
Peak τ_w [Nm]           |  x.xx            | N/A
NMPC solve time [ms]    |  x ± x           | N/A
QP solve time [ms]      |  x ± x           | x ± x
```

### Fichiers de sortie

```
results/logs/r5fix_multistep_nmpc.json     ← log NMPC 5 pas
results/logs/r5fix_multistep_lutze.json    ← log Lutze 5 pas
results/figures/fig_hw_comparison.pdf
results/figures/fig_hw_comparison.png
results/figures/fig_locomotion_multistep.pdf
results/figures/fig_locomotion_multistep.png
results/figures/fig_solver_performance.pdf  (optionnel)
```

---

## Notes pour Claude Code

### Pattern shell fiable
```bash
# Toujours écrire sur disque avant d'exécuter
cat > /home/claude/CrawlBot_control/script.py << 'EOF'
# contenu
EOF
PYTHONPATH=/home/claude/CrawlBot_control MUJOCO_GL=disabled python3 script.py
```

### Ne pas modifier les fichiers suivants
- `solvers/centroidal_nmpc.py` (sauf hw_min/max par défaut — Tâche 4)
- `solvers/wholebody_qp.py`
- `robot_interface.py`
- `models/VISPA_crawling.xml` (original préservé, travailler sur `_rwa3.xml`)

### Ordre d'exécution obligatoire
1. Tâche 1 (MJCF rwa3 + rwa4_pyramid)
2. Tâche 2 (mujoco_to_pinocchio)
3. Tâche 3 (AOCS dans _step)
4. Tâche 4 (hw_min/max ±5)
5. Tâche 5 (logging)
6. Tâche 6 (validation T1–T9)
7. → Validation humaine des courbes avant de passer à R6-new
8. R6-new (simulation 5 pas + figures)

### Dépendances
```bash
pip install pin casadi mujoco numpy matplotlib --break-system-packages -q
```

---

## Critère de succès global

R5-fix est **COMPLÉTÉ** si et seulement si :
- T1–T9 tous PASS
- Les courbes `hw_physical(t)` sur 3 pas restent dans `[-5, +5]` Nm·s avec violation rate < 5%
- Les courbes sont présentées à Idriss pour validation visuelle avant de passer à R6-new

R6-new est **COMPLÉTÉ** si et seulement si :
- Les figures `fig_hw_comparison` montrent clairement la différence NMPC vs Lutze
- Le tableau de métriques est complet
- Idriss valide les figures pour insertion dans le papier
