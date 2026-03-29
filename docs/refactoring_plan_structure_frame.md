# Plan de refactoring — Tout en repère structure

## 1. Constat

Le contrôleur actuel exprime toutes les grandeurs en **repère monde inertiel**.
Le robot est physiquement attaché à la structure via des weld constraints.
Quand la structure dérive (translation + rotation), le contrôleur voit des
"mouvements" qui n'existent pas dans le repère local de la structure et génère
des forces parasites pour les corriger.

Conséquences observées :
- Oscillations en DS settling (le contrôleur excite la structure)
- Accumulation de patchs (Fix 3, Fix R4, live anchors) pour compenser
- Incohérence entre les composants (swing planner en nominal, torso planner
  en structure, NMPC en monde)

## 2. Principe du refactoring

**Tout exprimer dans le repère structure (body frame de la poutre).**

Dans ce repère :
- Les ancres sont des constantes géométriques (CAD)
- Un robot docké en DS a une configuration articulaire constante
- Les trajectoires swing/torso sont des déplacements entre points fixes
- Aucune correction n'est nécessaire pour la dérive
- Les Fix 3, Fix R4, live anchors deviennent inutiles

## 3. Transformation MuJoCo → Repère structure

### 3.1 Grandeurs disponibles dans MuJoCo

```
p_struct = qpos[0:3]       # position CoM structure en monde
R_struct = quat_to_rot(qpos[3:7])  # rotation structure en monde
```

### 3.2 Transformation monde → structure

Pour tout point/vecteur en repère monde :
```
p_local = R_struct^T @ (p_world - p_struct)
v_local = R_struct^T @ (v_world - v_struct - omega_struct × (p_world - p_struct))
```

Pour les moments cinétiques (pseudo-vecteurs) :
```
L_local = R_struct^T @ L_world
```

### 3.3 Ce qui change dans `mujoco_to_pinocchio`

Actuellement : extrait la pose torso en monde, la passe à Pinocchio.
Après refactoring : exprime la pose torso **relative à la structure**.

```python
# Torso pose relative à la structure
p_torso_world = qpos[10:13]  # (layout RWA)
R_torso_world = quat_to_rot(qpos[13:17])
p_torso_local = R_struct^T @ (p_torso_world - p_struct)
R_torso_local = R_struct^T @ R_torso_world

# Pinocchio reçoit les coordonnées en repère structure
pin_q[0:3] = p_torso_local
pin_q[3:7] = rot_to_quat_xyzw(R_torso_local)
```

Les vitesses aussi doivent être transformées :
```python
v_torso_local = R_struct^T @ (v_torso_world - v_struct
                              - omega_struct × (p_torso_world - p_struct))
omega_torso_local = R_struct^T @ (omega_torso_world - omega_struct)
```

## 4. Ce qui change dans chaque composant

### 4.1 `mujoco_to_pinocchio()` / `pinocchio_to_mujoco()`

Ajouter `p_struct, R_struct, v_struct, omega_struct` comme arguments.
Transformer torso pose/twist en repère structure avant de passer à Pinocchio.

**Impact :** Pinocchio calcule TOUT en repère structure (r_com, v_com, L_com,
Jacobiens, etc.) de manière transparente.

### 4.2 `TorsoPlanner`

Les trajectoires sont DÉJÀ stockées en repère structure (Fix 3).
Le refactoring **supprime** la reconstruction monde dans `reference_at()`.
La sortie est directement en repère structure.

→ Supprimer les arguments `p_struct, R_struct, v_struct, omega_struct`
  de `reference_at()` et `com_reference_at()`.

### 4.3 `SwingPlanner`

Les ancres du scheduler sont en repère monde initial ≈ repère structure initial.
Avec Pinocchio en repère structure, les positions d'ancre du scheduler sont
directement les bonnes coordonnées (positions locales constantes).

→ Supprimer toute la machinerie live anchors, `_nominal_to_live`,
  `set_initial_struct_pose`.
→ `reference_at()` retourne des positions en repère structure directement.

### 4.4 `ContactScheduler` / `read_anchors_from_mujoco()`

Les ancres sont lues live depuis MuJoCo (en monde) puis utilisées par le NMPC.
Après refactoring : utiliser les positions **locales** constantes au lieu de
lire les sites MuJoCo.

→ Supprimer `read_anchors_from_mujoco()` dans `_step()`.
→ Utiliser `sched.anchors_a[idx]` directement (déjà en repère structure
  puisque la structure est initialement à l'identité dans MuJoCo... ou presque :
  elle est à pos=[0,0,-1.8]).

**Attention :** les ancres du scheduler sont en monde initial (structure à
pos=[0,0,-1.8]). Il faut vérifier si elles sont en monde ou en local.

### 4.5 `CentroidalNMPC`

Tout ce que Pinocchio fournit (r_com, v_com, L_com) est maintenant en repère
structure. Les paramètres (r_ref, r_C1, r_C2) aussi. Le NMPC n'a rien à
changer — il travaille dans un repère cohérent.

La correction orbitale `(r_com - r_mid) × Σf_j` utilise r_com et r_mid en
repère structure → correct (les deux sont dans le même repère, et le repère
structure est quasi-inertiel si la structure bouge peu).

**Point subtil :** le repère structure n'est PAS inertiel s'il tourne. Les
équations de Newton (`m a_com = Σf_j`) ne sont exactes que dans un repère
inertiel. Si la structure tourne de quelques degrés, l'erreur est faible
(forces de Coriolis/centrifuges proportionnelles à ω_struct² et ω_struct × v,
négligeables pour ω_struct petit). À valider expérimentalement.

### 4.6 `WholeBodyQP`

Reçoit ses entrées de Pinocchio (maintenant en repère structure) et du NMPC
(aussi en repère structure). Aucun changement nécessaire.

### 4.7 `AOCS` dans `_step()`

Le feedforward AOCS utilise `L_com` (maintenant en repère structure ≈ repère
monde si R_struct ≈ I). Le couple appliqué aux roues est en repère structure
(les roues sont sur la structure). Cohérent.

### 4.8 `_step()` dans `simulation_loop.py`

- Transformer l'état MuJoCo en repère structure dans `mujoco_to_pinocchio`
- Supprimer la lecture de `p_struct, R_struct` pour le torso planner
- Supprimer `read_anchors_from_mujoco` pour le NMPC
- Utiliser les ancres locales constantes pour le contact config
- Les sorties du QP (τ articulaires) sont indépendantes du repère

### 4.9 Logging

Les grandeurs loggées (r_com, p_torso, L_com, hw) seront en repère structure.
Pour les plots, on peut les reconvertir en monde si nécessaire, ou logger
dans les deux repères.

## 5. Ce qui est supprimé

- Fix 3 (trajectoire en repère structure + reconstruction monde) → inutile
- Fix R4 (ancres live depuis MuJoCo) → inutile
- `read_anchors_from_mujoco()` dans `_step()` → inutile
- `set_live_anchors()` / `_anchor_pos()` dans SwingPlanner → inutile
- `set_initial_struct_pose()` / `_nominal_to_live()` → inutile
- Arguments `p_struct, R_struct, v_struct, omega_struct` dans planners → inutile

## 6. Risques et points d'attention

### 6.1 Repère non-inertiel

Les équations du mouvement en repère structure ne sont exactes que si la
structure est quasi-inertielle (ω_struct ≈ 0). Pour ω_struct significatif,
il faut ajouter les termes de Coriolis et centrifuge :
```
m a_local = Σf_local - 2m ω × v_local - m ω × (ω × r_local) - m α × r_local
```

Pour notre cas (ω_struct < 10° ≈ 0.17 rad, v_robot < 0.1 m/s) :
- Coriolis: 2 × 71 × 0.17 × 0.1 ≈ 2.4 N (vs contact forces ~10 N)
- Centrifuge: 71 × 0.17² × 1 ≈ 2.1 N

Ces termes ne sont PAS négligeables à 14% ratio ! Il faudra peut-être les
compenser.

### 6.2 Ancres scheduler

Vérifier si `sched.anchors_a/b` sont en repère monde initial ou en repère
local structure. Si elles sont en monde initial (structure à [0,0,-1.8]),
il faut les convertir en local.

### 6.3 IK (`dock_configuration`)

L'IK travaille avec les frames Pinocchio. Si Pinocchio est en repère
structure, les targets SE3 doivent aussi être en repère structure.

### 6.4 Gripper distance / docking detection

`_gripper_distance()` utilise `mj_data.site_xpos` (monde). Le docking
detection doit rester en monde (c'est la distance physique réelle).

## 7. Ordre d'implémentation

1. Modifier `mujoco_to_pinocchio` pour transformer en repère structure
2. Adapter les ancres du scheduler en repère local
3. Simplifier TorsoPlanner (supprimer la reconstruction monde)
4. Simplifier SwingPlanner (supprimer la machinerie live)
5. Adapter `_step()` (supprimer Fix 3, Fix R4, live anchors)
6. Vérifier IK targets en repère structure
7. Tester avec 0.1% → 1% → 14%
8. Si nécessaire, ajouter compensation Coriolis/centrifuge
