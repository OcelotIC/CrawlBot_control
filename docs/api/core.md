# crawlbot.core

Interfaces fondamentales : modèle robot, conversions d'état, cinématique inverse.

---

## RobotInterface

**Fichier** : `crawlbot/core/robot_interface.py`

Wrapper Pinocchio calculant toutes les grandeurs nécessaires aux deux étages
du contrôleur. Un seul appel `update(q, v)` produit un `RobotState` complet.

### Construction

```python
robot = RobotInterface(urdf_path, tau_max=10.0, gravity='zero')
```

| Paramètre | Type | Description |
|-----------|------|-------------|
| `urdf_path` | str | URDF du robot (base libre free-flyer) |
| `tau_max` | float ou ndarray(12) | Limites de couple articulaire [Nm] |
| `gravity` | str | `'zero'` (orbite) ou `'earth'` (sol) |
| `torso_mass` | float, opt. | Override masse torso si URDF incorrecte |

### Appel principal : `update(q, v) -> RobotState`

Prend en entrée la configuration Pinocchio `(q, v)` en **repère structure**
et calcule :

```
pin.computeAllTerms(model, data, q, v)   ← H, C, J_com, CoM
pin.updateFramePlacements(model, data)    ← positions des frames
pin.computeJointJacobiansTimeVariation()  ← J̇·q̇
pin.centerOfMass(model, data, q, v, 0)    ← J̇_com·q̇
```

### RobotState (dataclass retournée)

| Champ | Taille | Description |
|-------|--------|-------------|
| `q` | (19,) | Configuration Pinocchio [pos(3), quat_xyzw(4), joints(12)] |
| `v` | (18,) | Vitesse généralisée [twist(6), q̇(12)] |
| `H` | (18,18) | Matrice de masse M(q) (CRBA, symétrisée) |
| `C` | (18,) | Termes de biais h(q,v) = Coriolis + gravité (RNEA) |
| `r_com` | (3,) | Position du CoM robot |
| `v_com` | (3,) | Vitesse du CoM : `J_com @ v` |
| `J_com` | (3,18) | Jacobienne du CoM |
| `Jdot_dq_com` | (3,) | `J̇_com · q̇` (pour accélération désirée) |
| `L_com` | (3,) | Moment centroïdal (partie angulaire de `data.hg`) |
| `J_tool_a`, `J_tool_b` | (6,18) | Jacobiennes des effecteurs (LOCAL_WORLD_ALIGNED) |
| `Jdot_dq_tool_a/b` | (6,) | `J̇_tool · q̇` |
| `oMf_tool_a/b` | SE3 | Placement des effecteurs dans le repère courant |
| `J_torso` | (6,18) | Jacobienne du torso |
| `total_mass` | float | Masse totale robot (calculée une fois) |

### Convention de repère

Toutes les sorties sont dans le **repère structure** (celui dans lequel
`q` et `v` sont exprimés). La conversion depuis/vers MuJoCo se fait
uniquement dans `state_conversions.py`.

---

## state_conversions

**Fichier** : `crawlbot/core/state_conversions.py`

Trois fonctions pures, sans état.

### `mujoco_to_pinocchio(mj_qpos, mj_qvel) -> (pin_q, pin_v)`

Transforme l'état MuJoCo (repère monde) en Pinocchio (repère structure).

**Layout MuJoCo RWA-3** (nq=29, nv=27) :
```
qpos: [struct_pos(3) struct_quat(4) rw_angles(3) torso_pos(3) torso_quat(4) joints(12)]
qvel: [struct_v(3) struct_omega(3) rw_vel(3) torso_v(3) torso_omega(3) joint_vel(12)]
```

**Algorithme** :
```
R_s = Quaternion(quat_struct).toRotationMatrix()

p_local = R_s^T · (p_torso - p_struct)              # position relative
R_local = R_s^T · R_torso                            # orientation relative
v_local = R_s^T · (v_torso - v_struct - ω_struct × Δp)  # vitesse relative
ω_local = R_s^T · (ω_torso - ω_struct)              # vitesse angulaire relative
```

Note : `qvel[0:6]` (structure free joint) est en **repère monde** (vérifié
empiriquement — cf. `tests/test_force_estimator.py` T2).

### `pinocchio_to_mujoco(pin_q, pin_v, struct_pos, struct_quat, rwa) -> (mj_qpos, mj_qvel)`

Transformation inverse. Utilisée uniquement pour l'initialisation (IK → MuJoCo).

### `quat_wxyz_to_euler_deg(qw, qx, qy, qz) -> ndarray(3,)`

Quaternion (w,x,y,z) → angles d'Euler (roll, pitch, yaw) en degrés.
Convention ZYX intrinsèque.

---

## IK

**Fichier** : `crawlbot/core/ik.py`

### `dock_configuration(model, anchor_a, anchor_b) -> ndarray(19,)`

Calcule une configuration valide avec les deux outils aux ancres spécifiées.

1. Part de la configuration neutre
2. Place le torso au milieu des deux ancres
3. Appelle `solve_ik` avec les deux cibles SE3
4. Lève `RuntimeError` si erreur > 1e-4

### `solve_ik(model, q0, targets, max_iter=500, tol=1e-8) -> (q, err)`

IK itérative par pseudo-inverse Jacobienne amortie :

```
Pour chaque itération :
    err_i = log6(oMf_current^{-1} · oMf_target)        # erreur SE3
    J_arm = getFrameJacobian(frame_id)[:, arm_dofs]      # Jacobienne 6×6 du bras
    dq_arm = (J^T J + λI)^{-1} J^T err                  # pseudo-inverse amortie
    dq_base = 0.3 · (J_base^T J_base + λI)^{-1} J_base^T err  # base conservatrice
    α = min(1, 0.5 / ||dq||)                             # pas adaptatif
    q ← integrate(q, α · dq)
```
