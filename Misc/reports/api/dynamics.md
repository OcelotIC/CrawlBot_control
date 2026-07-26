> **⚠ SUPERSEDED (2026-05-27).** This file is stale — it predates or does not
> track the reworked controller. For what the code actually does,
> `docs/architecture/STACK_OVERVIEW.md` is the code-ground-truth reference and
> supersedes any current-state claim here (e.g. the NMPC is 9-state
> `[r_com,v_com,L_com]`, not 12; module APIs and parameters have changed).

# crawlbot.dynamics

Dynamique contrainte (Pinocchio) pour la simulation indépendante de MuJoCo.

---

## VISPAConstrainedDynamics

**Fichier** : `crawlbot/dynamics/constrained_dynamics.py`

Dynamique avant contrainte pour le robot VISPA avec contacts rigides 6D
aux effecteurs. Utilise le solveur proximal de Pinocchio avec projection
SHAKE (position) + RATTLE (vitesse).

### Construction

```python
vcd = VISPAConstrainedDynamics(urdf_path='models/VISPA_crawling_fixed.urdf')
vcd.set_mode(LocomotionMode.ARM_A_DOCKED, anchors)
result = vcd.forward_dynamics(q, v, tau)
q_next, v_next = vcd.integrate(q, v, result.ddq, dt=0.001)
```

### Modes de locomotion

```python
class LocomotionMode(Enum):
    ARM_A_DOCKED   # bras A accroché, B libre (6 DOF contraints)
    ARM_B_DOCKED   # bras B accroché, A libre (6 DOF contraints)
    BOTH_DOCKED    # les deux accrochés (12 DOF contraints)
```

### Équation résolue

```
M(q) · q̈ + h(q,v) = S^T · τ_q + J_c^T · λ
J_c · q̈ + J̇_c · q̇ = 0
```

- `M` : matrice de masse (CRBA)
- `h` : termes de biais (Coriolis, gravité=0 en µg)
- `S^T = [0_{6×12}; I_{12}]` : sélection des actionneurs (pas de couple sur la base flottante)
- `J_c` : Jacobienne de contact empilée (6×nv par contact actif)
- `λ` : multiplicateurs de Lagrange (forces de contact)

### Méthodes

| Méthode | Description |
|---------|-------------|
| `set_mode(mode, anchors)` | Configure les contraintes holonomiques pour la phase courante |
| `forward_dynamics(q, v, tau)` | Résout l'accélération contrainte → `DynamicsResult(ddq, lambda_c, violation, iters)` |
| `integrate(q, v, ddq, dt)` | Euler semi-implicite + SHAKE/RATTLE → `(q_next, v_next)` |
| `compute_derivatives(q, v, tau)` | Dérivées analytiques `dddq/dq, dddq/dv, dddq/dtau, dlambda/d*` |
| `tool_poses(q)` | Placements SE3 des effecteurs |
| `tool_jacobians(q)` | Jacobiennes 6×nv (LOCAL_WORLD_ALIGNED) |
| `centroidal_momentum(q, v)` | Momentum centroïdal `[p_lin(3); L_com(3)]` |

### SHAKE / RATTLE

**SHAKE** (projection position) : itérations de Newton sur `c(q) = 0`
```
err = log6(oMf_tool^{-1} · oMf_target)
J = getFrameJacobian(frame_id, LOCAL)
dq = J^T · (J · J^T + εI)^{-1} · err
q ← integrate(q, dq)
```

**RATTLE** (projection vitesse) : projection sur ker(J_c)
```
dv = J^T · (J · J^T + εI)^{-1} · (J · v)
v ← v - dv
```

Tolérance : violation de contrainte < 1e-11.

### DynamicsResult (dataclass)

| Champ | Type | Description |
|-------|------|-------------|
| `ddq` | ndarray(18,) | Accélérations articulaires |
| `lambda_c` | ndarray(nc,) | Multiplicateurs de Lagrange (forces de contact) |
| `constraint_violation` | float | `‖log6(err)‖` max |
| `prox_iters` | int | Itérations du solveur proximal |

### AnchorConfig (dataclass)

```python
anchors = AnchorConfig(
    anchor_a=pin.SE3(np.eye(3), np.array([-0.4, 0.3, 0.0])),
    anchor_b=pin.SE3(np.eye(3), np.array([-0.4, -0.3, 0.0])))
```

---

## Usage typique

Ce module est utilisé principalement pour :
1. **Tests unitaires** : vérifier la dynamique indépendamment de MuJoCo
2. **Validation croisée** : comparer `lambda_c` (Pinocchio) avec `qfrc_constraint` (MuJoCo)
3. **Dérivées analytiques** : pour des algorithmes de contrôle nécessitant les sensibilités

Dans la boucle de simulation principale, c'est **MuJoCo** qui fait
l'intégration dynamique (via `mj_step`), pas Pinocchio.
