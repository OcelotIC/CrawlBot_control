# Generalized Momentum Observer for Contact Estimation

**Module**: `crawlbot/estimation/contact_estimator.py`

**References**:
- A. De Luca, R. Mattone, "Sensorless Robot Collision Detection and Hybrid Force/Motion Control", ICRA 2005
- A. De Luca et al., "Collision Detection and Safe Reaction with the DLR-III Lightweight Manipulator Arm", IROS 2006
- S. Haddadin, A. De Luca, A. Albu-Schaffer, "Robot Collisions: A Survey on Detection, Isolation, and Identification", IEEE T-RO, 2017

---

## 1. Motivation

The previous dock detection used a **10 Hz kinematic distance check** (`d < 5mm`) in the NMPC loop. This has three limitations for hardware deployment:

1. **100ms detection latency** -- at approach velocities of 4-6 mm/s, the arm moves 0.4-0.6mm between checks
2. **No force information** -- no way to confirm physical contact vs. FK coincidence
3. **FK noise vulnerability** -- joint encoder noise corrupts the computed gripper position

The Generalized Momentum Observer (GMO) provides **force-based contact detection at 100 Hz** without requiring a force/torque sensor, using only joint position, velocity, and commanded torque data already available in the control loop.

---

## 2. Equations of Motion

The VISPA crawler is a free-flying dual-arm robot with nv=18 velocity DOFs (6 base twist + 12 joint velocities). In orbital microgravity (`model.gravity = pin.Motion.Zero()`):

```
M(q) v_dot + C(q,v) v = S^T tau + J_c^T f_ext
```

| Symbol | Dimension | Description | Source |
|--------|-----------|-------------|--------|
| M(q) | (18,18) | Mass matrix | `pin.crba` -> `rs.H` |
| C(q,v) | (18,18) | Coriolis matrix | `pin.computeCoriolisMatrix` -> `rs.C_matrix` |
| v | (18,) | Generalized velocity | Pinocchio state |
| S | (12,18) | Actuation selection: `[0_{12x6} \| I_{12}]` | Base unactuated |
| tau | (12,) | Joint torques from QP | `wholebody_qp.solve()` |
| J_c | (6nc, 18) | Contact Jacobian | `robot.get_contact_jacobians()` |
| f_ext | (6nc,) | External contact wrench | Unknown |
| g(q) | (18,) | Gravity | **Zero** (orbital) |

Note: Pinocchio's `nle` = `C(q,v) v + g(q)` = `C(q,v) v` (since g=0). This is stored as `rs.C` (a vector). The full Coriolis *matrix* `C(q,v)` is computed separately via `pin.computeCoriolisMatrix` and stored as `rs.C_matrix`.

---

## 3. Generalized Momentum

Define the generalized momentum:

```
p = M(q) v     (18,)
```

Its time derivative:

```
p_dot = M v_dot + M_dot v
```

Substituting the EoM (`M v_dot = S^T tau + J_c^T f_ext - C v`):

```
p_dot = S^T tau + J_c^T f_ext - C v + M_dot v
      = S^T tau + J_c^T f_ext + (M_dot - C) v
```

### Key Identity (Christoffel-Symbol Coriolis)

Pinocchio computes the Coriolis matrix using Christoffel symbols of the first kind. A fundamental property of this factorization is:

```
M_dot = C + C^T
```

which means `M_dot - 2C` is skew-symmetric (the passivity property). Therefore:

```
(M_dot - C) v = C^T v
```

The momentum derivative becomes:

```
p_dot = S^T tau + C(q,v)^T v + J_c^T f_ext
        \_________ known _______/  \_ unknown _/
```

The first two terms are fully computable from available data. The external force `tau_ext = J_c^T f_ext` is the only unknown.

---

## 4. Observer Design

### Continuous-Time (De Luca 2006)

Define an auxiliary integrator `beta(t)` and residual `r(t)`:

```
beta_dot = S^T tau + C(q,v)^T v + r     (integrator with feedback)
p = M(q) v                              (measured generalized momentum)
r = K_O (p - beta)                      (residual signal, K_O > 0)
```

Initial conditions: `beta(0) = p(0) = M(q_0) v_0`, giving `r(0) = 0`.

### Convergence Proof

Taking the time derivative of `r`:

```
r_dot = K_O (p_dot - beta_dot)
      = K_O [(S^T tau + C^T v + tau_ext) - (S^T tau + C^T v + r)]
      = K_O (tau_ext - r)
```

This is a stable first-order linear ODE:

```
r_dot = -K_O r + K_O tau_ext
```

with eigenvalue `-K_O < 0`. For constant `tau_ext`:

```
r(t) = tau_ext (1 - e^{-K_O t})
```

**The residual converges exponentially to the generalized external force** with time constant `1/K_O`.

### Discrete-Time Implementation

Forward Euler at the QP rate (100 Hz, `dt = 0.01 s`):

```
beta_k = beta_{k-1} + dt * (S^T tau_{k-1} + C_k^T v_k + r_{k-1})
p_k = M(q_k) v_k
r_k = K_O * (p_k - beta_k)
```

Initialization: `beta_0 = M(q_0) v_0`, `r_0 = 0`.

---

## 5. Gain Selection

| K_O | Bandwidth | Time constant | Euler stable? | Notes |
|-----|-----------|---------------|---------------|-------|
| 50 | 8.0 Hz | 20 ms | Yes | Conservative, low noise |
| 80 | 12.7 Hz | 12.5 ms | Yes | **Default**. 1-2 QP steps |
| 100 | 15.9 Hz | 10 ms | Marginal | K_O * dt = 1.0 (limit) |

**Euler stability criterion**: `K_O * dt < 1`. At dt=0.01: K_O < 100.

**Trade-off**: Higher K_O gives faster detection but amplifies measurement noise. K_O = 80 provides ~12.5ms convergence (1-2 QP steps) with adequate noise margin.

---

## 6. Contact Force Extraction

The residual `r` (18,) converges to `tau_ext = J_c^T f_ext` -- the generalized external force projected into joint space. For swing arm contact detection:

```
r_swing = r[arm_v_slice]         (6,) -- residual on swing arm joints
||r_swing|| > F_threshold        => contact force detected
```

For full wrench estimation (optional, e.g., for impedance control):

```
f_ext_est = pinv(J_swing^T) @ r = (J_swing J_swing^T)^{-1} J_swing @ r
```

---

## 7. Contact State Machine

The state machine fuses the GMO residual with FK-based kinematic proximity:

```
NO_CONTACT --[d < 20mm]--> PROXIMITY --[contact_cond AND d < 5mm]--> CONTACT
     ^                                                                   |
     |                          [sustained 3 cycles (30ms)]              v
     +----------[d > 30mm]----------------------------------------- CONFIRMED
```

### Transition conditions

| Transition | Condition | Rationale |
|------------|-----------|-----------|
| NO_CONTACT -> PROXIMITY | d_FK < 20mm | Kinematic early warning; triggers approach velocity control |
| PROXIMITY -> CONTACT | contact_cond AND d_FK < 5mm | Physical contact range (matches weld_radius / latch engagement) |
| CONTACT -> CONFIRMED | Sustained 3 cycles (30ms) | Debounce transient FK noise |
| Any -> NO_CONTACT | d_FK > 30mm | Hysteresis prevents chattering at zone boundaries |

### d_contact = 5mm rationale

The CONTACT threshold must correspond to a distance where **physical contact is mechanically possible**. At 10mm the gripper is still in free space — declaring "contact" there conflates proximity with contact and wastes the GMO's force-detection capability. The 5mm threshold matches `weld_radius` (the geometric distance at which the latch mechanism can engage). On hardware with a docking cone, this would be set to the cone's capture radius (typically 1-3mm).

### Contact condition (mode-dependent)

- **Hardware** (`force_mode=True`): `||r_swing|| > F_threshold` AND `d_FK < d_contact` -- force residual confirms physical contact within latch range
- **Simulation** (`force_mode=False`): `d_FK < d_contact` -- kinematic only, since MuJoCo creates no contact force before weld activation. The threshold matches `weld_radius` so the GMO path and legacy path trigger at the same geometric distance, with the GMO adding 100 Hz rate + debounce

---

## 8. Relationship to Existing H_{r/O} Estimator

| Property | MomentumDisturbanceEstimator | GMO |
|----------|------------------------------|-----|
| State dimension | 3 (angular only) | 18 (full generalized) |
| Quantity observed | H_{r/O} = L_com + r x mv | p = M(q) v |
| Method | Finite-difference + EMA filter | First-order observer (integral form) |
| Output | dH_{r/O}/dt (AOCS feedforward) | J_c^T f_ext (contact force) |
| Purpose | Structure attitude control | Dock contact detection |
| Rate | 100 Hz | 100 Hz |
| Module | `crawlbot/aocs/force_estimator.py` | `crawlbot/estimation/contact_estimator.py` |

The two estimators are **complementary**. The GMO does NOT replace the existing `MomentumDisturbanceEstimator`.

---

## 9. Pinocchio Implementation Notes

### Computing C^T v

The Coriolis matrix is obtained via:

```python
pin.computeCoriolisMatrix(model, data, q, v)
C_matrix = data.C.copy()    # (nv, nv)
CT_v = C_matrix.T @ v       # (nv,)
```

This is separate from `computeAllTerms` (which computes M, nle, CoM, etc. but not the Coriolis matrix). The cost is O(n^2) for n=18 DOFs -- negligible compared to the QP solve (~1ms).

### Naming convention

- `rs.C` = `nle` vector (18,) = `C_matrix @ v` (bias term from RNEA)
- `rs.C_matrix` = full Coriolis matrix (18,18) from `computeCoriolisMatrix`
- `rs.H` = mass matrix M(q) (18,18) from CRBA

### Verification

```python
# These should be equal (since g=0):
assert np.allclose(rs.C_matrix @ v, rs.C)  # nle = Cv + g = Cv
```

---

## 10. Integration Points

### sim_loop.py

1. **Setup**: Instantiate `GeneralizedMomentumObserver` and `ContactStateMachine` after `MomentumDisturbanceEstimator`
2. **QP inner loop** (100 Hz): After `mj_step`, call `gmo.update(rs.H, rs.v, rs.C_matrix, tau_applied)`
3. **EXT phase**: Call `contact_sm.update(r_swing_norm, d_FK)` per QP step
4. **Phase transitions**: Call `gmo.reset(M, v)` and `contact_sm.reset()` when weld is deactivated (SS start)
5. **Dock detection**: When `use_gmo_dock=True`, `contact_sm.is_docked` triggers weld activation

### Config parameters (SimConfig)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gmo_K_O` | 80.0 | Observer gain [1/s] |
| `gmo_F_threshold` | 5.0 | Force residual threshold [N] |
| `gmo_d_proximity` | 0.020 | PROXIMITY threshold [m] |
| `gmo_d_contact` | 0.005 | CONTACT threshold [m] (= weld_radius) |
| `gmo_d_reset` | 0.030 | Hysteresis reset [m] |
| `gmo_debounce_count` | 3 | CONFIRMED cycles [@ 100Hz = 30ms] |
| `use_gmo_dock` | False | Enable GMO dock detection |

### Logged signals

| Signal | Description |
|--------|-------------|
| `gmo_residual_norm` | ||r|| (full 18D residual norm) |
| `gmo_swing_residual` | ||r[swing_joints]|| (swing arm projection) |
| `gmo_contact_state` | ContactState enum value (0-3) |
