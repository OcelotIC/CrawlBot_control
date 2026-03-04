# Test Plan — Lutze et al. (2023) Python Implementation

This document proposes tests to validate the Python translation against the
original MATLAB implementation and the paper's theoretical results.

---

## 1. Unit Tests — Mathematical Utilities

### 1.1 Quaternion Operations
- **quat_product**: Verify q ⊗ q_conj = [0,0,0,1] (identity).
- **quat_integrate**: Integrate constant omega for known time, compare to
  analytical rotation.
- **quat_error**: Verify zero error for identical quaternions; known error for
  90° rotation.
- **matrix_quat_prod_left/right**: Verify `[q1]_L @ q2 == quat_product(q1, q2)`.

### 1.2 Attitude Transformations
- **quat_dcm / dcm_quat**: Round-trip identity: `dcm_quat(quat_dcm(q)) == q`.
- **euler_dcm**: Known axis-angle pairs (e.g., 90° about z-axis).
- **angles321_dcm / dcm_angles321**: Round-trip identity for various angle sets.
- **angles123_dcm**: Cross-check against `scipy.spatial.transform.Rotation`.

### 1.3 Spatial Math
- **skew_sym**: Verify `skew_sym(a) @ b == np.cross(a, b)` for random vectors.
- **adjoint_se3**: Verify `Ad_g @ twist` produces correct transformed twist for
  known transforms. Check identity transform returns identity adjoint.

---

## 2. Unit Tests — SPART Kinematics & Dynamics

### 2.1 URDF Parser
- **urdf2robot**: Load `MAR_DualArm_6DoF.urdf`, verify:
  - `n_links_joints == 12` (or expected count from URDF).
  - `n_q == 12` (6 per arm).
  - Base link mass and inertia match URDF values.
  - Joint types all == 1 (revolute).
  - Joint axes match URDF.
  - Connectivity map: `branch`, `child`, `child_base` have correct shape and
    binary entries.

### 2.2 Forward Kinematics
- **Identity configuration** (qm=0): Verify link positions match URDF geometry.
- **Single-joint rotation**: Rotate one joint by 90°, verify the child link
  position and orientation are geometrically correct.
- **Consistency**: `rL[:, i] - rJ[:, parent_joint]` should equal `g[:, i]`.

### 2.3 Differential Kinematics
- **Bij symmetry**: If link i is NOT on the same branch as link j,
  `Bij[i][j]` should be zero.
- **Bi0 structure**: Top-left 3×3 should be identity; bottom-left should be
  skew-symmetric of `(r0 - rL[:, i])`.
- **P0 structure**: `P0 = [R0, 0; 0, I]`.

### 2.4 Velocities
- **Zero velocity**: With `u0=0, um=0`, all twists should be zero.
- **Base-only motion**: Set `u0` to known values with `um=0`, verify `t0 = P0 @ u0`
  and link twists are consistent with rigid-body motion.
- **Numerical Jacobian check**: Compare `tL` from Velocities against finite
  differences of Kinematics.

### 2.5 Inertia Projection (I_I)
- **Symmetry**: `I0` and each `Im[i]` should be symmetric.
- **Positive definiteness**: All inertia matrices should be positive definite.
- **Identity rotation**: If `R0 = I`, then `I0 == robot.base_link.inertia`.

### 2.6 Mass Composite Body (MCB)
- **Symmetry**: `M0_tilde` and `Mm_tilde[i]` should all be symmetric.
- **Positive definiteness**: All MCB matrices should be positive semi-definite.
- **Leaf links**: For leaf links, `Mm_tilde[i]` should equal `[Im[i], 0; 0, m*I]`.

### 2.7 Generalized Inertia Matrix (GIM)
- **Full GIM symmetry**: `H = [H0, H0m; H0m', Hm]` must be symmetric.
- **Positive definiteness**: H must be positive definite.
- **Dimensions**: `H0` is 6×6, `Hm` is n_q × n_q, `H0m` is 6 × n_q.

### 2.8 Convective Inertia Matrix (CIM)
- **Zero velocity**: When `u0=0, um=0`, C should be all zeros (no Coriolis
  terms without motion).
- **Dimensions**: `C0` is 6×6, `Cm` is n_q × n_q.

### 2.9 Geometric Jacobian
- **Numerical verification**: Compare `J0 @ u0 + Jm @ um` against twist
  computed by Velocities for the same point.
- **Zero columns**: Joints not on the kinematic chain to link i should produce
  zero columns in `Jm`.

### 2.10 Forward / Inverse Dynamics
- **Consistency**: `ID(FD(tau)) == tau` (round-trip). Apply known forces,
  compute accelerations, then apply inverse dynamics — should recover original
  forces.
- **Free-floating**: With zero external forces and torques, verify conservation
  of momentum (no net force => constant momentum).
- **Gravity-free**: In zero-g (no external wrenches), `tau` from ID should
  equal generalized Coriolis forces only.

---

## 3. Integration Tests — Controller & Simulation

### 3.1 Feedforward Wrench
- **Zero error**: When robot is at desired position/velocity, `F_d_r` should be
  near zero (only feedforward term, which is zero in this implementation).
- **Saturation**: Large position errors should be clamped to `e_pos_max = 10 m`.
- **Base wrench**: Identity quaternion (no rotation) => `F_d_b` rotation
  component should be near zero.

### 3.2 QP Controller
- **Non-optimized mode**: Verify `Fc = pinv(Ad') @ F_d_r` and saturation is applied.
- **Optimized mode**: Verify QP solution respects wrench bounds `F_min <= Fc <= F_max`.
- **Fallback**: When QP returns infeasible, verify fallback to pseudoinverse
  produces a valid (finite, bounded) wrench.
- **Build transform**: Verify `g_cb` is a valid SE(3) matrix (det(R) = 1,
  proper homogeneous form).

### 3.3 Dynamics Integration
- **Conservation**: With zero external forces and zero controller output, total
  angular momentum should remain constant.
- **Baumgarte stabilization**: Verify the stabilization term has correct structure.
- **Saturation**: Accelerations exceeding `accel_max` should be clipped.
- **Quaternion normalization**: After integration, `||quat|| == 1` to machine
  precision.

### 3.4 Full Simulation Loop
- **Short run**: Run 10 timesteps for each experiment, verify no NaN/Inf in any
  result field.
- **State structure**: Verify all result arrays have correct dimensions.
- **Reproducibility**: Two runs with same parameters produce identical results.

---

## 4. Paper Validation Tests (Cross-Reference with Lutze et al. 2023)

These tests validate that the implementation reproduces the paper's theoretical
claims and figures.

### 4.1 Experiment #1 — Straight Line Through CoM (Section 5.1)
- **Expected behavior**: Straight line through satellite CoM should produce
  minimal angular momentum disturbance due to symmetry (the robot's CoM
  crosses the satellite's CoM).
- **Key metric**: Both optimized and non-optimized should show similar `beta`
  (pitch) due to trajectory symmetry — the paper notes "limited improvement"
  for this case.
- **Verification**: `beta_max` improvement < 50% (as noted in the paper).

### 4.2 Experiment #2 — Offset Straight Line (Section 5.2)
- **Expected behavior**: Offset trajectory should cause larger angular momentum
  disturbance than Exp #1.
- **Key metric**: Optimized controller should significantly reduce `beta_max`
  compared to non-optimized (>50% improvement expected).
- **Contact forces**: Optimized should show different force profile (using
  contact to counteract disturbance).

### 4.3 Experiment #3 — Circular Arc (Section 5.3)
- **Expected behavior**: Arc motion on satellite edge creates continuous
  disturbance torque.
- **Key metric**: Significant reduction in satellite rotation angles with
  optimization.
- **RW saturation**: Verify reaction wheels are used within their capacity.

### 4.4 QP Formulation Validation (Equations 18-22)
- **Cost function decomposition**: Verify the QP Hessian H is the sum of three
  weighted quadratic terms:
  - `||Ad' Fc - F_d_r||²_Qr` (robot tracking)
  - `||Ad' Fc + F_d_b||²_Qb` (base stabilization)
  - `||Fc||²_Qc` (wrench regularization)
- **Wrench bounds**: All contact forces must stay within SI limits at every
  timestep.

### 4.5 Angular Momentum Conservation (eq. 1-5)
- **Total momentum**: `h_total = h_satellite + h_robot + h_RW` should be
  approximately constant throughout the simulation (no external torques in
  orbit).
- **Drift tolerance**: Allow up to 1% drift over 70s due to numerical
  integration errors.

### 4.6 Trajectory Tracking Quality
- **RMS tracking error**: Should be on the order of mm (< 100 mm for all
  experiments).
- **Optimized vs non-optimized**: Optimized controller may show slightly larger
  tracking error as a trade-off for reduced satellite disturbance.

---

## 5. MATLAB Cross-Validation Tests

These require running both MATLAB and Python on the same inputs and comparing
outputs.

### 5.1 Kinematics Cross-Check
- Load same URDF in both implementations.
- Use identical `R0, r0, qm`.
- Compare `RJ, RL, rJ, rL, e, g` — should match to machine precision (~1e-12).

### 5.2 Dynamics Cross-Check
- From identical initial states, compute one step of:
  - `compute_system_state` outputs (H0, Hm, C0, Cm, etc.)
  - `LutzeQPController.solve` outputs (Fc, tau)
  - `integrate_dynamics` outputs (next state)
- All should match to at least 1e-6 relative tolerance.

### 5.3 Full Simulation Cross-Check
- Run Experiment #2 for both implementations.
- Compare final satellite Euler angles — should match to within 1e-3 degrees.
- Compare tracking error RMS — should match to within 1%.

---

## 6. Performance & Robustness Tests

### 6.1 Numerical Stability
- **Ill-conditioned H**: Artificially create a near-singular GIM and verify
  regularization kicks in (Tikhonov regularization).
- **Large time step**: Run with `dt = 0.1` and verify simulation doesn't blow up
  (may not be accurate but should remain bounded).
- **Extended simulation**: Run for 200s (3× normal) and verify no drift or
  explosion.

### 6.2 Edge Cases
- **Zero joint velocities**: All `um = 0` throughout simulation.
- **Joint limits**: Joint positions approaching ±pi should not cause singularities.
- **Zero mass links**: Verify handling of links with very small mass (< 0.001 kg).

### 6.3 Performance Benchmarks
- **Single timestep**: Measure time for one `compute_system_state` + `solve` +
  `integrate_dynamics` cycle.
- **Full simulation**: Measure total wall time for each experiment.
- **Comparison**: Document Python vs MATLAB performance ratio.

---

## Test Priority

| Priority | Test Category | Rationale |
|----------|--------------|-----------|
| **P0** | 2.1 URDF Parser | Foundation — everything depends on correct model |
| **P0** | 2.2 Forward Kinematics | Core algorithm correctness |
| **P0** | 2.7 GIM | Dynamics accuracy depends on correct inertia |
| **P0** | 3.4 Full Simulation (no crash) | Basic sanity |
| **P1** | 1.x All unit math tests | Numerical correctness |
| **P1** | 2.x All SPART tests | Algorithm correctness |
| **P1** | 4.5 Momentum Conservation | Physical validity |
| **P1** | 3.2 QP Controller | Controller correctness |
| **P2** | 4.1-4.4 Paper validation | Research reproduction |
| **P2** | 5.x MATLAB cross-validation | Translation fidelity |
| **P3** | 6.x Performance & robustness | Production quality |
