# Test & Benchmark Findings Report

**92 tests, 16 benchmarks** — all passing.  
**Branch:** `claude/add-benchmarks-testing-OMdaG`  
**Date:** 2026-04-04  
**Environment:** Python 3.11, pinocchio 3.9.0, casadi 3.7.2, mujoco 3.6.0

---

## 1. Simulation Physics Findings

### 1.1 Momentum Spike — Box vs Norm Constraint

**Source:** `test_contact_dynamics.py::test_momentum_spike_box_vs_norm_documentation`

The NMPC and QP enforce per-component box constraints `|L_com,i| <= L_max`. At box corners, the Euclidean norm can reach `sqrt(3) * L_max`:

| Metric | Value |
|--------|-------|
| L_max per component | 5.0 Nm·s |
| ||L|| at box corner | 8.66 Nm·s |
| Overshoot ratio | **1.73x** |
| Documented peak (sim) | 8.82 Nm·s |

**Impact:** During weld release (SS->EXT transition), all three axes can activate simultaneously, pushing the norm to the geometric limit. Observed in 0.8% of NMPC steps with recovery in ~30ms.

**Recommendation:** Consider second-order cone (SOC) constraint `||L||_2 <= L_max`, or tighten per-component limit to `L_max / sqrt(3) = 2.89 Nm·s` (41% NMPC infeasibility reported at this setting).

---

### 1.2 NMPC L_dot Constraint — Spin-Only Limitation

**Source:** `test_nmpc_qp_consistency.py::test_nmpc_Ldot_constraint_documented`

The NMPC path constraint limits **centroidal (spin)** angular momentum rate only:

```
|L_dot_com,i| <= tau_w_max = 5 Nm    (per component)
```

The **orbital** term `d/dt[(r_com - r_mid) x sum(f_j)]` is NOT constrained directly. The actual wheel torque demand is spin + orbital, which can exceed `tau_w_max` significantly.

| Term | Typical magnitude |
|------|-------------------|
| Spin (L_dot_com) | ~5 Nm (constrained) |
| Orbital | 5-10+ Nm at 14% mass ratio (unconstrained) |
| Total wheel demand | Up to 15 Nm |

**Mitigation:** The corrected hw ODE (`hw_dot = -L_dot - orbital`) and the hw box constraint (`-5 <= hw <= +5 Nm·s`) indirectly limit the total demand over the prediction horizon.

---

### 1.3 QP vs MuJoCo Contact Force Gap

**Source:** `test_contact_dynamics.py::test_qp_mujoco_force_gap_documentation`

The QP optimizes virtual wrenches lambda as decision variables, but only sends joint torques `tau_q` to MuJoCo. MuJoCo's constraint solver independently determines the actual contact forces.

| Metric | Value |
|--------|-------|
| QP constraint on L_dot | 5 Nm |
| MuJoCo actual L_dot | 188 Nm |
| **Gap ratio** | **~33x** |

**Root cause:** Architectural design — the QP wrench lambda is a planning variable, not a physical actuator input. The physical contact forces emerge from MuJoCo's rigid constraint solver.

**Mitigation:** The `MomentumDisturbanceEstimator` computes H_{r/O} directly from kinematics (r_com, v_com, L_com), bypassing the QP/MuJoCo gap entirely.

---

### 1.4 NMPC/QP Momentum Dynamics Inconsistency

**Source:** `test_nmpc_qp_consistency.py::test_nmpc_hw_stays_in_envelope`

| Stage | hw dynamics | Includes orbital? |
|-------|------------|-------------------|
| NMPC (Stage 1) | `hw_dot = -L_dot - (r_com - r_mid) x Sigma_f` | Yes |
| QP (Stage 2) | `Delta_hw = -dt * M_lambda * lambda` | No |

**Measured:** < 1 Nm·s prediction gap per NMPC cycle (10 QP steps at 100 Hz).

**Tested:** NMPC hw trajectory stays within safety-margin-adjusted bounds across full horizon.

---

### 1.5 Spin/Orbital Decomposition — Verified

**Source:** `test_momentum.py`

The decomposition `H_{r/O} = L_com + r_com x (m * v_com)` is an exact identity. Verified across 5 random configurations to machine precision (< 1e-12). The orbital term grows proportionally with CoM displacement from the structure origin.

---

## 2. Code-Level Findings

### 2.1 Silent Exception Swallowing

**Source:** `test_liabilities.py::TestL1`

| Location | Behavior | Risk |
|----------|----------|------|
| `sim_loop.py:529-532` | NMPC failure -> zero wrenches, `nmpc_ok=False` | No log output |
| `sim_loop.py:595-597` | QP failure -> zero torques, `qp_ok=False` | No log output |

The solver-level `info.success=False` detection works correctly (tested with infeasible hw and NaN inputs). The problem is that `sim_loop` catches all exceptions silently and continues with fallback values.

**Recommendation:** Add `logging.warning()` with the exception message in each `except Exception` block.

---

### 2.2 Config Validation Gaps

**Source:** `test_liabilities.py::TestL6`

| Config issue | Accepted? | Risk |
|-------------|-----------|------|
| `dt_nmpc=0.1, dt_qp=0.03` (ratio=3.33) | Yes, silently | Incorrect sub-stepping in QP inner loop |
| `hw_min > hw_max` (inverted bounds) | Yes, silently | NMPC infeasibility, unpredictable behavior |
| `dt_qp = 0` | Yes, silently | Division by zero in timing calculations |

**Recommendation:** Add `__post_init__` validation to `SimConfig` dataclass.

---

### 2.3 Warm-Start Resilience — No Issue Found

**Source:** `test_liabilities.py::TestL5`

Tested sequence: feasible solve -> infeasible solve -> feasible solve. The third solve **succeeds**, indicating IPOPT gracefully ignores corrupted dual variables from failed intermediate solves. The unconditional warm-start storage at `nmpc_solver.py:457-460` is not a practical issue.

---

### 2.4 Numerical Edge Cases — All Safe

**Source:** `test_liabilities.py::TestL2, TestL3`

| Edge case | Guard | Status |
|-----------|-------|--------|
| Near-singular IK Jacobian | Regularization `1e-4 * I` | No NaN/Inf |
| Zero torso mass | `max(mass, 1e-6)` | No crash |
| Gimbal lock quaternion | `np.clip(sinp, -1, 1)` | No NaN, pitch = 90 deg |
| Near-zero quaternion (1e-10) | No explicit guard | Produces finite values |
| Wrong solver input dimensions | CasADi catches | Exception raised |

---

## 3. Physics Invariants — All Verified

| Property | Tolerance | Status |
|----------|-----------|--------|
| Mass matrix symmetry `H = H^T` | 1e-12 | Pass (5 random configs) |
| Mass matrix positive definite | all eigenvalues > 0 | Pass |
| CoM velocity `v_com = J_com @ v` | 1e-12 | Pass (4 configs) |
| Linear momentum `p = m * v_com` | 1e-8 | Pass |
| Angular momentum `L_com = h[3:6]` | 1e-12 | Pass (5 configs) |
| Bias `C = 0` at zero velocity | 1e-12 | Pass |
| Skew antisymmetry | exact | Pass |
| Momentum map dimensions | exact | Pass (single + double) |
| Frame round-trip mj->pin->mj | 1e-10 | Pass (0/10/20/30 deg rotation) |
| Coriolis velocity term | 1e-10 | Pass |
| Quaternion wxyz <-> xyzw | exact | Pass |

---

## 4. Benchmark Results

### 4.1 Component Timing

| Component | Median | p95 | Budget | Margin |
|-----------|--------|-----|--------|--------|
| NMPC cold start | 47.4 ms | 48.2 ms | 500 ms | 10.5x |
| NMPC warm start | 0.91 ms | 1.01 ms | 50 ms | 55x |
| NMPC full trajectory | 0.88 ms | 0.98 ms | 100 ms | 113x |
| RobotInterface.update() | 40 us | 59 us | 2 ms | 50x |
| Contact Jacobians | 2.9 us | 3.5 us | 1 ms | 345x |
| State conv mj->pin | 35 us | 52 us | 100 us | 2.9x |
| State conv pin->mj | 13 us | 22 us | 100 us | 7.5x |
| IK dock_configuration | 3.7 ms | 6.4 ms | 5 ms | 1.4x |
| TorsoPlanner.reference_at() | 1.2 us | 1.4 us | 50 us | 42x |
| ContactScheduler.config_at() | 4.4 us | 10.4 us | 10 us | 2.3x |
| Contact sequence (N=20) | 86 us | 113 us | 200 us | 2.3x |
| H_{r/O} estimator update | 41 us | 64 us | 100 us | 2.4x |
| AOCS command | 4.8 us | 5.1 us | 50 us | 10.4x |

### 4.2 Real-Time Budget Analysis

One NMPC cycle (10 Hz) = 100 ms budget:

```
NMPC solve (1x warm):          0.91 ms
QP inner loop (10 iterations):
  robot.update():              0.40 ms  (10 x 40 us)
  H_{r/O} estimator:           0.41 ms  (10 x 41 us)
  AOCS command:                0.05 ms  (10 x 5 us)
  ─────────────────────────────
  Total (without QP solve):    1.77 ms
  Headroom for QP solve:       98.2 ms  (9.8 ms per QP step)
```

**Finding:** The non-QP components consume only 1.8% of the real-time budget. The QP solver (WholeBodyQP) has 9.8 ms per step available, well within the 5-10 ms target.

### 4.3 Bottleneck Candidates

| Component | Margin | Risk |
|-----------|--------|------|
| IK dock_configuration | 1.4x | Closest to budget, p95 exceeds 5ms |
| ContactScheduler.config_at() | 2.3x | Moderate, could degrade with longer plans |
| State conv mj->pin | 2.9x | Moderate, involves quaternion algebra |

---

## 5. Summary of Actionable Items

### Fix (quick wins):
1. **Add logging to sim_loop exception handlers** — 2 lines, high impact for debugging
2. **Add SimConfig.__post_init__ validation** — catch dt ratio, hw bounds, zero dt

### Monitor (design limitations, not bugs):
3. **Box vs norm constraint gap** — 1.73x overshoot at corners, recovers in 30ms
4. **NMPC/QP hw inconsistency** — < 1 Nm·s per cycle, absorbed by hw box constraint
5. **QP/MuJoCo force gap** — 33x, handled by H_{r/O} estimator
6. **L_dot spin-only constraint** — orbital unconstrained, absorbed by hw ODE

### Investigate (potential improvements):
7. **SOC momentum constraint** — would eliminate box-corner excursions but may cause infeasibility
8. **IK timing** — p95 at 6.4ms exceeds 5ms budget, consider iteration limit tuning
