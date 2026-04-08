# Momentum-Aware Multi-Step Crawling

## Overview

This document describes three controller modifications that enable reliable multi-step locomotion for the VISPA crawling robot on a free-floating structure. The modifications address the coupled robot-structure dynamics that cause docking failures and structural rotation accumulation during multi-step gaits.

---

## 1. Modifications

### 1.1 Structure Disturbance Constraint (Ḣ_s)

**Files:** `centroidal_nmpc.py`, `config.py`, `sim_loop.py`

**Problem:** The existing NMPC bounds the robot's centroidal angular momentum rate `L̇_com` to protect the robot's reaction wheels, but does not bound the torque delivered to the structure. These quantities differ by the lever arm term:

```
Ḣ_s = L̇_com + r_com × (m · a_com)
```

More precisely, the structure disturbance torque about its CoM is:

```
Ḣ_s = Σⱼ [r_Cⱼ × fⱼ + τⱼ]
```

where `r_Cⱼ` are contact positions relative to the structure CoM (origin in R_s). The existing `L̇_com` constraint uses `(r_Cⱼ - r_com)` as lever arms; the new `Ḣ_s` constraint uses `r_Cⱼ` directly. With anchors at z = -1.775m from the structure CoM (the crawling panel), even small contact forces create large structural torques.

**Implementation:** Six bilateral path constraints `|Ḣ_s,i| ≤ tau_struct_max` added to the NMPC. Conditionally included only when `tau_struct_max` is finite, avoiding CasADi numerical issues with `inf`.

**Configuration:**
- `SimConfig.tau_struct_max: float = np.inf` (disabled by default)
- `CentroidalNMPCConfig.tau_struct_max: float = np.inf`

**Effect:** When active, the NMPC planner generates contact wrenches that respect the structure's AOCS torque authority. The robot naturally slows down at anchors with large lever arms (panel extremes at ||r_C|| = 2.68m) to keep `||Ḣ_s||` bounded.

### 1.2 Inter-Step Settling

**Files:** `sim_loop.py`, `config.py`

**Problem:** After docking, the gait immediately starts the next step. Residual joint velocities, wheel momentum, and structural angular rates from the previous step carry over, degrading the next step's approach accuracy.

**Implementation:** A configurable settle phase inserted after each dock event (before `step_idx` increments). Uses the existing `settle_mode=True` path in the whole-body QP: skips torso/CoM tasks, applies pure velocity damping (Kd=10, alpha=1e3).

A cumulative `t_offset` tracks settle durations and shifts all subsequent gait plan timestamps, preventing phase truncation.

**Configuration:**
- `SimConfig.t_settle_inter: float = 0.0` (disabled by default)

**Effect at 1% mass ratio (3-step):**

| `t_settle_inter` | Docks | Struct rotation |
|-------------------|-------|-----------------|
| 0s | 2/3 | 26.9deg |
| 2s | 3/3 | 4.9deg |

At 0.1%, settling is not needed (3/3 without).

### 1.3 Close-Approach QP and EXT Phase Fixes

**Files:** `sim_loop.py`

**Problem:** The 6D end-effector tracking upgrade introduced two docking regressions:

1. **Angular Jacobian competition:** During close approach (d < 20mm), the orientation component of the 6D EE task consumed DOFs needed for position convergence. At the approach minimum, the tool had ~46deg orientation error; the QP traded position accuracy for orientation correction.

2. **CoM task vs EE task:** The CoM task (priority 1, weight 1e2) conflicts with the EE approach (priority 2, weight 1e4). At 1% mass ratio, each push toward the anchor disturbs the CoM via the stance contact reaction force. The CoM task corrects this by pulling the EE back, creating a sustained oscillation at the coupled robot-structure natural frequency (~0.7Hz, period ~1.4s). The gripper oscillates between 5-8mm, never crossing the 5mm dock threshold.

**Implementation (three sub-fixes):**

**(a) Angular Jacobian zeroing:** When d < 20mm, the angular rows (3:6) of `J_ee` and `Jdot_dq_ee` are zeroed. The 6D task interface is preserved but all DOFs serve position convergence.

**(b) Close-approach QP (`qp_approach`):** A third QP variant with EE-dominant weights: `alpha_ee * 10`, `alpha_com * 0.1`, `alpha_torso * 0.1`. Activated when d < 20mm via a latched `EXT_CLOSE` phase flag. The latch prevents mode-switching oscillation between QP variants when the gripper bounces above/below the threshold.

**(c) Approach velocity:** Proportional gain 0.5 with 2mm/s floor: `v_mag = max(0.5 * d, 0.002)`. Applied as a 6D twist reference `[v_approach; 0₃]`.

**Effect:**

| Mass ratio | Before | After |
|-----------|--------|-------|
| 0.1% 1-step | 0/1 (6.6mm) | 1/1 (2.4mm) |
| 0.1% 3-step | 0/3 | 3/3 (rot=2.6deg) |
| 1% 1-step | 0/1 (5.1mm) | 1/1 (4.8mm) |
| 1% 3-step + settle=2s | 0/3 | 3/3 (rot=4.9deg) |

---

## 2. Limits

### 2.1 Mass Ratio Ceiling

The controller achieves 3/3 docks at 0.1% and 1% mass ratios but fails at 8% (structure 901 kg, robot 74 kg). At 8%, the structure tumbles to 180deg and NMPC failures reach ~45%. The reaction forces from crawling are too large relative to the structure's inertia. The `tau_struct_max` constraint (Section 1.1) is designed to address this but has not yet been tested with finite values.

### 2.2 Dock Threshold Sensitivity

Docking relies on a kinematic distance threshold (`weld_radius = 5mm`). The close-approach QP gets the gripper to 2-5mm depending on mass ratio. At 1%, the margin is thin (4.8mm vs 5mm threshold). The old GMO (Generalized Momentum Observer) contact detection triggered at 6-7mm regardless of mass ratio, but it no longer triggers reliably with the 6D EE controller because the higher joint torques from orientation tracking mask the contact momentum residual.

### 2.3 Settle Duration Sensitivity at 0.1%

At 0.1% mass ratio, intermediate settle durations (1-3s) degrade performance (1/3 docks) compared to no settle (3/3) or long settle (5s, 3/3). The settle phase disrupts the controller state (NMPC warm start, torso planner reference) in a way that the next step starts from a suboptimal initial condition. At 1%, settling is always beneficial because the velocity damping outweighs the state disruption.

### 2.4 6D Orientation at Dock

The angular Jacobian is zeroed during close approach (d < 20mm), meaning orientation tracking is effectively disabled in the final phase. The tool arrives at the anchor with ~46deg orientation error. For the simulated weld constraint this is acceptable, but a real HOTDOCK mechanism may require tighter orientation alignment.

### 2.5 Fixed QP Weights

The close-approach QP uses static weight ratios (10x EE, 0.1x CoM). These were tuned empirically for the 0.1%-1% range. At higher mass ratios, the optimal ratio may differ. There is no online adaptation of task weights based on the current dynamics.

### 2.6 No Structure AOCS Model

The `tau_struct_max` constraint bounds the disturbance torque but does not model the structure's AOCS response. The constraint assumes the AOCS can absorb any torque below the bound, but real reaction wheels have momentum saturation, rate limits, and desaturation constraints that are not captured.

---

## 3. Roadmap

### Phase 1: Ḣ_s Constraint Validation (immediate)

**Goal:** Demonstrate bounded structural rotation at 1% and 8% mass ratios using `tau_struct_max`.

- Parametric sweep: `tau_struct_max in {1, 2, 5, 10} Nm` at 1% and 8%
- For each value, record: dock success, step times, max structural rotation, max `||Ḣ_s||`, NMPC feasibility
- Identify the tightest constraint that still achieves 3/3 docks
- Combined test: best `tau_struct_max` + `t_settle_inter`
- Generate figures: structural rotation vs step number, Ḣ_s time series, Pareto (tau_struct_max vs rotation vs crawl time)

### Phase 2: GMO Dock Recovery

**Goal:** Restore sensorless contact detection for robust docking regardless of mass ratio.

- Diagnose why the GMO residual no longer triggers with 6D EE tracking (hypothesis: higher joint torques from orientation tracking inflate the internal momentum estimate, masking the contact residual)
- Option A: filter the GMO residual to remove the orientation-tracking component
- Option B: use a separate momentum observer that only tracks the swing arm's contribution
- Option C: hybrid dock detection: kinematic (d < threshold) OR GMO, whichever triggers first
- Target: reliable dock detection at 6-8mm for all mass ratios

### Phase 3: Orientation-Aware Close Approach

**Goal:** Achieve < 5deg orientation error at dock without sacrificing position convergence.

- Phased approach: far (d > 20mm) use full 6D tracking; close (d < 20mm) use 3D position with soft orientation penalty; final (d < 5mm) pure position
- Or: separate EE position (priority 2) and EE orientation (priority 3) into distinct QP tasks with independent weights
- Validate against HOTDOCK mechanism orientation tolerance (TBD from hardware specs)

### Phase 4: Adaptive Close-Approach Weights

**Goal:** Replace fixed weight ratios with online adaptation based on coupled dynamics.

- Estimate the robot-structure coupling from the contact Jacobian and mass ratio
- Scale `alpha_com` inversely with coupling strength: when the EE approach strongly disturbs the CoM (high mass ratio), reduce CoM weight automatically
- Or: use a nullspace approach where the EE task is projected into the CoM-consistent subspace

### Phase 5: Structure AOCS Integration

**Goal:** Replace the scalar `tau_struct_max` bound with a realistic AOCS model.

- Model reaction wheel dynamics: `I_w * ω̇_w = -τ_w`, with `|ω_w| ≤ ω_max` (momentum saturation) and `|τ_w| ≤ τ_w_max` (rate limit)
- Add wheel momentum `h_w_struct` as NMPC state, with desaturation policy
- Constraint becomes `|τ_w,i| ≤ τ_w_max AND |h_w,i| ≤ h_w_max` instead of the current `|Ḣ_s,i| ≤ tau_struct_max`
- Validate: 6-step traversal with bounded wheel momentum and desaturation between steps

### Phase 6: 8% Mass Ratio Demonstration

**Goal:** Achieve multi-step locomotion at 8% mass ratio (structure 901 kg, robot 74 kg).

- Requires: Ḣ_s constraint (Phase 1) + AOCS model (Phase 5) + adaptive weights (Phase 4)
- Expected trade-off: significantly slower crawling (the NMPC must respect tight structural torque bounds, emergent behavior is gentler motions at large lever arms)
- Quantify: total crawl time vs mass ratio, dock margin vs mass ratio
- Generate publication figure: Pareto frontier of mass ratio vs crawl performance

---

## 4. Configuration Reference

| Parameter | Default | Effect |
|-----------|---------|--------|
| `tau_struct_max` | `inf` | Structure disturbance torque bound [Nm]. `inf` = disabled. |
| `t_settle_inter` | `0.0` | Inter-step settle duration [s]. `0` = disabled. Recommended: `2.0` for mass ratio > 0.5%. |
| `ext_alpha_ee` | `1e4` | EXT phase EE task weight. Close-approach QP uses 10x this value. |
| `ext_alpha_com` | `1e2` | EXT phase CoM task weight. Close-approach QP uses 0.1x this value. |
| `weld_radius` | `0.005` | Kinematic dock threshold [m]. |
| `ext_Kp_ee` | `25.0` | EXT phase EE position gain. |
| `ext_Kd_ee` | `15.0` | EXT phase EE velocity gain. |

### Recommended Configurations

**0.1% mass ratio (demonstration/validation):**
```python
cfg = SimConfig()  # all defaults
```

**1% mass ratio (nominal mission):**
```python
cfg = SimConfig(t_settle_inter=2.0)
```

**Higher mass ratios (requires Phase 1 validation):**
```python
cfg = SimConfig(t_settle_inter=2.0, tau_struct_max=5.0)  # TBD
```
