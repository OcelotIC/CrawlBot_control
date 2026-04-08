# Momentum-Aware Multi-Step Crawling

## Overview

This document describes controller modifications that enable multi-step locomotion for the VISPA crawling robot on a free-floating structure. The modifications address the coupled robot-structure dynamics that cause docking failures and structural rotation accumulation during multi-step gaits.

**Current status (7-DOF, 5mm kinematic dock threshold):**

| Mass ratio | 1-step | 3-step | 3-step + settle=2s |
|-----------|--------|--------|-------------------|
| 0.1% (71100 kg) | 1/1 | 1/3 | **3/3** |
| 1% (7110 kg) | 1/1 | **3/3** | **3/3** |
| 8% (901 kg) | 0/1 | 0/3 | 0/3 |

---

## 1. Modifications

### 1.1 Structure Disturbance Constraint (Ḣ_s)

**Files:** `centroidal_nmpc.py`, `config.py`, `sim_loop.py`

**Problem:** The existing NMPC bounds the robot's centroidal angular momentum rate `L̇_com` to protect the robot's reaction wheels, but does not bound the torque delivered to the structure. The structure disturbance torque about its CoM is:

```
Ḣ_s = Σⱼ [r_Cⱼ × fⱼ + τⱼ]
```

where `r_Cⱼ` are contact positions relative to the structure CoM (origin in R_s). The existing `L̇_com` constraint uses `(r_Cⱼ - r_com)` as lever arms; the new `Ḣ_s` constraint uses `r_Cⱼ` directly. With anchors at z = -1.775m from the structure CoM (the crawling panel), even small contact forces create large structural torques.

**Implementation:** Six bilateral path constraints `|Ḣ_s,i| ≤ tau_struct_max` added to the NMPC. Conditionally included only when `tau_struct_max` is finite, avoiding CasADi numerical issues with `inf`.

**Configuration:**
- `SimConfig.tau_struct_max: float = np.inf` (disabled by default)
- `CentroidalNMPCConfig.tau_struct_max: float = np.inf`

**Status:** Implemented, builds correctly, backward compatible (73 unit tests pass). **Not yet tested with finite values.** The expected behavior (NMPC generating gentler wrenches at large lever arms) is unvalidated.

### 1.2 Inter-Step Settling

**Files:** `sim_loop.py`, `config.py`

**Problem:** After docking, the gait immediately starts the next step. Residual joint velocities, wheel momentum, and structural angular rates from the previous step carry over, degrading the next step's approach accuracy.

**Implementation:** A configurable settle phase inserted after each dock event (before `step_idx` increments). Uses the existing `settle_mode=True` path in the whole-body QP: skips torso/CoM tasks, applies pure velocity damping (Kd=10, alpha=1e3).

A cumulative `t_offset` tracks settle durations and shifts all subsequent gait plan timestamps. Without this offset, the settle time eats into the next step's SS phase (e.g., a 6s swing becomes 1.7s with 2s settle).

**Configuration:**
- `SimConfig.t_settle_inter: float = 0.0` (disabled by default)

**Measured effect (3-step):**

| Mass ratio | settle=0 | settle=2s |
|-----------|----------|-----------|
| 0.1% | 1/3 | **3/3** (rot=6.2deg) |
| 1% | **3/3** (rot=2.4deg) | **3/3** (rot=2.1deg) |

At 1%, settling is optional (3/3 without). At 0.1%, settling is required for multi-step — the reason is unclear (the structure barely moves at 0.1%) and warrants investigation.

### 1.3 Close-Approach Docking Fixes

**Files:** `sim_loop.py`  
**`wholebody_qp.py` is unchanged.**

**Problem:** The 6D end-effector tracking upgrade (commit `c5e2f7d`, prior session) broke docking. Two independent issues:

**Issue A — Orientation competing with position:** During close approach, the 6D EE task's orientation component consumed DOFs needed for position convergence. The tool approaches with ~46deg orientation error; the QP traded position accuracy for orientation correction. This prevented the gripper from reaching the 5mm dock threshold.

**Fix:** During close approach (d < 20mm), set `R_ee_ref = R_ee_actual` — the actual tool orientation. This zeros the orientation error in the 6D PD law, so the orientation rows only compensate Coriolis terms. The full 6D Jacobian is preserved for regularization. No QP modification needed.

**What was tried and didn't work:**
- Separate position (priority 2) and orientation (priority 3) QP tasks: position-only task lost the regularization from angular Jacobian rows, making convergence worse
- Same-priority split with weight ratios: orientation error is in radians (~0.8) vs position in meters (~0.005), so orientation dominates even at 1% weight
- Lowering `ext_alpha_com` globally: destabilizes the far approach where CoM tracking is essential

**Issue B — CoM task fighting EE approach:** At 1% mass ratio, the gripper oscillates between 5-8mm instead of converging. Each push toward the anchor creates a reaction force that disturbs the CoM. The CoM task (priority 1, weight 1e2) corrects this by pulling the EE back, creating a sustained oscillation at the coupled robot-structure natural frequency (~0.7Hz).

**Fix:** A third QP variant (`qp_approach`) with rebalanced weights: `alpha_ee * 10`, `alpha_com * 0.1`, `alpha_torso * 0.1`. Activated when d < 20mm via a latched flag (once entered, stays active for the rest of the EXT phase to prevent mode-switching oscillation). This is not a hack — it's a necessary weight rebalancing for the docking phase where EE convergence matters more than CoM maintenance.

**What was tried and didn't work:**
- Lowering `ext_alpha_com` for the entire EXT phase: destabilizes far approach (d > 20mm) where CoM tracking keeps the robot stable

**Additional fix — Approach velocity:** Proportional gain 0.5 with 2mm/s floor: `v_mag = max(0.5 * d, 0.002)`. The original gain (0.2) created exponential decay too slow to cross 5mm within the EXT timeout.

---

## 2. Limits

### 2.1 Mass Ratio Ceiling: 8% Fails

The controller fails at 8% (structure 901 kg, robot 74 kg). The structure tumbles to 180deg and NMPC failures reach ~45%. The `tau_struct_max` constraint is designed to address this but is untested. It may also not be sufficient — even with bounded structural torque, the close-approach QP weight ratios were tuned for 0.1%-1% and may not work at 8%.

### 2.2 0.1% 3-Step Without Settle: 1/3

At 0.1%, the structure barely moves (71100 kg), yet steps 1-2 fail without settling. This is unexpected and suggests a controller state issue (stale NMPC warm start, torso planner reference, or accumulated numerical drift) rather than a dynamics problem. Root cause is uninvestigated.

### 2.3 Dock Threshold Sensitivity

The gripper reaches 4.8-4.9mm at both mass ratios — barely under the 5mm threshold. The old GMO contact detection triggered at 6-7mm regardless of mass ratio but no longer works reliably with the 6D EE controller (hypothesis: higher joint torques from orientation tracking mask the contact momentum residual).

### 2.4 Orientation at Dock

Setting `R_ee_ref = R_ee_actual` during close approach disables orientation tracking. The tool arrives with ~46deg orientation error relative to identity. Acceptable for simulated weld constraints; likely unacceptable for real HOTDOCK hardware which requires alignment.

### 2.5 Fixed Close-Approach QP Weights

The weight ratios (10x EE, 0.1x CoM, 0.1x Torso) are empirically tuned for 0.1%-1%. No online adaptation. The 20mm activation threshold is also a magic number.

### 2.6 Ḣ_s Constraint: Zero Validation

The constraint is implemented and builds correctly but has never been activated. We do not know if:
- The NMPC remains feasible with tight bounds
- The emergent behavior (slower crawling at large lever arms) actually occurs
- It helps at 8% mass ratio

---

## 3. Roadmap

### Phase 1: Fix 0.1% 3-Step Without Settle

**Priority: high.** This is a regression that should not exist — at 0.1% the structure is effectively fixed.

- Diagnose why steps 1-2 fail: log NMPC warm start quality, torso planner state, and EE reference trajectory at the start of each step
- Compare state at step 1 start (after step 0 dock) vs step 0 start (initial): what's different?
- Likely fix: reset NMPC warm start or torso planner after each dock

### Phase 2: Ḣ_s Constraint Validation

**Priority: high.** The constraint is the main deliverable of this session; it needs data.

- Parametric sweep: `tau_struct_max in {1, 2, 5, 10} Nm` at 1% and 8%
- For each value, record: dock success, step times, max structural rotation, max `||Ḣ_s||`, NMPC feasibility rate
- Combined test: best `tau_struct_max` + `t_settle_inter`
- If feasibility drops below 90%, the bounds are too tight — either relax or investigate NMPC solver options

### Phase 3: GMO Dock Recovery

**Priority: medium.** Would increase dock margin from <1mm to ~2mm.

- Diagnose why GMO no longer triggers: compare momentum residual with/without R_ee_ref matching
- Simplest fix: hybrid detection — dock when kinematic (d < weld_radius) OR GMO confirms contact, whichever comes first
- This avoids needing to fix the GMO itself

### Phase 4: Orientation-Aware Close Approach

**Priority: low (unless HOTDOCK requires it).**

Setting `R_ee_ref = R_ee_actual` sacrifices orientation for position. Recovery options:

- **NOT viable:** Separate position/orientation into distinct QP tasks with different priorities or weights. Tried during this session — fails due to radians-vs-meters unit mismatch and loss of angular Jacobian regularization.
- **Potentially viable:** Phased approach — full 6D at d > 20mm, R_ee_ref matching at d < 20mm, then a post-dock orientation correction phase before weld activation.
- **Potentially viable:** Normalize orientation error by `Kp_ee_ang` scaling so it's comparable in magnitude to position error. Requires careful gain tuning.

### Phase 5: 8% Mass Ratio

**Priority: depends on mission requirements.**

Requires at minimum Ḣ_s constraint (Phase 2) and possibly:
- Adaptive close-approach weights (scale CoM relaxation with mass ratio)
- Structure AOCS model in the NMPC (wheel momentum as state, desaturation policy)
- Significantly longer crawl times (the NMPC must respect tight torque bounds)

This is the hardest problem and may not be achievable with the current controller architecture.

---

## 4. Configuration Reference

| Parameter | Default | Effect |
|-----------|---------|--------|
| `tau_struct_max` | `inf` | Structure disturbance torque bound [Nm]. `inf` = disabled. Untested with finite values. |
| `t_settle_inter` | `0.0` | Inter-step settle duration [s]. `0` = disabled. |
| `ext_alpha_ee` | `1e4` | EXT phase EE task weight. Close-approach QP uses 10x. |
| `ext_alpha_com` | `1e2` | EXT phase CoM task weight. Close-approach QP uses 0.1x. |
| `weld_radius` | `0.005` | Kinematic dock threshold [m]. |

### Tested Configurations

**0.1% mass ratio — 3/3 requires settling:**
```python
cfg = SimConfig(t_settle_inter=2.0)
```

**1% mass ratio — 3/3 works without settling:**
```python
cfg = SimConfig()  # all defaults
# or with settling for lower structural rotation:
cfg = SimConfig(t_settle_inter=2.0)
```

**8% mass ratio — does not work with any configuration tested.**

---

## 5. Files Changed (vs main)

| File | Change | Clean? |
|------|--------|--------|
| `centroidal_nmpc.py` | Ḣ_s constraint (conditional, backward compatible) | Yes |
| `config.py` | `tau_struct_max`, `t_settle_inter` params | Yes |
| `sim_loop.py` | Approach velocity, R_ee_ref matching, close-approach QP, inter-step settling with t_offset | Functional but carries complexity (3 QP variants, latched phase flag, 20mm magic number) |
| `wholebody_qp.py` | **Unchanged** | — |
