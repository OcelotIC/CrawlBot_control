# Roadmap: Lutze Baseline on VISPA Robot in MuJoCo

## Goal
1. Adapt Lutze et al. (2023) single-step QP controller for the VISPA dual-arm robot
2. Run it inside the same MuJoCo simulation as MPC_crawling for a fair journal comparison

---

## Phase 0 — Centroidal Approximation Layer (match Lutze's abstraction level)

Lutze models the robot as a **rigid body** (point mass + inertia cube).
To reach the same abstraction on VISPA, we need functions that reduce the
12-DOF dual-arm robot to centroidal quantities — the same quantities Lutze's
QP operates on.

### 0.1  `centroidal_model.py` — rigid-body reduction of VISPA
- Input: Pinocchio RobotState (q, v, H, J_com, contact frames)
- Compute:
  - `r_com, v_com` — CoM position/velocity (already in robot_interface)
  - `L_com` — centroidal angular momentum (already in robot_interface)
  - `I_centroidal` — locked-inertia tensor about CoM
    (`pin.ccrba(model, data, q, v)` gives the centroidal momentum matrix Ag;
     the 3×3 upper-left block of Ag relates ω to L)
  - `m_robot` — total mass (constant, 71 kg)
- Output: `CentroidalState(r_com, v_com, L_com, I_locked, m)`

### 0.2  `contact_adjoint.py` — SE(3) wrench mapping via Pinocchio
- For each active gripper frame (`tool_a`, `tool_b`):
  - Get `oMf = data.oMf[frame_id]` (SE3 placement in world frame)
  - Compute Ad_gc = oMf.toActionMatrixInverse().T  (wrench from contact → world)
  - Or equivalently: wrench at satellite CoM = Ad_gc^{-T} @ F_contact
- Output: `Ad_a (6×6)`, `Ad_b (6×6)` adjoint matrices for each contact

### 0.3  `momentum_map.py` — contact wrench → angular momentum rate
- From Pinocchio contact positions and CoM:
  - `dL/dt = sum_j [ (r_Cj - r_com) × f_j + tau_j ]`
  - Assemble M_lambda (3×12) matrix mapping [Fc_a, Fc_b] → L_dot
- This is the centroidal dynamics that Lutze's QP implicitly uses

---

## Phase 1 — Lutze QP Controller (Pinocchio-native)

Port the core algorithm: **single-step wrench optimization** that minimizes
structure disturbance while tracking the desired robot motion.

### 1.1  `lutze_feedforward.py` — desired wrench computation
- **Robot tracking wrench** F_d_r (Lutze eq. 12-14):
  - PD on CoM/torso position: `F_d_r = [0; Kr*(r_ref - r_com) + Dr*(v_ref - v_com)]`
  - Same as Lutze but using Pinocchio's CoM
- **Structure stabilization wrench** F_d_b (Lutze eq. 16):
  - PD on structure attitude: `F_d_b = [Kb*e_quat + Db*(-omega_struct); 0]`
  - Uses structure quaternion from MuJoCo state
- **Swing arm tracking wrench** F_d_swing:
  - PD on end-effector position (Cartesian impedance)
  - Only during single-support phase

### 1.2  `lutze_qp.py` — wrench optimization QP
Extend Lutze's 6-dim QP to **dual-contact** (12-dim decision variable):

```
Decision: Fc = [Fc_a(6), Fc_b(6)]     (12-dim)

min  ||Ad_a' Fc_a + Ad_b' Fc_b - F_d_r||²_Qr     (tracking)
   + ||M_lambda @ Fc||²_Qb                         (minimize L_dot)
   + ||Fc||²_Qc                                     (regularization)

s.t.  -F_max <= Fc_a <= F_max      (SI traction/bending limits)
      -F_max <= Fc_b <= F_max
      |L_dot| <= tau_w_max          (optional: momentum rate constraint)
```

Key differences from original Lutze:
- **12-dim** decision variable (dual arm) vs 6-dim (single contact)
- **M_lambda** (momentum map) replaces Ad_b for satellite stabilization
  — directly minimizes angular momentum rate rather than a feedback wrench
- During single support: zero the inactive contact → reduces to 6-dim
- Uses OSQP or qpOASES (same solver as MPC_crawling's Stage 2)

### 1.3  `lutze_joint_torques.py` — wrench → joint torques
Convert optimal contact wrenches to joint commands:
```
tau = J_tool_a^T @ Fc_a + J_tool_b^T @ Fc_b
```
- Uses Pinocchio's `J_tool_a (6×18)`, `J_tool_b (6×18)` contact Jacobians
- Extract joint torques: `tau_joints = tau[6:]` (first 6 are unactuated base)
- Clip to actuator limits (±10 Nm)
- **No whole-body QP** — this is the key simplification vs MPC_crawling

### 1.4  `lutze_swing_controller.py` — swing arm Cartesian impedance
During single support, the swing arm needs separate control:
- Compute swing arm Jacobian from Pinocchio
- Cartesian PD: `F_swing = Kp*(p_ref - p_ee) + Kd*(v_ref - v_ee)`
- Add to joint torques: `tau += J_swing^T @ F_swing`
- Trajectory: quintic interpolation to next anchor (reuse SwingPlanner)

---

## Phase 2 — Integration with MuJoCo Simulation

### 2.1  `sim_lutze.py` — main simulation script
Mirror `sim_torso6d.py` structure but replace the control pipeline:

```python
while t < T_MAX:
    # 1. State from MuJoCo → Pinocchio
    q_pin, v_pin = mj_to_pin(mj_data)
    rs = robot.update(q_pin, v_pin)

    # 2. Structure state from MuJoCo
    q_struct, omega_struct = get_structure_state(mj_data)

    # 3. Reference generation (reuse TorsoPlanner / ContactScheduler)
    r_ref, v_ref = planner.get_reference(t, phase)
    p_ee_ref, v_ee_ref = swing_planner.get_reference(t, phase)

    # 4. Lutze feedforward
    F_d_r, F_d_b = lutze_feedforward(rs, r_ref, v_ref, q_struct, omega_struct)

    # 5. Lutze QP
    Fc_opt = lutze_qp(F_d_r, F_d_b, Ad_a, Ad_b, M_lambda, phase)

    # 6. Joint torques
    tau = lutze_joint_torques(Fc_opt, rs, phase)
    tau += swing_impedance(rs, p_ee_ref, v_ee_ref, phase)
    tau = np.clip(tau, -tau_max, tau_max)

    # 7. Apply to MuJoCo
    mj_data.ctrl[:12] = tau
    mujoco.mj_step(mj_model, mj_data)
```

### 2.2  Reuse from MPC_crawling (no reimplementation)
- `robot_interface.py` — Pinocchio state computation
- `convert.py` — MuJoCo ↔ Pinocchio conversion
- `contact_scheduler.py` — gait timing, anchor grid, phase management
- `swing_planner.py` — end-effector trajectory for swing arm
- `torso_planner.py` — torso/CoM reference generation
- `ik.py` — inverse kinematics for dock configurations
- URDF/MJCF robot models

### 2.3  `plot_comparison.py` — head-to-head visualization
Generate matched figures for the journal:
- Panel 1: Structure attitude (α, β, γ) — Lutze vs MPC
- Panel 2: Angular momentum (L_com norm) — Lutze vs MPC
- Panel 3: Wheel momentum accumulation — Lutze vs MPC
- Panel 4: Joint torques — Lutze vs MPC
- Panel 5: Tracking error (torso/CoM) — Lutze vs MPC
- Panel 6: Contact wrenches — Lutze vs MPC

---

## Phase 3 — Validation & Comparison Scenarios

### 3.1  Single-step locomotion (anchor 3a → 4b)
- Same scenario as MPC_crawling's validated case
- Metrics: docking success, structure drift, max |L_com|, max |tau|
- Expected: Lutze baseline may fail to dock or exceed momentum limits
  (no horizon planning → reactive only)

### 3.2  Multi-step locomotion (3+ steps)
- Extended crawling sequence along the structure
- Metrics: cumulative structure attitude drift, RW momentum accumulation
- Shows advantage of proactive (MPC) vs reactive (Lutze) momentum planning

### 3.3  Ablation studies
- Lutze QP with vs without momentum rate constraint
- Lutze QP with different Qb weights
- MPC with only Stage 2 (no NMPC) — "fair" single-step QP comparison
- Sensitivity to actuator torque limits (10 Nm tight, 50 Nm loose)

---

## Implementation Order (dependency graph)

```
Phase 0.1 (centroidal_model)  ──┐
Phase 0.2 (contact_adjoint)  ──┤
Phase 0.3 (momentum_map)     ──┼── Phase 1.2 (lutze_qp)
Phase 1.1 (lutze_feedforward) ─┘        │
                                         ├── Phase 1.3 (joint_torques)
Phase 1.4 (swing_controller) ───────────┤
                                         │
                              Phase 2.1 (sim_lutze) ── Phase 2.3 (plots)
                                         │
                              Phase 3.x (scenarios)
```

## File placement

All new files go into `/home/user/MPC_crawling/lutze_baseline/`:
```
lutze_baseline/
├── __init__.py
├── centroidal_model.py      # Phase 0.1
├── contact_adjoint.py       # Phase 0.2
├── momentum_map.py          # Phase 0.3
├── lutze_feedforward.py     # Phase 1.1
├── lutze_qp.py              # Phase 1.2
├── lutze_joint_torques.py   # Phase 1.3
├── lutze_swing_controller.py # Phase 1.4
├── sim_lutze.py             # Phase 2.1
└── plot_comparison.py       # Phase 2.3
```

## Estimated complexity
- Phase 0: ~150 lines (thin wrappers around Pinocchio)
- Phase 1: ~400 lines (core algorithm)
- Phase 2: ~300 lines (simulation glue + plotting)
- Phase 3: configuration + running experiments
- **Total new code: ~850 lines**
