# Codebase Explanation: Lutze et al. (2023) MIRROR Framework

## Overview

This repository reproduces and extends the controller from **Lutze et al. (2023)**, *"Optimization of multi-arm robot locomotion to reduce satellite disturbances during in-orbit assembly"* (IEEE Aerospace Conference 2023). The paper proposes a **single-step QP wrench optimizer** (called MIRROR) that minimizes disturbances to a satellite while a dual-arm robot crawls along its surface.

The codebase has three layers:

1. **MATLAB original** — Direct MATLAB implementation using the SPART dynamics library
2. **Python translation** — Standalone Python port of the MATLAB code (SPART-based)
3. **lutze_baseline** — Pinocchio/MuJoCo adaptation for the real VISPA robot, designed as a baseline for comparison against an MPC controller

---

## Repository Structure

```
Lutze2023mirror/
├── MATLAB layer (original paper reproduction)
│   ├── RUN_BENCHMARK.m              # Main entry point
│   ├── config/
│   │   ├── setup_environment.m      # Satellite, actuator, QP parameters
│   │   └── setup_trajectories.m     # 5th-order polynomial CoM trajectories
│   ├── control/
│   │   ├── LutzeQPController.m      # Core QP controller class
│   │   └── compute_feedforward_wrench.m  # PD tracking wrench computation
│   ├── dynamics/
│   │   ├── compute_system_state.m   # Full kinematics/dynamics via SPART
│   │   ├── compute_contact_jacobian.m
│   │   └── compute_angular_momentum.m
│   ├── simulation/
│   │   ├── simulate_experiment.m    # Main simulation loop
│   │   └── integrate_dynamics.m     # Euler integration with Baumgarte stabilization
│   ├── visualization/
│   │   ├── plot_satellite_rotation.m
│   │   ├── plot_contact_wrenches.m
│   │   ├── plot_tracking_error.m
│   │   ├── plot_momentum_saturation.m
│   │   └── compute_performance_metrics.m
│   ├── utils/
│   │   ├── adjoint_SE3.m            # SE(3) adjoint for wrench transformation
│   │   └── quat_utils.m            # Quaternion operations
│   └── src/                         # SPART library (kinematics/dynamics engine)
│       ├── kinematics_dynamics/     # GIM, CIM, Jacobians, forward dynamics
│       ├── casadi_func/             # CasADi-compatible SPART functions
│       ├── robot_model/             # URDF parser, connectivity maps
│       └── attitude_transformations/
│
├── Python layer (standalone translation)
│   └── python/
│       ├── run_benchmark.py         # Python equivalent of RUN_BENCHMARK.m
│       ├── config/
│       │   ├── environment.py       # Same parameters, Python dicts
│       │   └── trajectories.py      # Same polynomial trajectories
│       ├── control/
│       │   ├── controller.py        # LutzeQPController (scipy.optimize.quadprog)
│       │   └── feedforward.py       # compute_feedforward_wrench
│       ├── dynamics/
│       │   ├── system_state.py      # compute_system_state
│       │   ├── contact_jacobian.py
│       │   └── angular_momentum.py
│       ├── simulation/
│       │   ├── experiment.py        # simulate_experiment
│       │   └── integrator.py        # integrate_dynamics (no RW, paper-accurate)
│       ├── spart/                   # Python SPART library
│       │   ├── robot_model.py       # URDF parser (urdf2robot)
│       │   ├── kinematics.py        # Forward kinematics
│       │   ├── diff_kinematics.py   # Differential kinematics
│       │   ├── velocities.py        # Twist propagation
│       │   ├── inertia.py           # Spatial inertias
│       │   ├── gim.py               # Generalized Inertia Matrix
│       │   ├── cim.py               # Convective Inertia Matrix (Coriolis)
│       │   ├── jacobian.py          # Geometric Jacobians
│       │   ├── dynamics.py          # Forward/inverse dynamics
│       │   ├── accelerations.py     # Acceleration computations
│       │   └── attitude.py          # DCM/quaternion conversions
│       ├── utils/
│       │   ├── quaternion.py        # Quaternion operations
│       │   └── spatial.py           # SE(3) adjoint, skew, etc.
│       ├── tests/                   # pytest suite (105 tests)
│       └── visualization/
│           ├── plots.py             # 4-panel figure generation
│           └── metrics.py           # Performance table
│
├── lutze_baseline/ (VISPA robot adaptation for MPC comparison)
│   ├── __init__.py
│   ├── centroidal_model.py          # Phase 0.1: Multi-body → centroidal reduction
│   ├── contact_adjoint.py           # Phase 0.2: SE(3) wrench mapping via Pinocchio
│   ├── momentum_map.py              # Phase 0.3: Contact wrench → L_dot map
│   ├── lutze_feedforward.py         # Phase 1.1: PD tracking wrench
│   ├── lutze_qp.py                  # Phase 1.2: 12-dim dual-contact QP
│   ├── lutze_joint_torques.py       # Phase 1.3: Wrench → joint torques
│   ├── lutze_swing_controller.py    # Phase 1.4: Cartesian impedance for swing arm
│   ├── sim_lutze.py                 # Phase 2.1: Full MuJoCo simulation
│   ├── plot_comparison.py           # Phase 2.3: Lutze vs MPC figures
│   ├── test_phase0.py              # Smoke tests for centroidal layer
│   └── test_phase1.py              # Smoke tests for controller pipeline
│
├── URDF_models/                     # Robot descriptions
│   ├── MAR_DualArm_6DoF.urdf       # Paper's 200kg satellite + 12-DOF dual-arm
│   ├── MAR_DualArm_3DoF.urdf       # Simplified 6-DOF variant
│   └── VISPA_crawling.urdf         # Real VISPA robot for MPC comparison
│
├── models/                          # MuJoCo-compatible models
│   ├── VISPA_crawling.xml           # MJCF model for MuJoCo simulation
│   └── VISPA_crawling_fixed.urdf    # Corrected URDF for Pinocchio
│
├── results/
│   ├── data/                        # Saved simulation results (.mat, .pkl)
│   ├── figures/                     # Paper reproduction figures
│   └── comparison/                  # Lutze vs MPC comparison plots
│
├── ROADMAP.md                       # Implementation plan for VISPA adaptation
├── lutze_paper.pdf                  # Original paper
└── README.md
```

---

## The MIRROR Algorithm (Lutze et al. 2023)

### Problem Statement

A dual-arm robot crawls along the surface of a satellite in orbit. Each step consists of:
1. **Dual support**: Both grippers anchored; torso advances forward
2. **Single support**: One gripper releases, swings to the next anchor

The challenge: **every force the robot exerts on the satellite creates a reaction torque** that rotates the satellite. Reaction wheels can absorb some angular momentum, but they saturate. The goal is to plan contact wrenches that move the robot while minimizing satellite disturbance.

### Core Optimization (QP)

At each control timestep, the controller solves a **quadratic program** over the 6-DOF contact wrench `Fc = [torque(3); force(3)]`:

```
min  ||Ad^T Fc - F_d_r||²_Qr          ← track desired robot motion
   + ||Ad^T Fc - (-F_d_b)||²_Qb       ← minimize satellite disturbance
   + ||Fc||²_Qc                        ← regularize contact wrench

s.t. F_min <= Fc <= F_max              ← structural interface limits
```

Where:
- **`Ad` (6×6)**: SE(3) adjoint matrix mapping contact wrench to satellite CoM wrench
- **`F_d_r` (6×1)**: Desired robot tracking wrench (PD on CoM position)
- **`F_d_b` (6×1)**: Desired satellite stabilization wrench (PD on satellite attitude)
- **`Qr, Qb, Qc`**: Weight matrices balancing tracking vs. disturbance minimization

The key insight: the three objectives compete. Pure tracking (`Qr` dominant) moves the robot but disturbs the satellite. Pure stabilization (`Qb` dominant) freezes everything. The QP finds the Pareto-optimal tradeoff.

### Data Flow

```
trajectory(t) ──→ feedforward_wrench ──→ [F_d_r, F_d_b]
                                              │
robot_state ──→ SE(3) adjoint ──→ Ad          │
                                   │          │
                                   ▼          ▼
                              QP solver ──→ Fc_opt
                                   │
                                   ▼
                  Fc_opt ──→ J^T @ Fc_opt ──→ tau_joints
                                   │
                                   ▼
                            integrate_dynamics ──→ next_state
```

---

## MATLAB Implementation Details

### `RUN_BENCHMARK.m`
Entry point. Loads the 6-DOF dual-arm URDF via SPART's `urdf2robot`, runs two simulations (non-optimized vs. QP-optimized) for each of three experiments, generates comparison figures.

### `setup_environment.m`
Defines the satellite (2040 kg, 7.5m radius cylinder), Standard Interconnect limits (3000N / 300Nm), reaction wheel configuration (4 wheels in pyramid), and QP gains.

### `setup_trajectories.m`
Three experiments from the paper:
- **Exp 1**: Straight line crossing satellite CoM (15m, hardest — minimal lever arm)
- **Exp 2**: Offset straight line (3m diagonal)
- **Exp 3**: 20° circular arc on satellite edge

All trajectories use 5th-order polynomial interpolation ensuring zero velocity/acceleration at boundaries:
```
s(τ) = 10τ³ - 15τ⁴ + 6τ⁵     (τ = t/T_motion)
```

### `LutzeQPController.m`
The controller class. Key methods:
- `solve()`: Computes feedforward wrench, then either solves QP (optimized) or uses pseudo-inverse (non-optimized)
- `solve_qp()`: Builds Hessian `H = A_r^T Qr A_r + A_b^T Qb A_b + Qc`, solves with MATLAB's `quadprog`
- `build_transform()`: Constructs SE(3) transform from robot contact frame to satellite base

### `compute_system_state.m`
The most complex function. Calls the full SPART pipeline (CasADi-compatible versions):
1. `Kinematics_casadi` → link positions/orientations
2. `DiffKinematics_casadi` → twist propagation matrices
3. `Velocities_casadi` → body twists
4. `I_I_casadi` → spatial inertias in world frame
5. `MCB_casadi` → composite body inertias
6. `GIM_casadi` → generalized inertia matrix `H = [H0, H0m; H0m^T, Hm]`
7. `CIM_casadi` → Coriolis matrix `C = [C0, C0m; Cm0, Cm]`
8. Contact Jacobian for the end-effector
9. Angular momentum decomposition (satellite + robot + RW)

### `integrate_dynamics.m`
Forward Euler integration with Baumgarte constraint stabilization:
```
M q̈ = τ + J^T Fc - C q̇ + J^T γ_stab
```
The satellite inertia is augmented into the SPART base (`H0 += H0_sat`), so the coupled equations of motion naturally conserve angular momentum. **No reaction wheels** — matching the paper's assumption of pure momentum conservation.

---

## Python Translation

Direct 1:1 translation of every MATLAB function into Python/NumPy. The SPART library is fully re-implemented in `python/spart/`. Key differences from MATLAB:

- Uses `scipy.optimize.minimize` with bounds (fallback from `quadprog`)
- Robot model stored as Python dicts instead of MATLAB structs
- All cell-array indexing translated to list indexing
- Quaternion convention preserved (SPART: `[q1,q2,q3,q4]` with `q4` = scalar)

### Important: No Reaction Wheels

The Python integrator (`python/simulation/integrator.py`) deliberately removes all RW dynamics to match the paper. The satellite's angular velocity changes only through contact force reactions — this is the regime where the QP optimization has the most impact.

---

## lutze_baseline: VISPA Robot Adaptation

This is the most practically important layer. It adapts the Lutze algorithm for the **real VISPA dual-arm crawling robot** (71 kg, 12-DOF) running in MuJoCo simulation, designed for head-to-head comparison against a hierarchical MPC controller.

### Architecture

The key architectural decision: **Lutze QP replaces only the top-level wrench planner (NMPC), while the low-level Whole-Body QP remains identical** to the MPC controller. This isolates the comparison to reactive (Lutze) vs. proactive (MPC) wrench planning.

```
┌─────────────────────────────────────────────────┐
│  MPC Controller                                  │
│  ┌──────────────┐    ┌──────────────────────┐   │
│  │ Centroidal   │    │                      │   │
│  │ NMPC (10Hz)  │───→│  Whole-Body QP       │   │
│  │ (CasADi)     │    │  (100Hz, OSQP)       │──→│ tau_joints
│  └──────────────┘    │                      │   │
│                      └──────────────────────┘   │
├─────────────────────────────────────────────────┤
│  Lutze Baseline                                  │
│  ┌──────────────┐    ┌──────────────────────┐   │
│  │ Lutze QP     │    │                      │   │
│  │ (10Hz)       │───→│  Whole-Body QP       │   │
│  │ (OSQP)       │    │  (100Hz, OSQP)       │──→│ tau_joints
│  └──────────────┘    │  [SAME as MPC]       │   │
│                      └──────────────────────┘   │
└─────────────────────────────────────────────────┘
```

### Phase 0: Centroidal Approximation Layer

The paper models the robot as a rigid body. To use the same algorithm on a 12-DOF articulated robot, we reduce it to centroidal quantities:

**`centroidal_model.py`** — Wraps Pinocchio's `computeCentroidalMap()` to extract:
- `r_com, v_com` — CoM position/velocity
- `L_com` — centroidal angular momentum
- `I_locked` — the 3×3 locked inertia tensor (upper-left of centroidal momentum matrix `Ag`)
- `mass` — total robot mass

**`contact_adjoint.py`** — Builds 6×6 wrench transformation matrices from each contact frame to the world frame using Pinocchio SE3 placements:
```python
A = [R,   skew(p) @ R]    # co-adjoint (wrench transform)
    [0,       R       ]
```

**`momentum_map.py`** — Constructs `M_lambda` (3×12), the matrix mapping stacked contact wrenches to angular momentum rate:
```
L_dot = sum_j [ tau_j + (r_Cj - r_com) × f_j ]
```
Each 3×6 block: `M_j = [I_3, skew(r_Cj - r_com)]`

### Phase 1: Lutze QP Controller

**`lutze_feedforward.py`** — PD tracking wrench:
```python
F_d_r = [0;  Kr*(r_ref - r_com) + Dr*(v_ref - v_com)]   # robot tracking
F_d_b = [Kb*e_att + Db*(-omega);  0]                      # structure stabilization
```

**`lutze_qp.py`** — The core 12-dimensional QP (dual-contact extension):
```
Decision: Fc = [Fc_a(6), Fc_b(6)]     (12-dim)

min  ||A_wrench @ Fc - F_d_r||²_Qr     (tracking)
   + ||M_lambda @ Fc||²_Qb             (minimize L_dot)
   + ||Fc||²_Qc                         (regularization)

s.t.  -F_max <= Fc <= F_max
      (optional) |M_lambda @ Fc| <= tau_w_max
```

Key extension from paper: **M_lambda replaces Ad_b** for the satellite stabilization term — instead of minimizing a feedback wrench, it directly minimizes the angular momentum rate. This is physically more meaningful for the multi-body case.

**`lutze_joint_torques.py`** — Simple `tau = J_a^T @ Fc_a + J_b^T @ Fc_b`, extracting actuated joints.

**`lutze_swing_controller.py`** — Cartesian impedance for the swing arm during single-support: `tau = J^T @ [0; Kp*e + Kd*ė]`.

### Phase 2: MuJoCo Simulation

**`sim_lutze.py`** — Full simulation mirroring the MPC's `sim_torso6d.py`:
- Same MuJoCo setup, inverse kinematics, contact scheduler, swing planner
- Lutze QP runs at 10Hz providing `lambda_ref` (wrench references)
- WholeBodyQP runs at 100Hz converting wrenches to joint torques (identical to MPC)
- Logs all quantities for comparison

**`plot_comparison.py`** — Generates 6-panel comparison figures (docking convergence, torso advancement, peak torques, structure drift, angular momentum, tracking error).

### Results

| Metric | Lutze Baseline | MPC Controller |
|--------|---------------|----------------|
| Docking distance | 3.0 mm | 3.0 mm |
| Docking time | 7.6 s | 7.6 s |
| max \|L_com\| | **6.5 Nms** (exceeds 5 Nms limit) | **3.25 Nms** (within limit) |
| Structure drift | 5.8 cm | 5.8 cm |
| Torso advancement | +45.5 cm | +45.5 cm |

The key finding: **Lutze's reactive approach achieves equivalent tracking and docking but violates the angular momentum constraint** because it cannot plan ahead. The MPC proactively distributes wrenches over a horizon to stay within bounds.

---

## Robot Models

### MAR_DualArm_6DoF (Paper)
- 200 kg satellite (2040 kg for the full satellite model)
- 12-DOF dual arm (6 per arm)
- Satellite modeled as SPART base with augmented inertia
- Large Standard Interconnect limits (3000N, 300Nm)

### VISPA (Real Robot)
- 71 kg total mass
- 12-DOF dual arm (6 revolute joints per arm)
- Kinematic chain: Link_0 (torso) → {Link_1..6_a, Link_1..6_b} → {tool_a, tool_b}
- Joint limits: ±π rad, 50 Nm effort, 0.094 rad/s velocity
- Pinocchio: loads with FreeFlyer base (nq=19, nv=18, 14 joints)
- MuJoCo: uses companion MJCF (VISPA_crawling.xml)

### URDF Fixes Applied
The `VISPA_crawling_fixed.urdf` contains three corrections:
1. Added missing `<link name="world"/>` (referenced by `world-link0` joint)
2. Fixed `tool_a`/`tool_b` self-closing `<link/>` tags that orphaned their `<inertial>` children
3. Reduced tool inertias to 0.0001 kg·m² (minimal end-effector mass)

---

## Key Mathematical Concepts

### SE(3) Adjoint for Wrenches
A wrench `F = [τ; f]` at a contact frame is mapped to the world frame via:
```
F_world = Ad_{g}^{-T} @ F_contact
```
where `g ∈ SE(3)` is the contact-to-world transformation. The matrix:
```
Ad^{-T} = [R,   [p]× R]
           [0,      R   ]
```
This is implemented in both `utils/adjoint_SE3.m` (MATLAB) and `lutze_baseline/contact_adjoint.py` (Python/Pinocchio).

### Centroidal Momentum Map
The relationship between contact wrenches and angular momentum rate at the system CoM:
```
L̇ = Σ_j [τ_j + (r_Cj - r_com) × f_j]
```
Assembled as `L̇ = M_λ @ Fc` where `M_λ = [I₃, [r_C - r_com]×]` for each contact.

### SPART Dynamics
The Spatial Algebra and Rigid-body Toolbox (SPART) computes:
- **GIM** (Generalized Inertia Matrix): `H = [H0, H0m; H0m^T, Hm]`
- **CIM** (Convective Inertia Matrix): `C` (Coriolis/centrifugal effects)
- Equation of motion: `H q̈ + C q̇ = τ + J^T F_ext`

---

## Dependencies

### MATLAB
- SPART toolbox (included in `src/`)
- Optimization Toolbox (`quadprog`)

### Python (standalone)
- NumPy, SciPy
- matplotlib (visualization)

### Python (VISPA / lutze_baseline)
- Pinocchio (`pip install pin`)
- MuJoCo (`pip install mujoco`)
- qpsolvers + OSQP (`pip install qpsolvers osqp`)
- CasADi (`pip install casadi`)
- NumPy, matplotlib

---

## How to Run

### Paper reproduction (MATLAB)
```matlab
cd Lutze2023mirror
RUN_BENCHMARK(2)   % Experiment 2 (offset trajectory)
```

### Paper reproduction (Python)
```bash
cd Lutze2023mirror
python python/run_benchmark.py 2
```

### VISPA simulation (Lutze baseline)
```bash
cd MPC_crawling
python -m lutze_baseline.sim_lutze --urdf models/VISPA_crawling_fixed.urdf
```

### Comparison plot
```bash
python -m lutze_baseline.plot_comparison sim_lutze_log.json sim_torso6d_log.json
```

### Tests
```bash
cd MPC_crawling
python -m lutze_baseline.test_phase0 --urdf models/VISPA_crawling_fixed.urdf
python -m lutze_baseline.test_phase1 --urdf models/VISPA_crawling_fixed.urdf
```
