# VISPA Whole-Body Controller — Comparison Study

Hierarchical whole-body controller for the **VISPA dual-arm crawling robot** operating in microgravity, with a comparative evaluation of two wrench planning strategies:

1. **CNMPC approach** — Centroidal Nonlinear MPC over a receding horizon, proactively planning momentum-feasible contact wrenches.
2. **Lutze baseline** — Single-step QP wrench optimizer adapted from [Lutze et al. (2023)](https://core.ac.uk/download/pdf/564843368.pdf), reactive wrench optimization without horizon planning.

Both controllers share identical infrastructure (WholeBodyQP, MuJoCo simulation, Pinocchio dynamics, contact scheduler, trajectory planners) and differ **only** in Stage 1 wrench reference generation. This isolates the comparison to: **proactive (CNMPC) vs reactive (Lutze) wrench planning**.


## Features

- **Two-stage hierarchical control**: Centroidal NMPC (10 Hz) + Whole-body QP (100 Hz)
- **Multi-step locomotion**: Closed-loop 3-step crawling with alternating arm swings
- **Cooperative torso task**: 6D torso tracking (position + orientation via SLERP) replacing static CoM regulation
- **Real docking validation**: Gripper-to-anchor convergence < 5 mm without teleportation
- **Momentum-aware planning**: Angular momentum envelope constraints (L_max = 5 Nms) in NMPC horizon
- **Publication-quality figures**: Automated generation of comparison plots and LaTeX tables
- **Baseline comparison**: Fair Lutze et al. (2023) reimplementation sharing all infrastructure except Stage 1


## Architecture

```
                     ┌─────────────────────────┐
                     │   ContactScheduler       │
                     │  (gait timing, phases)    │
                     └──────┬──────────────────┘
                            │
           ┌────────────────┼────────────────┐
           ▼                ▼                ▼
  TorsoPlanner      SwingPlanner     LocomotionPlanner
  (quintic+SLERP)   (quintic+bump)   (CoM reference)
           │                │                │
           ▼                │                │
 ┌─────────────────────┐    │                │
 │  Stage 1 (10 Hz)    │    │                │
 │                     │    │                │
 │  NMPC:  CentroidalNMPC ──── CasADi/IPOPT │
 │  Lutze: LutzeQP ──────────── qpsolvers   │
 │                     │    │                │
 │  → λ_ref, a_com_ff  │    │                │
 └─────────┬───────────┘    │                │
           ▼                ▼                │
 ┌──────────────────────────────────────────┐
 │  Stage 2 — WholeBodyQP (100 Hz)          │
 │  qpOASES — torso 6D + EE + posture       │
 │  → τ (12 joint torques)                  │
 └──────────────┬───────────────────────────┘
                ▼
 ┌──────────────────────────────────────────┐
 │  MuJoCo Simulation                       │
 │  penalty contacts, free-floating (500 kg)│
 └──────────────┬───────────────────────────┘
                ▼
 ┌──────────────────────────────────────────┐
 │  RobotInterface (Pinocchio wrapper)      │
 │  CRBA, RNEA, Jacobians, centroidal       │
 └──────────────────────────────────────────┘
```


## Quick Start

### Dependencies

```bash
pip install -r requirements.txt
```

### Run the CNMPC controller (single step)

```bash
python scripts/sim_torso6d.py --urdf models/VISPA_crawling_fixed.urdf \
                               --mjcf models/VISPA_crawling.xml --plot
```

### Run multi-step locomotion (3 steps)

```bash
MUJOCO_GL=disabled python scripts/run_r6_full_sim.py
```

### Run the Lutze baseline

```bash
python -m lutze_baseline.sim_lutze --urdf models/VISPA_crawling_fixed.urdf \
                                    --mjcf models/VISPA_crawling.xml --plot
```

### Generate publication figures

```bash
MUJOCO_GL=disabled python scripts/run_r7_figures.py
```

### Run tests

```bash
PYTHONPATH=. python -m pytest tests/ -v
```


## Repository Structure

```
├── robot_interface.py           # Pinocchio wrapper: CRBA, RNEA, Jacobians, centroidal state
├── dynamics.py                  # Constrained dynamics (Lagrange multipliers, SHAKE/RATTLE)
├── contact_scheduler.py         # Gait plan, phase timing, weld activation
├── locomotion_planner.py        # CoM quintic reference trajectory
├── swing_planner.py             # End-effector quintic with clearance bump
├── torso_planner.py             # Torso 6D: quintic position + SLERP orientation
├── simulation_loop.py           # MuJoCo closed-loop orchestrator (two-stage controller)
├── convert.py                   # MuJoCo ↔ Pinocchio state conversion
├── ik.py                        # Inverse kinematics for dock configurations
│
├── solvers/                     # Two-stage controller algorithms
│   ├── centroidal_nmpc.py       #   Stage 1: CasADi/IPOPT centroidal NMPC
│   ├── wholebody_qp.py          #   Stage 2: qpOASES whole-body QP
│   ├── nmpc_solver.py           #   Generic NMPC backend (CasADi NLP)
│   ├── hierarchical_qp.py       #   Generic HQP backend
│   └── contact_phase.py         #   Contact phase definitions & momentum map
│
├── lutze_baseline/              # Lutze et al. (2023) baseline implementation
│   ├── sim_lutze.py             #   Reactive baseline simulation
│   ├── lutze_qp.py              #   Single-step QP wrench optimizer
│   ├── lutze_feedforward.py     #   PD-based feedforward wrench generation
│   ├── centroidal_model.py      #   Centroidal rigid-body reduction
│   ├── contact_adjoint.py       #   SE(3) adjoint wrench mapping
│   ├── momentum_map.py          #   Contact wrenches → angular momentum rate
│   ├── lutze_joint_torques.py   #   Map wrenches to joint torques
│   ├── lutze_swing_controller.py#   Swing arm Cartesian impedance
│   └── plot_comparison.py       #   Side-by-side comparison plots
│
├── scripts/                     # Executable simulation & plotting scripts
│   ├── sim_torso6d.py           #   Single-step CNMPC simulation
│   ├── plot_torso6d.py          #   5-panel visualization
│   ├── run_r6_full_sim.py       #   Full pipeline: multi-step + comparison
│   └── run_r7_figures.py        #   Publication-quality PDF/PNG figures
│
├── tests/                       # Test suite
│   ├── conftest.py              #   Path setup for pytest
│   ├── test_integration.py      #   Full pipeline integration tests
│   ├── test_torso_task.py       #   Torso 6D cooperative task tests
│   ├── test_multi_step.py       #   Multi-step locomotion (2 steps)
│   ├── test_r3_fixes.py         #   IK start/end pose validation
│   ├── test_r4_fixes.py         #   Live anchor position validation
│   ├── test_r5_closed_loop.py   #   3-step closed-loop validation
│   ├── test_phase0.py           #   Centroidal model unit tests
│   └── test_phase1.py           #   Full Lutze pipeline unit tests
│
├── models/                      # Robot simulation models
│   ├── VISPA_crawling.xml       #   MuJoCo model (structure + anchors)
│   └── VISPA_crawling_fixed.urdf#   Fixed-base URDF for Pinocchio
│
├── URDF_models/                 # Original URDF models + mesh geometry
│   ├── *.urdf                   #   MAR & VISPA variants
│   └── meshes/                  #   DAE & STL mesh files
│
├── results/                     # Simulation outputs
│   ├── logs/                    #   JSON simulation logs
│   └── figures/                 #   Generated plots (PNG/PDF)
│
├── docs/                        # Documentation & references
│   ├── multistep_handoff.md     #   Multi-step controller design notes
│   ├── r5_validation_report.md  #   Closed-loop validation report
│   ├── VISPA_Controller_Documentation.pdf
│   └── lutze_paper.pdf          #   Lutze et al. (2023) reference
│
├── requirements.txt             # Python dependencies
└── LICENSE
```


## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `TORSO_MASS` | 40 kg | Central torso mass |
| `TAU_MAX` | 10 Nm | Space-grade actuator limit |
| `T_SWING` | 6.0 s | Single-support phase duration |
| `TORSO_FRAC` | 0.70 | Fraction of full IK displacement |
| `WELD_R` | 5 mm | Docking success threshold |
| `L_MAX` | 5.0 Nms | Angular momentum envelope limit |
| `DT_NMPC` | 0.1 s | Stage 1 control rate (10 Hz) |
| `DT_QP` | 0.01 s | Stage 2 control rate (100 Hz) |
| `N_HORIZON` | 8 | NMPC prediction horizon steps |


## Simulation Results (3-Step Locomotion)

| Metric | Value |
|--------|-------|
| Docks achieved | 3/3 |
| Dock distances | 3.3 mm, 4.8 mm, 4.8 mm |
| Peak angular momentum | 15.65 Nms |
| Peak joint torque | 10.0 Nm (at limit) |
| Structure drift | 11.3 cm |
| NMPC infeasibility | 5.2% (15/288) |
| Simulation duration | 28.7 s |


## References

- Lutze et al. (2023), "Optimization of multi-arm robot locomotion to reduce satellite disturbances during in-orbit assembly", IEEE Aerospace Conference.
- Bellicoso et al. (2018), "Dynamic locomotion through online nonlinear motion optimization", IEEE-RA Letters.
- Orin et al. (2013), "Centroidal dynamics of a humanoid robot", Autonomous Robots.
