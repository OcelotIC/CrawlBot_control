# VISPA Whole-Body Controller — Comparison Study

Hierarchical whole-body controller for the **VISPA dual-arm crawling robot** operating in microgravity, with a comparative evaluation of two wrench planning strategies:

1. **NMPC approach** (`sim_torso6d.py`) — Centroidal Nonlinear MPC over a receding horizon, proactively planning momentum-feasible contact wrenches.
2. **Lutze baseline** (`lutze_baseline/sim_lutze.py`) — Single-step QP wrench optimizer adapted from [Lutze et al. (2023)](https://core.ac.uk/download/pdf/564843368.pdf), reactive wrench optimization without horizon planning.

Both controllers share identical infrastructure (WholeBodyQP, MuJoCo simulation, Pinocchio dynamics, contact scheduler, trajectory planners) and differ **only** in Stage 1 wrench reference generation. This isolates the comparison to: **proactive (NMPC) vs reactive (Lutze) wrench planning**.


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
pip install numpy pinocchio casadi mujoco matplotlib qpsolvers osqp
```

### Run the NMPC controller (your approach)

```bash
python sim_torso6d.py --urdf models/VISPA_crawling_fixed.urdf --mjcf models/VISPA_crawling.xml --plot
```

### Run the Lutze baseline

```bash
python -m lutze_baseline.sim_lutze --urdf models/VISPA_crawling_fixed.urdf --mjcf models/VISPA_crawling.xml --plot
```

### Generate comparison plots

```bash
python -m lutze_baseline.plot_comparison sim_lutze_log.json sim_torso6d_log.json
```

### Run tests

```bash
python -m pytest test_integration.py test_torso_task.py test_multi_step.py -v
python -m pytest lutze_baseline/test_phase0.py lutze_baseline/test_phase1.py -v
```


## File Layout

```
├── sim_torso6d.py              # NMPC simulation (Stage 1: CentroidalNMPC)
├── plot_torso6d.py             # 6-panel visualisation for NMPC run
│
├── robot_interface.py          # Pinocchio wrapper: CRBA, RNEA, Jacobians
├── contact_scheduler.py        # Gait plan, phase timing, weld activation
├── locomotion_planner.py       # CoM quintic reference trajectory
├── swing_planner.py            # End-effector quintic with clearance bump
├── torso_planner.py            # Torso 6D: quintic position + SLERP orientation
│
├── simulation_loop.py          # Generic simulation orchestrator
├── dynamics.py                 # VISPA constrained dynamics (Pinocchio)
├── convert.py                  # MuJoCo ↔ Pinocchio state conversion
├── ik.py                       # Inverse kinematics for dock configurations
│
├── solvers/
│   ├── centroidal_nmpc.py      # Stage 1: CasADi/IPOPT centroidal NMPC
│   ├── wholebody_qp.py         # Stage 2: qpOASES whole-body QP
│   ├── nmpc_solver.py          # Generic NMPC backend (CasADi NLP)
│   ├── hierarchical_qp.py     # Generic HQP backend
│   └── contact_phase.py       # Contact definitions, momentum map
│
├── lutze_baseline/
│   ├── sim_lutze.py            # Lutze simulation (Stage 1: LutzeQP)
│   ├── lutze_qp.py             # Single-step QP wrench optimizer
│   ├── lutze_feedforward.py    # PD-based feedforward wrench computation
│   ├── centroidal_model.py     # Centroidal rigid-body reduction
│   ├── contact_adjoint.py      # SE(3) adjoint wrench mapping
│   ├── momentum_map.py         # Contact wrenches → angular momentum rate
│   ├── plot_comparison.py      # Side-by-side comparison plots
│   ├── test_phase0.py          # Unit tests: centroidal model
│   └── test_phase1.py          # Unit tests: full pipeline
│
├── models/
│   ├── VISPA_crawling.xml      # MuJoCo model (structure + anchors)
│   └── VISPA_crawling_fixed.urdf  # Fixed-base URDF for Pinocchio
│
├── URDF_models/                # Original URDF models + meshes
├── results/                    # Pre-computed figures and comparison plots
│
├── test_integration.py         # Integration tests: full pipeline
├── test_torso_task.py          # Torso 6D task unit tests
├── test_multi_step.py          # Multi-step locomotion tests
│
├── requirements.txt            # Python dependencies
├── lutze_paper.pdf             # Lutze et al. (2023) reference paper
└── VISPA_Controller_Documentation.pdf  # Controller documentation
```


## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `TORSO_MASS` | 40 kg | Central torso mass |
| `TAU_MAX` | 10 Nm | Space-grade actuator limit |
| `T_SWING` | 6.0 s | Single-support phase duration |
| `TORSO_FRAC` | 0.70 | Fraction of full IK displacement |
| `WELD_R` | 5 mm | Docking success threshold |
| `DT_NMPC` / `DT_LUTZE` | 0.1 s | Stage 1 control rate |
| `DT_MJ` | 0.01 s | MuJoCo timestep / Stage 2 rate |


## References

- Lutze et al. (2023), "Optimization of multi-arm robot locomotion to reduce satellite disturbances during in-orbit assembly", IEEE Aerospace Conference.
- Bellicoso et al. (2018), "Dynamic locomotion through online nonlinear motion optimization", IEEE-RA Letters.
- Orin et al. (2013), "Centroidal dynamics of a humanoid robot", Autonomous Robots.
