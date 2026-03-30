# VISPA Whole-Body Controller

Hierarchical whole-body controller for the **VISPA dual-arm crawling robot** operating in microgravity on a free-floating structure, with reaction wheel attitude control (AOCS).

## Architecture

```
                    Structure frame (all internal quantities)
                         │
TorsoPlanner ──→ CoM ref ──→ CentroidalNMPC (10 Hz)
                                    │
                          r_com_plan, v_com_plan
                          λ_ref, a_com_ff
                                    │
                                    ▼
                             WholeBodyQP (100 Hz)
                                    │
                              τ_q (12 joints)
                                    │
                      ┌─────────────┼──────────────┐
                      ▼             ▼              ▼
                MuJoCo ctrl   AOCS (τ_w)    Weld constraints
                 [0:12]        [12:15]       (contact forces)
```

**Two-stage cascade**: centroidal NMPC plans momentum-feasible trajectories,
whole-body QP tracks them at high rate. All quantities are in **structure body
frame** — conversions to/from MuJoCo world frame happen only at the interfaces.

3 orthogonal reaction wheels absorb robot angular momentum to maintain
spacecraft pointing during crawling.


## Quick Start

```bash
pip install pin casadi mujoco numpy matplotlib

# Run 3-step crawling simulation
PYTHONPATH=. MUJOCO_GL=disabled python scripts/compare_aocs.py

# Run tests
PYTHONPATH=. MUJOCO_GL=disabled python tests/test_r5fix_rwa.py
PYTHONPATH=. MUJOCO_GL=disabled python tests/test_force_estimator.py
```


## Package Structure

```
crawlbot/                            # Main package
├── core/
│   ├── robot_interface.py           # Pinocchio wrapper (CRBA, RNEA, Jacobians, CoM)
│   ├── state_conversions.py         # MuJoCo ↔ Pinocchio frame conversions
│   └── ik.py                        # Inverse kinematics for dock configurations
│
├── planning/
│   ├── contact_scheduler.py         # Gait plan, phase timing, anchor management
│   ├── torso_planner.py             # 6D torso trajectory (quintic + SLERP)
│   ├── swing_planner.py             # End-effector swing with clearance bump
│   └── locomotion_planner.py        # CoM reference trajectory
│
├── solvers/
│   ├── centroidal_nmpc.py           # Stage 1 — Centroidal NMPC (CasADi/IPOPT)
│   ├── wholebody_qp.py             # Stage 2 — Whole-body QP (qpOASES)
│   ├── nmpc_solver.py              # Generic NMPC backend
│   ├── hierarchical_qp.py          # Generic HQP backend
│   └── contact_phase.py            # Contact config and momentum map
│
├── aocs/
│   └── force_estimator.py          # H_{r/O} momentum disturbance estimator
│
├── dynamics/
│   └── constrained_dynamics.py     # SHAKE/RATTLE constrained forward dynamics
│
└── simulation/
    ├── config.py                   # SimConfig dataclass (all parameters)
    ├── logging.py                  # SimLog dataclass (time-series data)
    ├── sim_loop.py                 # SimulationLoop (NMPC+QP+AOCS orchestration)
    └── plotting.py                 # 9-panel diagnostic plot

scripts/                             # Executable scripts
├── compare_aocs.py                  # AOCS variant comparison at configurable mass ratio
├── sim_torso6d.py                   # Single-step CNMPC simulation
├── run_r6_full_sim.py               # Multi-step simulation pipeline
└── run_r7_figures.py                # Publication figure generation

models/                              # MuJoCo MJCF + Pinocchio URDF
├── VISPA_crawling_rwa3.xml          # Default: 3 RWA, 7110 kg structure
├── VISPA_crawling_rwa3_8pct.xml     # 8% ratio: 888 kg structure
├── VISPA_crawling_rwa3_8pct_hw50.xml  # 8% ratio, 50 Nms wheels
├── VISPA_crawling.xml               # No wheels variant
└── VISPA_crawling_fixed.urdf        # Robot-only URDF for Pinocchio

tests/                               # Validation suite
├── test_r5fix_rwa.py                # RWA model + docking (23 tests)
├── test_force_estimator.py          # H_{r/O} estimator + sign validation (12 tests)
├── test_integration.py              # Full pipeline integration
├── test_r5_closed_loop.py           # 3-step closed-loop validation
├── test_multi_step.py               # Multi-step locomotion
└── ...

docs/                                # Technical documentation
├── controller_architecture.md       # NMPC+QP data flow and constraints
├── momentum_conservation_analysis.md  # Orbital correction derivation
├── force_estimator_note.md          # H_{r/O} estimator theory and validation
├── status_report.md                 # Session status and open problems
└── ...

lutze_baseline/                      # Reference implementation (Lutze et al. 2023)
results/                             # Simulation logs (JSON) and figures (PNG)
```

Root-level `.py` files are **compatibility shims** that re-export from `crawlbot/`
so existing scripts and tests work without import changes.


## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Robot mass | ~71 kg | Torso (40 kg) + 2 arms (31 kg) |
| Structure mass | 500–71000 kg | Configurable (8%–0.1% ratio) |
| `tau_max` | 20 Nm | Joint torque limit |
| `hw_min/max` | ±5 Nms | Reaction wheel momentum envelope |
| `tau_w_max` | 5 Nm | Wheel torque rate limit |
| `L_max` | 10 Nms | Robot angular momentum limit |
| `dt_nmpc` | 0.1 s | Stage 1 rate (10 Hz) |
| `dt_qp` | 0.01 s | Stage 2 rate (100 Hz) |
| `t_swing` | 6.0 s | Single-support phase duration |
| `weld_radius` | 5 mm | Docking success threshold |


## AOCS — Reaction Wheel Attitude Control

The robot's motion generates angular momentum that must be absorbed by the
spacecraft reaction wheels to maintain pointing. Two components:

- **Spin** (`L_com`): centroidal angular momentum from joint motion
- **Orbital** (`r_com × m·v_com`): from CoM translation along the structure

The AOCS provides feedforward rejection of the spin component via
`τ_w = -L̇_com_est - K_hw · hw_error`. The orbital component is managed by
the NMPC's trajectory planning (hw box constraint with corrected dynamics).

See `docs/force_estimator_note.md` for the full theoretical derivation.


## Simulation Results (3-Step, 7110 kg structure)

| Metric | Value |
|--------|-------|
| Docks achieved | 3/3 |
| Max structure rotation | ~24° |
| Peak `\|\|hw\|\|` | 0.5 Nms |
| Peak `\|\|L_com\|\|` | 8.1 Nms |
| NMPC infeasibility | 3/447 (0.7%) |
| QP failures | 0 |


## References

- Lutze et al. (2023), "Optimization of multi-arm robot locomotion to reduce
  satellite disturbances during in-orbit assembly", IEEE Aerospace Conference.
- Orin et al. (2013), "Centroidal dynamics of a humanoid robot", Autonomous Robots.
- De Luca & Mattone (2005), "Sensorless robot collision detection and hybrid
  force/motion control", IEEE ICRA.
