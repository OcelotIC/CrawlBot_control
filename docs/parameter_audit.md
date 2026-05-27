> **⚠ SUPERSEDED (2026-05-27).** This file is stale — it predates or does not
> track the reworked controller. For what the code actually does,
> `docs/architecture/STACK_OVERVIEW.md` is the code-ground-truth reference and
> supersedes any current-state claim here (e.g. the NMPC is 9-state
> `[r_com,v_com,L_com]`, not 12; module APIs and parameters have changed).

# Parameter Audit — M-1.4

Generated: 2026-04-10 | Reference: SimConfig (`crawlbot/simulation/config.py`)

## Single Source of Truth

`SimConfig` is the canonical parameter set. When `SimulationLoop` runs, it
passes SimConfig values to all subsystems (NMPC, QP, swing planner, AOCS),
overriding their local defaults. The local defaults in `CentroidalNMPCConfig`,
`WholeBodyQPConfig`, etc. are fallbacks for standalone usage only.

## SimConfig vs Spec Cross-Reference

| Parameter | SimConfig | Spec (§) | Match? |
|-----------|-----------|----------|--------|
| Robot mass | ~71 kg (computed) | §0.4: ~71 kg | Yes |
| hw_max | ±5.0 Nms | §4.6: ±5 Nms | Yes |
| tau_w_max | 5.0 Nm | §5.1: 5 Nm | Yes |
| dt_nmpc | 0.1 s | §0.5: 0.1 s | Yes |
| dt_qp | 0.01 s | §0.5: 0.01 s | Yes |
| NMPC horizon N | 8 | §5.1: 8 | Yes |
| tau_max (joint) | 20.0 Nm | SimConfig | — |
| L_max | 10.0 Nms | SimConfig | — |
| rwa_I_w | 0.01 kg·m² | §4.6 | Yes |

## Discrepancies Found

### 1. Swing Clearance Default (cosmetic)

- `swing_planner.py:40` — `DEFAULT_CLEARANCE = 0.08` m
- `config.py:111` — `swing_clearance = 0.03` m
- **Impact**: None at runtime (SimConfig overrides via sim_loop.py:123)
- **Action**: Updated DEFAULT_CLEARANCE to match SimConfig for consistency

### 2. Solver Config Defaults vs SimConfig

These defaults in solver Config dataclasses differ from SimConfig but are
always overridden when run through SimulationLoop:

| Config class | Field | Default | SimConfig | Overridden? |
|-------------|-------|---------|-----------|-------------|
| CentroidalNMPCConfig | robot_mass | 90.0 | computed ~71 | Yes (sim_loop:178) |
| CentroidalNMPCConfig | N | 20 | nmpc_N=8 | Yes (sim_loop:176) |
| CentroidalNMPCConfig | dt | 0.05 | nmpc_dt=0.1 | Yes (sim_loop:177) |
| CentroidalNMPCConfig | f_max | 3000.0 | nmpc_f_max=25.0 | Yes (sim_loop:180) |
| CentroidalNMPCConfig | tau_max | 300.0 | nmpc_tau_max=8.0 | Yes (sim_loop:180) |
| WholeBodyQPConfig | tau_max | 50.0 | tau_max=20.0 | Yes (sim_loop:259) |
| RobotInterface | tau_max | 10.0 | — | Not used for clipping |

### 3. IK Hardcoded Parameters (ik.py)

These are internal to the IK solver and not physics parameters:

| Value | Location | Purpose |
|-------|----------|---------|
| 1e-4 | lines 107,398,412 | Arm Jacobian regularization |
| 1e-3 | lines 115,401,415 | Base Jacobian regularization |
| 2000 | lines 162,256 | Max dock IK iterations |
| 500 | line 214 | Max inner IK iterations |
| 300 | line 386 | Max waypoint IK iterations |
| 0.3, 0.6 | line 242 | Multi-start height offsets |

These are solver internals, not physics parameters. They do not belong in
SimConfig but should be documented if they cause convergence issues.

### 4. Torso Planner Ramp Fraction

- `torso_planner.py:232` — `ramp = 0.35`
- Not in SimConfig. This is a trajectory shape parameter (35% of duration for
  trapezoidal ramp-up/down). Could be added if tuning is needed.

## MJCF vs SimConfig

| Parameter | MJCF (rwa3.xml) | SimConfig |
|-----------|-----------------|-----------|
| RW ctrlrange | ±5 Nm | tau_w_max=5.0 Nm | Match (fixed in M-1.1) |
| RW armature | 0.01 | rwa_I_w=0.01 | Match |
| Joint ctrlrange | ±50 Nm | tau_max=20.0 Nm | MJCF wider (software-limited) |
