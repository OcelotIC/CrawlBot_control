# M7 — DS settle d0_a0: power balance + dynamics residual

Diagnostic only, single variant `d0_a0` (damping=0, armature=0). The script mutates the MJCF in place and restores it on exit; the original on-disk state at script entry is preserved (try/finally).

## Run summary

| field | value |
|---|---|
| n_steps_run | 51 |
| T_start [J] | 0.000000e+00 |
| T_end [J] | 1.914203e-01 |
| T_settle threshold [J] | 5.224006e-10 |
| lambda_min(H_entry) | 1.044801e-03 |
| exit_reason | plateau |
| alpha (cfg.alpha_passivity) | 1.0 |
| (damping, armature) | (0, 0) |

## AOCS status during the settle

`SimulationLoop._run_ds_passivity_loop` (sim_loop.py:483-642) does not call `compute_aocs_command_legacy_corrected`; the wheel-actuator slot is hard-zeroed at line 632 (`self.mj_data.ctrl[n_j:n_j + 3] = 0.0`). The AOCS call sites are at sim_loop.py:1898 and :1917 inside `_step()`, which the inter-step settle bypasses entirely. This diagnostic mirrors that wiring.

- Unique values written to `mj_data.ctrl[n_j:n_j+3]` across the 50 ticks: `[(0.0, 0.0, 0.0)]`
- `compute_aocs_command_legacy_corrected` calls observed: 0 (function never invoked from the settle loop)

## Augmented per-tick columns (over the full trace)

Trace CSV: `results/M7_settle_diag/d0_a0/power_balance_trace.csv`. Columns: `k, t_s, T_full_J, T_jj_J, dq_dot_tau_W, alpha_times_T_jj_W, passivity_lhs_W, constraint_slack_W, tau_max_Nm, P_weld_W, P_aocs_W, dyn_residual_norm`.

### P_weld = (J_c · q̇)^T · λ  [W]

- min  = `+0.000000e+00`
- max  = `+2.941290e+02`
- mean = `+2.303287e+02`

### P_aocs = τ_w · ω_s  [W]

- min  = `+0.000000e+00`
- max  = `+0.000000e+00`
- mean = `+0.000000e+00`

(τ_w identically 0 throughout the settle — see AOCS status above.)

### Dynamics residual  ‖H·q̈ + C − B·τ_q − J_c^T·λ‖

q̈ via Pinocchio finite-difference: `qdd = (v_post - v) / dt_qp`. `B = [[0_{6×14}]; [I_{14×14}]]`. `H, C` from `RobotInterface.update`. `J_c` from `get_contact_jacobians(True, True)` (DS, both arms welded). `λ` from the QP's `lam_sol` return.

- min  = `2.462040e-05`
- max  = `3.560446e+02`
- mean = `2.689102e+02`

### Passivity LHS (for cross-reference)

- max(LHS) = `+0.000000e+00` W
- min(LHS) = `-3.037878e+02` W

## Selected per-tick samples

Every 5th row (first 11 entries) plus the last row.

| k | t [s] | T_full [J] | P_weld [W] | P_aocs [W] | dyn residual | passivity LHS [W] | tau_max [Nm] |
|---|---|---|---|---|---|---|---|
| 0 | 0.000 | 0.0000e+00 | +0.0000e+00 | +0.0000e+00 | 2.4620e-05 | +0.0000e+00 | 8.9484e-14 |
| 5 | 0.050 | 5.7432e-08 | +1.3555e-03 | +0.0000e+00 | 1.8206e+00 | -1.3601e-03 | 1.6096e-01 |
| 10 | 0.100 | 2.2270e-01 | +2.7278e+02 | +0.0000e+00 | 3.5604e+02 | -2.8418e+02 | 2.0000e+01 |
| 15 | 0.150 | 1.8448e-01 | +2.7648e+02 | +0.0000e+00 | 3.1840e+02 | -2.8575e+02 | 2.0000e+01 |
| 20 | 0.200 | 1.8595e-01 | +2.7720e+02 | +0.0000e+00 | 3.2035e+02 | -2.8565e+02 | 2.0000e+01 |
| 25 | 0.250 | 1.5914e-01 | +2.6941e+02 | +0.0000e+00 | 3.2169e+02 | -2.8025e+02 | 2.0000e+01 |
| 30 | 0.300 | 2.0619e-01 | +2.8036e+02 | +0.0000e+00 | 3.1819e+02 | -2.8795e+02 | 2.0000e+01 |
| 35 | 0.350 | 1.6497e-01 | +2.6780e+02 | +0.0000e+00 | 3.1569e+02 | -2.7873e+02 | 2.0000e+01 |
| 40 | 0.400 | 2.0551e-01 | +2.7678e+02 | +0.0000e+00 | 3.1422e+02 | -2.8508e+02 | 2.0000e+01 |
| 45 | 0.450 | 1.8708e-01 | +2.7253e+02 | +0.0000e+00 | 3.1020e+02 | -2.8150e+02 | 2.0000e+01 |
| 49 | 0.490 | 2.0242e-01 | +2.7565e+02 | +0.0000e+00 | 3.0877e+02 | -2.8310e+02 | 2.0000e+01 |
