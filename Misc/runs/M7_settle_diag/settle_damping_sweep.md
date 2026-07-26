# M7 — DS settle damping/armature sweep

Diagnostic only. The MJCF file (`models/VISPA_crawling_rwa3.xml`) is mutated in-place per variant by writing new `damping` / `armature` values into the `default class="robot_joint"` block, then restored to the on-disk state observed at script entry. The settle loop mirrors `SimulationLoop._run_ds_passivity_loop` (`sim_loop.py:483-642`) one-for-one but logs five extra columns per QP tick. Constraint slack is computed as `-(dq_j·tau + 2α·T_jj)` from the QP's returned `tau`; positive = constraint inactive, ~0 = binding, negative = the constraint was violated by the solver. No qpOASES dual values are extracted (the wrapper does not expose them).

## Per-variant summary

| variant | damping | armature | T_settle [J] | T_start [J] | T_end [J] | exit_reason | n_steps |
|---|---|---|---|---|---|---|---|
| d0_a0 | 0.0 | 0.0 | 5.2240e-10 | 0.0000e+00 | 7.6224e-01 | plateau | 51 |
| d0p005_a0p005 | 0.005 | 0.005 | 5.2240e-10 | 0.0000e+00 | 2.1105e-01 | plateau | 51 |
| d0p01_a0p01 | 0.01 | 0.01 | 5.2240e-10 | 0.0000e+00 | 1.2616e-01 | plateau | 51 |
| d0p05_a0p05 | 0.05 | 0.05 | 5.2240e-10 | 0.0000e+00 | 7.3355e-12 | target_met | 11 |
| d0_a0p05 | 0.0 | 0.05 | 5.2240e-10 | 0.0000e+00 | 7.4623e-12 | target_met | 11 |

`alpha_passivity` (cfg.alpha_passivity) used in all variants = 1.

## Per-variant constraint behaviour (variant `d0_a0`)

`d0_a0` rows logged: 50.

Per-tick LHS sign breakdown:

- violated  (lhs > 1e-9 W):   0 / 50
- binding   (|lhs| ≤ 1e-9 W): 3 / 50
- inactive  (lhs < -1e-9 W):  47 / 50

`max(lhs)` = 0.0000e+00 W,  `min(slack)` = -0.0000e+00 W,  `max(|lhs|)` = 3.1005e+02 W.

## Overlay plot

`Misc/runs/M7_settle_diag/passivity_trace.png` — top: `T_full(t)` on a symlog scale; bottom: passivity LHS on a symlog scale (zero line marked).

## Per-variant trace files

| variant | trace |
|---|---|
| d0_a0 | `Misc/runs/M7_settle_diag/d0_a0/passivity_trace.csv` |
| d0p005_a0p005 | `Misc/runs/M7_settle_diag/d0p005_a0p005/passivity_trace.csv` |
| d0p01_a0p01 | `Misc/runs/M7_settle_diag/d0p01_a0p01/passivity_trace.csv` |
| d0p05_a0p05 | `Misc/runs/M7_settle_diag/d0p05_a0p05/passivity_trace.csv` |
| d0_a0p05 | `Misc/runs/M7_settle_diag/d0_a0p05/passivity_trace.csv` |

Columns: `k, t_s, T_full_J, T_jj_J, dq_dot_tau_W, alpha_times_T_jj_W, passivity_lhs_W, constraint_slack_W, tau_max_Nm`.

## Variant `d0_a0` is the trace requested by item (1)

CSV: `Misc/runs/M7_settle_diag/d0_a0/passivity_trace.csv` — aliased per spec to `Misc/runs/M7_settle_diag/passivity_trace.csv` (also written).
