# M7 — EE position bisection summary

Per `docs/architecture/M7_EE_POSITION_BISECTION.md`. Three
v21-baseline cases share initial state, contact config
(SINGLE_A), and SwingPlanner EE reference; they differ only
in which subsystems run during SS. Case D is read from
`Misc/runs/M7_abort_diag/R1_baseline/sim_log.json` (commit
`6128db9`) — no re-simulation. Metrics are computed over
`t ∈ [0, T_step]` with T_step = 7.284 s.

| case | description | ee_pos_peak_SS [mm] | ee_pos_at_T_step [mm] | tau_q_peak_SS [Nm] | Δ from prev [mm] |
|---|---|---|---|---|---|
| A_swing | standalone, SwingPlanner EE | 3.82 | 3.08 | 0.60 | — |
| B_minus | + NMPC, torso const | 4.59 | 3.73 | 0.60 | +0.77 |
| B_v21 | + mapping (planned-δ) | 164.79 | 135.84 | 2.54 | +160.20 |
| D | full sim_loop SS (from R1) | 162.38 | 120.92 | 3.16 | -2.41 |

## Invariants

All three v21 cases share: initial qpos checksum 5.3040941901,
T_step = 7.284331 s, identical contact config (SINGLE_A) and
SwingPlanner EE reference. NMPC/mapping call counts match the
spec: A_swing (0/0), B_minus (78/0), B_v21 (78/778).
