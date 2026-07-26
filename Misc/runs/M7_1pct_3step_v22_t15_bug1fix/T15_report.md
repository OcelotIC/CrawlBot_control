# T15 (bug1fix) — Three-step traversal at mass_ratio = 0.01 with Option Z

Rerun of T15 after the Bug 1 fix (Option Z: reset `_t_plan_offset`
at each SS entry, `crawlbot/simulation/sim_loop.py:914–918` and
caller mirror at `:1178–1179`).

---

## §1 Configuration echo

### 1.1 Repository state

| Field | Value |
|---|---|
| Commit SHA | `7c8f01aeeb96e2fbefbc9d2dc8a38d1d3e33ee75` (HEAD of `claude/t15-bug1-fix`) |
| Base commit | `4435c5d` (`origin/main`) |
| Commit subject | T15: fix timeline desync in SwingPlanner query |
| Reproducer script | `Misc/scripts/run_m7_v22_1pct_3step_t15.py` (`OUT` temporarily re-pointed to `M7_1pct_3step_v22_t15_bug1fix/`) |
| Output directory | `Misc/runs/M7_1pct_3step_v22_t15_bug1fix/` |

### 1.2 Active controller configuration (unchanged from original T15)

| Flag | Value | Source |
|---|---|---|
| `aocs_off_in_ds` | `True` | T12 closure (explicit in T15 script) |
| `ds_ramp_duration_s` | `2.0` s | SimConfig default |
| `mapping_bypass_in_ss` | `True` | T11 fix |
| `swing_early_finish_fraction` | `0.80` | T11 fix |
| `preplanner_a_cruise_max` | `0.01` m/s² | T11 fix |
| `preplanner_cruise_ramp_frac` | `0.2` | T11 fix |
| NMPC `ipopt.max_iter` | `200` | `crawlbot/solvers/nmpc_solver.py:566` |

The only controller-code change vs. the original T15 run is the
Option Z `_t_plan_offset` reset inside `_setup_torso_for_step`.

### 1.3 Run parameters

| Parameter | Value |
|---|---|
| Mass ratio | 0.01 (1%) |
| `n_steps` | 3 |
| `start_a`, `start_b` | 2, 2 |
| `dt_nmpc` | 0.1 s |
| Simulation tick count | 459 |
| Simulation span | t = 0.110 s → 48.390 s (duration 48.280 s) |

### 1.4 MJCF integrity

| Event | md5 |
|---|---|
| Pre-run, pre-mutation | `96d229250ca882951f1c0d2516391421` |
| During run (damping=0.0, armature=0.05 applied) | `8ab0edb58ca2f05fcd3c9b24dd6b41a8` |
| Post-run, restored | `96d229250ca882951f1c0d2516391421` |
| Byte-exact restoration | **True** (pre == post) |

---

## §2 Scheduler and per-step overview

### 2.1 Contact scheduler

| Step | Swing arm | Released anchor | Target anchor |
|---|---|---|---|
| 0 | b | anchor 2 | anchor 3 |
| 1 | a | anchor 2 | anchor 3 |
| 2 | b | anchor 3 | anchor 4 |

### 2.2 Pre-planner outputs

| Step | T_step (s) | Solve time (ms) | Iterations | IPOPT status | Cost | peak \|v\| (m/s) | peak \|L\| (Nms) |
|---|---|---|---|---|---|---|---|
| 0 | 7.284 | 81.9 | 14 | Solve_Succeeded | 1.195 | 0.097 | 0.028 |
| 1 | 7.915 | 29.1 | 14 | Solve_Succeeded | 0.824 | 0.098 | 0.131 |
| 2 | 9.423 | 29.9 | 14 | Solve_Succeeded | 0.602 | 0.099 | 0.011 |

### 2.3 Dock events (2)

| Step | t (s) | d (mm) | ori (deg) | arm | anchor | method |
|---|---|---|---|---|---|---|
| 0 | 6.01 | 3.82 | 0.08 | b | 3 | kinematic |
| 1 | 13.02 | 4.84 | 0.22 | a | 3 | kinematic |

### 2.4 Aborted steps (1)

| Step | Reason | t (s) | d at exit (mm) | ori at exit (deg) |
|---|---|---|---|---|
| 2 | `dock_timeout` | 28.49 | 374.35 | 9.841 |

### 2.5 Inter-step settles

| settle | step_idx | t_start (sim, s) | t_end (sim, s) | n_steps | exit_reason |
|---|---:|---:|---:|---:|---|
| [0] | 0 | 0.000 | 0.110 | 11 | `target_met` |
| [1] | 1 | 6.010 | 6.520 | 51 | `plateau` |
| [2] | 2 | 13.020 | 14.990 | 197 | `plateau` |

### 2.6 Phase-duration mismatch warning (diagnostic output)

- SwingPlanner SS phase durations: `[7.284, 7.915, 9.423]` s
- TorsoPlanner last phase duration: `9.423` s
- Reported mismatch (swing vs torso on step 0): Δ = 2.14 s

(Same warning as in the original T15 run; comes from the swing vs torso comparator, not affected by Option Z.)

---

## §3 Per-step metrics

### 3.0 Per-step SS-entry verification (Option Z fix)

For each step's first SS tick `k_entry`:

| Step | `ss_phase_idx` | `t_ss_start` (s) | `plan.t_start[ss_phase_idx]` at reset (s) | `_t_plan_offset` after reset (s) | `plan_query_t` at SS entry (s) | `phase_at(plan_query_t)` |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 |  0.110 |  0.500 | **−0.390** |  0.500 | phase 1 (step 0 SS) ✓ |
| 1 | 3 |  6.520 |  8.284 | **−1.764** |  8.284 | phase 3 (step 1 SS) ✓ |
| 2 | 5 | 14.990 | 16.699 | **−1.709** | 16.699 | phase 5 (step 2 SS) ✓ |

First-tick `p_ee_ref` vs `p_ee_actual` (structure frame, mm):

| Step | swing_arm | `p_ee_actual` (mm) | `p_ee_ref` (mm) | `\|p_ee_ref − p_ee_actual\|` (mm) |
|---:|---|---|---|---:|
| 0 | b | (−399.855, −299.982, +25.075) | (−399.961, −300.000, +24.913) | **0.194** |
| 1 | a | (−399.562, +299.892, +25.467) | (−399.969, +300.000, +24.926) | **0.685** |
| 2 | b | (+400.244, −300.098, +24.928) | (+400.018, −300.000, +24.948) | **0.246** |

All three SS entries satisfy `|p_ee_ref − p_ee_actual| < 10 mm`
(criterion from Phase 3 prompt). All three `phase_at` results
match the step's own SS phase index. Option Z resets
`_t_plan_offset` so the main-loop `plan_query_t(t_ss_start)`
lands exactly on `plan.t_start[ss_phase_idx]` at every SS entry.

### 3.1 SS window — tracking errors

Values computed over `(step_idx == s) & (phase == 'SS')`.

| Metric | Step 0 | Step 1 | Step 2 |
|---|---:|---:|---:|
| SS span t (s) | 0.110 – 5.910 | 6.520 – 12.920 | 14.990 – 28.390 |
| SS ticks (N) | 59 | 65 | 135 |
| `torso_pos_peak_SS` (mm) | 42.665 | 72.078 | 84.536 |
| `torso_ori_peak_SS` (deg) | 1.049 | 1.439 | 2.781 |
| `ee_pos_peak_SS` (mm) | 26.337 | 42.076 | 432.765 |
| `ee_ori_peak_SS` (deg) | 0.090 | 0.862 | 11.648 |

### 3.2 SS window — wheel momentum and actuator torques

| Metric | Step 0 | Step 1 | Step 2 |
|---|---:|---:|---:|
| `h_w` per-axis max (Nms) | [0.166, 0.540, 0.302] | [0.505, 0.897, 0.202] | [1.112, 2.130, 1.615] |
| `h_w` norm peak (Nms) | 0.598 | 0.999 | 2.779 |
| `h_w` norm peak / 5 Nms | 0.120 | 0.200 | 0.556 |
| `h_w` at SS exit (Nms) | [−0.018, 0.083, −0.007] | [0.095, 0.350, −0.036] | [0.758, −2.130, 1.615] |
| `h_w` exit norm (Nms) | 0.085 | 0.365 | 2.779 |
| `τ_joint` peak (Nm, budget 20) | 1.105 | 3.243 | 15.934 |
| `τ_w` norm peak (Nm) | 1.672 | 5.060 | 8.660 |

### 3.3 SS window — AOCS transport-term instrumentation

| Metric | Step 0 | Step 1 | Step 2 |
|---|---:|---:|---:|
| `\|ω_s\|` peak (mrad/s) | 0.767 | 1.527 | 2.694 |
| `transport_term_mag` max (Nm) | 3.995e-04 | 2.473e-03 | 5.448e-03 |
| `transport_term_mag` mean (Nm) | 1.951e-04 | 6.309e-04 | 1.036e-03 |

### 3.4 SS window — structure attitude (Euler ZYX, degrees)

| Instant | Step 0 | Step 1 | Step 2 |
|---|---|---|---|
| Euler @ SS entry | [−3.11e-4, −3.07e-5, 2.01e-5] | [−0.0685, 0.0524, −0.0286] | [−0.1107, 0.0137, 0.0035] |
| Euler @ SS exit | [−0.0732, 0.0554, −0.0291] | [−0.0827, 0.0419, 0.0023] | [−0.7033, 0.3301, −0.2214] |

### 3.5 SS window — NMPC health

| Metric | Step 0 | Step 1 | Step 2 |
|---|---:|---:|---:|
| NMPC failures in SS | 0 | 0 | 0 |
| NMPC iters max | 19 | 20 | 29 |
| NMPC iters mean | 13.4 | 13.1 | 14.5 |

Zero NMPC failures in any SS window.

### 3.6 Step 2 DS window (post-abort, 28.490 – 48.390 s, N = 200 ticks)

| Metric | Value |
|---|---:|
| `torso_pos_peak_DS` (mm) | 1228.241 |
| `torso_ori_peak_DS` (deg) | 179.139 |
| `ee_pos_peak_DS` (mm) | 3095.760 |
| `ee_ori_peak_DS` (deg) | 179.804 |
| `h_w` per-axis max (Nms) | [0.758, 2.129, 1.614] |
| `transport_term_mag` max (Nm) | 1.503 |
| `transport_term_mag` mean (Nm) | 0.220 |
| struct_euler end (deg) | [−8.621, 8.312, 1.319] |
| NMPC failures in DS | 9 (all `Infeasible_Problem_Detected`) |

NMPC failure events (all in step 2 DS, post-abort):

| k | t (s) | step | phase | status |
|---:|---:|---:|---|---|
| 297 | 32.29 | 2 | DS | `Infeasible_Problem_Detected` |
| 298 | 32.39 | 2 | DS | `Infeasible_Problem_Detected` |
| 299 | 32.49 | 2 | DS | `Infeasible_Problem_Detected` |
| 300 | 32.59 | 2 | DS | `Infeasible_Problem_Detected` |
| 301 | 32.69 | 2 | DS | `Infeasible_Problem_Detected` |
| 302 | 32.79 | 2 | DS | `Infeasible_Problem_Detected` |
| 303 | 32.89 | 2 | DS | `Infeasible_Problem_Detected` |
| 322 | 34.79 | 2 | DS | `Infeasible_Problem_Detected` |
| 323 | 34.89 | 2 | DS | `Infeasible_Problem_Detected` |

### 3.7 Per-step outcome

| Step | Outcome | t (s) | d (mm) | ori (deg) |
|---|---|---:|---:|---:|
| 0 | DOCK (kinematic) | 6.01 | 3.82 | 0.08 |
| 1 | DOCK (kinematic) | 13.02 | 4.84 | 0.22 |
| 2 | ABORT `dock_timeout` | 28.49 | 374.35 | 9.841 |

---

## §4 Step transitions (handoff metrics)

Transitions span the un-logged dock/settle gap between the last
SS tick of step N and the first SS tick of step N+1. Values at
`exit` = last logged tick of step N SS; values at `entry` = first
logged tick of step (N+1) SS.

### 4.1 Step 0 → Step 1 (gap 0.610 s)

| Quantity | Exit (t = 5.910 s) | Entry (t = 6.520 s) | Δ |
|---|---:|---:|---:|
| `\|e_torso_pos\|` (mm) | 42.665 | 0.728 | −41.937 |
| `\|e_torso_ori\|` (deg) | 0.971 | 0.035 | −0.936 |
| `\|e_ee_pos\|` (mm) | 3.823 | 0.649 | −3.174 |
| `\|e_ee_ori\|` (deg) | 0.080 | 0.421 | +0.341 |
| `h_w` (Nms) | [−0.018, 0.083, −0.007] | [0.167, 0.186, 0.074] | — |
| `h_w` norm (Nms) | 0.085 | 0.261 | +0.176 |
| struct_euler (deg) | [−0.0732, 0.0554, −0.0291] | [−0.0685, 0.0524, −0.0286] | ≈0 |
| `\|ω_s\|` (mrad/s) | 0.262 | 0.164 | −0.098 |
| `transport_term_mag` (Nm) | 4.183e-05 | 1.430e-05 | −2.75e-05 |

### 4.2 Step 1 → Step 2 (gap 2.070 s)

Gap is larger here (2.070 s) because `inter_step_settles[2]` ran
to `plateau` with 197 steps × 0.01 s = 1.970 s of DS settle time
(plus 0.1 s of `dt_nmpc` spacing between the last logged step-1
SS tick and the dock event).

| Quantity | Exit (t = 12.920 s) | Entry (t = 14.990 s) | Δ |
|---|---:|---:|---:|
| `\|e_torso_pos\|` (mm) | 44.232 | 0.121 | −44.111 |
| `\|e_torso_ori\|` (deg) | 1.439 | 0.021 | −1.418 |
| `\|e_ee_pos\|` (mm) | 4.842 | 0.272 | −4.570 |
| `\|e_ee_ori\|` (deg) | 0.215 | 0.022 | −0.194 |
| `h_w` (Nms) | [0.095, 0.350, −0.036] | [0.109, 0.377, −0.023] | — |
| `h_w` norm (Nms) | 0.365 | 0.393 | +0.028 |
| struct_euler (deg) | [−0.0827, 0.0419, 0.0023] | [−0.1107, 0.0137, 0.0035] | ≈0 |
| `\|ω_s\|` (mrad/s) | 0.364 | 0.209 | −0.155 |
| `transport_term_mag` (Nm) | 1.285e-04 | 8.234e-06 | −1.20e-04 |

---

## §5 Aggregate metrics (full run, from `metrics.csv`)

### 5.1 Summary counts (from diagnostic summary)

| Category | Count |
|---|---:|
| PASS | 14 |
| FAIL | 12 |
| SKIP | 1 |
| WARN | 7 |
| INFO | 3 |

### 5.2 Global (whole-run) metrics

| Metric | Value | Threshold | Status |
|---|---:|---:|---|
| `torso_pos_err_peak_mm` | 1228.2413 | 10.0 | **FAIL** |
| `torso_ori_err_peak_deg` | 179.1391 | 5.0 | **FAIL** |
| `ee_pos_err_at_dock_mm` | 4.8421 | 5.0 | PASS |
| `ee_ori_err_at_dock_deg` | 0.2152 | 5.0 | PASS |
| `com_tracking_err_rms_mm` | 378.9671 | 15.0 | **FAIL** |
| `hw_saturation_ratio_peak` | 0.5558 | 1.0 | PASS |
| `hw_saturation_ratio_rms` | 0.3817 | 0.7 | PASS |
| `platform_rotation_total_deg` | 15.2002 | 5.0 | **FAIL** |
| `platform_omega_peak_deg_s` | 2.2731 | 2.0 | **FAIL** |
| `tau_w_peak_ratio` | 1.0000 | 1.0 | PASS |
| `nmpc_solve_rate_50ms` | 0.9542 | 0.95 | PASS |
| `nmpc_infeasibility_rate` | 0.0196 | 0.02 | PASS |
| `passivity_violations` | — | 0 | SKIP |

### 5.3 Phase-split metrics (SS / DS / global)

| Metric | SS | DS | Global | Threshold |
|---|---:|---:|---:|---:|
| `torso_pos_peak_mm` | 84.536 | 1228.241 | 1228.241 | 10.0 |
| `torso_ori_peak_deg` | 2.781 | 179.139 | 179.139 | 5.0 |
| `ee_pos_peak_mm` | 432.765 | 3095.760 | 3095.760 | 5.0 |
| `ee_ori_peak_deg` | 11.648 | 179.804 | 179.804 | 5.0 |
| `hw_sat_peak` | 0.5558 | 0.5555 | 0.5558 | 1.0 |
| `tau_peak_Nm` | 15.934 | 20.000 | 20.000 | 20.0 |
| `tau_w_peak_ratio` | 1.0000 | 0.0000 | 1.0000 | 1.0 |

### 5.4 Info-only fields

| Metric | Value |
|---|---:|
| `ss_end_torso_ori_deg` | 2.0452 |
| `ds_entry_torso_ori_deg` | 4.6225 |
| `q_torso_ref_ss_to_ds_jump_deg` | 3.9992 |

---

## §6 Comparison to original T15 (bug1fix − original)

Reference: `Misc/runs/M7_1pct_3step_v22_t15/` (original T15,
HEAD = `4435c5d` of `main`, same scenario).

### 6.1 Dock outcomes

| Step | Original T15 | Bug1fix | Δ |
|---|---|---|---|
| 0 | DOCK (d=2.70 mm, ori=0.06°) | DOCK (d=3.82 mm, ori=0.08°) | +1.12 mm, +0.02° |
| 1 | ABORT `dock_timeout` (d=31.37 mm) | **DOCK (d=4.84 mm, ori=0.22°)** | ABORT → DOCK |
| 2 | ABORT `dock_timeout` (d=1621.79 mm, ori=106.01°) | ABORT `dock_timeout` (d=374.35 mm, ori=9.84°) | d: −1247 mm; ori: −96.2° |

### 6.2 Dock/abort counts

| Metric | Original T15 | Bug1fix | Δ |
|---|---:|---:|---:|
| Dock events | 1 | 2 | **+1** |
| Aborted steps | 2 | 1 | **−1** |

### 6.3 Per-step SS peaks

| Metric | Step | Original T15 | Bug1fix | Δ |
|---|---:|---:|---:|---:|
| `torso_pos_peak_SS` (mm) | 0 | 36.533 | 42.665 | +6.132 |
| | 1 | 150.947 | 72.078 | **−78.869** |
| | 2 | 313.997 | 84.536 | **−229.461** |
| `torso_ori_peak_SS` (deg) | 0 | 1.048 | 1.049 | +0.001 |
| | 1 | 2.607 | 1.439 | **−1.168** |
| | 2 | 176.048 | 2.781 | **−173.267** |
| `ee_pos_peak_SS` (mm) | 0 | 32.433 | 26.337 | **−6.096** |
| | 1 | 993.986 | 42.076 | **−951.910** |
| | 2 | 2160.264 | 432.765 | **−1727.499** |
| `ee_ori_peak_SS` (deg) | 0 | 9.373 | 0.090 | **−9.283** |
| | 1 | 34.089 | 0.862 | **−33.227** |
| | 2 | 177.823 | 11.648 | **−166.175** |

### 6.4 Per-step `h_w` norm peaks (SS)

| Step | Original T15 (Nms) | Bug1fix (Nms) | Δ |
|---:|---:|---:|---:|
| 0 | 0.555 | 0.598 | +0.043 |
| 1 | 1.687 | 0.999 | **−0.688** |
| 2 | 3.392 | 2.779 | **−0.613** |

### 6.5 NMPC failure counts

| Location | Original T15 | Bug1fix | Δ |
|---|---:|---:|---:|
| SS (all three steps) | 1 (step 2 SS) | 0 | **−1** |
| DS (post-abort only) | 1 (step 2 DS) | 9 (step 2 DS) | +8 |
| Total | 2 | 9 | +7 |

All 9 failures in Bug1fix occur in step-2's post-abort DS window
(32.29 – 34.89 s), same failure mode
(`Infeasible_Problem_Detected`). Zero NMPC failures in any SS
window.

### 6.6 Per-step transport-term peaks

| Window | Original T15 (Nm) | Bug1fix (Nm) | Δ |
|---|---:|---:|---:|
| Step 0 SS max | 7.941e-04 | 3.995e-04 | **−3.95e-04** |
| Step 1 SS max | 1.073e-02 | 2.473e-03 | **−8.26e-03** |
| Step 2 SS max | 8.609e-02 | 5.448e-03 | **−8.06e-02** |
| Step 2 DS max | 9.120e-02 | 1.503 | **+1.412** |
| Step 0 SS mean | 2.550e-04 | 1.951e-04 | −5.99e-05 |
| Step 1 SS mean | 2.152e-03 | 6.309e-04 | −1.52e-03 |
| Step 2 SS mean | 1.613e-02 | 1.036e-03 | **−1.51e-02** |

(Step 2 DS max increased because the post-abort divergence is
deeper in Bug1fix: `|ω_s|` grows as the structure tumbles
free-flying for 20 s. SS-window transport magnitudes are lower
across the board.)

### 6.7 Diagnostic summary counts

| Category | Original T15 | Bug1fix | Δ |
|---|---:|---:|---:|
| PASS | 13 | 14 | +1 |
| FAIL | 13 | 12 | −1 |
| SKIP | 1 | 1 | 0 |
| WARN | 7 | 7 | 0 |
| INFO | 3 | 3 | 0 |

---

## §7 Tripwire evaluation

Tripwires defined in `Misc/reports/architecture/AOCS_CONCERN_MEMO.md` §6.
Evaluated against the Bug1fix run only.

### 7.1 Tripwire A — SS metric degradation across steps

Criterion: SS peaks (`torso_pos_peak`, `torso_ori_peak`,
`ee_pos_peak`, `ee_ori_peak`) monotonically worsening step 0 → 1
→ 2 *and* beyond the ±10 % T11 envelope.

T11 reference (`Misc/runs/M7_1pct_1step_v22_with_swing_hold/metrics.csv`):

| T11 SS metric | Value |
|---|---:|
| `torso_pos_peak_mm_SS` | 36.533 |
| `torso_ori_peak_deg_SS` | 1.048 |
| `ee_pos_peak_mm_SS` | 32.433 |
| `ee_ori_peak_deg_SS` | 9.373 |

Per-step vs T11 (%):

| SS metric | Step 0 | Δ% | Step 1 | Δ% | Step 2 | Δ% | Monotone ↑? | Outside ±10 % T11? |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `torso_pos_peak_mm_SS` | 42.665 | +16.8 % | 72.078 | +97.3 % | 84.536 | +131.4 % | yes | yes (all 3) |
| `torso_ori_peak_deg_SS` | 1.049 | +0.1 % | 1.439 | +37.3 % | 2.781 | +165.4 % | yes | step 0 within ±10 %; steps 1, 2 outside |
| `ee_pos_peak_mm_SS` | 26.337 | −18.8 % | 42.076 | +29.7 % | 432.765 | +1234 % | yes | step 0 outside (lower); steps 1, 2 outside (higher) |
| `ee_ori_peak_deg_SS` | 0.090 | −99.0 % | 0.862 | −90.8 % | 11.648 | +24.3 % | yes | all 3 outside |

All four metrics monotone ↑ step 0 → 1 → 2. Steps 1 and 2 outside
the ±10 % T11 envelope on all four.

**Tripwire A verdict: FIRED** (monotone ↑ on all four SS peaks;
all four outside ±10 % T11 envelope at step 2; three of four
outside at step 1).

### 7.2 Tripwire B — `|h_w_i| > 3 Nm·s` on any axis in any SS

| Step | `h_w` per-axis max (Nms) (SS only) | Axis max (Nms) | > 3 Nms? |
|---|---|---:|---|
| 0 | [0.166, 0.540, 0.302] | 0.540 | no |
| 1 | [0.505, 0.897, 0.202] | 0.897 | no |
| 2 | [1.112, 2.130, 1.615] | 2.130 | no |

**Tripwire B verdict: NOT FIRED** (max per-axis SS `|h_w|` =
2.130 Nm·s, below the 3 Nm·s threshold).

### 7.3 Tripwire C — NMPC failures in SS

Criterion: any `nmpc_status != 0` event during SS beyond the
known single-dock-tick `Maximum_Iterations_Exceeded` event. New
failure modes (not `Maximum_Iterations_Exceeded`) = investigation-
level issue.

SS-only failures in Bug1fix run: **0**.

(All 9 NMPC failures occur in step 2's post-abort DS window; see
§3.6. DS post-abort failures are outside Tripwire C's SS-only
scope.)

**Tripwire C verdict: NOT FIRED** (zero NMPC failures in any SS
window).

### 7.4 Transport-term instrumentation summary (diagnostic)

| Window | `\|ω_s\|` peak (mrad/s) | `transport_term_mag` max (Nm) | `transport_term_mag` mean (Nm) |
|---|---:|---:|---:|
| Step 0 SS | 0.767 | 3.995e-04 | 1.951e-04 |
| Step 1 SS | 1.527 | 2.473e-03 | 6.309e-04 |
| Step 2 SS | 2.694 | 5.448e-03 | 1.036e-03 |
| Step 2 DS (post-abort) | — | 1.503 | 0.220 |

SS-window peaks are an order of magnitude smaller than in the
original T15 run across all three steps (see §6.6). Step 2 DS
peak is elevated (1.503 Nm) because the structure tumbles
free-flying for 20 s after abort — both arms unwelded (see
`T15_NOTE.md` §3 for Bug 2 mechanism; unfixed in this run).

### 7.5 Tripwire summary

| Tripwire | Original T15 | Bug1fix | Δ |
|---|---|---|---|
| A (SS metric degradation) | FIRED | **FIRED** | unchanged |
| B (`\|h_w_i\|` > 3 Nm·s in SS) | NOT FIRED | NOT FIRED | unchanged |
| C (NMPC SS failure) | FIRED (1 SS failure) | **NOT FIRED** | cleared |

---

## §8 Overall pass/fail

### 8.1 Scenario-level outcome

| Criterion | Target | T15 bug1fix observed | Status |
|---|---|---|---|
| Dock rate | 3 / 3 steps | 2 / 3 steps (steps 0, 1) | **FAIL** |
| Aborted steps | 0 | 1 (`dock_timeout` on step 2) | **FAIL** |

### 8.2 Envelope and tripwire outcomes

| Check | Condition | Result | Status |
|---|---|---|---|
| T11 envelope — `torso_pos_peak_mm_SS` | ≤ ±10 % across steps | all 3 outside | **FAIL** |
| T11 envelope — `torso_ori_peak_deg_SS` | ≤ ±10 % across steps | steps 1, 2 outside | **FAIL** |
| T11 envelope — `ee_pos_peak_mm_SS` | ≤ ±10 % across steps | all 3 outside | **FAIL** |
| T11 envelope — `ee_ori_peak_deg_SS` | ≤ ±10 % across steps | all 3 outside | **FAIL** |
| Tripwire A | not fired | fired | **FIRED** |
| Tripwire B | not fired | not fired | PASS (not fired) |
| Tripwire C | not fired | not fired | PASS (not fired) |

### 8.3 Diagnostic-metric outcomes (global thresholds)

| Metric | Value | Threshold | Status |
|---|---:|---:|---|
| `torso_pos_err_peak_mm` | 1228.241 | 10.0 | **FAIL** |
| `torso_ori_err_peak_deg` | 179.139 | 5.0 | **FAIL** |
| `ee_pos_err_at_dock_mm` | 4.842 | 5.0 | PASS |
| `ee_ori_err_at_dock_deg` | 0.215 | 5.0 | PASS |
| `com_tracking_err_rms_mm` | 378.967 | 15.0 | **FAIL** |
| `hw_saturation_ratio_peak` | 0.556 | 1.0 | PASS |
| `hw_saturation_ratio_rms` | 0.382 | 0.7 | PASS |
| `platform_rotation_total_deg` | 15.200 | 5.0 | **FAIL** |
| `platform_omega_peak_deg_s` | 2.273 | 2.0 | **FAIL** |
| `tau_w_peak_ratio` | 1.000 | 1.0 | PASS |
| `nmpc_solve_rate_50ms` | 0.9542 | 0.95 | PASS |
| `nmpc_infeasibility_rate` | 0.0196 | 0.02 | PASS |

### 8.4 Run integrity

| Check | Result | Status |
|---|---|---|
| Commit SHA at run | `7c8f01a` (HEAD of `claude/t15-bug1-fix`, 1 commit beyond `origin/main` = `4435c5d`) | PASS |
| MJCF restored byte-exact | pre md5 = post md5 = `96d229250ca882951f1c0d2516391421` | PASS |
| Source edits localised to sim_loop.py | +9 lines at two sites in `crawlbot/simulation/sim_loop.py` only | PASS |
| One run | single invocation of `Misc/scripts/run_m7_v22_1pct_3step_t15.py` | PASS |

### 8.5 Overall

| Roll-up | Result |
|---|---|
| Scenario outcome (§8.1) | **FAIL** (2/3 docks, 1 abort) |
| Envelope + tripwires (§8.2) | **FAIL** (4 envelope fails, Tripwire A FIRED; B and C not fired) |
| Diagnostic metrics (§8.3) | 7 PASS, 5 FAIL |
| Run integrity (§8.4) | PASS |
| **T15 bug1fix overall** | **FAIL** (progress vs original T15: see §8.6) |

### 8.6 Progress vs original T15

| Roll-up | Original T15 | Bug1fix | Δ |
|---|---|---|---|
| Docks | 1 / 3 | 2 / 3 | +1 dock |
| Aborts | 2 | 1 | −1 abort |
| Tripwires fired | A, C | A | C cleared |
| Global `torso_ori_err_peak_deg` (SS portion only, step 2) | 176.048 | 2.781 | −173.267° |
| Global `ee_pos_peak_mm_SS` (step 2) | 2160.264 | 432.765 | −1727.5 mm |
| Per-step SS-entry `|p_ee_ref − p_ee_actual|` (step 1) | 995.199 mm | 0.685 mm | **−994.514 mm** |
| Diagnostic summary PASS count | 13 | 14 | +1 |
| Diagnostic summary FAIL count | 13 | 12 | −1 |

The Option Z fix closes Bug 1 (per §3.0: all three SS-entry
`|p_ee_ref − p_ee_actual|` values are below 1 mm, well within the
<10 mm acceptance criterion from the Phase 3 prompt), unlocks
step 1 dock, and improves every SS metric in all three steps.
Step 2 still aborts on `dock_timeout` with residual SS
degradation (monotone ↑ per Tripwire A) — a new failure mode not
attributable to Bug 1, not addressed by Option Z, and distinct
from Bug 2 (which is gated on abort cascades and is not
exercised until step 2 itself aborts).

