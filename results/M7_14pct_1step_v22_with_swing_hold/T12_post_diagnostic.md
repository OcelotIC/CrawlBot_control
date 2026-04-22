# T12 — Post-hoc DS1 divergence diagnostic

Source data: `results/M7_14pct_1step_v22_with_swing_hold/sim_log.json`
(read-only; no re-run, no re-instrumentation, no parameter sweeps).
Generated: 2026-04-21.

Figures produced:

- `figs_post/D1_reference_zoom.png`  (t ∈ [5.5, 10.0] s)
- `figs_post/D1_reference_full.png`  (t ∈ [0.11, 25.91] s)
- `figs_post/D3_hw_growth.png`       (DS1 window, t ∈ [6.01, 25.91] s)

All attitudes are Euler ZYX intrinsic (roll, pitch, yaw) in degrees, produced
by the same `quat_wxyz_to_euler_deg` convention used internally by the
controller (`crawlbot/core/state_conversions.py:154`).

---

## D1 — DS1 reference setpoint trajectory

### Fields available in `SimLog`

| Quantity                 | Logged?  | Field name                     |
|---|---|---|
| torso attitude — actual  | yes      | `q_torso` (wxyz quaternion)    |
| torso attitude — reference | yes   | `q_torso_ref` (wxyz quaternion)|
| structure attitude — actual | yes  | `struct_quat`, `struct_euler_deg` |
| structure attitude — reference | **no**   | — *(not logged)*        |

`SimLog` does not carry a structure-attitude reference. The AOCS target in
`SimConfig.aocs_hw_target` (`crawlbot/simulation/config.py:66-77`) is the
**zero wheel-momentum vector** `[0,0,0]` Nms — the AOCS loop regulates `h_w`,
not structure attitude. The structure attitude therefore has no reference
trajectory in this log; only the actual evolution is plottable.

### Numerical table (nearest log ticks to the requested times)

All values in degrees, Euler ZYX.

| t [s] | torso_roll_act | torso_pitch_act | torso_yaw_act | torso_roll_ref | torso_pitch_ref | torso_yaw_ref | struct_roll_act | struct_pitch_act | struct_yaw_act |
|---|---|---|---|---|---|---|---|---|---|
|  5.910 | +42.5566 | +37.9057 |  −4.6400 | +41.9400 | +37.0816 |  −5.5979 |  −0.4203 |  +0.2831 |  −0.3647 |
|  6.010 | +42.5940 | +37.8396 |  −4.6227 | +41.9400 | +37.0816 |  −5.5979 |  −0.4317 |  +0.2864 |  −0.3671 |
|  6.110 | +42.6240 | +37.7834 |  −4.5958 | +41.9400 | +37.0816 |  −5.5979 |  −0.4407 |  +0.2863 |  −0.3684 |
| 25.910 | +32.3453 | +30.7607 | −12.8928 | +40.9788 | +37.5277 |  −6.7162 | +33.0298 | +13.6863 | +18.3503 |

### Single-tick deltas across the SS→DS1 transition (k=58 → k=59, Δt = 0.1 s)

| Channel                | Δroll [deg] | Δpitch [deg] | Δyaw [deg] |
|---|---|---|---|
| `q_torso` actual       | +0.0374     | −0.0661      | +0.0173    |
| `q_torso_ref` reference | 0.0000     |  0.0000      |  0.0000    |
| `struct_euler_deg` actual | −0.0114  | +0.0033      | −0.0024    |

The torso reference changes by exactly zero across the SS→DS1 boundary; the
actual torso attitude changes by sub-degree increments (all components
|Δ| < 0.07°); the structure attitude changes by sub-hundredth-of-a-degree
increments. No step discontinuity exceeds the single-tick drift present in
the surrounding ticks.

### Divergence behavior over DS1 (numerical summary from the plot data)

Using the reference held constant at its SS-exit value (as shown in the
`q_torso_ref` column above, identical at t = 5.91 / 6.01 / 6.11 / 25.91 s
to within the plot-level precision), the DS1 error growth on each torso
Euler component is:

| Component | actual(6.01) − ref(6.01) [deg] | actual(25.91) − ref(25.91) [deg] |
|---|---|---|
| roll  | +0.6540 | −8.6335 |
| pitch | +0.7580 | −6.7670 |
| yaw   | +0.9752 | −6.1766 |

Structure attitude (no reference logged):

| Component | struct_euler(6.01) [deg] | struct_euler(25.91) [deg] |
|---|---|---|
| roll  | −0.4317 | +33.0298 |
| pitch | +0.2864 | +13.6863 |
| yaw   | −0.3671 | +18.3503 |

See `figs_post/D1_reference_zoom.png` for the [5.5, 10.0] s window and
`figs_post/D1_reference_full.png` for the whole horizon. Both figures
overlay `q_torso` (actual, solid) with `q_torso_ref` (reference, dashed) on
the upper axis and plot `struct_euler_deg` (actual only) on the lower axis
with the SS→DS1 boundary (t = 6.01 s) marked with a dotted vertical line.

---

## D2 — NMPC `status = 1` at k = 65

### Logged quantities at k = 65

| Field            | Value                |
|---|---|
| `t`              | 6.610 s              |
| `phase`          | `DS` (DS1)           |
| `nmpc_status`    | 1                    |
| `nmpc_time_ms`   | 218.10               |
| `nmpc_cost`      | 72.9978              |

Context window (7 ticks either side of k = 65):

| k   | t [s] | phase | status | nmpc_time_ms | nmpc_cost |
|---|---|---|---|---|---|
|  58 | 5.910 | SS    | 0 |  23.33 | 280.630 |
|  59 | 6.010 | DS    | 0 |  25.63 | 264.980 |
|  60 | 6.110 | DS    | 0 |  35.14 | 264.401 |
|  61 | 6.210 | DS    | 0 |  34.31 | 265.261 |
|  62 | 6.310 | DS    | 0 |  34.03 | 265.987 |
|  63 | 6.410 | DS    | 0 |  35.36 | 266.401 |
|  64 | 6.510 | DS    | 0 |  35.82 | 266.575 |
| **65** | **6.610** | **DS** | **1** | **218.10** | **72.998** |
|  66 | 6.710 | DS    | 0 |  93.50 |  70.747 |
|  67 | 6.810 | DS    | 0 |  41.23 |  70.867 |
|  68 | 6.910 | DS    | 0 |  42.67 |  70.942 |
|  69 | 7.010 | DS    | 0 |  39.91 |  70.995 |
|  70 | 7.110 | DS    | 0 |  41.62 |  71.049 |
|  71 | 7.210 | DS    | 0 |  53.47 |  71.110 |
|  72 | 7.310 | DS    | 0 |  54.47 |  71.176 |

### Status code semantics (from `crawlbot/simulation/logging.py:148` and `crawlbot/simulation/sim_loop.py:1540-1610`)

`SimLog.nmpc_status` is a reduced 3-code enum declared as
`# 0=ok, 1=max_iter, 2=infeasible`. The encoding in the sim loop is
`0` on `info_n.success == True`; otherwise `2` if the IPOPT
`info_n.status` string contains `'infeasib'`, else `1`. Code `1` is
therefore **any non-success, non-infeasible termination** (which for IPOPT
most commonly means iteration-limit reached, but also includes
`Solve_Succeeded = False` for a handful of other IPOPT return codes).

### DS1 first-tick check

| Quantity | Value |
|---|---|
| k at first DS1 tick | 59 |
| t at first DS1 tick | 6.010 s |
| NMPC status at k = 59 | 0 |

**k = 65 is the 7th DS1 tick** (`k − k_ds1_first = 65 − 59 = 6`, counting
the first DS1 tick as tick 1, the failing tick is the 7th). It is **not**
the immediate post-dock tick.

### Fields explicitly NOT logged in `SimLog`

The following detail is required to fully characterise the failure but is
not captured by `SimLog`:

- **IPOPT return-status string.** `NMPCSolveInfo.status`
  (`crawlbot/solvers/nmpc_solver.py:450`) carries the IPOPT-level string
  (e.g. `Maximum_Iterations_Exceeded`, `Restoration_Failed`), but the sim
  loop reduces it to the 3-code enum before logging; the raw string is
  discarded.
- **Iteration count.** `NMPCSolveInfo.iterations` is carried in memory by
  the solver wrapper, but `SimLog` has no field for it. Not logged.
- **Warm-start delta (solution change from k = 64 primal).** Not logged;
  `SimLog` carries only per-tick `lambda_ref` (the QP-facing wrench), not
  the full NMPC primal/dual warm-start state.
- **Per-constraint violation vector.** Not logged.
- **NMPC solver stats (`solver_stats` dict from CasADi/IPOPT).** Not
  logged.

### Observations strictly from the logged data

- NMPC cost jumps from 266.575 (k = 64) → 72.998 (k = 65, status = 1) →
  70.747 (k = 66, status = 0) and stays in the ~70–71 band for the
  remainder of DS1. The pre-failure cost band (~265) and post-failure
  band (~70) differ by a factor of ≈ 3.8.
- Solve-time profile: the surrounding ticks run at 25–36 ms; k = 65 takes
  **218.10 ms** (≈ 6× the surrounding median) and the immediately
  following tick k = 66 takes 93.50 ms (≈ 2.6×); from k = 67 onward the
  solver returns to the 40–55 ms band.
- No other `nmpc_status ≠ 0` events in the log (single occurrence).

---

## D3 — Wheel-momentum growth profile during DS1

Window: t ∈ [6.010, 25.910] s, 200 log ticks.
Source: `log.hw_physical` (Nms, structure-body components).

| Quantity                | Value |
|---|---|
| \|h_w\|(t = 6.010 s)    | 0.2074 Nms |
| \|h_w\|(t = 25.910 s)   | 6.2827 Nms |
| Δ\|h_w\| over DS1       | +6.0753 Nms |
| DS1 duration            | 19.90 s |

### Fits to \|h_w\|(t) over DS1

| Model                  | Parameters | R² | Residual RMS [Nms] |
|---|---|---|---|
| **Linear** `a + b τ`   | a = +0.2501,  b = +0.3166 Nms/s | 0.9921 | 0.1628 |
| **Exponential** `A exp(k τ) + C` | A = 0.2634,  k = +0.2028 1/s,  C = +0.1683 | −0.8195 | 2.4756 |
| **Piecewise-linear (1 breakpoint, continuous)** | t_b = 16.41 s (τ_b = 10.40 s from DS1 start); b₁ = +0.3684 Nms/s (t ∈ [6.01, 16.41]), b₂ = +0.2573 Nms/s (t ∈ [16.41, 25.91]) | 0.9997 | 0.0291 |

τ is time relative to DS1 start (τ = t − 6.010 s). The exponential fit
has R² < 0, i.e. worse than the constant-mean baseline for this record.
The piecewise-linear fit's RMS (29 µNms) is ~5.6× better than the plain
linear fit and achieves R² = 0.9997. All three fits use the same 200
log-tick sample; the exponential fit is done in log-space of
(y − min(y) + ε) with ε = 10⁻⁹.

### d\|h_w\|/dt averages

| Window                          | Rate [Nms/s] |
|---|---|
| Overall DS1 (t ∈ [6.01, 25.91]) | +0.3053 |
| First 2 s of DS1 (t ∈ [6.01, 8.01]) | +0.2373 |
| Last 2 s of DS1  (t ∈ [23.91, 25.91]) | +0.2719 |

### Per-component rates (hw_i(end_DS1) − hw_i(start_DS1)) / ΔtDS1

| Component | Rate [Nms/s] |
|---|---|
| hw_x | −0.2004 |
| hw_y | −0.2389 |
| hw_z | −0.0828 |

Components individually drift negative; the magnitude |h_w| grows because
the components drift monotonically in sign from near-zero at DS1 start to
(−3.98, −4.55, −1.70) Nms at DS1 end.

### Cross-check: ∫ τ_w dt vs Δh_w (DS1 window)

`log.tau_w` is the logged AOCS wheel-torque command `tau_w_cmd`
(`crawlbot/simulation/sim_loop.py:2069`, held from the most recent AOCS
sub-tick; sampled at the 10 Hz log cadence, not the 100 Hz physics cadence).

| Component | Δh_w (end − start) [Nms] | ∫ τ_w dt (trap.) [Nms] |
|---|---|---|
| x | −3.9882  | −94.3253 |
| y | −4.7549  | −99.5000 |
| z | −1.6477  | −35.4187 |

The signs agree on all three components; the magnitudes disagree by a
factor of ≈ 20–24 (∫ τ_w dt is numerically larger than Δh_w). The
trapezoidal integral is computed over the DS1 window on the 10 Hz log
samples of `tau_w` (no access to the 100 Hz physics-step samples, which
are not persisted to `SimLog`). The numerical identity
Δh_w = ∫ h_w_dot dt holds at the physics cadence, not necessarily at the
logged-command cadence; the discrepancy above is the cross-check value as
requested and is reported without interpretation.

See `figs_post/D3_hw_growth.png` for the plot (|h_w|, hw_x, hw_y, hw_z
with the three fits overlaid and the ±5 Nms box bounds marked).

---

## Fields NOT captured (consolidated)

The following would have been useful for this diagnostic but are not
present in `SimLog` or the on-disk artefacts, and §5 prohibits adding
instrumentation post-hoc:

- Structure-attitude reference trajectory.
- NMPC IPOPT return-status string, iteration count, per-constraint
  violations, warm-start delta.
- NMPC primal/dual solution across knots (only `lambda_ref` at k = 0 of
  each NMPC call is logged).
- Physics-cadence `tau_w` samples (only the 10 Hz logged command snapshot
  is persisted; `physics_trace.pkl` covers only SS for this run).
- Process-level start/end wall-clock timestamps.
