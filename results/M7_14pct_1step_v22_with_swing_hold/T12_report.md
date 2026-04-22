# T12 — Single-step dock at mass_ratio = 0.14

Run script: `scripts/run_m7_v22_14pct_with_swing_hold.py`
Output dir: `results/M7_14pct_1step_v22_with_swing_hold/`
Generated: 2026-04-21

---

## Config echo

### Exact edit applied

The `mass_ratio` knob is not a `SimConfig` field; per `CLAUDE_CODE_HANDOFF.md`
§line 39 parametric variations are applied at the runtime model. In T11 the
MJCF declares the floating structure as `mass="7110"` (ratio ≈ 71/7110 =
0.010). To take mass_ratio 0.01 → 0.14, the structure body mass is scaled by
exactly 1/14 (per user instruction, the principal inertia tensor is scaled by
the same factor so the shape is preserved). The mutation is applied to the
MJCF on disk for the duration of the run and byte-exactly restored in the
`finally` block (same transient-mutation pattern already used for
`damping`/`armature`; see `scripts/run_m7_v22_with_swing_hold.py`).

`models/VISPA_crawling_rwa3.xml` (transient, 3-line context):

```xml
    <body name="structure" pos="0 0 -1.8">
      <freejoint name="structure_free"/>
-     <inertial pos="0 0 0" mass="7110"
-               fullinertia="597 1493 1777 0 0 0"/>
+     <inertial pos="0 0 0" mass="507.8571428571"
+               fullinertia="42.6428571429 106.6428571429 126.9285714286 0 0 0"/>
```

Post-load verification (from `mj_name2id` on body `structure`):

- `body_mass[structure] = 507.857142857` kg
- `body_inertia[structure] = (126.929, 106.643, 42.643)` kg·m² (MuJoCo's
  descending principal-axis ordering)

Restoration check after the `finally` block: the MJCF file matches the
original byte-for-byte (`[mjcf restored byte-exactly] True` printed at
script exit).

### T11 fixes — active/inactive verification

| Fix | Expected | Actual | Status |
|---|---|---|---|
| Pinocchio `model.armature` = `[0]*6 + [0.05]*14` | yes | `first 6 = 0`, `last 14 = 0.05` | PASS |
| MJCF arm joints: damping=0, armature=0.05 | yes | regex mutation applied once; joint-class `robot_joint` default rewritten | PASS |
| `SimConfig.mapping_bypass_in_ss = True` | True | True | PASS |
| `SimConfig.swing_early_finish_fraction = 0.80` | 0.80 | 0.80 | PASS |
| `ss_alpha_wrench = 0.01` | 0.01 | 0.01 | PASS |
| `alpha_com_soft = 0.0` | 0.0 | 0.0 | PASS |
| 7-DOF arms (`n_arm_a = n_arm_b = 7`) | 7 | 7 | PASS |
| `preplanner_a_cruise_max = 0.01` (cruise-box) | 0.01 | 0.01 | PASS |
| `preplanner_cruise_ramp_frac = 0.2` | 0.2 | 0.2 | PASS |
| `tau_max = 20.0 Nm` | 20.0 | 20.0 | PASS |

### Run parameters

- Seed: not set explicitly; MuJoCo+numpy defaults (deterministic from
  current MJCF state and `sim.setup(start_a=2, start_b=2, n_steps=1)`)
- Total sim duration: 25.91 s (259 ticks × dt_qp = 0.1 s cadence at the log
  level; physics stepped at 100 Hz internally, logged at 10 Hz)
- `mass_ratio` value read at runtime:
    - structure body mass = 507.857143 kg
    - robot total mass (non-structure, non-RWA) = 71.056 kg
    - robot / structure_body = 0.139913
    - robot / (structure_body + RWAs) = 0.139501
    - (reference: target mass_ratio literal = 0.14)

---

## Dock event

| Field | Value |
|---|---|
| Activated | yes |
| abort code / reason | — (no abort) |
| `t_dock` | 6.010 s |
| `dock_d` | 0.860 mm |
| `dock_ori` | 0.100 deg |
| Activation path | kinematic |

Raw event record:
`{'t': 6.01, 'step': 0, 'd_mm': 0.86, 'ori_deg': 0.1, 'arm': 'b', 'anchor': 3, 'method': 'kinematic'}`

---

## Relative velocity audit at dock activation instant

**Frame.** All vectors are in the **structure body frame**, which is how
controller-internal quantities (`p_ee`, `q_ee`, `p_torso`, …) are logged
(see `crawlbot/core/state_conversions.py:3`). In this frame the anchor
site (`anchor_4b`, scheduler index 3) is rigidly attached to the origin of
the frame and its structure-frame velocity is identically zero; relative
velocity at the gripper reduces to finite-difference of `log.p_ee` and
`log.q_ee` directly. Sign convention: the `approach` direction is
`p_anchor − p_gripper`; a *positive* along-approach component means the
gripper is closing on the anchor (approaching); a *negative* component
means it is opening up (receding).

The dock-activation instant is the last SS tick (`k = 58`, `t = 5.91 s`,
where `d_grip_swing = 0.862 mm`); the dock event is recorded at the SS→DS
transition at `t = 6.01 s`. Central finite differences at this tick
straddle `[k−1, k+1]` = `[SS, DS]`, i.e. the instant the weld activates.

| Quantity | Value |
|---|---|
| approach distance \|p_anchor − p_gripper\| | 0.8621 mm |
| linear relative speed \|v_rel\| | 13.3823 mm/s |
| along-approach component (sign) | +10.4469 mm/s (approaching) |
| linear components (x, y, z) | (+11.7154, +2.5422, +5.9474) mm/s |
| angular relative speed \|ω_rel\| | 7.3855 mrad/s |
| angular components (x, y, z) | (+0.7601, +0.0694, −7.3459) mrad/s |

---

## SS metrics (SS entry → SS exit, k ∈ [1, 58], t ∈ [0.11, 5.91] s)

| Field | Value |
|---|---|
| `torso_pos_peak` | 34.7665 mm |
| `torso_ori_peak` | 1.1227 deg |
| `ee_pos_peak` | 32.7864 mm |
| `ee_ori_peak` | 9.3418 deg |
| `ee_ori_at_T_step` | 0.0951 deg |

(`T_step = 7.284 s` from the pre-planner. SS terminates at the dock-gate
transition at `t = 5.91 s` ≈ 0.81·T_step, which is consistent with
`swing_early_finish_fraction = 0.80`; `ee_ori_at_T_step` is reported at the
last SS log tick.)

---

## DS1 metrics (post-dock settle, k ∈ [59, 258], t ∈ [6.01, 25.91] s)

| Field | Value |
|---|---|
| `torso_pos_peak_DS1` | 22.9960 mm |
| `torso_ori_peak_DS1` | 9.9199 deg |
| `ee_pos_peak_DS1` | 3.1988 mm |
| `ee_ori_peak_DS1` | 0.0821 deg |

### Residual at end of DS1 budget (`k = 258`, `t = 25.91 s`)

| Field | Value |
|---|---|
| \|torso_pos\| | 2.2762 mm |
| \|torso_ori\| | 9.9199 deg |
| \|ee_pos\| | 0.0073 mm |
| \|ee_ori\| | 0.0626 deg |

### Settle time to thresholds

Thresholds reused from `run_diagnostics` (the same thresholds that produced
the T11 metrics.csv columns `torso_pos_peak_mm_DS = 10`,
`torso_ori_peak_deg_DS = 5`, `ee_pos_peak_mm_DS = 5`, `ee_ori_peak_deg_DS = 5`):

- `|torso_pos| < 10 mm`
- `|torso_ori| < 5 deg`
- `|ee_pos| < 5 mm`
- `|ee_ori| < 5 deg`

Result: **all four thresholds are NOT simultaneously met at the end of
DS1.** The `torso_ori` error grows monotonically through DS1 and reaches
its peak of 9.9199 deg at the last DS1 tick (`t = 25.91 s`), exceeding the
5 deg threshold for the entirety of DS1. `torso_pos`, `ee_pos`, and
`ee_ori` are all below their thresholds by the end of DS1. No
four-threshold-simultaneous "settle time" is defined for this run.

---

## Wheel momentum `h_w`

Source: `log.hw_physical` (Nms, structure-frame body components hx, hy, hz).
Horizon = `[t_0, t_end] = [0.11, 25.91] s`.

| Instant | t [s] | \|h_w\| [Nms] | (hx, hy, hz) [Nms] |
|---|---|---|---|
| \|h_w\|_max | 25.910 | 6.2827 | (−3.9832, −4.5532, −1.6957) |
| at t_dock (k=58) | 5.910 | 0.2416 | (−0.1250, +0.2017, −0.0457) |
| at end of DS1 (k=258) | 25.910 | 6.2827 | (−3.9832, −4.5532, −1.6957) |

Box bound: ±5 Nms per component. Fraction of horizon where any
\|h_w_i\| > 4 Nms (saturation-proximity): **9.65 %** (25/259 ticks).

(The \|h_w\|_max instant coincides with the end of DS1; `h_w` is still
growing at the horizon terminus.)

---

## Scheduler / abort state

| Field | Value |
|---|---|
| Abort during SS | no |
| Abort during DS1 | no |
| `log.aborted_steps` | `[]` |
| Scheduler advance (dock recorded) | yes (1 event, at t = 6.01 s, arm = b, anchor = 3) |
| Unique phase strings in log | `['DS', 'SS']` |

---

## NMPC / QP health

| Field | Value |
|---|---|
| Solve count | 259 |
| Solve rate < 50 ms (fraction) | 0.9846 (255 / 259) |
| Max solve time | 218.10 ms |
| Time of max solve | t = 6.610 s (DS1, 0.60 s after dock) |
| QP failures | 0 / 259 |
| NMPC failures | 1 / 259 (at k = 65, t = 6.61 s, DS1; `nmpc_status = 1`; the failing call is also the max-solve-time call) |
| Dynamics residual max over horizon | not captured by the current logger (the `SimLog` writes `qfrc_constraint_torque` but no per-knot dynamics-residual trace; `physics_trace.pkl` carries `qdd_t`/`tau_q` but no explicit residual field) |

---

## Wall clock

| Field | Value |
|---|---|
| Total run time | not captured explicitly by the logger |
| Σ NMPC solve time | 6.39 s |
| Σ QP solve time | 20.96 s |
| Σ solver time (lower bound on wall clock) | 27.34 s |

(The `SimLog` schema in `crawlbot/simulation/logging.py` records per-call
`nmpc_time_ms` and `qp_time_ms` but does not record a process-level
start/end timestamp. §5.7 of the T12 spec prohibits instrumentation
rewrites, so the wall-clock value is reported as "not captured" rather
than instrumented retroactively.)
