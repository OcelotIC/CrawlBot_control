# T15-post-2 — Step 2 SS mid-trajectory characterization

Read-only analysis of `results/M7_1pct_3step_v22_t15_bug1fix/` to
characterize step 2's SS failure (`dock_timeout` at t = 28.49 s,
d = 374.35 mm, ori = 9.84°). Bug 1 is closed (T15_report.md §3.0:
SS-entry reference agreement < 1 mm on all three steps). This
report localizes what fails during the swing.

**Inputs.**
- `results/M7_1pct_3step_v22_t15_bug1fix/sim_log.json`
- `results/M7_1pct_3step_v22_t15_bug1fix/physics_trace.pkl`
  (SS-only, 130 entries at ~0.2 s spacing spanning t ∈ [0.11, 28.39] s;
  **does not cover any DS window**)
- `crawlbot/core/robot_interface.py` (for joint-slice conventions)
- `crawlbot/planning/swing_planner.py` / `contact_scheduler.py` (for
  anchor table and `reference_at` shape)
- `crawlbot/simulation/sim_loop.py` (for `_get_ee_data`)

**Conventions.**
- All positions in mm, structure-local frame (Pinocchio `oMf`
  with structure body as origin; log.p_ee is `rs.oMf_tool_{a,b}.translation`
  per `sim_loop.py:2211–2217`).
- Quaternions wxyz throughout; EE orientation error = 2·acos(|q1·q2|).
- Anchor indices are 0-based scheduler indices; scheduler idx `k` =
  MJCF site `anchor_{k+1}{arm}`.
- Arm-Pinocchio joint slices (from `robot_interface.py`): arm A =
  `q[7:14]`, arm B = `q[14:21]`. Joints (×7 per arm): J1, J2,
  J_swivel, J3, J4, J5, J6. All arm-joint position limits are ±π rad.

---

## Q1 — EE reference vs actual through step 2 SS

### Q1.1 — Time series (first SS tick every ~1 s)

| t (s) | \|ref−act\| (mm) | ee_ori_err (°) | `p_ee_actual` (mm) | `p_ee_ref` (mm) |
|---:|---:|---:|---|---|
| 14.99 | 0.246 | 0.022 | (+400.24, −300.10, +24.93) | (+400.02, −300.00, +24.95) |
| 15.99 | 9.475 | 0.048 | (+419.98, −295.48, +27.45) | (+419.73, −300.00, +19.12) |
| 16.99 | 15.782 | 0.053 | (+504.15, −290.03, +18.66) | (+508.73, −300.00, +7.32) |
| 17.99 | 12.002 | 0.072 | (+666.25, −294.31, +7.29) | (+669.61, −300.00, −2.73) |
| 18.99 | 115.631 | 3.984 | (+759.65, −290.88, −50.10) | (+865.49, −300.00, −4.43) |
| 19.99 | 255.232 | 8.428 | (+811.93, −289.07, −103.18) | (+1043.62, −300.00, +3.32) |
| 20.99 | 367.017 | 11.073 | (+828.65, −293.71, −144.10) | (+1159.12, −300.00, +15.45) |
| 21.99 | 420.706 | 11.587 | (+820.65, −293.84, −160.75) | (+1198.56, −300.00, +24.01) |
| 22.99 | 432.765 | 11.063 | (+809.65, −288.42, −161.49) | (+1200.00, −300.00, +25.00) |
| 23.99 | 425.555 | 10.769 | (+809.60, −284.43, −143.65) | (+1200.00, −300.00, +25.00) |
| 24.99 | 416.326 | 10.631 | (+816.68, −284.89, −136.76) | (+1200.00, −300.00, +25.00) |
| 25.99 | 397.835 | 10.359 | (+829.10, −288.47, −118.43) | (+1200.00, −300.00, +25.00) |
| 26.99 | 385.065 | 10.124 | (+842.06, −293.50, −116.83) | (+1200.00, −300.00, +25.00) |
| 27.99 | 376.563 | 9.897 | (+851.57, −299.03, −117.81) | (+1200.00, −300.00, +25.00) |
| 28.39 | 374.749 | 9.841 | (+852.90, −300.91, −116.27) | (+1200.00, −300.00, +25.00) |

### Q1.2 — EE position error behaviour

| Statistic | Value |
|---|---:|
| min | 0.246 mm (at t = 14.99 s, SS entry) |
| max | **432.765 mm** (at t = 22.99 s, k_ss index 80 — during quintic tail / ref saturation) |
| mean | 267.995 mm |
| fraction of ticks with err increasing | 0.619 |
| value at final SS tick | 374.749 mm |

The error grows roughly monotonically from SS entry up to a peak
at t ≈ 22.99 s, then decreases slightly as the actual arm drifts
back toward the (now-static) reference at `anchors_b[4]`.

### Q1.3 — EE orientation error behaviour

| Statistic | Value |
|---|---:|
| min | 0.014° |
| max | 11.648° (at t = 21.69 s) |
| mean | 7.257° |
| value at final SS tick | 9.841° |

### Q1.4 — Abort-tick state

At the last logged SS tick (`k = 258`, `t = 28.390 s`); the
`dock_timeout` abort is logged at `t = 28.49 s`, one tick later.

| Quantity | Value |
|---|---|
| `p_ee_actual` (arm B tool, struct frame, mm) | (+852.899, −300.908, −116.269) |
| `p_ee_ref` (struct frame, mm) | (+1200.000, −300.000, +25.000) = `anchors_b[4]` |
| `q_ee_actual` (wxyz) | (0.9963, 0.0002, 0.0858, 0.0017) |
| `q_ee_ref` (wxyz) | (1.0000, 0, 0, 0) |
| `\|Δp\|` (mm) | 374.749 |
| EE orientation error (°) | 9.841 |
| `log.d_grip_swing[k_last]` (mm) | 374.749 (gripper-to-target-anchor; matches `|Δp|`) |

---

## Q2 — Swing reference trajectory shape

### Q2.1 — Reference sampled at phase fractions of T_step

Step 2: `T_step = 9.423 s`, `t_ss_start = 14.990 s`,
`swing_early_finish_fraction = 0.80` so the quintic completes at
`t = t_ss_start + 0.8 · T_step ≈ 22.52 s`.

| sample point | t (s) | `p_ee_ref` (mm) |
|---|---:|---|
| SS entry | 14.990 | (+400.02, −300.00, +24.95) |
| 25 % T_step | 17.390 | (+565.89, −300.00, +2.64) |
| 50 % T_step | 19.690 | (+995.11, −300.00, +0.20) |
| 75 % T_step | 22.090 | (+1199.32, −300.00, +24.41) |
| 100 % T_step | 24.390 | (+1200.00, −300.00, +25.00) (saturated) |
| SS exit (abort tick) | 28.390 | (+1200.00, −300.00, +25.00) (saturated) |

### Q2.2 — Path vs straight-line anchor-to-anchor

| Quantity | Value |
|---|---:|
| `anchors_b[3]` (scheduler idx 3 = MJCF `anchor_4b`) | (+400, −300, +25) mm |
| `anchors_b[4]` (scheduler idx 4 = MJCF `anchor_5b`) | (+1200, −300, +25) mm |
| Straight-line distance | 800.000 mm |
| Max perpendicular deviation of reference from straight line | 29.995 mm |
| Max reference z above anchor plane | 0.000 mm |
| Min reference z relative to anchor plane | −29.995 mm |

Interpretation note (as facts only): the swing reference dips
**below** the anchor plane by up to ~30 mm, i.e. the away-normal
direction in `SwingPlanner` is −z in the structure frame. No
loops, overshoots, or kinks — the reference is a single quintic-
with-bump segment.

### Q2.3 — Shape comparison to steps 0 and 1

| Step | ref start (mm) | ref end (mm) | Δx span | Δy span | Δz span | max z above plane | min z rel. plane |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | (−400, −300, +25) | (+400, −300, +25) | 799.96 | 0.00 | 30.00 | 0.00 | −30.00 |
| 1 | (−400, +300, +25) | (+400, +300, +25) | 799.97 | 0.00 | 29.99 | 0.00 | −29.99 |
| 2 | (+400, −300, +25) | (+1200, −300, +25) | 799.98 | 0.00 | 30.00 | 0.00 | −30.00 |

All three references have the same qualitative shape (800-mm
x-span, 0 y-span, 30-mm below-plane z clearance bump). Step 2's
reference is shifted in x by +800 mm from step 0's but is
otherwise identical in shape.

---

## Q3 — Arm B joint state through step 2 SS

Extracted from `physics_trace.pkl` entries within step 2 SS
(68 entries, ~0.2 s spacing, covering t ∈ [14.99, 28.39] s).

### Q3.1 — Joint angles (arm B = swinging arm)

Endpoints:

| joint | q @ SS entry (rad) | q @ SS exit (rad) |
|---|---:|---:|
| J1_b | −0.1160 | −1.8841 |
| J2_b | −2.3046 | −2.9604 |
| J_swivel_b | −0.5129 | −1.3916 |
| J3_b | −1.1720 | −0.0567 |
| J4_b | +0.4473 | −0.0553 |
| J5_b | +1.2784 | +1.0189 |
| J6_b | +0.3899 | −0.0135 |

Per-joint min/max across step 2 SS window:

| joint | min (rad) | max (rad) | range (rad) | within 5 % of ±π? |
|---|---:|---:|---:|---|
| J1_b | −2.3534 | −0.1136 | 2.2398 | no |
| J2_b | **−2.9959** | −2.2999 | 0.6960 | **yes** (|−2.9959| > 0.95·π = 2.985) |
| J_swivel_b | −1.9163 | −0.4845 | 1.4318 | no |
| J3_b | −1.1740 | +0.2420 | 1.4160 | no |
| J4_b | −0.3045 | +0.7886 | 1.0931 | no |
| J5_b | +0.9106 | +1.3103 | 0.3997 | no |
| J6_b | −0.0400 | +0.3899 | 0.4299 | no |

**J2_b** reaches −2.9959 rad during step 2 SS, which is 0.146 rad
from the lower limit of −π (4.63 % of π). All other joints stay
outside the 5 % envelope.

### Q3.2 — Joint velocities

`sim_log.json` does not contain a per-tick joint-velocity field
(no `qd_joints_b` or equivalent). `physics_trace.pkl` items
contain `q`, `qdd_t`, `tau_q`, `lambda`, `contact_fL`,
`cond_J_t`, `sig_min_J_t`, `cond_NJe`, `sig_min_NJe`,
`torso_debug`, and `t`/`phase` — **no joint-velocity field**.
Per-joint velocity saturation cannot be assessed from existing
logs.

### Q3.3 — Manipulability (available conditioning proxies)

True swing-arm (arm B) EE manipulability
`√(det(J_arm_B · J_arm_B^T))` is **not logged** and cannot be
reconstructed without forward-kinematics calls beyond this task's
read-only scope.

`physics_trace` carries the following kinematic-conditioning
scalars per captured SS tick (see `sim_loop.py:1884–1912`):

- `sig_min_J_t` / `cond_J_t` — min singular value and condition
  number of `J_torso` (6-row torso Jacobian).
- `sig_min_NJe` / `cond_NJe` — min singular value and condition
  number of the stance-arm EE Jacobian projected into torso's
  null space (i.e. characterises stance-arm capacity after
  torso-tracking demand is met).

These reflect torso controllability and the *stance* arm's
configuration (arm A during step 2 SS), not the swinging arm B.

At 5 evenly spaced instants through step 2 SS:

| t (s) | `sig_min_J_t` | `cond_J_t` | `sig_min_NJe` | `cond_NJe` |
|---:|---:|---:|---:|---:|
| 14.99 | 1.0000 | 1.0000 | 1.7944e−01 | 1.0297e+01 |
| 18.19 | 1.0000 | 1.0000 | 1.1865e−01 | 1.5951e+01 |
| 21.59 | 1.0000 | 1.0000 | 1.4089e−01 | 1.3264e+01 |
| 24.99 | 1.0000 | 1.0000 | 1.1421e−01 | 1.6587e+01 |
| 28.39 | 1.0000 | 1.0000 | **4.9326e−02** | **3.8393e+01** |

`sig_min_J_t` is constant at 1.0 throughout (torso Jacobian is
well-conditioned). `sig_min_NJe` decreases from 0.179 at SS entry
to 0.049 at abort (−73 %); `cond_NJe` correspondingly grows from
10.3 to 38.4.

---

## Q4 — SS-entry configurations across all three steps

Using `physics_trace` entries at the first SS tick of each step:
step 0 `(t=0.11)`, step 1 `(t=6.62)`, step 2 `(t=14.99)`. Note
the step-1 SS entry tick is read from the nearest physics_trace
sample (t_trace=6.62 vs log SS-entry at 6.52).

### Q4.1 — Arm A joint configuration at SS entry (rad)

| joint | step 0 | step 1 | step 2 | Δ (s2−s0) |
|---|---:|---:|---:|---:|
| J1_a | −0.9496 | −0.9899 | **+0.6786** | +1.6282 |
| J2_a | +1.6769 | +1.7105 | +1.7976 | +0.1208 |
| J_swivel_a | −1.1870 | −1.1891 | −1.5598 | −0.3728 |
| J3_a | −2.0364 | −1.9928 | −2.5826 | −0.5462 |
| J4_a | +0.0307 | +0.0079 | −0.5988 | −0.6295 |
| J5_a | −0.1966 | −0.2756 | **+1.8634** | +2.0600 |
| J6_a | −0.0669 | −0.0584 | −0.5658 | −0.4989 |

Step 2 arm-A configuration differs from step 0 by > 1.5 rad on
J1_a and > 2.0 rad on J5_a; other joints by 0.12 – 0.63 rad.
Steps 0 and 1 arm-A configurations are within 0.08 rad of each
other on all seven joints.

### Q4.2 — Arm B joint configuration at SS entry (rad)

| joint | step 0 | step 1 | step 2 | Δ (s2−s0) |
|---|---:|---:|---:|---:|
| J1_b | +0.1423 | −0.1777 | −0.1160 | −0.2583 |
| J2_b | −1.2227 | −2.4076 | −2.3046 | −1.0820 |
| J_swivel_b | −0.8595 | −0.5575 | −0.5129 | +0.3467 |
| J3_b | −1.7253 | −0.9828 | −1.1720 | +0.5534 |
| J4_b | −0.2160 | +0.4707 | +0.4473 | +0.6634 |
| J5_b | +0.7284 | +1.1927 | +1.2784 | +0.5501 |
| J6_b | +0.8265 | +0.3698 | +0.3899 | −0.4366 |

Step 2 arm-B configuration is close to step 1 arm-B (max 0.19 rad
difference per joint) but differs from step 0 by up to 1.08 rad
(J2_b).

### Q4.3 — Torso pose at SS entry

| field | step 0 (t = 0.11 s) | step 1 (t = 6.52 s) | step 2 (t = 14.99 s) |
|---|---|---|---|
| `struct_pos` (m) (MuJoCo structure body) | (−1.8e-7, −1.5e-7, −1.8000) | (−0.00140, −0.0000, −1.80034) | (−0.00257, +0.00017, −1.80100) |
| `struct_euler_deg` (°, Z-Y-X) | (−2.6e-4, −7.6e-5, +3.6e-5) | (−0.0685, +0.0524, −0.0286) | (−0.1107, +0.0137, +0.0035) |
| `p_torso` (m, Pinocchio torso body in struct frame) | (+0.1201, −0.0352, −0.8594) | (+0.1521, −0.0204, −0.8371) | (+0.1841, −0.0225, −0.8062) |

Torso (Pinocchio body within structure) advances in +x by +32.0 mm
from step 0 to step 1 and a further +32.0 mm from step 1 to step 2;
z (height above structure floor) rises by +22.3 mm then +30.9 mm.
Structure body in world frame moves < 3 mm total between the
three SS entries. Structure attitude (Euler) is ≤ 0.1 ° at all
three entries.

### Q4.4 — Configuration-type summary

- Steps 0 and 1: both arms in near-mirror configurations. Arm-A
  and arm-B joint-angle pairs at SS entry differ predominantly by
  sign on J1 (y-side dependency). Arm-A step-1-vs-step-0 per-joint
  delta ≤ 0.08 rad on all seven joints; arm-B step-1-vs-step-0
  delta up to 1.19 rad on J2_b (reflecting the stance-anchor
  change from (2,2) at step 0 to (2,3) at step 1).
- Step 2: arm-A configuration diverges from the step-0/step-1
  pattern by > 1.5 rad on two joints (J1_a, J5_a) and ≥ 0.35 rad
  on four others. Arm-B configuration at step 2 closely matches
  arm-B at step 1 (max per-joint delta 0.19 rad). Torso has
  advanced +64 mm in x over two successful docks.

---

## Q5 — Abort-moment snapshot

At last logged SS tick `k = 258`, `t = 28.390 s` (the
`dock_timeout` abort event is logged at `t = 28.49 s`, i.e. one
tick later; no SS log tick at the abort instant itself).

### Q5.1 — Arm B tool state

| Quantity | Value |
|---|---|
| Position (struct frame, mm) | (+852.899, −300.908, −116.269) |
| Orientation (wxyz) | (0.9963, 0.000213, 0.08576, 0.001739) |

### Q5.2 — Target `anchors_b[4]`

| Quantity | Value |
|---|---|
| Position (struct frame, mm) | (+1200.000, −300.000, +25.000) |
| Orientation | identity (R = I, q_wxyz = (1, 0, 0, 0)) |

### Q5.3 — Tool → anchor vector

| Component | Value |
|---|---:|
| Δx (mm, axial) | +347.101 (anchor ahead of tool; tool has undershot in +x) |
| Δy (mm, lateral) | +0.908 (tool y essentially on anchor's y-plane) |
| Δz (mm, vertical) | +141.269 (anchor above tool; tool is below anchor plane) |
| \|Δ\| (mm) | 374.749 |
| Character | undershoot in x (347 mm short) **and** below-plane in z (141 mm below) |

Cross-check: `log.d_grip_swing[258] = 374.749 mm` = `|Δ|`.

### Q5.4 — Arm B joint configuration at abort

| joint | q (rad) at abort (t = 28.39 s) | q (rad) at step-2 SS entry (t = 14.99 s) | Δ (abort − entry) (rad) |
|---|---:|---:|---:|
| J1_b | −1.8841 | −0.1160 | −1.7681 |
| J2_b | −2.9604 | −2.3046 | −0.6558 |
| J_swivel_b | −1.3916 | −0.5129 | −0.8787 |
| J3_b | −0.0567 | −1.1720 | +1.1153 |
| J4_b | −0.0553 | +0.4473 | −0.5027 |
| J5_b | +1.0189 | +1.2784 | −0.2596 |
| J6_b | −0.0135 | +0.3899 | −0.4034 |

Per-joint limit-distance at abort (|q − sign(q)·π| or shortest
distance to ±π, whichever is smaller):

| joint | distance to nearest ±π (rad) | within 5 % of ±π? |
|---|---:|---|
| J1_b | 1.258 (to −π) | no |
| J2_b | 0.181 (to −π) | no (5.8 % of π; **within 5 %** was transiently hit earlier, see Q3.1) |
| J_swivel_b | 1.750 (to −π) | no |
| J3_b | 3.085 (to −π) | no |
| J4_b | 3.087 (to −π) | no |
| J5_b | 2.123 (to +π) | no |
| J6_b | 3.128 (to −π) | no |

### Q5.5 — Reconstructed "nominal arrival configuration" against `anchors_b[4]`

The IK `q_end` that `_setup_torso_for_step` computed against
(`end_a = 3`, `end_b = 4`) is not stored per-step in
`sim_log.json`. The log's `preplanner_T_steps` entry is present
(`T_step_2 = 9.423 s`) but the corresponding `q_end` vector is
not written to the JSON. **Nominal arrival configuration cannot
be reconstructed from existing logs** without rerunning the IK,
which is outside this task's read-only scope.

---

*End of T15-post-2 report.*
