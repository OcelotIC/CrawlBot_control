# Q1/Q2 trackability diagnostic — what can the closed-loop follow?

**Branch:** `claude/step2-path-diagnostic` @ `3ce416e`+
**Source data:** `results/M7_1pct_3step_v22_t15_midwaypoint/` (Q1) and
seven single-step T15-equivalent runs in `results/diagnostic_q2/` (Q2).
**Date:** 2026-04-25.

This report answers two questions left open by the Phase-7
mid-waypoint failure:

- **Q1**: is the 177° step-1 orientation error a representation /
  SLERP-hemisphere bug, or a real kinematic divergence?
- **Q2**: how far can the IK place ``q_end`` from ``q_start``
  before the closed-loop controller fails to track?

---

## §1  Q1 — step-1 orientation failure diagnosis

### §1.1  Reference vs actual orientation traces

From `sim_log.json` of the mid-waypoint run, step-1 SS window
(t ∈ [12.22, 26.32] s, swing arm = a):

| metric                                | torso  | swing-EE |
|---------------------------------------|-------:|---------:|
| max commanded reorient (ref vs ref(0))| 10.0°  | 12.1°    |
| max ref-vs-actual error (geodesic)    | 178.6° | 178.8°   |
| time of first divergence (>30°)       | 13.52 s (1.30 s into SS) | similar |
| q_torso_ref / q_ee_ref sign-flips     | **0** / **0** | **0** / **0** |
| q_torso_actual / q_ee_actual sign-flips | 3 / 6 | — |

Plots: `q1_step1_torso_ori_traces.png`,
`q1_step1_swing_ori_traces.png`.

**The reference quaternion stream is well-formed.** Zero sign
flips across 142 SS samples; the commanded reorient never exceeds
12°. The actual quaternion stream shows 3 + 6 sign flips — these
reflect *real* ~180° rotations of the structure body under the
cascading dynamics, not representation artifacts.

### §1.2  SLERP hemisphere check

`crawlbot/planning/torso_planner.py` and
`crawlbot/planning/swing_planner.py` use `pin.log3(R0^T R1)` which
returns axis-angle in `[−π, π]`. Per-sub-segment, this always
yields the geodesic (shorter-arc) integration; no quaternion sign
ambiguity is exposed at the API level.

For step 1's three orientation waypoints (start, mid, end) the
quaternion inner products are:

| pair                | torso  | swing-EE |
|---------------------|-------:|---------:|
| `⟨q_start, q_mid⟩`  | +0.997 | +0.996   |
| `⟨q_mid, q_end⟩`    | +0.997 | +0.996   |
| `⟨q_start, q_end⟩`  | +1.000 | +1.000   |

All three pairs same-sign → the same hemisphere is consistently
chosen. The piecewise SLERP would not produce a 180° spurious
rotation even if it had a hemisphere bug; the bug isn't possible
here because all three waypoint quaternions lie within a
~5°-radius cluster.

### §1.3  Replayed piecewise SLERP

Script: `scripts/diagnostic_q1_slerp_repro.py` reconstructs the
piecewise SLERP path using the same `pin.log3 / pin.exp3` machinery
the planners use, with the actual step-1 mid-waypoint waypoints.

Outputs:
- `diagnostic/q1_slerp_repro_torso.png` — replay peaks at **9.5°**
  from start.
- `diagnostic/q1_slerp_repro_swing.png` — replay peaks at **9.96°**
  from start.

Both replay traces stay below ~10° throughout, matching the
commanded-reorient values from §1.1. The geodesic path the
planners would have produced, given the actual mid-waypoint, is
nowhere near 180°.

### §1.4  Q1 verdict — Q1-C confirmed

| Verdict | Description                          | Status     |
|:-------:|:-------------------------------------|:----------:|
| Q1-A    | SLERP sub-segments different hemispheres | rejected |
| Q1-B    | R_mid sign issue elsewhere            | rejected |
| **Q1-C**| Genuine kinematic divergence; no representation bug | **confirmed** |

**Mechanism.** The reference is small (≤12° reorient). The actual
diverges from the reference within 1.3 s of SS start — the
controller cannot keep up, and the errors compound until the
structure body has flipped ~180°. The mid-waypoint reshape's
aggressive translation (812 mm `dp_torso` vs the natural 595 mm)
induces inertial loads on the structure that the AOCS (reaction
wheels, capped at ±5 Nms) cannot counteract; the body tumbles
through the 14-s SS window. The 177° at exit is the integral of
that uncontrolled rotation.

**Implication for Option B'.** A representation fix won't help
because the bug isn't representation. The cost function in
`manipulability_config_mid_waypoint` needs to constrain or
penalize *dynamic* trackability (path length, momentum loading,
torque envelope), not just kinematic manipulability.

---

## §2  Q2 — empirical trackability bound

_To be populated when the sweep completes._

### §2.1  Method

Single-step T15-equivalent simulation (`scripts/run_m7_single_step.py`,
n_steps=1, start_a=2, start_b=2). The closed-loop IK
``dock_configuration_fixed_rotation`` is monkey-patched to return
an α-extrapolation of the natural ``q_end``:

- torso position: `q_start[:3] + α · (q_natural[:3] − q_start[:3])`
- torso quaternion: SLERP-extrapolated via `pin.log3 / pin.exp3`
- arm joints: `q_start[7:] + α · (q_natural[7:] − q_start[7:])`

Sweep over α ∈ {0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0}. α=1.0 is
the byte-identical baseline (sanity).

### §2.2  Results

| α    | d_arm [rad] | d_torso_pos [m] | e_torso_peak [mm] | e_swing_peak [mm] | e_swing_at_dock [mm] | outcome |
|-----:|------------:|----------------:|------------------:|------------------:|---------------------:|:--------|
| 0.50 | 1.344       | 0.296           | 48.7              | 44.9              | 4.23                 | DOCKED |
| 0.75 | 2.016       | 0.443           | 46.1              | 32.2              | 3.71                 | DOCKED |
| 1.00 | 2.688       | 0.591           | 42.6              | 26.8              | 4.20                 | DOCKED (baseline) |
| 1.25 | 3.360       | 0.739           | 40.5              | 23.7              | **0.65**             | DOCKED |
| 1.50 | 4.032       | 0.887           | 46.1              | 21.2              | 2.00                 | DOCKED |
| 1.75 | 4.704       | 1.034           | 54.5              | 19.0              | 2.73                 | DOCKED |
| 2.00 | 5.376       | 1.182           | 62.6              | 21.8              | 2.80                 | DOCKED |

Plot: `diagnostic/q2_tracking_vs_distance.png` (4-panel).

**All 7 α values dock cleanly** at the single-step T15 scenario.
The closed-loop controller successfully tracks displacements from
0.5× to 2.0× the natural one — including 1.18 m of torso position
displacement and 5.38 rad of arm-joint displacement.

The dock distance does not degrade monotonically with α. Tightest
dock is at α=1.25 (0.65 mm); the baseline α=1.00 sits at 4.20 mm.
This is consistent with the dock detector firing whenever the
swing arm is briefly within the criterion — the trajectory shape
matters more than the endpoint position. Pre-dock peak swing-EE
error is *bounded* across all α, ranging from 19 mm (α=1.75) to
45 mm (α=0.5). The torso peak grows mildly with α (43 → 63 mm)
but is always recoverable.

The note on `torso_rot_distance_perturbed_deg = 0.00°` for every
α: at this single-step scenario `start_a=2, start_b=2`, the
natural fixed-rotation IK keeps torso orientation identical
between q_start and q_end_natural (no reorient required); SLERP
extrapolation of identity yields identity for all α. So the
sweep does not exercise the rotation-tracking axis. A follow-up
on a step that requires non-trivial reorient could close that
gap; not in scope here.

### §2.3  Mid-waypoint joint-space distance — comparison

For step 1 of the mid-waypoint run (the catastrophic 177°-flip
step), the mid-waypoint optimiser proposed:

| quantity                                  | mid-waypoint step-1 | Q2 α=2.00 (all-DOCKED) |
|-------------------------------------------|--------------------:|-----------------------:|
| `q_start[:3]`                             | [0.149, −0.014, −0.797] | (different scenario) |
| `q_mid[:3]` proposed by IK                | [0.898, −0.444, −0.128] | n/a |
| `\|\|q_mid_torso − q_start_torso\|\|`     | **1.093 m**         | **1.182 m**            |

**The mid-waypoint torso displacement (1.093 m) is *within* the
empirically-validated trackable range** demonstrated by Q2 (1.18 m
at α=2.0, all docked).

So the step-1 failure cannot be attributed to displacement
magnitude — the controller has demonstrated it can track that far.
Something else about the mid-waypoint shape must be the trigger.

The structural difference between the Q2 and mid-waypoint cases:

- **Q2**: the perturbed `q_end` lies on the line from `q_start`
  through `q_end_natural`. The trajectory is a single quintic in
  the same direction. Just longer.
- **Mid-waypoint**: `q_mid` is geometrically *off* the
  `q_start → q_end_natural` line. The piecewise quintic produces
  a kinked trajectory — first segment goes in one direction,
  second in another. Step-1's `q_mid` y-coordinate is −0.444 m
  while the natural trajectory's y stays near zero (mass and
  gait are nearly symmetric). The mid-waypoint forces a 0.44 m
  lateral excursion that has no counterpart in any successful
  step.

### §2.4  Q2 verdict — Q2-C with refinement

| Verdict | Description                          | Status     |
|:-------:|:-------------------------------------|:----------:|
| Q2-A    | Sharp trackability threshold; mid-waypoint exceeded it | rejected |
| Q2-B    | Smooth degradation; mid-waypoint moderately exceeded ideal | rejected |
| **Q2-C**| Mid-waypoint within distance bounds; failure has different cause | **confirmed** |

The cause is **path orthogonality / direction**, not displacement
distance. A `||q_mid − q_start||² ≤ d_max²` constraint or
penalty (Option B' as the brief framed it) would *not* fix the
failure mode this report identifies — Q2 shows the controller is
fine with that magnitude.

What *would* address it:

1. **Off-axis penalty**: `||q_mid − [(q_start + q_end)/2]||²` — penalises
   q_mid's deviation from the midpoint of the natural geodesic.
   This pushes the optimum toward `q_mid ≈ midpoint_of_natural_path`,
   reducing the mid-waypoint to a no-op (which is what we want
   on steps where the natural path is already feasible).
2. **Path-length penalty**: ``∫₀¹ ‖q̇(τ)‖ dτ`` over the piecewise
   quintic. Penalises the extra arc-length the kink introduces.
3. **Trajectory optimisation (Option C)**: drop the "two quintic
   segments through a single waypoint" structure and optimise the
   full reference trajectory with explicit cost on tracking error
   (e.g. direct collocation with NMPC bandwidth as a constraint).

---

## §3  Combined disposition

### §3.1  What the data says

- **Q1**: 177° step-1 orientation is genuine open-loop
  divergence, not a representation bug. The reference is small
  (≤12° reorient, 0 sign flips); the controller cannot keep up
  with the dynamics induced by the mid-waypoint detour.
- **Q2**: along-axis displacement up to 2.0× the natural one is
  fully trackable on the single-step scenario. The mid-waypoint's
  1.09 m torso displacement is *within* this range.
- **Combined**: the trackability failure is not bounded by
  displacement magnitude. It is bounded by *path shape* — the
  kink the mid-waypoint introduces forces the controller through
  a transient with high momentum loading the AOCS cannot recover
  from in the SS time budget.

### §3.2  Implication for Option B'

Option B' as the brief originally scoped it (add
`||q_mid − q_start||²` to the cost) **will not work**. The Q2
data directly rules it out: that distance is empirically tracked.
Option B' needs to evolve to one of the §2.4 alternatives —
ideally an off-axis or path-length penalty rather than a
distance penalty.

### §3.3  Implication for Option C

Option C (full trajectory optimisation with tracking-bandwidth
constraints) remains the principled fix and is consistent with
both Q1 and Q2 findings. It explicitly handles:

- Path shape via the optimisation variable (full trajectory
  rather than a single mid-waypoint).
- Tracking bandwidth via explicit constraints on
  `||q̇_ref(t)|| ≤ v_max(state)` or equivalent.
- AOCS / momentum loading via a constraint on
  `||L_dot_ref(t)|| ≤ τ_w_max`.

The cost is implementation effort (direct collocation /
short-horizon NMPC over the SS, with the existing torso planner
as a warm start). Whether this is worth the effort depends on
the alternative paths:

### §3.4  Cheaper alternative — gait-level fix

A gait-level fix that avoids (3,4) altogether (multi-segment SS,
virtual transit anchor) does **not** require any new cost
function. It re-routes the geometric problem rather than solving
it. Given that the Phase-7 mid-waypoint regressed steps that
previously docked, a gait-level fix is also less risk of
introducing new failure modes.

Recommendation: scope **gait-level fix first** (cheap, low risk).
If gait-level fix doesn't close T15 step 2, escalate to **off-axis
penalty Option B''** (single-line cost-function tweak in
`manipulability_config_mid_waypoint`). Reserve **full trajectory
optimisation Option C** for the case where neither suffices, since
its cost is significantly higher.

### §3.5  Specific next-prompt scope (out of scope here)

The next prompt should pick *one* of:

(a) **Gait-level**: scope a multi-segment SS or virtual transit
    anchor for (3,4); read-only diagnostic first to confirm
    feasibility.
(b) **Off-axis B''**: ~10-line edit to
    `manipulability_config_mid_waypoint`'s cost; re-run T15 with
    `mid_waypoint_force_on=True` to validate.
(c) **Trajectory optimisation C**: full short-horizon TO over
    SS; significant new module; ~1 day of work.

This diagnostic prompt deliberately does not implement any of
them.

---

**Status:** Q1 complete (Q1-C). Q2 complete (Q2-C with
refinement: failure is path-shape, not distance). §3 disposition
written. Diagnostic complete; stopping per the brief.
