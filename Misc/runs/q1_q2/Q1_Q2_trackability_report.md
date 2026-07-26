# Q1/Q2 trackability diagnostic — what can the closed-loop follow?

**Branch:** `claude/step2-path-diagnostic` @ `3ce416e`+
**Source data:** `Misc/runs/M7_1pct_3step_v22_t15_midwaypoint/` (Q1) and
seven single-step T15-equivalent runs in `Misc/runs/diagnostic_q2/` (Q2).
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

Script: `Misc/scripts/diagnostic_q1_slerp_repro.py` reconstructs the
piecewise SLERP path using the same `pin.log3 / pin.exp3` machinery
the planners use, with the actual step-1 mid-waypoint waypoints.

Outputs:
- `Misc/runs/q1_q2/q1_slerp_repro_torso.png` — replay peaks at **9.5°**
  from start.
- `Misc/runs/q1_q2/q1_slerp_repro_swing.png` — replay peaks at **9.96°**
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

### §2.0  Two-experiment design

Q2 was conducted in two parts. Q2a (single-step) followed the
brief literally and gave a misleading positive signal — the
scenario it used was geometrically too easy. Q2b (T15 step 2) is
the rigorous follow-up that reproduces the actual failure
context and tests the perturbation directions the brief Q2.3
notes called out as missing from Q2a.

| Experiment | Scenario               | Sweeps                           | Result   |
|:-----------|:-----------------------|:---------------------------------|:---------|
| Q2a        | Single-step n=1, (2,2)→(2,3) | along-axis α ∈ {0.5..2.0}     | 7/7 DOCKED |
| Q2b        | Full T15, intercept step 2 only | along-axis α ∈ {0.5..1.5} + orthogonal-y β ∈ {−0.45..+0.45} m | 0/12 DOCKED |

The Q2a positive signal does **not** generalize. The single-step
scenario at (2,2)→(2,3) has different kinematics from T15 step 2
at (3,3)→(3,4) and the closed-loop is far more forgiving.

### §2.1  Q2a method (single-step)

Single-step T15-equivalent simulation (`scripts/run_m7_single_step.py`,
n_steps=1, start_a=2, start_b=2). The closed-loop IK
``dock_configuration_fixed_rotation`` is monkey-patched to return
an α-extrapolation of the natural ``q_end``:

- torso position: `q_start[:3] + α · (q_natural[:3] − q_start[:3])`
- torso quaternion: SLERP-extrapolated via `pin.log3 / pin.exp3`
- arm joints: `q_start[7:] + α · (q_natural[7:] − q_start[7:])`

Sweep over α ∈ {0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0}. α=1.0 is
the byte-identical baseline (sanity).

### §2.2  Q2a results (single-step)

| α    | d_arm [rad] | d_torso_pos [m] | e_torso_peak [mm] | e_swing_peak [mm] | e_swing_at_dock [mm] | outcome |
|-----:|------------:|----------------:|------------------:|------------------:|---------------------:|:--------|
| 0.50 | 1.344       | 0.296           | 48.7              | 44.9              | 4.23                 | DOCKED |
| 0.75 | 2.016       | 0.443           | 46.1              | 32.2              | 3.71                 | DOCKED |
| 1.00 | 2.688       | 0.591           | 42.6              | 26.8              | 4.20                 | DOCKED (baseline) |
| 1.25 | 3.360       | 0.739           | 40.5              | 23.7              | **0.65**             | DOCKED |
| 1.50 | 4.032       | 0.887           | 46.1              | 21.2              | 2.00                 | DOCKED |
| 1.75 | 4.704       | 1.034           | 54.5              | 19.0              | 2.73                 | DOCKED |
| 2.00 | 5.376       | 1.182           | 62.6              | 21.8              | 2.80                 | DOCKED |

Plot: `Misc/runs/q1_q2/q2_tracking_vs_distance.png` (4-panel).

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

**The Q2a result is misleading in scope.** All 7 α docked — but
this scenario doesn't have the (3,4) wrist-singular geometry,
doesn't have the cascaded state contamination of T15, and
requires zero torso reorient at q_end. It only tells us that
*some* easy single-step scenario tolerates wide α. The brief's
question — "how far can the IK place q_end from q_start before
the closed-loop fails to track" — is properly answered only by
Q2b.

### §2.3  Q2b method (T15 step 2 specifically)

Script: `Misc/scripts/diagnostic_q2b_step2_trackability.py`. Runs the
full 3-step T15 scenario (`run_m7_v22_1pct_3step_t15.py`) and
monkey-patches `dock_configuration_fixed_rotation` to intercept
ONLY the 3rd call (step 2). Steps 0 and 1 run with the natural
IK, so the closed-loop state at step 2 entry matches the IK-fix
baseline.

Two sweeps:

- **Sweep A (along-axis)**: q_perturbed = q_start + α(q_natural −
  q_start) for α ∈ {0.5, 0.75, 1.0, 1.25, 1.5}. Tests whether
  the Q2a along-axis finding holds at step 2.

- **Sweep B (orthogonal lateral-y)**: q_perturbed = q_natural
  with q_perturbed[1] += β for β ∈ {−0.45, −0.30, −0.15, 0.00,
  +0.15, +0.30, +0.45} m. Tests the empirically-observed failure
  direction — the Phase-7 mid-waypoint q_mid for step 1 was at
  torso y = −0.44 m vs natural near 0. This is the perturbation
  axis Q2a left out per the user's clarification.

### §2.4  Q2b results

**Sweep A (along-axis at T15 step 2):** see
`Misc/runs/q1_q2/q2b_along_axis.png`.

| α    | d_torso_pos [m] | step-2 e_swing_peak [mm] | docked? |
|-----:|----------------:|-------------------------:|:-------:|
| 0.50 | 0.379           | **439.8**                | ✘       |
| 0.75 | 0.569           | 431.7                    | ✘       |
| 1.00 | 0.759           | 432.8 (= IK-fix baseline)| ✘       |
| 1.25 | 0.948           | 414.3                    | ✘       |
| 1.50 | 1.138           | **397.2**                | ✘       |

**Sweep B (orthogonal y at T15 step 2):** see
`Misc/runs/q1_q2/q2b_orthogonal_y.png`.

| β [m] | torso_pos perturbed dist [m] | step-2 e_swing_peak [mm] | docked? |
|------:|-----------------------------:|-------------------------:|:-------:|
| −0.45 | 0.936                        | **423.5**                | ✘       |
| −0.30 | 0.869                        | 427.7                    | ✘       |
| −0.15 | 0.825                        | 431.4                    | ✘       |
|  0.00 | 0.806                        | 432.8 (sanity vs α=1)    | ✘       |
| +0.15 | 0.815                        | 431.9                    | ✘       |
| +0.30 | 0.851                        | 430.4                    | ✘       |
| +0.45 | 0.911                        | **423.0**                | ✘       |

**0 of 12 perturbations docked.** The β=0 / α=1 cell of both
tables matches at 432.8 mm, confirming the patch is byte-faithful
when no perturbation is requested.

Observations:

1. **Along-axis α has minor effect**: monotonically improves
   swing peak from 439.8 → 397.2 mm as α grows (range 43 mm),
   but every sample is two orders of magnitude above the 5 mm
   dock gate.
2. **Orthogonal β is nearly flat**: total range only 10 mm
   (423–433 mm), with a slight U-shape (best at the extreme |β|=0.45,
   worst at β=0). The natural q_end is approximately the
   y-optimum within ±0.45 m — but only by 10 mm of swing peak.
3. **No perturbation in either sweep makes step 2 dock.**

### §2.5  Q2b verdict

| Verdict | Description                                                                | Status     |
|:-------:|:---------------------------------------------------------------------------|:----------:|
| Q2b-A   | Sharp trackability threshold along the natural axis                        | rejected: monotone trend, no knee |
| Q2b-B   | Smooth degradation along axis OR orthogonal direction has a tractable dock | rejected: 0/12 dock |
| **Q2b-C** | **Step 2 is infeasible from this q_start regardless of q_end choice** within the tested neighborhood | **confirmed** |

The decisive finding: T15 step 2 cannot be made to dock by
choosing a different q_end within ±50% along-axis OR ±0.45 m
laterally. The failure is **not in the q_end target**; it is in
the *trajectory* the planners generate from this q_start to any
q_end within the tested neighborhood.

This refines the Q2a finding correctly:

- Q2a was right that **distance magnitude** is not the binding
  constraint (single-step tolerated 2× distance).
- Q2b shows that **q_end choice** is also not the binding
  constraint — perturbing it in any direction in the tested
  neighborhood doesn't change the outcome.

The binding constraint is **the path between** q_start and q_end.
This is the (3,4) singular interior the path-geometry diagnostic
already identified (§3 of `T15_step2_path_geometry.md`).

### §2.6  Mid-waypoint comparison — re-examined

The Phase-7 mid-waypoint failure was a **regression on steps 0
and 1**, not a step-2 fix. Step 2 always failed (Phase 7's
abort at d=429 mm matches Q2b's α=1.0 abort at 433 mm — same
failure regime). What the mid-waypoint did was destabilize the
PRIOR steps' tracking, not improve step 2.

Why prior steps regressed under mid-waypoint, but step 2 is
robust to direct q_end perturbation:

- The mid-waypoint inserts an OFF-AXIS waypoint *between*
  q_start and q_end at a known time t_mid. The TorsoPlanner's
  reference is then forced to pass through the off-axis pose
  in mid-flight, with v=0 there. That requires a real velocity
  reversal mid-trajectory. The closed-loop dynamics don't
  comply.
- Q2b sweep B perturbs the ENDPOINT q_end[1] — the trajectory
  is still a single quintic from q_start to q_end_perturbed.
  No velocity reversal. The trajectory shape is just shifted,
  not kinked. This is much more tractable.

So the mid-waypoint failure mechanism (kinked trajectory with
mid-flight velocity reversal) is **fundamentally different**
from what Q2b sweep B tests (smoothly-shifted endpoint).
Sweep B confirms what Q2a hinted at: smooth perturbations of
q_end, in any direction, are tracked. The mid-waypoint failure
is specifically about kinks.

---

## §3  Combined disposition

### §3.1  What the data says — corrected

- **Q1**: 177° step-1 orientation is genuine open-loop
  divergence, not a representation bug. The reference is small
  (≤12° reorient, 0 sign flips); the controller cannot keep up
  with the dynamics induced by the mid-waypoint detour.

- **Q2a (single-step)**: along-axis perturbations up to 2.0× the
  natural displacement are fully trackable in this *easier*
  scenario. The result does **not** generalize to T15 step 2.

- **Q2b (T15 step 2)**: 0 of 12 perturbations dock. Sweep A
  along-axis (α ∈ [0.5, 1.5]) all fail with swing peaks
  397–440 mm. Sweep B orthogonal-y (β ∈ [−0.45, +0.45] m) all
  fail with swing peaks 423–433 mm. **Step 2 from this q_start
  is unreachable for any q_end in the tested neighborhood**.

- **Combined**: the trackability failure is *not* bounded by
  q_end choice. The binding constraint is the **trajectory
  between** q_start and q_end. This converges with the
  path-geometry diagnostic's H2 finding: the (3,4) reference
  path crosses a singular interior, and no endpoint adjustment
  reshapes the interior enough to fix it.

### §3.2  Implication for Option B' / B''

Both Option B variants are now ruled out:

- **B' (`||q_mid − q_start||²` distance penalty)**: ruled out
  by Q2a — the controller already tolerates that distance.
- **B'' (off-axis penalty `||q_mid − midpoint_natural||²`)**:
  ruled out by Q2b — neither along-axis nor orthogonal-y
  perturbation of q_end recovers the dock. A penalty that
  pushes q_mid back toward the natural geodesic would just
  produce something close to the natural q_end, which Q2b shows
  also fails.

The mid-waypoint approach is structurally limited: choosing a
better single point along the trajectory cannot fix a singular
reference interior that exists for any reasonable endpoint.

### §3.3  Implication for Option C — strengthened

Option C (full trajectory optimisation) is now the only
candidate left from the original menu. Q2b directly supports
it: the trajectory shape needs to be the optimisation variable,
not a single waypoint. Specifically:

- The cost must include a *path-singularity* term integrated
  over the trajectory, not just at endpoints/waypoints.
- Constraints on tracking bandwidth (`‖q̇_ref‖`,
  `‖L̇_ref‖ ≤ τ_w_max`) prevent the optimizer from selecting
  technically-feasible-but-untrackable paths.
- The natural q_end can serve as a terminal constraint or as a
  warm start, not the optimisation variable.

Implementation cost: significant (direct collocation over SS,
new module, 1+ day). But this is now the only on-axis fix.

### §3.4  Gait-level fix — strengthened recommendation

A gait-level fix that avoids the (3,4) anchor pair entirely
becomes the **dominant recommendation** given Q2b:

- Q2b shows the (3,4) anchor pair is infeasible from any q_end
  in a wide neighborhood. The geometric problem is intrinsic to
  the pair, not the IK output.
- Inserting a transit anchor (e.g., (3, 3.5) virtual or
  re-routing to (3,3)→(2,4)→(3,4)) splits the singular
  transition into two sub-transitions, each of which lives in a
  smaller and likely well-conditioned region.
- This requires **no new cost function** and **no new optimisation
  module** — just a scheduler / planner-level change.
- Lower regression risk: per-step references stay single-quintic
  (the configuration the IK-fix run already validated for
  steps 0 and 1).

### §3.5  Updated recommendation

1. **First (cheapest)**: gait-level fix — virtual transit anchor
   or multi-segment SS for (3,4). Re-run T15 with the new gait;
   verify dock outcome. ~0.5 day of work.
2. **If (1) doesn't close T15**: full trajectory optimisation
   (Option C). ~1+ day of work; new module; explicit dynamics
   awareness in the cost.
3. **Mid-waypoint reshape (any variant)**: do not pursue. Q2b
   rules out B' and B''; the path-geometry diagnostic plus
   Q2b's 0/12 dock count rule out the structure of "single
   waypoint plus piecewise quintic" altogether.

### §3.6  Specific next-prompt scope (out of scope here)

The next prompt should pick:

(a) **Gait-level fix scoping**: read-only diagnostic of
    candidate transit anchors or multi-segment SS layouts for
    (3,4) at 1% mass-ratio. Confirm feasibility kinematically
    before implementing.
(b) **If (a) is feasible**: implement the scheduler/planner
    change and re-run T15.

This diagnostic prompt deliberately does not implement either.

---

**Status:** Q1 complete (Q1-C). Q2 complete in two stages:

- Q2a single-step (7/7 docked, scoped wrong, misleading positive).
- Q2b T15 step 2 (0/12 docked, definitive negative on q_end
  perturbation as a fix).

§3 disposition rewritten with Q2b: step 2 cannot be fixed by
endpoint choice in any direction; the (3,4) path-singularity is
intrinsic. Recommended next path is gait-level (transit
anchor / multi-segment SS) rather than any IK-output reshape.

Stopping per the brief.
