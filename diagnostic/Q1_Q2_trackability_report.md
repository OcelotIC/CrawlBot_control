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

_To be filled in._

### §2.3  Mid-waypoint joint-space distance

_To be filled in._

### §2.4  Q2 verdict

_To be filled in._

---

## §3  Combined disposition

_To be filled in once Q2 completes._

---

**Status:** Q1 complete (verdict Q1-C). Q2 sweep in progress.
