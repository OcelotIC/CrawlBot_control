# Phase-0 pre-flight findings: stance deviation along the joint-space geodesic

**Branch:** `claude/step2-path-diagnostic`
**Plan reference:** `/root/.claude/plans/magical-munching-book.md` §4.0, §2.9
**Date:** 2026-04-27.
**Verdict:** the §7 fix as written **fails** the gate. Of the
three candidate mitigations tested, **task-space-aware
constrained smoothing** is the clear winner — stance
constraint to <40 μm and world-frame swing/torso path inflation
≤+1.2 % on all three T15 steps.

---

## Three candidates tested

| Method | Algorithm | Stance gate? | Step-2 swing-EE path inflation |
|--------|-----------|:------------:|--------------------------------|
| Raw `pin.interpolate` | The §7 fix as written | **FAIL** (Δ = 588 mm) | n/a |
| **Local projection** | per-τ 1-task IK seeded from raw q(τ); plan §2.9 | PASS (<1 μm) | **+105 %** ❌ rejected |
| **Laplacian smoothing** | iterate: replace q[k] with constraint projection of joint-space midpoint of neighbours | PASS (<40 μm) | **+86 %** marginal |
| **Task-space smoothing** | iterate: 3-task IK with stance pinned + torso/swing targets at task-space midpoint of neighbours | PASS (<40 μm) | **+0.2 %** ✓ recommended |

The task-space smoother converges in 120 iterations (~0.3 s on
this machine) with **zero** 3-task IK fallbacks across all
samples × iterations on all three steps.

---

## Headline numbers

| Step | Raw Δ_stance max | Local proj max | Laplacian max | **Task-space max** | **Swing-EE inflation (task-space)** |
|------|------------------|----------------|----------------|----------------------|-------------------------------------|
| 0 (end (2,3)) | **96.0 mm** | 0.80 μm | 0.060 μm | **0.0 mm** | **−14.6 %** (shorter than raw) |
| 1 (end (3,3)) | **152.8 mm** | 0.76 μm | 37.7 μm | **0.038 mm** | **−1.9 %** |
| 2 (end (3,4)) | **587.7 mm** | 0.30 μm | 37.2 μm | **0.037 mm** | **+0.2 %** |

The task-space smoother makes the projected path nearly
indistinguishable from the world-frame straight line — for
step 0 it is *shorter* than the raw chord (because the raw
chord is straight in joint space, not world space).

---

## Method

For each T15 step, with `q_start` from the IK-fix run's
physics_trace at SS-entry and `q_end` from
`manipulability_config_trajectory` (cached as
`step{0,1,2}_q_end.npz`):

```
1. Raw chord:           q(τ) = pin.interpolate(model, q_start, q_end, s(τ))
2. Local projection:    1-task IK from seed q(τ) with stance pinned
3. Laplacian smoothing: iterate q[k] ← stance-project(midpoint(q[k-1], q[k+1]))
4. Task-space smoothing: iterate q[k] ← 3-task IK with stance pinned +
                                         torso/swing targets at task-space
                                         midpoint(neighbours)
```

Δ_stance(τ) = ||FK[stance].translation − p_anchor|| at 21
uniformly-spaced τ samples. Path lengths computed by summing
‖p[k+1] − p[k]‖ over the world-frame xyz of torso and
swing-EE.

The script: `scripts/diagnostic_stance_deviation_along_geodesic.py`.

---

## Why task-space smoothing wins

The constraint manifold `M = {q : FK[stance](q) = p_anchor}`
is 14-dimensional, embedded in the 20-dim tangent space of Q.
Three notions of "closest path on M" between q_start and q_end:

1. **Per-point Cartesian projection of the chord** (= local
   projection): each q_chord(τ) projected independently to its
   nearest point on M (under pseudo-inverse local distance).
   Result follows the manifold's *transverse* curvature — the
   chord pierces M at q_start and q_end and bulges away at
   interior τ; projection traces this bulge along M.
   Path is constraint-feasible but unnecessarily long.

2. **Intrinsic geodesic on M** (≈ Laplacian smoothing
   converged): the shortest curve on M connecting q_start to
   q_end, measured by joint-space metric. Discrete
   approximation: each interior q[k] is replaced by the
   constraint projection of its joint-space neighbours'
   midpoint. Converges to a curve along the manifold's
   intrinsic geodesic structure.
   Better than (1) but still long because *joint-space
   metric* doesn't penalise long world-frame paths.

3. **Task-space-shortest curve on M** (= task-space smoothing):
   minimise Σ ‖p_torso[k+1] − p_torso[k]‖² + ‖p_swing[k+1] −
   p_swing[k]‖² subject to q[k] ∈ M. This finds the curve on M
   whose *world-frame* image is shortest. Key insight: in this
   problem, the "swing-EE moves from anchor_b[3] to
   anchor_b[4]" is a *world-frame* objective — minimising
   joint-space arc length doesn't help.

The 14-DOF M manifold is wide enough that a path satisfying
"swing-EE stays close to the chord in world frame" exists for
all three steps. Task-space smoothing finds it.

---

## Implementation implications

The plan §7 fix needs revision:

- Drop `q(τ) = pin.interpolate(q_start, q_end, s(τ))` as the
  reference source.
- Add a planner-side **task-space-smoothed constrained
  geodesic generator**: at SS-entry, solve the iterative
  smoothing problem on a 21-sample τ grid (~0.3 s overhead)
  and cache the resulting q-sequence. At runtime, the planner
  interpolates between adjacent samples (joint-space SLERP via
  `pin.interpolate`) and produces references via FK.
- Cost: ~0.3 s once per SS-entry. Zero runtime cost in the QP
  loop.

**Revised plan effort estimate:** the original plan §6 budget of
0.5 day for the §2.9 mitigation grows to ~1.5 days because:

- The smoothing routine itself: 0.5 day (port the diagnostic's
  `_smoothed_constrained_geodesic_taskspace` to
  `crawlbot/core/ik.py` or a new module).
- Caching + interpolation in TorsoPlanner / SwingPlanner:
  0.5 day (extend the FK reference generator from the original
  plan to read q from the smoothed sequence rather than from
  `pin.interpolate(q_start, q_end, s(τ))`).
- Tests + closed-loop validation: 0.5 day.

Net new effort vs original plan: +1 day. Total revised estimate:
~6.5–7 days.

The mathematical derivation in plan §2 is preserved verbatim
for everything **except** equation (1). The sequence q_smooth(τ)
replaces `pin.interpolate(...)` as the source of q(τ); all
downstream FK + tangent + Jacobian construction is identical.
The chain rule for v_full(τ) and a_full(τ) needs a small
adjustment: `dq_geo` is no longer constant — it varies with τ.
That changes eq. (3)–(4) to:

```
dq_local(τ_k → τ_{k+1}) := pin.difference(model, q_smooth[k], q_smooth[k+1])
v_full(τ) ≈ dq_local at τ / Δτ_grid  (numerical / spline fit)
a_full(τ) ≈ d(v_full)/dτ
```

i.e., velocities/accelerations come from finite-differencing
the smoothed q-sequence, then optionally fitted to a spline for
analytic derivatives. The QP only needs the value at integer
QP ticks; numerical derivative quality is fine.

---

## Files

```
Misc/runs/diagnostic/stance_deviation_along_geodesic/
  step{0,1,2}_data.json                    raw + projected + Laplacian + task-space per-τ
  step{0,1,2}_q_end.npz                    cached q_end from re-IK
  all_steps_delta_stance.png               raw vs projected Δ_stance
  all_steps_fk_smoothness.png              FK[torso/swing] xyz over τ
  summary.txt                              one-page verdict (auto-generated)
  PHASE0_FINDINGS.md                       this report
scripts/
  diagnostic_stance_deviation_along_geodesic.py
```
