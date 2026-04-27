# Phase-0 pre-flight findings: stance deviation along the joint-space geodesic

**Branch:** `claude/step2-path-diagnostic`
**Plan reference:** `/root/.claude/plans/magical-munching-book.md` §4.0
**Date:** 2026-04-27.
**Verdict:** the §7 fix as written is **not sufficient on its own**.
The §2.9 stance-projection mitigation **is required** and is
demonstrated below to recover the constraint to <1 μm. There is
a secondary cost (path-length inflation on step 2) that warrants
discussion before continuing to Phase 1.

---

## Method

For each T15 step, with q_start drawn from the IK-fix run's
physics_trace at SS-entry and q_end re-computed by
`manipulability_config_trajectory` with the same parameters:

```
q(τ)   = pin.interpolate(model, q_start, q_end, s(τ))           (raw geodesic)
q_p(τ) = 1-task IK from seed q(τ) with stance EE pinned          (projected geodesic)
        to anchor_a[end_a] (or anchor_b[end_b] when stance='b')
```

Δ_stance(τ) = ||FK[stance].translation − p_anchor|| at 21
uniformly-spaced τ samples.

The script: `scripts/diagnostic_stance_deviation_along_geodesic.py`.
Outputs: this directory.

---

## Results

| Step | Raw Δ_stance max | Projected Δ_stance max | proj_dq_norm max | Torso path (raw → proj) | Swing path (raw → proj) |
|------|------------------|------------------------|------------------|-------------------------|-------------------------|
| 0 (end (2,3))  | **96.0 mm**  at τ=0.50  | 0.80 μm | 0.084 rad | 637 → 660 mm (+4 %)    | 941 → 980 mm (+4 %)    |
| 1 (end (3,3))  | **152.8 mm** at τ=0.45  | 0.76 μm | 0.280 rad | 853 → 946 mm (+11 %)   | 819 → 965 mm (+18 %)   |
| 2 (end (3,4))  | **587.7 mm** at τ=0.50  | 0.30 μm | 0.824 rad | 1099 → 1698 mm (+54 %) | 812 → 1661 mm (+105 %) |

Gate (50 mm on raw Δ_stance): **fails on all 3 steps**.
Projection IK convergence: **21/21 samples on all 3 steps**.

`proj_dq_norm` is the joint-space distance ‖q_p − q_raw‖ at
the worst τ. All three are ≤1 rad — the projection is a small
local correction, not a global re-optimisation.

---

## Interpretation

### Finding 1 — `pin.interpolate` does not preserve the stance constraint.

The constraint manifold `{q : FK[stance](q) = p_anchor}` is a
non-flat 14-dimensional submanifold of Q. The SE(3)+linear
interpolation between two points on the manifold cuts a chord
through the ambient space; the chord leaves the manifold for
any non-trivial curvature. For step 2's
(q_start_3,4 → q_end_3,4) chord, the interior leaves by 588 mm
in stance-EE world position.

This means the §7 plan, applied directly with
`q(τ) = pin.interpolate(q_start, q_end, s(τ))`, generates
references derived from FK on configurations that **the QP cannot
follow** — the QP enforces the welded stance constraint as
*hard*, so it cannot reach a configuration that places the
stance EE 588 mm from its anchor. The QP would respond by
trading off torso and swing tracking, recreating exactly the
failure mode the §7 fix was designed to eliminate.

### Finding 2 — A 1-task IK projection of q(τ) onto the constraint manifold works.

For each τ, solve a 1-task IK with stance pinned to the anchor,
seeded from the raw q(τ). The result q_p(τ) is on the
constraint manifold to numerical tolerance (<1 μm position
residual, all 63 samples converged). The seed-from-raw choice
matters: warm-starting from the previous τ's q_p collapses the
projection to a single point (zero path) because the warm-start
already satisfies the constraint and the IK exits with no work.
Seeding from raw q(τ) gives a τ-varying projection.

### Finding 3 — The projected swing path is up to 2.05× longer than the raw chord on step 2.

Step 2's swing-EE world-frame path along the projected geodesic
is 1661 mm vs the raw chord's 812 mm. Step 0/1 inflation is
modest (4 %, 18 %). This is the price the projection pays: by
pinning stance, the swing/torso must take a longer detour
through Cartesian space to reach q_end while remaining on the
constraint manifold.

For step 2 with T_phase ≈ 17 s, the average swing-EE speed goes
from 48 mm/s → 98 mm/s. Still within reasonable arm bandwidth,
and FAR better than the current architecture, where the
*reference itself is infeasible* — the QP has no path to
track at all.

### Finding 4 — Step-to-step Δ doubles on the projected step 2.

Per-sample step-to-step Δ on the projected path peaks at 208 mm
(torso) and 211 mm (swing) for step 2, vs 102 / 79 mm on the
raw chord. The projection is *less* spatially smooth than the
raw geodesic, especially around the singular interior at τ≈0.5.
This is the τ region where the path-geometry diagnostic
(`results/diagnostic/T15_step2_path_geometry.md` §3.2) found
`w_ideal` collapsing 6 orders of magnitude — the constraint
manifold is locally deformed there, and the projection bends
around the deformation.

Whether the QP can track this less-smooth projected reference
is **the open question** that only closed-loop simulation can
answer. The smooth quintic time-scaling `s(τ)` flattens
endpoint accelerations, but the spatial geometry of the
projected path is determined by the manifold curvature.

---

## Implications for the implementation plan

1. The §7 fix **must be augmented** with the §2.9 projection
   step. Plan-Phase-0 gate has effectively delivered its
   verdict: not "proceed as planned" but "proceed with the
   mitigation".
2. The mitigation as scoped (~0.5 day in plan §6) is now
   primary work, not optional. Effort estimate revises to
   ~6.5–7 days.
3. **New open question for the user**: the projected reference
   for step 2 is 2× longer than the raw chord. Three options:
   - **(A)** Implement projection as scoped in §2.9; close-loop
     simulate; if AOCS / NMPC don't saturate, ship.
     Lowest effort, highest risk of "tracks fine but slowly".
   - **(B)** Augment with a path-length cost in the projection:
     among constraint-feasible q_p(τ), pick the one closest to
     a smoothed neighbour. Adds a smoothing pass at SS-entry.
     Medium effort, smaller path inflation.
   - **(C)** Drop joint-space-quintic entirely and switch to
     constraint-aware trajectory optimisation: minimise path
     length subject to stance constraint at every τ. This is
     plan-§7's "Option C" but for path generation, not q_end.
     High effort, best path quality.
4. Recommendation: **Option A**. The path-inflation cost at
   step 0/1 is small (4 %, 18 %); only step 2 doubles. Step 2
   is the failing case anyway, and even a 2× longer trackable
   reference is incomparably better than today's untrackable
   one. If closed-loop validates, ship; if it saturates,
   revisit with B or C.

---

## Files

```
results/diagnostic/stance_deviation_along_geodesic/
  step{0,1,2}_data.json                    raw + projected per-τ data
  step{0,1,2}_q_end.npz                    cached q_end from re-IK
  all_steps_delta_stance.png               raw vs projected Δ_stance
  all_steps_fk_smoothness.png              FK[torso/swing] xyz over τ
  summary.txt                              one-page verdict
  PHASE0_FINDINGS.md                       this report
scripts/
  diagnostic_stance_deviation_along_geodesic.py
```
