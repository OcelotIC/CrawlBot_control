# Active-DS torso advance + reachability-triggered SS — proposed gait architecture

**Plan-of-record location:** `Misc/reports/architecture/active_ds_torso_advance.md`
**Branch this proposes:** new branch `claude/active-ds-architecture` forked from main at the merge of PR #15 (commit `59de7b2`).
**Authors of evidence:** investigation on `claude/fk-bypass-aware-tuning`
(commits `ae08221` and prior).
**Status:** proposal. Not yet implemented.
**Date:** 2026-04-28.

This document proposes a *gait-level* architecture upgrade that
addresses the residual T15 step-2 failure (and generalises to any
anchor traversal that requires significant body translation). The
existing FK-reference architecture from PR #15 is **preserved
verbatim** — this proposal adds an active-DS layer above it, not in
place of it.

---

## §0  TL;DR

The current crawler treats every DS phase identically — a
**passive settling** loop that drives kinetic energy to a target
threshold while both arms are welded to anchors. The torso has no
active reference during DS. SS triggers immediately when DS settles.

This conflates three semantically distinct DS classes:

| DS class | When | Purpose | Correct behaviour |
|---|---|---|---|
| Initial | post-setup, before step 0 | Bring system to rest from initialisation transients | Passive settle ✓ |
| Inter-step | between step k and step k+1 | Prepare body for next swing motion | **Active torso advance + reachability-triggered SS** |
| Terminal | after final step | Bring system to rest at destination | Passive settle ✓ |

The current implementation uses the *passive-settle* policy for all
three. The two outer classes are correct; the inter-step class is
the bug.

The consequence: every body translation needed for the next step's
swing-EE reachability is forced into the SS phase, *simultaneously*
with the swing arm's large-amplitude motion. The QP must
multiplex 6-DoF torso tracking + 6-DoF swing-EE tracking + stance
contact + CoM + L_com + posture in a single SS window. For modest
anchor pairs (steps 0/1: ~150–250 mm body recoil) the QP handles it.
For aggressive pairs (step 2: 591 mm body recoil + 800 mm swing) the
QP's multi-task balance can't satisfy the dock criterion in time.

The fix is to **split the body translation work**:

- **DS (inter-step) becomes ACTIVE.** Both arms remain welded.
  The torso advances along a planned reference toward a "ready
  pose" for the next step. The QP cost has *only one* primary
  task (torso 6D tracking) plus the stance constraints — no
  competing swing-arm task.
- **SS triggers reachability-based.** When the swing arm can
  reach the next anchor in T_step from the current body pose
  (1-task IK feasibility check), undock the swing arm and
  proceed with the FK-ref-driven SS phase.
- **SS becomes the small-amplitude phase.** With body
  pre-positioned, the swing arm has at most 200–400 mm to
  travel and the body only needs ~100 mm of refinement. The
  QP's task balance is much easier; existing FK refs and
  smoother work as-is.
- **Terminal DS unchanged.** After the final SS, run the
  existing passive-settle to bring the system to rest at the
  destination.

Step 2's (3,4) body recoil of 591 mm is split: ~500 mm during
DS (with no swing competing), ~91 mm during SS (alongside the
arm motion). The 800 mm swing-EE motion stays in SS but starts
from a body pose where the arm only needs to extend ~300 mm in
world frame to reach the anchor.

This is a known gait pattern in legged locomotion (the
"double-support body advance" that quadrupeds and bipeds use to
shift CoM under stance feet before lifting the swing leg). The
crawler's gait scheduler (`ContactScheduler`) already contains
the temporal phase structure; only the DS phase's *behaviour*
needs upgrading, plus a reachability gate at the DS→SS
transition.

---

## §1  The current locomotion plan, in code

The locomotion plan is built in
`crawlbot/planning/contact_scheduler.py::ContactScheduler.plan_traversal`
as a sequence of `GaitPhase` objects. Each phase carries
`phase: ContactPhase` (one of `DOUBLE`, `SINGLE_A`, `SINGLE_B`)
and a `duration` field. The scheduler's traversal is
deterministic given (start_a, start_b, n_steps):

```
phases = [
    GaitPhase(DOUBLE, dt_ds=0.5),                # DS_0: initial
    GaitPhase(SINGLE_A or SINGLE_B, dt_ss=0),    # SS_0: step 0
    GaitPhase(DOUBLE, dt_ds=0.5),                # DS_1: inter-step
    GaitPhase(SINGLE_..., dt_ss=0),              # SS_1: step 1
    ...                                          # ...
    GaitPhase(SINGLE_..., dt_ss=0),              # SS_{n-1}: step n-1
    GaitPhase(DOUBLE, dt_ds=0.5),                # DS_n: terminal
]
```

The `dt_ss` fields are zero placeholders that get replaced per-step
by the pre-planner's `T_step` via `plan.set_step_duration(idx,
T_step)`. The `dt_ds` fields stay at their default `0.5 s` —
this is *not* the real DS duration, just a fallback for timeline
queries; the actual DS exit is energy-based, not time-based.

### §1.1  How the sim_loop consumes the plan

`sim_loop.py::run` walks the plan phases via:

```python
while i < len(phases):
    gp = phases[i]
    if gp.phase.value == 'double':
        # Look ahead for next SS phase
        if i + 1 < len(phases) and phases[i+1].phase.value != 'double':
            ss_gp = phases[i+1]
            ...
            # ── 1. DS — energy-based exit (spec §7.1.1) ──────────
            ds_result = self._run_ds_passivity_loop(
                contact_config=cc_ds, max_steps=cfg.n_ds_max_steps,
                epsilon_v=cfg.settle_inter_epsilon_v, ...)
            # ── 2. SS — pre-planner solve, then closed-loop SS ──
            ...
        else:
            # No SS follows → terminal DS, also passive settle
            ...
```

The key observation: **the same `_run_ds_passivity_loop` is called
for both inter-step DS and terminal DS** (sim_loop.py:1426 and
:1764). There is no policy distinction between "transit between
SS phases" and "arrive at destination".

### §1.2  What `_run_ds_passivity_loop` actually does

Per `sim_loop.py:503–668`, the loop:

1. Builds a `ContactConfig` with both arms welded.
2. Runs the QP in `settle_mode=True, passivity_active=True`. This
   tells the QP to **dissipate kinetic energy** as fast as the
   passivity constraint allows. **No torso reference is provided;
   no NMPC call is made.**
3. Iterates at `dt_qp=0.01 s` until exit:
   - **target_met**: `T_kinetic < T_settle = 0.5 · ε_v² · λ_min(H)`
   - **plateau**: no progress over `plateau_window` steps.
   - **max_steps**: safety cap (`cfg.n_ds_max_steps = 5000`).

The thresholds:
```
ε_v = cfg.settle_inter_epsilon_v   (~ 1e-3 m/s)
T_settle ≈ 0.5 · 1e-6 · λ_min(H)   (~ 1e-7 to 1e-6 J)
```

So the loop exits when the system is *essentially at rest* (joint
velocities < ~1 mm/s in mass-weighted norm). The torso has not
moved during DS; the body, joints, and arms are all where they
were at SS-exit, just bled of velocity.

### §1.3  Empirical inter-step DS durations on T15

From the merged-main FK run (`results/M7_1pct_3step_v22_t15_fk/sim_log.json`):

| Phase | t [s] | Duration | Exit reason |
|---|---|---|---|
| DS_0 (initial) | 0.00 → 0.10 | 0.10 s | target_met (post-settle is already at rest) |
| SS_0 (step 0) | 0.10 → 7.21 | 7.11 s | DOCK at d=2.91 mm |
| DS_1 (inter-step) | 7.21 → 7.71 | 0.50 s | target_met (residuals from step 0 dock) |
| SS_1 (step 1) | 7.71 → 19.22 | 11.51 s | DOCK at d=4.84 mm |
| DS_2 (inter-step) | 19.22 → 19.92 | 0.70 s | target_met |
| SS_2 (step 2) | 19.92 → 36.61 | 16.69 s | TIMEOUT at d=412 mm |

The inter-step DS phases (DS_1, DS_2) consume 0.5–0.7 s each.
**During this time, the torso does not move.** It just dissipates
the small post-dock kinetic energy. The body pose at the end of
DS_k is essentially the same as at the end of SS_k (the prior
step's dock pose), modulo a few mm of weld-engagement transient.

### §1.4  The cost of doing all body translation in SS

Step 2's (3,4) anchor pair requires the body to translate ~591 mm
during the step (CoM-conservation calculation; see synthesis
§0.4). The pre-planner allocates `T_step = 12.77 s` for SS_2.
The QP must, in those 12.77 s, *simultaneously*:

1. Hold the stance arm 'a' welded at anchor_a[3].
2. Track the FK torso reference (linear ramp 591 mm + ~3°
   angular).
3. Track the FK swing-EE reference (800 mm + asymmetric clearance
   bump).
4. Hold the CoM on its trajectory.
5. Hold the angular momentum on its trajectory.
6. Maintain joint posture / arm configuration.
7. Stay within wrench / torque / momentum limits.

The QP solves this as a quadratic program with weighted task
costs in a hierarchy (P1: torso, P2: swing null-space-projected
against torso, P3: posture, etc.). With seven simultaneous
demands competing for the same 14 DoF of arm joint actuation +
6 DoF of body floating, the QP's cost minimum lands ~410 mm short
of the swing target.

By contrast, in the QP isolation test (`Misc/runs/M7_step2_isolation/`),
the *same* anchor pair from a clean (3,3) post-settle starting
state reaches **20 mm short** of the swing target with default
margin, and **docks at 4.76 mm** with extended `t_ss_margin = 20 s`.
The difference is the absence of accumulated dynamics from prior
steps — momentum carry-over, joint velocities, AOCS reaction-wheel
state. The QP works fine *if* it doesn't have to fight residuals
on top of all seven concurrent demands.

The architectural insight: the inter-step DS phase is the *correct*
place to discharge the prior-step residuals AND to advance the
body so the upcoming SS doesn't have an impossible task balance.
The current passive settle does only the first half (residual
discharge) and leaves the body translation entirely to SS.

---

## §2  Proposed architecture

The proposal modifies the inter-step DS phase only. Initial DS
and terminal DS keep their current passive-settle behaviour.

### §2.1  Inter-step DS — three sub-phases

```
DS_k (inter-step, between SS_{k-1} dock and SS_k undock):

  ┌─ §2.1.1: short residual settle ────────────────────────────────┐
  │   - Existing _run_ds_passivity_loop, capped at 100–200 ms.     │
  │   - Drives the post-dock impact transient out of the system.   │
  │   - Exits when T_kin < threshold (same target as today's       │
  │     terminal settle, but with a faster cap so we don't bleed   │
  │     time when residuals are already small).                    │
  └────────────────────────────────────────────────────────────────┘
                                ↓
  ┌─ §2.1.2: active torso advance ─────────────────────────────────┐
  │   - Plan q_target_DS: the body pose at which the next SS's    │
  │     swing arm can reach its target with the rest of the SS    │
  │     budget. Computed by §3 reachability gate.                 │
  │   - Build a torso reference from current pose to q_target_DS  │
  │     under the constraint that BOTH arms stay welded.          │
  │   - Run the QP with this reference as the primary task.       │
  │     Stance contacts on both arms enforced by ContactConfig.   │
  │     No swing-arm task (both arms are stance during DS).       │
  │   - Execute until §2.1.3 reachability gate passes OR cap.     │
  └────────────────────────────────────────────────────────────────┘
                                ↓
  ┌─ §2.1.3: reachability gate / DS exit ──────────────────────────┐
  │   - Continuously evaluate: can the swing arm reach the next   │
  │     anchor in T_step from the current body pose? (1-task IK   │
  │     feasibility check or workspace reachability sphere.)      │
  │   - If yes, undock the swing arm and transition to SS.        │
  │   - If max DS duration cap is reached without passing the     │
  │     gate, undock anyway (the body got as close as it could;   │
  │     SS does the residual; same failure mode as today's QP    │
  │     for that anchor pair).                                    │
  └────────────────────────────────────────────────────────────────┘
                                ↓
                             SS phase (existing FK-ref machinery)
```

### §2.2  Choosing the DS-end body pose `q_target_DS`

The objective: pick `q_target_DS` such that the upcoming SS
has the best possible task balance for the QP to solve.

**Definition of "best":** minimise the maximum task demand the
QP will face during SS_k. Given the FK-ref architecture, the
SS task demand is dominated by the swing arm's required arc
length in world frame. So `q_target_DS` should put the body in
a pose from which the swing-EE *natural geodesic* to anchor_b[i+1]
is short.

**Concrete construction (recommended):**

1. Run `manipulability_config_trajectory(model, anchor_a[i], anchor_b[i+1], q_start=q_pq_live)`
   — the same IK call SS uses today — to get `q_end_SS`. This is
   the dock target.
2. Define `q_target_DS = pin.interpolate(q_pq_live, q_end_SS, β)`
   for some `β ∈ (0, 1)`. Typical value: `β = 0.7` (advance the
   body 70% of the way to the SS target during DS).
3. Project `q_target_DS` to the **double-stance** constraint
   manifold via a 2-task IK (both arms pinned at their current
   anchors) seeded by `q_target_DS`. The projected
   `q_target_DS_proj` may differ from `q_target_DS` because the
   double-stance manifold is more constrained than single-stance
   (12 DoF locked vs 6 DoF locked).

The `β = 0.7` heuristic comes from the observation that the
double-stance manifold's "reach" (how far the body can advance
under both-arms-welded) is bounded by the arms' joint limits.
For T15 anchor pairs spaced 0.8 m apart, with arms of ~1.7 m
reach, the double-stance manifold permits ~70 % of the body
translation needed before the trailing arm hits its workspace
limit. Step 2 needs 591 mm of body translation total; with
β = 0.7 the DS would do ~415 mm and SS would do the remaining
~175 mm — both within comfortable QP tracking range.

`β` can be made anchor-pair-aware in a follow-up: for shorter
traversals (steps 0/1 in T15) use β = 0.3; for longer (step 2,
or higher mass ratio scenarios) use β = 0.7+.

### §2.3  DS torso reference

Once `q_target_DS_proj` is selected, the DS torso reference is
constructed analogously to the FK-ref machinery used in SS:

```
q_DS_seq[k] for k = 0, 1, ..., n_DS-1:
    a smoothed q-sequence on the DOUBLE-stance constraint
    manifold from q_pq_live to q_target_DS_proj.
```

The smoother of `crawlbot/planning/constrained_geodesic.py`
already supports "constraint manifold smoothing"; it just
needs an additional version that pins **two** EE poses
(stance_a + stance_b) instead of one. Call it
`smoothed_constrained_geodesic_double_stance`. The
implementation is a copy of the existing function with a
2-task IK in place of the 1-task projection.

The **DS torso reference** at any τ_DS ∈ [0, 1] is:
```
q(τ_DS) = piecewise pin.interpolate(q_DS_seq, τ_DS)
torso_ref(τ_DS) = FK[fid_torso](q(τ_DS))     # 6D pose + twist
```

This is identical in shape to the existing SS torso reference
(plan_v2 §2.6). The QP consumes it the same way.

### §2.4  Reachability gate

At each QP tick during DS_k §2.1.2, evaluate whether the
upcoming SS can plausibly succeed from the current body pose.
Two complementary checks:

**Cheap check — workspace sphere.** Compute
`d_reach = ‖p_torso_now − anchor_b[i+1]‖`. If `d_reach <
arm_max_reach − safety_margin`, the swing arm CAN physically
extend to the anchor. If `d_reach ≥ arm_max_reach`, it cannot.

For T15 with arm reach ≈ 1.7 m and safety margin 0.1 m, the
cheap check is `d_reach < 1.6 m`. From the initial body pose
to anchor_b[4] this is ~1.4 m — already passes. So this check
is a "no impossibility" guard; it doesn't say the arm can
DOCK in T_step, only that it CAN reach in principle.

**Expensive check — task-space-time reachability.** Run a
1-task IK with the swing arm pinned at anchor_b[i+1]
(stance still welded both sides). If the IK converges with
manipulability w_min ≥ threshold (~1e-2), the arm can dock.
If not, more body advance is needed.

This gate fires at every QP tick during DS_k §2.1.2; the
DS phase exits to SS the moment the gate passes.

### §2.5  Three-class DS dispatch

The fix to `sim_loop.py`'s phase loop (around line 1390):

```python
def _ds_class(i_phase, n_phases):
    if i_phase == 0:
        return 'initial'   # passive settle (existing behaviour)
    if i_phase == n_phases - 1:
        return 'terminal'  # passive settle (existing behaviour)
    return 'inter_step'    # active torso advance (new)

while i < len(phases):
    gp = phases[i]
    if gp.phase.value == 'double':
        ds_class = _ds_class(i, len(phases))
        if ds_class in ('initial', 'terminal'):
            ds_result = self._run_ds_passivity_loop(...)
        else:  # 'inter_step'
            # 2.1.1: brief residual settle (cap 200 ms)
            self._run_ds_passivity_loop(..., max_steps=20)
            # 2.1.2: plan q_target_DS, build smoothed q_DS_seq
            q_DS_seq, _ = smoothed_constrained_geodesic_double_stance(
                ..., q_start=pq_live, q_target=q_target_DS_proj)
            # 2.1.3: active torso tracking + reachability gate
            self._run_ds_active_advance(
                q_DS_seq=q_DS_seq, swing_arm=ss_gp.swing_arm,
                target_anchor=anchor_b_or_a[swing_to_idx],
                T_DS_max=cfg.t_ds_active_max)
    elif gp.phase.value != 'double':
        # SS phase — existing FK-ref machinery, unchanged
        ...
```

The new helper `_run_ds_active_advance` is the inter-step DS
runtime loop. It runs the QP at `dt_qp` with the DS torso
reference, evaluates the reachability gate every N ticks
(every ~50 ms is fine), and exits when the gate passes or the
cap is hit. NMPC may or may not be active during DS; see §3.5.

### §2.6  Terminal DS — explicit class

The terminal DS keeps its current behaviour (passive settle to
rest at the destination), with one improvement: it should
explicitly carry the LAST step's `swing_arm` reference until
the dock is confirmed, then fully release. This is a small
extension; not strictly required for the active-DS proposal.

### §2.7  What the SS phase does under this architecture

Largely unchanged. The pre-planner still runs at SS-entry to
choose T_step. The smoother still produces the FK q_seq. The
QP still tracks FK refs.

The only change: at SS-entry, `pq_live` is now the post-DS-advance
state (not the post-dock state from the prior step). The
smoother takes the same (pq_live, q_end_SS) pair and produces a
shorter q_seq because pq_live is closer to q_end_SS in joint space.

The expected outcome for step 2 with β = 0.7:
- pq_live at SS-entry of step 2: body has advanced ~415 mm of
  the 591 mm needed.
- Smoother produces a q_seq with body advancing the residual
  ~176 mm + arm extending the residual ~250 mm in world frame.
- SS task balance becomes much closer to step 0/1's regime.
- Dock criterion crosses at d ≤ 5 mm well before T_step + margin.

---

## §3  Math derivation

### §3.1  Notation

Reusing the conventions from synthesis §0 and plan-v2 §2.1:

- Configuration manifold `Q = SE(3) × T¹⁴`. `nq = 21`, `nv = 20`.
- Frame IDs: `fid_a` (tool A), `fid_b` (tool B), `fid_torso`.
- Tangent vector `v ∈ ℝ²⁰` = `[v_b(3), ω_b(3), v_arm(14)]` in
  Pinocchio LOCAL convention for the free-flyer block.
- Anchor poses in initial-structure-local frame:
  `anchor_a[i]`, `anchor_b[j]`.
- `M_double(q_a_anchor, q_b_anchor) := { q ∈ Q :
    FK[fid_a](q) = q_a_anchor AND
    FK[fid_b](q) = q_b_anchor }` — the double-stance
  constraint manifold (12 DoF locked, body has 6 DoF − 6 wrench-
  redundancy + 8 arm-redundancy = ~8 effective DoF along which
  it can move).

### §3.2  The inter-step DS planning problem

Given:
- `q_DS_start = pq_live` at the start of the inter-step DS
  (post-prior-SS-dock + brief residual settle). Both arms welded
  at their current anchors `anchor_a[i_a]`, `anchor_b[i_b]`. So
  `q_DS_start ∈ M_double(anchor_a[i_a], anchor_b[i_b])`.
- `q_SS_end = q_end_SS` from `manipulability_config_trajectory(...)`
  for the upcoming SS_k. This is on the manifold of the *next* SS
  phase: stance arm at `anchor_a[i_a]` (or `anchor_b[i_b]`,
  whichever is staying), swing arm at the new anchor. So `q_SS_end
  ∈ M_double(anchor_a[i_a'], anchor_b[i_b'])` for the new pair —
  but the swing arm's old anchor is no longer pinned at q_SS_end.

In other words: at the start of DS, the swing arm is *currently*
welded at its old anchor. At the end of DS, the swing arm
*should still be* welded at its old anchor (DS doesn't release
yet). At the end of SS, the swing arm is welded at its new
anchor.

So during DS, the configuration must remain on
`M_double(anchor_a[i_a], anchor_b[i_b])` (the *current* pair).
The DS-end target `q_target_DS_proj` is therefore computed by:

1. Take `q_target_DS_unconstrained = pin.interpolate(q_DS_start,
   q_SS_end, β)` for some `β ∈ (0, 1)`.
2. Project to `M_double(anchor_a[i_a], anchor_b[i_b])` via a
   2-task IK pinning **the current** stance pair.

The projection is well-defined because at q_DS_start the system
is already on the manifold; for small `β` the projection is a
small local correction; for larger `β` the projection may push
the body further from `q_target_DS_unconstrained` because
constraint curvature is larger.

### §3.3  The DS reference path

Apply the §2.2 task-space smoothing to construct
`q_DS_seq[k] for k = 0, 1, ..., N_DS-1` on
`M_double(anchor_a[i_a], anchor_b[i_b])`. The smoother
minimises world-frame torso arc length subject to the double-
stance constraint at every interior k.

**Algorithm sketch (mirrors plan-v2 §2.2 with two stance
constraints):**

```
algorithm  smoothed_constrained_geodesic_double_stance(
    q_DS_start, q_target_DS_proj, fid_stance_a, fid_stance_b,
    fid_torso, n_tau=11, n_iter=80, tol=1e-5):
    
    s_grid = quintic time-scaling samples on [0, 1]
    q_seq = [project_to_double_stance(
                pin.interpolate(q_DS_start, q_target_DS_proj, s),
                stance_a_target=anchor_a[i_a],
                stance_b_target=anchor_b[i_b])
             for s in s_grid]
    q_seq[0] = q_DS_start   # endpoint pinned
    q_seq[-1] = q_target_DS_proj   # endpoint pinned
    
    for it in 0..n_iter-1:
        for k in 1..n_tau-2:
            # Task-space midpoint for torso ONLY (not swing — both
            # are stance during DS)
            target_torso = SE3 midpoint of (FK[torso](q_seq[k-1]),
                                            FK[torso](q_seq[k+1]))
            target_a = SE3(I, anchor_a[i_a])  # pinned
            target_b = SE3(I, anchor_b[i_b])  # pinned
            q_new = ik_three_tasks(model, fid_torso, fid_a, fid_b,
                                   target_torso, target_a, target_b,
                                   seed=q_seq[k])
            q_seq[k] = q_new
        if max τ-update over interior < tol: break
    
    return q_seq
```

The 3-task IK during smoothing pins both stance arms (the
constraint we want to enforce throughout DS) plus the torso
target (the objective we want to minimise arc length on).
Compare this to the SS smoother which has stance + torso +
swing (where swing is moving toward a target).

`N_DS = 11` (instead of SS's 21) is enough resolution for the
shorter DS phase. `n_iter = 80` is conservative; convergence is
usually faster on the double-stance manifold than on the
single-stance manifold (more constraints ⇒ smaller search
space).

### §3.4  DS dynamic feasibility — torque & momentum

During DS_k §2.1.2, the QP applies torques to drive the system
along the DS reference. The dynamic constraints are the same as
SS:

- `tau_q ∈ [-tau_max, +tau_max]^14` (joint torque limit, 20 Nm).
- `tau_w ∈ [-tau_w_max, +tau_w_max]^3` (AOCS torque limit, 5 Nm).
- `hw ∈ [hw_min, hw_max]^3` (AOCS momentum limit, ±5 Nms).
- `‖λ‖ ∈ ‖[λ_a, λ_b]‖ ≤ wrench_max` (contact wrench limit).

The key advantage during DS: **only one task is moving** (torso).
The QP has 6-DoF of feedforward demand, not 12. Joint torques
will be lower; AOCS will be less saturated; wrench loads will be
smaller because the body's CoM motion is the only momentum
source, not the swing arm.

Quantitative prediction for T15 step 2's DS_2 (with β=0.7):
- Body translation during DS: ~415 mm
- Body angular displacement: ~2° (small fraction of the SS-side ~3°)
- DS duration: ~5 s (so body avg speed ~83 mm/s, similar to SS-current)
- Peak joint torque: well under 4 Nm (single-task QP is much
  cheaper than the 7-task SS QP)
- Peak AOCS torque: smaller than SS because no swing arm to
  compensate (probably ~2 Nm)

### §3.5  NMPC in inter-step DS — yes or no?

Two options:

**Option A: NMPC OFF during inter-step DS.** Same as today's
passive settle. The DS QP runs in `settle_mode=False,
passivity_active=False`, with the new torso reference. This is
the simpler implementation. The CoM and L_com tasks are dropped
from the QP cost during DS (no NMPC means no `r_com_ref`,
`v_com_ref`, `lambda_ref`, `L_com_ref`).

**Option B: NMPC ON during inter-step DS.** The NMPC is given
a reduced state (no swing arm dynamics) and a torso-tracking
reference. It computes contact-wrench feedforwards. The QP
includes the CoM/L_com costs.

For the first cut, **Option A is recommended.** The DS phase
is short (few seconds), the body motion is modest, and NMPC's
value-add is marginal compared to the implementation cost. The
QP's torso-tracking + wrench-regularisation should be enough.
Option B can be a follow-up if Option A's tracking is poor.

### §3.6  Reachability gate math (§2.4 detail)

At each QP tick during DS_k §2.1.2, evaluate whether the swing
arm can reach `anchor_b[i_b']` in time T_step from the current
body pose `q_now`. The gate has two components:

**Workspace sphere (cheap, always run first):**
```
d_reach = ‖FK[fid_torso](q_now).translation - anchor_b[i_b']‖
gate_workspace = (d_reach < arm_max_reach - safety_margin)
                                                ^^^^^^^^^^^^^
                                               ~ 0.1 m (10 % of reach)
```

If this fails, the body is geometrically too far from the
anchor; continue DS advance.

**1-task IK reachability (expensive, run only when gate_workspace
passes):**
```
q_test = solve_ik(model, q_now,
                  targets={fid_swing: SE3(I, anchor_b[i_b'])},
                  max_iter=200, tol=1e-4)
gate_ik = (q_test.converged AND
           σ_min(J_swing(q_test)) >= w_min_threshold)
```

Where `w_min_threshold ≈ 0.01` to 0.05 (well-conditioned
swing-arm Jacobian at the candidate dock pose).

If both gates pass: **trigger SS undock.** Else continue DS.

The 1-task IK is cheap (~1–10 ms) at most. Running it every
~50 ms (every 5th QP tick at 100 Hz) is negligible overhead.

### §3.7  Math soundness summary

| Item | Claim | Verification |
|---|---|---|
| §3.2 | `q_target_DS_proj ∈ M_double(...)` to numerical tol | Test D.1 |
| §3.3 | DS smoother converges with ≤40 μm stance compliance | Test D.2 (analogue of Phase-0) |
| §3.4 | Inter-step DS QP has lower peak torques than SS | Run-time measurement |
| §3.5 | Option A (NMPC off) DS produces stable torso tracking | Test D.3 closed-loop |
| §3.6 | Reachability gate fires correctly when DS body advance is sufficient | Test D.4 (synthetic poses) |

---

## §4  Files to modify

### §4.1  New module: `crawlbot/planning/double_stance_planner.py`

Public API (~250 lines):

| Function | Purpose |
|---|---|
| `compute_q_target_DS(model, q_DS_start, q_SS_end, beta, fid_a, fid_b, anchor_a_now, anchor_b_now)` | Eq. §2.2: pin.interpolate from q_DS_start to q_SS_end at β, project to M_double via 2-task IK. |
| `smoothed_constrained_geodesic_double_stance(model, q_DS_start, q_target_DS_proj, fid_torso, fid_a, fid_b, anchor_a_now, anchor_b_now, n_tau=11, n_iter=80, tol=1e-5)` | Eq. §3.3: task-space smoother with both-stance constraints. Returns `q_DS_seq + info`. |
| `reachability_gate(model, q_now, fid_swing, anchor_swing_target, arm_max_reach, w_min_threshold)` | Eq. §3.6: workspace sphere + 1-task IK feasibility check. Returns `(passed, q_test, info)`. |

The first two reuse `project_to_stance` and `ik_three_tasks`
patterns from `crawlbot/planning/constrained_geodesic.py`.

### §4.2  Edit: `crawlbot/planning/torso_planner.py`

Add a new method `add_phase_double_stance(t_start, t_end,
q_DS_seq, n_tau)` mirroring the existing `add_phase` for SS but
flagging `phase['kind'] = 'DS_active'`. The reference_at logic
branches: DS_active uses `q_DS_seq` interpolation, SS uses the
existing FK ref machinery. Effectively a small generalisation
of the existing `use_fk` flag to `kind ∈ {SS_FK, DS_active,
legacy_slerp}`.

### §4.3  Edit: `crawlbot/simulation/sim_loop.py`

Three changes:

1. **`_ds_class(i_phase, n_phases)` helper** (new, ~10 lines).
   Returns 'initial', 'inter_step', or 'terminal' based on
   phase index.

2. **Phase-loop dispatch** (sim_loop.py:1390 area, ~30 lines).
   The `if gp.phase.value == 'double':` branch chooses behaviour
   based on `_ds_class`. Initial/terminal call `_run_ds_passivity_loop`;
   inter_step calls the new `_run_ds_active_advance`.

3. **`_run_ds_active_advance`** (new method, ~120 lines).
   Implements §2.1.1 + §2.1.2 + §2.1.3:
   - Brief residual settle (~100–200 ms).
   - Plan q_target_DS_proj via §4.1's
     `compute_q_target_DS`.
   - Build q_DS_seq via §4.1's `smoothed_constrained_geodesic_double_stance`.
   - Wire `torso_planner.add_phase_double_stance(...)`.
   - Run QP loop at dt_qp, evaluating reachability gate at every
     ~50 ms. Exit when gate passes or T_DS_active_max hits.

### §4.4  Edit: `crawlbot/simulation/config.py`

Add the active-DS knobs:

```python
# Active-DS torso advance (inter-step DS phases only).
# Initial and terminal DS keep their passive-settle behaviour.
ds_active_enabled: bool = False    # Default off; enable per-runner.
ds_active_beta: float = 0.7         # §2.2: fraction of SS body
                                    # advance to do during DS.
ds_active_n_tau: int = 11           # §3.3: smoother τ-grid.
ds_active_n_iter: int = 80          # §3.3: smoother max iters.
ds_active_residual_settle_max_ms: int = 200  # §2.1.1
t_ds_active_max: float = 5.0        # §2.5: DS active-advance cap [s]

# Reachability gate (§3.6).
ds_reach_arm_max_reach: float = 1.7
ds_reach_safety_margin: float = 0.1
ds_reach_w_min_threshold: float = 0.02
ds_reach_check_every_ticks: int = 5  # eval gate every N QP ticks
```

The default `ds_active_enabled = False` keeps the existing
passive-settle behaviour byte-identical for everything except
the new test runner.

### §4.5  Files NOT touched

- `crawlbot/planning/constrained_geodesic.py` — already has the
  primitives we need (`project_to_stance`, `ik_three_tasks`).
- `crawlbot/planning/swing_planner.py` — SS phase unchanged.
- `crawlbot/solvers/wholebody_qp.py` — same QP, same task stack;
  during DS_active the swing-arm task is simply not added (only
  torso + 2× stance).
- `crawlbot/solvers/centroidal_nmpc.py` — Option A from §3.5
  (NMPC off in inter-step DS). No NMPC change needed.
- `crawlbot/core/ik.py` — no IK change.
- MJCF, URDF, gait scheduler temporal structure — all unchanged.

---

## §5  Tests

New test file: `tests/test_active_ds.py`. Six tests.

| ID | Validates |
|---|---|
| D.1 | `compute_q_target_DS` projects onto M_double to ≤ 50 μm stance compliance for synthetic (q_DS_start, q_SS_end) pairs across β ∈ {0.3, 0.5, 0.7, 0.9}. |
| D.2 | `smoothed_constrained_geodesic_double_stance` returns a sequence with both endpoints exact, ≤ 50 μm stance compliance for both arms at every interior k, ≤ 110 % world-frame torso arc length inflation vs raw chord. |
| D.3 | Closed-loop sanity: drive a synthetic q_DS_start to q_target_DS in 5 s under double-stance constraints; assert max torso tracking error < 50 mm, joint torque peak < 10 Nm, AOCS τ_w peak < 5 Nm. |
| D.4 | `reachability_gate` returns False before sufficient body advance, True after. Use synthetic body poses sweeping toward the anchor. |
| D.5 | T15 multi-step 3-step run with `ds_active_enabled=True`: assert step 2 docks at d ≤ 5 mm, all three steps complete, total run time ≤ baseline + 6 s (extra DS time budget). |
| D.6 | T15 multi-step with `ds_active_enabled=False`: byte-identical to merged-main behaviour (regression test). |

D.5 is the headline. D.6 protects the existing pipeline.

### §5.1  Closed-loop validation runs

| Runner | Purpose | Expected |
|---|---|---|
| `scripts/run_m7_v22_1pct_3step_t15_active_ds.py` | T15 with active DS | All 3 steps dock; step 2 d ≤ 5 mm |
| `scripts/run_m7_v22_14pct_3step_t16_active_ds.py` | Higher mass ratio (T16) | All 3 steps dock; AOCS within budget |
| `scripts/diagnostic_ds_torso_advance.py` | Per-step DS budget breakdown: how much body translation each DS does, peak torque/AOCS, residual SS demand | Verify β=0.7 split is reasonable across step 0/1/2 |

---

## §6  Implementation phases and effort

Phase 0 of this plan would be a *pre-flight* analogous to plan-v2 §4.0:
write a diagnostic that, given a captured (q_DS_start, q_SS_end) pair
from the merged-main FK run, computes q_target_DS_proj and the
DS smoother output, and reports torso advance vs SS residual. This
proves the geometric premise (β=0.7 is achievable on the
double-stance manifold) before any sim_loop edits land.

### Phase 0 — Pre-flight (0.5 day)

- Write `scripts/diagnostic_ds_torso_advance.py`.
- For T15 step 2 (and steps 0, 1 for context): compute
  q_target_DS_proj at β ∈ {0.3, 0.5, 0.7, 0.9}; report 2-task
  IK convergence, double-stance compliance, world-frame torso
  advance.
- **Exit gate:** β=0.7 is achievable on M_double for all 3 steps
  (2-task IK converges with both stance residuals ≤ 50 μm). If
  failed, lower β until feasible; if unfeasible at any β,
  the manifold is too constrained for this fix and we re-think.

### Phase 1 — Double-stance smoother + reachability gate (1.5 days)

- Implement `crawlbot/planning/double_stance_planner.py` with the
  three public functions from §4.1.
- Write tests D.1, D.2, D.4.
- **Exit gate:** D.1, D.2, D.4 pass on synthetic q-pairs.

### Phase 2 — TorsoPlanner DS_active mode (0.5 day)

- Add `add_phase_double_stance` and the `kind` flag to phase
  dict.
- Reuse `frame_reference_at_tau` from
  `crawlbot/planning/constrained_geodesic.py` (no change to
  that module).
- **Exit gate:** synthetic test calling `reference_at` over a
  DS_active phase returns sensible torso refs; legacy code
  paths byte-identical.

### Phase 3 — sim_loop active-DS dispatch (1 day)

- Implement `_ds_class`, `_run_ds_active_advance`,
  reachability-gate runtime loop.
- Wire `cfg.ds_active_enabled` flag.
- **Exit gate:** test D.6 byte-identical with flag off; test
  D.3 closed-loop sanity passes.

### Phase 4 — T15 closed-loop validation (1 day)

- Write `scripts/run_m7_v22_1pct_3step_t15_active_ds.py`.
- Run T15 with active DS enabled.
- Per-step breakdown via `diagnostic_ds_torso_advance.py`.
- Tune β if needed (start at 0.7, sweep if necessary).
- **Exit gate:** test D.5 passes (all 3 steps dock at d ≤ 5 mm).

### Phase 5 — Default flip + T16 regression (0.5 day)

- Flip `cfg.ds_active_enabled = True` default.
- Verify T15 baseline runs still pass (regression).
- Run T16 (14% mass ratio) closed-loop; confirm step 2 dock and
  AOCS within budget.
- **Exit gate:** all existing T15 tests pass with active DS as
  default; T16 step 2 docks at d ≤ 5 mm.

### Phase 6 — Documentation + merge (0.5 day)

- Write `T15_active_ds_validation_report.md` (the closing
  report analogue to PR #15's).
- Update `CLAUDE.md` "Current Milestone" + parameter table.
- Update synthesis at
  `Misc/reports/architecture/T15_step2_diagnosis_and_resolution.md` with a
  §12 "Resolution: active DS" pointing at the new report.
- **Exit gate:** branch ready for PR.

**Total effort: ~5 days** of focused engineering, plus 0.5–1 day
buffer for IK/QP convention surprises (Phase 1 is the highest-
risk part — the 2-task IK on the double-stance manifold may
have multiplicity or singularity issues we haven't seen).

---

## §7  Risks and mitigations

| ID | Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|---|
| R1 | β=0.7 is not achievable on M_double for T15 step 2 (the manifold is too constrained or the 2-task IK fails) | Low–Medium | High | Phase 0 pre-flight catches this. If failed, lower β until 2-task IK converges. If even β=0.3 fails, the architecture must be reconsidered (e.g. relaxed stance constraints during DS, fork the double-stance manifold per anchor pair). |
| R2 | Active DS torso reference's velocity profile destabilises the QP | Low | Medium | Same risk as plan-v2's R3 (C⁰-vs-C² reference smoothness). Solution: fit a C² spline through q_DS_seq if needed. The DS phase has more time per arc length than SS, so velocity demands are smaller; the issue is less likely to bite. |
| R3 | Residual settle (§2.1.1) at the start of DS isn't long enough to discharge the post-dock kinetic energy, so DS active-advance starts with non-zero velocity | Medium | Low | Cap the residual settle at 200 ms but require T_kin < target before advancing. If target not met, extend cap. Worst case: a few extra ms of settle time per step. |
| R4 | Reachability gate fires too late (DS overruns) or too early (SS undocks before body is positioned well enough) | Medium | Medium | The gate has a `T_ds_active_max` cap (5 s default) that triggers SS undock even if gate fails. Conservative `w_min_threshold = 0.02` to avoid false positives. Run-time tunable. |
| R5 | Inter-step time grows by 5 s (DS_active_max) per step, total run time grows ~10–15 s for T15 | Low (purely cosmetic) | Low | Acceptable trade-off: 10 s longer run time for 3-step robustness. Not a real-time constraint. |
| R6 | The double-stance manifold for some anchor pairs is locally singular (joint limit / collision) — IK fails | Low (workspace volumes are well-separated for T15 mass ratio) | High | Phase 0 pre-flight + a per-step manifold check during planning. If a step's M_double has dimension < 6 (over-constrained), fall back to passive settle for that step. |
| R7 | NMPC drift over the longer DS phase if Option A (NMPC off) is chosen — the L_com state at SS-entry is no longer fresh | Low | Low | The NMPC re-plans at SS-entry anyway. The DS-OFF is fine for short DS (~5 s); if active DS exceeds 10 s, revisit. |
| R8 | Active DS interacts with AOCS hw bookkeeping — wheel momentum drifts during DS body advance, leaving SS in a non-zero hw state | Medium | Low | The QP's wrench regularisation handles this implicitly; AOCS τ_w during DS is well within budget per §3.4 prediction. If hw drift becomes a problem, add an explicit hw-target task to the DS QP cost (drives hw → 0 at end of DS). |

---

## §8  Open questions

These are honest unknowns the implementation will discover. None
should block Phase 0; they're things to keep eyes on during Phases
1–5.

### §8.1  How to choose β per anchor pair

The plan uses a fixed β = 0.7 with a placeholder for "anchor-pair-
aware β". The right choice may depend on:
- The anchor-pair distance (T15: 0.8 m; later configs may differ).
- The mass ratio (T16's higher mass ratio means more body recoil
  per arm motion, which may want a smaller β).
- The double-stance manifold's local geometry at q_DS_start.

A simple anchor-pair-aware policy: β proportional to the
required body translation, capped at 0.85. Alternatively: solve
a small optimisation at SS-entry to pick β minimising the SS
peak-task-demand. The first is cheap; the second is principled
but slower.

For the first cut: hard-code β = 0.7 and observe. Tune in
Phase 4 if T16 (or other regimes) reveal it's wrong.

### §8.2  Reachability gate vs T_DS_active_max — which dominates?

Two exit conditions for DS_active: (a) gate passes, (b) cap.

Ideally (a) always fires before (b). If (b) fires first, the
body didn't advance enough — SS will inherit a harder-than-
needed task. Diagnostic logging should record which fired and
the eventual SS dock outcome, so we can tune the cap or β.

### §8.3  Should DS_active be opt-in per step or globally?

Globally `cfg.ds_active_enabled` is the simplest. Per-step
opt-in would let the user say "use active DS only for the
hardest steps" — but choosing which step is hard is itself a
diagnostic step. For the first cut, global on/off; per-step
gate can be added if regressions on easy steps appear.

### §8.4  Higher mass ratios (T16, T17, T18)

The T15 1% scenario is mass-ratio-light: arm motion creates
small body recoil. T16 14% has 14× more recoil per arm motion;
the SS task balance is much harder. Active DS should generalise
to higher mass ratios because:

- The double-stance manifold's kinematics are mass-independent
  (geometry only).
- The QP's torque demand during DS scales with mass, but DS has
  no swing arm task competing — there's headroom.

But the *required* β will likely grow with mass ratio. T16's
step 2 may need β = 0.85 or higher. Phase 5 of the
implementation runs T16 to confirm.

### §8.5  Multi-step T15 step-1 regression?

Step 1 (T15) had 995 mm of mid-traj e_ee_pos in the merged-main
FK run but still docked at 4.84 mm. With active DS, step 1 will
do part of its body advance during DS_1, leaving a smaller SS
task. **It should dock cleaner than 4.84 mm** under active DS.
But if step 1 regresses (e.g., DS_1 introduces some new mode),
that's a regression to address.

### §8.6  Synthesis update

The synthesis at
`Misc/reports/architecture/T15_step2_diagnosis_and_resolution.md`
identified the structural failure mode (kinematically-uncoupled
refs at SS interior τ) and proposed §7's FK-ref fix. PR #15
delivered that fix. The synthesis's §3 narrative should be
extended in a §12 once active DS lands, framing it as: "PR #15
eliminated the *kinematic* infeasibility; active DS eliminates
the *task-balance* impossibility that remained."

---

## §9  Artifact index

### §9.1  This proposal

- `Misc/reports/architecture/active_ds_torso_advance.md` (this file)

### §9.2  Predecessor work referenced

- `Misc/reports/architecture/T15_step2_diagnosis_and_resolution.md` —
  synthesis (PR #15 base).
- `docs/architecture/IK_FORMULATION.md` — IK formulation spec.
- `results/M7_1pct_3step_v22_t15_fk/CLOSING_REPORT.md` — PR #15
  closing report (the failure-mode pinpointing that motivated
  this proposal).

### §9.3  Diagnostic data this proposal builds on

- `results/M7_1pct_3step_v22_t15_fk/{sim_log.json,
  physics_trace.pkl}` — multi-step FK baseline (steps 0/1 dock,
  step 2 fails 412 mm).
- `Misc/runs/M7_step2_isolation/A_bypass_on/{sim_log.json,
  physics_trace.pkl}` — clean-(3,3) isolation run (step 2 fails
  20 mm).
- `Misc/runs/M7_step2_isolation_xlong_hold/xlong_hold/{sim_log.json,
  physics_trace.pkl}` — clean-(3,3) with 20 s margin (step 2
  docks at 4.76 mm; proves the QP works given enough time and a
  clean start).
- `Misc/runs/M7_step2_isolation_aocs_unleashed/aocs_unleashed/...`
  — AOCS at 100 Nm / 100 Nms; same 20 mm result (rules out
  AOCS budget).
- `Misc/runs/M7_1pct_3step_v22_t15_fk_margin5/...` — multi-step with
  `t_ss_margin = 5 s`; step 2 still 425 mm (rules out time
  budget alone).

These collectively prove:
1. Step 2 fails specifically because of the multi-step momentum
   carry-over interacting with the SS task balance (not due to
   AOCS budget, joint torque budget, time budget, or the FK
   refs themselves).
2. Step 2 is achievable from a clean start (4.76 mm dock).
3. The remaining gap from "clean start" to "multi-step
   feasibility" is the gait architecture's failure to use DS for
   body translation.

### §9.4  Files to be added / edited (planned)

| Path | Status |
|---|---|
| `crawlbot/planning/double_stance_planner.py` | NEW (~250 lines) |
| `crawlbot/planning/torso_planner.py` | edit (+~80 lines for DS_active mode) |
| `crawlbot/simulation/sim_loop.py` | edit (+~150 lines for `_ds_class`, `_run_ds_active_advance`) |
| `crawlbot/simulation/config.py` | edit (+9 fields) |
| `tests/test_active_ds.py` | NEW (~400 lines, 6 tests) |
| `scripts/run_m7_v22_1pct_3step_t15_active_ds.py` | NEW (~80 lines) |
| `scripts/run_m7_v22_14pct_3step_t16_active_ds.py` | NEW (~80 lines) |
| `scripts/diagnostic_ds_torso_advance.py` | NEW (~200 lines) |
| `results/M7_1pct_3step_v22_t15_active_ds/T15_ACTIVE_DS_REPORT.md` | NEW (closing report) |

**Total new code: ~1300 lines** including tests and reports;
**edits to existing code: ~240 lines.**

### §9.5  Branch lineage

```
main (post-PR-15: 59de7b2)
  └── claude/active-ds-architecture (this proposal)
        ├── Phase 0 pre-flight diagnostic
        ├── double_stance_planner.py
        ├── TorsoPlanner DS_active mode
        ├── sim_loop dispatch + active-advance loop
        ├── 6 tests
        ├── T15 closed-loop validation
        └── T16 regression
```

---

**End of proposal.** Phase 0 pre-flight is the next concrete
action; if its exit gate passes, Phases 1–5 follow per §6.
