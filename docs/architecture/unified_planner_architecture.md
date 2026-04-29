# Unified planning architecture — proposal

**Plan-of-record location:** `docs/architecture/unified_planner_architecture.md`
**Status:** proposal, not implemented.
**Date:** 2026-04-28.
**Predecessors (read first):**
- `docs/architecture/T15_step2_diagnosis_and_resolution.md` — synthesis from PR #15.
- `docs/architecture/active_ds_torso_advance.md` — proposal that hit a fundamental architectural wall in Phase 4.
- `results/M7_1pct_3step_v22_t15_fk/CLOSING_REPORT.md` — PR #15 closing.
- This file's predecessors live on branches `claude/step2-path-diagnostic`, `claude/fk-bypass-aware-tuning`, `claude/active-ds-architecture`. Read those branch tips for the full diagnostic chain.

---

## §0  TL;DR

The control stack (whole-body QP + AOCS) is sound. Multiple
ablations on the failing T15 step-2 case have ruled out QP
tuning, AOCS budget, joint torque budget, and time budget as
the binding constraint. **The failure is in the planning layer
and in the contract between the planning layer and the
controllers.**

What's actually wrong:

1. **Seven user-toggled bypass flags** plus two QP cost-mode
   flags plus a hard-coded NMPC bypass during DS phases. Each
   flag was correct in isolation; together they make the system
   compose differently under different runs. This is the codebase
   smell the user identified ("we turn it on and off whenever
   it's needed").

2. **Planning is fragmented across five modules** that each
   produce one slice of the reference: `coarse_preplanner`
   (CoM + T_step + λ_ref), `TorsoPlanner` (torso 6D), `SwingPlanner`
   (swing-EE 6D), `CoMToTorsoMapping` (M5 layer), the new
   `constrained_geodesic` smoother. They have inconsistent
   conventions, overlapping responsibilities, and per-flag
   on/off behaviour. There is no single source of truth for
   "what should the system be doing at time t".

3. **The DS phase is treated as second-class.** The pre-planner
   only sizes SS phases. The smoother only produces SS q-sequences.
   The torso planner has no DS reference. The QP runs in
   `settle_mode=True` during DS, which switches its cost. The
   NMPC is bypassed entirely. Every active-DS attempt has had to
   re-invent first-class DS planning from scratch.

4. **The NMPC↔QP interface is opportunistic, not a contract.**
   NMPC outputs (`r_com_ref`, `v_com_ref`, `λ_ref`, `a_com_ff`,
   `L_com_ref`) are *only sometimes* meaningful. The QP can run
   with NMPC off (DS settle: caller passes zeros), with NMPC on
   (SS), or with the NMPC's outputs ignored (mapping bypass).
   No invariant exists that the controller can rely on.

The proposed fix is foundational, not incremental:

- **One unified planner** that emits a continuous reference
  `(q(t), v(t), λ_ff(t))` for the entire gait — DS, SS, and
  transitions — defined by the contact schedule, the kinematics,
  and the dynamics. No phase-specific bypasses; no flag-driven
  branches.
- **One NMPC contract**: always consumes the planner's reference,
  always emits a wrench/momentum ref. DS becomes a phase where the
  planner's reference happens to demand near-zero motion — the
  NMPC sees this and emits the appropriate near-zero wrench.
- **One QP contract**: always tracks NMPC's outputs subject to
  the contact constraints. No `settle_mode`, no `passivity_active`
  flags. Settling is just a planner output that says "stay here";
  the QP tracks "stay here".

The seven user-toggled flags collapse to one (or zero). The
QP's two cost-mode flags disappear. The NMPC bypass disappears.
The planning surface shrinks to one module with one contract.

**This is not an incremental fix — it is an architecture rewrite.**
Estimated scope: 4–8 weeks, two engineers if external; 6–12 weeks
solo. Several existing modules get deleted in the process; the
diff is net-negative in lines of code. The QP and AOCS layers
are *preserved* (this proposal does not touch them).

---

## §1  What's actually broken

### §1.1  The bypass-flag pathology

```
Toggle                              Purpose                                    Bypasses
─────────────────────────────────────────────────────────────────────────────────────────
cfg.use_trajectory_aware_ik         Phase-4 IK improvement                     conditional q_end source
cfg.reference_source                PR-15 FK refs                              conditional reference path
   = 'task_space' | 'joint_space_fk'
cfg.mapping_bypass_in_ss            Freeze linear torso ref                    M5 mapping (linear)
cfg.use_m2_stack                    M5 mapping on/off (global)                 M5 mapping (everywhere)
cfg.aocs_off_in_ds                  AOCS off in DS                             AOCS torque
cfg.use_path_feasibility_check      Option-B mid-waypoint gate                 smoother
cfg.use_mid_waypoint_reshape        Option-B mid-waypoint                      smoother
cfg.ds_active_enabled               Active-DS torso advance                    passive settle
qp.solve(settle_mode=True/False)    QP cost mode                               tracking
qp.solve(passivity_active=...)      QP energy dissipation                      tracking
NMPC bypass during DS               (hard-coded in sim_loop.py:1418)           NMPC entirely
```

Eleven independent toggles. Each was added to fix a specific
problem and is correct under specific conditions. The combined
state space — 2¹¹ = 2048 cfg-flag combinations — is not
exhaustively tested. In practice, each runner script picks one
combination by hand. Different runners pick different
combinations. Comparing two runs requires checking all eleven
flags first.

### §1.2  Planning fragmentation

The reference signal (q, v, a, λ) at any time t comes from up to
five modules, each owning one slice:

```
coarse_preplanner   → r_com(t), v_com(t), λ_ref(t), T_step
                       (CasADi/IPOPT, runs once per SS at SS-entry)
TorsoPlanner        → p_torso_ref(t), R_torso_ref(t), v_torso_ref(t),
                       a_torso_ff(t)
                       (FK on smoothed q_seq, OR legacy SLERP, OR DS_active mode)
SwingPlanner        → p_ee_ref(t), R_ee_ref(t), v_ee_ref(t), a_ee_ff(t)
                       (FK on smoothed q_seq, OR legacy SLERP-with-bump)
CoMToTorsoMapping   → r_b_ref(t) (alternative torso linear ref)
                       (wired via cfg.use_m2_stack; may be bypassed via
                        cfg.mapping_bypass_in_ss; may be replaced by
                        the FK ref under reference_source='joint_space_fk')
constrained_geodesic→ q_seq for SS (PR #15)
                     q_DS_seq for DS-active (active_ds branch, partial)
```

The interactions are not commutative. Running with
`reference_source='joint_space_fk' AND mapping_bypass_in_ss=True`
gives different behaviour than `joint_space_fk AND
mapping_bypass_in_ss=False` — discovered the hard way during the
fk-bypass-aware-tuning branch's investigation. Neither was
documented.

There is no single function `get_reference(t)` that returns the
authoritative (q, v, a, λ) at time t. There are five planners,
each with its own concept of "what t is" (the SwingPlanner uses
a `t_plan_offset`, the TorsoPlanner uses absolute t, the
preplanner uses a fixed-horizon t-relative-to-SS-start). Time
synchronisation is itself fragile.

### §1.3  DS as second-class

The synthesis of how DS is handled today, traced from
`sim_loop.py:1390+`:

| Aspect                       | SS phase                       | DS phase                 |
|------------------------------|--------------------------------|--------------------------|
| Pre-planner runs?            | Yes (every SS-entry)           | **No**                   |
| Smoother runs?               | Yes (FK refs)                  | **No** (legacy mode)     |
| TorsoPlanner reference?      | Yes (FK or SLERP)              | **No** (`_hold_reference()` returns p_t0) |
| SwingPlanner reference?      | Yes (per swing arm)            | **No** (held)            |
| NMPC?                        | Active                         | **Bypassed**             |
| QP cost mode?                | `settle_mode=False`            | `settle_mode=True`       |
| QP passivity constraint?     | Off                            | `passivity_active=True`  |
| Exit condition?              | Time + dock criterion          | Energy + plateau         |

DS is essentially "the QP runs in a different mode with no
references and a different cost." The planning layer has no
voice during DS. Active-DS proposals (the `active_ds_torso_advance.md`
branch) had to *re-invent* a torso reference for DS from
scratch, layered on top of the existing passive-settle code.

### §1.4  The NMPC↔QP contract is opportunistic

NMPC produces five quantities the QP consumes:

```
r_com_ref,  v_com_ref,  λ_ref,  a_com_ff,  L_com_ref
```

In SS the NMPC is run at 10 Hz; the QP at 100 Hz consumes the
*last* NMPC output (held between solves). In DS the NMPC is
bypassed and the caller of `qp.solve` passes zeros / current
values for these fields. The QP doesn't know whether its
inputs come from NMPC or from a bypass — it just consumes them.

Consequence: a QP run that should track a CoM trajectory might
in fact be tracking zeros, depending on which caller invoked it.
This is invisible from the QP's side. There's no invariant like
"`r_com_ref` is always a valid CoM reference signal".

The hardcoded NMPC bypass in DS is the most egregious instance,
but the same opportunism appears at SS handoffs (M5 mapping
bypass overrides the NMPC's CoM-related outputs with the
TorsoPlanner's torso linear ref).

### §1.5  Why this matters for step 2

Step 2's failure is the *symptom* of all four issues compounding:

- The pre-planner sizes T_step assuming a QP that can perfectly
  track all references, but the QP is doing seven concurrent
  tasks under a flag combination that wasn't exhaustively
  validated.
- The smoother produces a q_seq that's kinematically valid in
  isolation, but the planner pipeline doesn't deliver it
  consistently to the QP (mapping bypass overrides parts of it).
- DS_1 and DS_2 don't pre-position the body, so SS_2 has to do
  the entire 591 mm body recoil in 12.77 s alongside the 800 mm
  swing — a task balance the QP can't satisfy.
- The NMPC's wrench feedforwards aren't computed for DS phases,
  so the system enters SS_2 without the momentum-management
  wrench prepared.

Each ablation we ran on the post-PR-15 branches (`gain
softening`, `bypass off`, `margin extension`, `AOCS unleashed`,
`active-DS at β=0.7`) addressed one of these four issues in
isolation. None composed cleanly because the underlying
architecture doesn't compose cleanly.

---

## §2  Design principles for the unified architecture

These are the invariants the new architecture must satisfy.
They constrain every later decision.

### §2.1  One reference, continuous over the whole gait

There is exactly one function

```
get_reference(t) → (q_ref(t), v_ref(t), a_ff(t), λ_ff(t), contact_config(t))
```

that returns the authoritative reference at any time `t ∈ [0, T_total]`,
covering DS_0, SS_0, DS_1, SS_1, …, DS_n. The reference is C¹
across phase boundaries (position and velocity continuous; the
contact set may change discontinuously, but the *configuration
trajectory* through that change is continuous).

No phase-specific bypasses. No "DS gets no reference". No
flag-driven branches that change what the reference is. The
caller (NMPC, QP, log, plot) gets the same answer regardless
of which branch of the call stack invoked `get_reference`.

### §2.2  No bypasses, no cost-mode flags

The QP has one cost. It tracks the references it's given. It
has zero `settle_mode`, `passivity_active`, or
`mapping_bypass` flags. If the planner says "stay at p_t0",
the QP tracks "stay at p_t0" using the same cost it uses to
track "move to p_t1" — the difference is in the references,
not in the cost.

Likewise the NMPC has one mode. It always consumes the
planner's reference and emits a wrench/momentum reference. No
"NMPC bypassed during DS". DS is just a phase where the
planner's reference happens to demand near-zero motion.

### §2.3  DS is first-class

The planner produces references for DS phases identically to
how it produces references for SS phases. The same `q_seq`
machinery emits DS sub-paths and SS sub-paths from a single
trajectory optimisation. The contact schedule (which arms are
welded when) determines the constraint set; everything else
flows from that.

A "passive settle" becomes a planner output that says "stay at
the current pose with zero velocity for ε seconds". A "torso
advance during DS" becomes a planner output that says "advance
the torso along this trajectory while both arms remain
welded". The QP tracks both identically.

### §2.4  The planner respects the dynamics it's planning for

The current pre-planner sizes T_step from CoM-acceleration
limits but ignores the QP's tracking bandwidth and the AOCS
torque budget. The unified planner must produce trajectories
that the controller stack can actually deliver. This means
either:

- (a) The planner's trajectory optimisation includes the QP's
  task-priority hierarchy as a constraint (i.e. plans for
  feasibility under the actual controller).
- (b) The planner exposes a feasibility-check primitive, and the
  controller can request a re-plan if it detects infeasibility
  during execution.

(a) is more rigorous; (b) is more practical. Option (b)
matches the existing pre-planner's role (offline) plus a new
"replan trigger" path. Option (a) requires the QP's cost
landscape to be differentiable enough for the planner to use
it — likely too expensive for the IPOPT-based pre-planner.

The proposal goes with (b) initially.

### §2.5  Single source of truth for time

There is exactly one time clock. Every reference query, every
log timestamp, every phase boundary is expressed in this clock.
No `t_plan_offset`, no `t_relative_to_SS_start`, no
`tq_planner = min(tq, ss_end - 1e-3)`. Phases are intervals on
the same clock, full stop.

### §2.6  The architecture survives the gait

The current planner stack is hard-coded to T15-style traversals
(start at (2,2), step 0/1/2, end at (3,4)). The unified
architecture must handle:

- Variable n_steps (1, 3, 10, …).
- Variable mass ratio (T15 1 %, T16 14 %, future heavier).
- Variable anchor pair geometry (the current scheduler assumes
  anchors are evenly spaced along x; not always true).
- Step retries (a step that fails at first attempt should be
  re-plannable from the failed state, not abort the run).

This rules out hard-coded `T15_STEPS` tables, hard-coded β
values per step index, hard-coded n_tau choices. Every
parameter the planner uses must be derived from the gait or
the dynamics, or be a single global tunable.

### §2.7  Composability over flexibility

The current architecture's eleven flags exist because every new
behaviour was added as an opt-in to preserve byte-identical
old behaviour. Each new opt-in compounds the test surface.

The unified architecture takes the opposite view: **one default
behaviour, no opt-ins**. Old behaviours can be reproduced by
configuring the planner's *parameters* (gains, weights, time
budgets) — not by toggling code paths. Cfg has 5–10 parameters,
not 50.

This means the migration is *destructive*: old runners stop
producing identical output to today. That's intentional. The
old behaviours that mattered (PR-15's FK refs producing 2.91 mm
step-0 docks, e.g.) are reproduced as the default of the new
planner; the rest go away.

---

## §3  The unified planner contract

### §3.1  Inputs

```
plan_gait(
    model: pin.Model,
    gait_schedule: list[GaitPhase],          # contact set per phase
    q_initial: ndarray (nq,),                # post-setup state
    cfg: PlannerConfig,
) → GaitTrajectory
```

`gait_schedule` is the existing `ContactScheduler.plan` output —
a list of (contact_set, anchor_indices) ordered phases. The
planner does NOT assume DS/SS distinction; it sees only contact
sets that change over time.

### §3.2  Output: GaitTrajectory

```
class GaitTrajectory:
    """Continuous reference for the entire gait.

    Time domain: [0, T_total]. Discretised at planner-resolution
    samples (typically 20–50 Hz, more than the QP's 100 Hz so the
    QP can interpolate, less than the QP so the planner is feasible
    to compute offline).
    """

    t_grid: ndarray (N,)              # planner-resolution timestamps
    q_seq: list of (nq,) ndarray      # configuration at each t_k
    v_seq: list of (nv,) ndarray      # twist at each t_k
    a_seq: list of (nv,) ndarray      # acceleration at each t_k
    contact_seq: list[ContactSet]     # which arms welded at each t_k
    anchor_seq: list[dict]            # anchor poses at each t_k

    # Pre-computed dynamics quantities (NMPC consumes these directly)
    r_com_seq: ndarray (N, 3)
    v_com_seq: ndarray (N, 3)
    L_com_seq: ndarray (N, 3)
    lambda_seq: ndarray (N, 12)       # per-phase contact wrenches
    a_com_ff_seq: ndarray (N, 3)

    def at(self, t: float) → Reference: ...   # interpolated query
```

A single `at(t)` method returns the full reference at time `t`.
This is the *one* reference function the QP, NMPC, log, and
plotting code all consume.

### §3.3  How the planner builds GaitTrajectory

The planner runs once at gait-start (offline). It does NOT
re-plan during execution unless the controller signals
infeasibility (§3.6).

```
algorithm  plan_gait(model, gait_schedule, q_initial, cfg):
    # Step 1: per-phase configuration optimisation.
    # For each phase, compute q_phase_end given q_phase_start
    # and the contact set + anchor poses for that phase.
    q_seq_phases = []
    q_at = q_initial
    for phase in gait_schedule:
        q_phase_end = solve_phase_endpoint(
            model, q_at, phase.contact_set, phase.anchors)
        q_seq_phases.append((q_at, q_phase_end, phase))
        q_at = q_phase_end

    # Step 2: per-phase trajectory smoothing on the contact-set
    # constraint manifold. Generalises the SS smoother (single
    # stance pinned) and the DS smoother (both stance pinned)
    # into one function parameterised by the contact set.
    full_q_seq = []
    full_t_grid = [0.0]
    for (q_start, q_end, phase) in q_seq_phases:
        q_phase_seq = smooth_on_manifold(
            model, q_start, q_end,
            constraint_set=phase.contact_set,
            anchors=phase.anchors,
            duration=phase.duration,
            n_tau=cfg.n_tau_per_phase)
        full_q_seq.extend(q_phase_seq)
        t_phase = np.linspace(full_t_grid[-1],
                              full_t_grid[-1] + phase.duration,
                              cfg.n_tau_per_phase)
        full_t_grid.extend(t_phase[1:].tolist())

    # Step 3: derive (v, a, λ) by finite-difference + dynamics.
    full_v_seq = compute_velocities(full_q_seq, full_t_grid)
    full_a_seq = compute_accelerations(full_v_seq, full_t_grid)
    full_lambda_seq = solve_inverse_dynamics(
        model, full_q_seq, full_v_seq, full_a_seq, contact_sets)

    # Step 4: derive centroidal quantities.
    r_com_seq, v_com_seq, L_com_seq = compute_centroidal(
        model, full_q_seq, full_v_seq)

    return GaitTrajectory(t_grid=full_t_grid, q_seq=full_q_seq, ...)
```

Three observations:

1. **`smooth_on_manifold`** generalises both
   `smoothed_constrained_geodesic` (1 stance) and
   `smoothed_constrained_geodesic_double_stance` (2 stances).
   The contact set determines which task constraints are
   active. PR #15's smoother becomes a special case.

2. **`solve_phase_endpoint`** generalises the existing
   `manipulability_config_trajectory` and `dock_configuration`
   IK calls. Per phase: pin all welded arms at their anchors;
   solve for q with maximal manipulability and smooth
   continuation from q_phase_start. SS phases pin one arm; DS
   phases pin two arms. The function signature is the same.

3. **No T_step distinction between DS and SS.** Each phase has
   a duration that comes from the gait schedule. The duration
   may itself be planned by an outer trajectory optimisation
   (§3.4), but at the per-phase level there's no
   "T_step is for SS only".

### §3.4  Phase duration: from CoM acceleration limit, ALL phases

Today the pre-planner sizes T_step (SS only) from a CoM
acceleration limit. The unified planner does this for every
phase:

```
For each phase (DS or SS):
    Δp_torso = ‖p_torso_phase_end − p_torso_phase_start‖
    a_max = cfg.preplanner_a_cruise_max
    T_phase = sqrt( 4 · Δp_torso / a_max )    # trapezoidal-cruise approx

For phases where Δp_torso ≈ 0 (a passive-settle DS):
    T_phase = cfg.t_settle_min (e.g. 0.5 s)
```

DS phases that need to advance the body (the previous
"active-DS" use case) get a non-trivial duration automatically
because Δp_torso is non-trivial. DS phases that don't need to
move get a short fixed duration.

The total gait duration is `Σ T_phase`. The planner's offline
solve takes ~1 s wall-clock for a 3-step gait — same order as
today.

### §3.5  How NMPC consumes GaitTrajectory

```
nmpc.set_reference(gait_trajectory)        # called once per gait
loop at 10 Hz:
    ref = gait_trajectory.at(t_now + horizon)
    nmpc.solve(current_state, ref)         # always; no bypass
```

The NMPC's signature doesn't change. What changes is what's
upstream: the NMPC always gets a meaningful reference, including
during DS phases (where the reference happens to be near-zero
motion). The NMPC just solves; it doesn't know or care about
the phase.

### §3.6  Infeasibility-triggered replan

If during execution the controller detects:

- The QP returns infeasible for `cfg.replan_n_consecutive` ticks
  (default 3), or
- The tracking error exceeds `cfg.replan_tracking_threshold` for
  any task,

then the controller signals the planner to **replan from the
current state** for the rest of the gait. The replanner takes
the same inputs as the original planner but with `q_initial =
current state` and `gait_schedule = remaining phases`.

This handles step retries, drift accumulation, and unexpected
contact behaviour. The current architecture has *no* replan
path — once a step's pre-planner runs, the system is locked
into that plan until SS-exit, no matter what happens in
closed-loop.

---

## §4  The NMPC↔QP contract

### §4.1  NMPC inputs (always)

```
nmpc.solve(
    state: CentroidalState,           # current (r_com, v_com, L_com)
    horizon_ref: list of Reference,   # gait_trajectory.at(t_now + k·dt) for k in 0..N
)
→ NMPCOutput(r_com_ref_dt, v_com_ref_dt, lambda_ref_dt, a_com_ff_dt, L_com_ref_dt)
```

The NMPC receives a horizon's worth of references from the
gait trajectory. It computes the optimal wrench/momentum
trajectory subject to (a) achieving those references, (b)
respecting AOCS torque/momentum limits. Output is the
first-step wrench/momentum reference for the QP.

**No `if phase == 'DS': skip nmpc.solve(...)`** anywhere in the
codebase. The NMPC always runs. If the gait trajectory says
"stay still", the NMPC's wrench output is near-zero — which is
fine.

### §4.2  QP inputs (always)

```
qp.solve(
    state: RobotState,
    nmpc_output: NMPCOutput,          # always meaningful, always present
    contact_config: ContactConfig,    # from gait_trajectory.contact_seq[k]
    gait_ref: Reference,              # at current t (interpolated)
)
→ tau, qdd, lambda_qp
```

The QP cost has a fixed structure: track torso, track swing-EE
(if there is one — determined by contact_set, NOT a flag),
track CoM, track L_com, posture, regularise wrench. **All
weights are constants from cfg.** No `settle_mode`, no
`passivity_active`.

When the gait_ref says "swing arm has no target" (DS phase), the
swing-EE task is automatically inactive because the contact_config
has both arms welded. The QP's existing logic (J_ee=None disables
the EE task) handles this naturally — we just remove the
*caller-side* flag that decided when to pass J_ee=None and let
the contact_config drive it.

### §4.3  What disappears from the QP signature

```
# Removed (caller-side decision moved into gait_ref / contact_config)
- settle_mode: bool
- passivity_active: bool
- p_torso_ref / R_torso_ref / v_torso_ref / a_torso_ff   (now from gait_ref)
- p_ee_ref / R_ee_ref / v_ee_ref / a_ee_ff               (now from gait_ref)
- r_com_ref / v_com_ref / lambda_ref / a_com_ff          (now from nmpc_output)
- L_com_ref                                              (now from nmpc_output)
```

The QP becomes a pure function: `(state, nmpc_output, contact_config,
gait_ref) → (tau, qdd, lambda)`. No mode parameters.

### §4.4  AOCS

AOCS is unchanged. It already has a clean contract: read
`L_com` from state estimation, react with reaction-wheel torques
within budget. The proposal does not touch the AOCS layer.

The `cfg.aocs_off_in_ds` flag goes away as a side-effect: if
the gait trajectory commands no body rotation during DS, AOCS
sees no momentum demand and emits ~zero torque automatically.

---

## §5  Migration path

This is a multi-week effort. The migration is destructive — old
runners will not be byte-identical to today. The migration is
phased so each phase is independently testable and the system
remains runnable at every commit.

### §5.1  Phase A — Build the unified planner *alongside* the old stack (~1.5 weeks)

Goal: ship `crawlbot/planning/unified_planner.py` exporting
`plan_gait(model, gait_schedule, q_initial, cfg) →
GaitTrajectory`. The new planner runs in isolation; it doesn't
yet feed the controllers.

Subgoals:

- A.1  Generalise the smoother to `smooth_on_manifold(model,
       q_start, q_end, constraint_set, anchors, duration,
       n_tau)`. Today's `smoothed_constrained_geodesic` and
       `smoothed_constrained_geodesic_double_stance` become
       special cases (constraint_set = {fid_a} or {fid_a,
       fid_b}).
- A.2  Generalise the IK to `solve_phase_endpoint(model, q_start,
       constraint_set, anchors)`. Wraps the existing
       `dock_configuration` / `manipulability_config_trajectory`.
- A.3  Implement `plan_gait(...)` per §3.3.
- A.4  Implement `GaitTrajectory.at(t)` interpolation.
- A.5  Tests: U.1 endpoint exactness, U.2 constraint-set
       satisfaction at every k, U.3 phase-boundary continuity
       (C¹ across DS↔SS).

Phase-A deliverable: a function the user can call to produce a
GaitTrajectory for any T15/T16/etc. gait, validated against the
existing PR-15 q_seq for SS phases (byte-identical for that case).

### §5.2  Phase B — Wire the unified planner into NMPC (~1 week)

Goal: NMPC consumes GaitTrajectory.at(t) for its horizon
references, replacing the existing pre-planner output + planner
fragments.

Subgoals:

- B.1  Add `nmpc.set_reference(gait_trajectory)` API.
- B.2  Modify `nmpc.solve(...)` to read references from the
       trajectory, not from caller-passed args.
- B.3  Remove the hard-coded "NMPC bypassed during DS" code in
       sim_loop. NMPC now runs continuously.
- B.4  Tests: NMPC closed-loop on a T15 gait under the new
       planner; assert wrench reference is sensible during DS
       (~zero motion → ~zero wrench).

Phase-B deliverable: NMPC always runs, consumes a unified
reference. The QP is still consuming the old pre-planner +
TorsoPlanner + SwingPlanner outputs at this stage — the bridge
hasn't been built yet, but NMPC is decoupled.

### §5.3  Phase C — Wire the unified planner into the QP (~1.5 weeks)

Goal: QP reads gait_ref.at(t) for tracking targets, replacing
the per-planner-module inputs.

Subgoals:

- C.1  Modify `qp.solve(...)` to take a unified `gait_ref` arg.
       Internally derive p_torso_ref, p_ee_ref, etc. from it.
       Old args remain accepted (backward-compat) but emit
       DeprecationWarning.
- C.2  Modify sim_loop to pass gait_ref to qp.solve, not the
       individual planner outputs.
- C.3  Remove the `settle_mode` and `passivity_active` flags.
       The QP cost is now a single fixed function.
- C.4  Tests: byte-identical T15 dock outcomes for steps 0/1
       under the new wiring (validates that the unified planner's
       SS sub-paths reproduce PR-15's FK ref behaviour).

Phase-C deliverable: the entire control stack runs against a
single reference source. The eleven flags from §1.1 collapse
to one cfg pointer (which planner; default unified).

### §5.4  Phase D — Remove the legacy planners (~0.5 weeks)

Goal: delete `coarse_preplanner`, `TorsoPlanner` (legacy
SLERP path), `SwingPlanner` (legacy SLERP+bump path),
`CoMToTorsoMapping`. The unified planner replaces all of them.
The `constrained_geodesic` module is repurposed to provide the
generalised `smooth_on_manifold` primitive.

Net code change: −2,000 to −3,000 lines (rough estimate; the
deleted modules sum to ~3,500 lines today).

### §5.5  Phase E — T15 step-2 closed-loop validation (~0.5 weeks)

With the unified planner producing first-class DS references,
re-run T15 and verify step 2 docks. If it does not, the
unified planner's per-phase duration and contact-set choices
need tuning — but the failure mode is now diagnosable in one
place (the planner) rather than across eleven flags.

### §5.6  Phase F — T16 + step retries + replan (~1 week)

Goal: validate higher mass ratios + the §3.6 infeasibility-
triggered replan. T16's 14% mass ratio amplifies every
momentum effect; a fragile architecture would fail there. If
T16 docks all 3 steps, the architecture has demonstrated
generality.

### §5.7  Phase G — Documentation + closing report (~0.5 weeks)

- Closing report.
- Update CLAUDE.md milestone line.
- Update synthesis at
  `docs/architecture/T15_step2_diagnosis_and_resolution.md`
  with §13 "Resolution: unified planning".
- Branch ready for PR.

**Total: ~6.5 weeks single-engineer, ~3.5 weeks two-engineer.**

---

## §6  What gets deleted

Cleanup is half the value. Phases A–D collectively remove these
artefacts; this is the planned diff at the end of Phase D.

### §6.1  Modules deleted

| Path | Lines | Replaced by |
|---|---:|---|
| `crawlbot/planning/coarse_preplanner.py` | ~600 | `unified_planner.plan_gait` (T_phase from §3.4) |
| `crawlbot/planning/torso_planner.py` (legacy SLERP path) | ~400 | `GaitTrajectory.at(t)` |
| `crawlbot/planning/swing_planner.py` (legacy SLERP path) | ~500 | `GaitTrajectory.at(t)` |
| `crawlbot/core/com_to_torso_mapping.py` (M5 mapping) | ~350 | unused under unified pipeline |

### §6.2  Cfg flags removed

```
- cfg.use_trajectory_aware_ik
- cfg.reference_source
- cfg.mapping_bypass_in_ss
- cfg.use_m2_stack
- cfg.aocs_off_in_ds
- cfg.use_path_feasibility_check
- cfg.use_mid_waypoint_reshape
- cfg.ds_active_enabled
- cfg.ds_active_beta, ds_active_n_tau, ds_active_n_iter, …
- cfg.geodesic_n_tau, geodesic_n_iter, geodesic_tol
- cfg.preplanner_a_cruise_max, preplanner_cruise_ramp_frac, ...
```

…and roughly 20 more cfg fields that exist only to parameterise
behaviours that no longer have switches. Net cfg reduction:
from ~120 fields to ~50.

### §6.3  QP signature simplifications

```python
# Before
qp.solve(
    q_t, dq_t, q, dq,
    r_com_ref, v_com_ref, lambda_ref, a_com_ff,
    H_robot, C_robot, J_com, Jdot_dq_com,
    contact_config, J_contacts, Jdot_dq_contacts,
    hw_current, hw_min, hw_max,
    r_com, L_com_current,
    J_ee, Jdot_dq_ee, p_ee_ref, R_ee_ref, v_ee_ref, a_ee_ff, p_ee, R_ee,
    J_torso, Jdot_dq_torso, p_torso, R_torso,
    p_torso_ref, R_torso_ref, v_torso_ref, a_torso_ff,
    H_base_swing, swing_v_slice,
    settle_mode=False, passivity_active=False,
)

# After
qp.solve(
    state: RobotState,
    nmpc_output: NMPCOutput,
    contact_config: ContactConfig,
    gait_ref: Reference,
)
```

40+ args → 4. Internal logic stays the same; just decoupled from
caller-side decisions.

### §6.4  Removed from sim_loop.py

```
_run_ds_passivity_loop                 — replaced by unified QP loop
_run_ds_active_advance                 — replaced by unified QP loop
_setup_torso_for_step                  — replaced by gait-trajectory query
_planned_arm_config                    — replaced by gait_ref.at(t).q
_ds_class                              — no class distinction; one loop
mapping bypass branches                — gone
all settle_mode toggles in qp.solve calls — gone
```

The phase loop in sim_loop.py:1390+ becomes a clean ~50-line
loop that, at each tick, queries `gait_ref.at(t)`, calls
`nmpc.solve()` (every 10 ticks), calls `qp.solve()`, applies
the torque to MuJoCo, and steps. No conditional logic on phase
type.

---

## §7  Risks and mitigations

| ID | Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|---|
| R1 | The unified `plan_gait` IPOPT solve doesn't converge for some gait + initial-state combinations | Medium | High | Phase A.5 tests cover synthetic adversarial cases. Add a fallback to per-phase IK if the trajectory optimisation fails for any phase. |
| R2 | The first-class DS reference produces torso motions that, in closed-loop, produce drift the QP can't reject | Medium | Medium | The infeasibility-triggered replan (§3.6) addresses this. If a phase's tracking error grows, replan from current state. Validates in Phase E. |
| R3 | Removing `settle_mode` regresses settling behaviour (the legacy passivity-active mode dissipated kinetic energy aggressively; tracking mode may not) | Medium | Low | Settling becomes a planner output: a phase whose reference is "stay at current pose, zero velocity, for ε seconds". The QP tracks this with PD gains; energy dissipation is automatic. Validate by comparing terminal-DS T_kin trajectories before vs after migration. |
| R4 | NMPC running continuously (no DS bypass) costs more compute time | Low | Low | NMPC at 10 Hz is cheap. Phase B validates the wall-clock cost of always-on NMPC. |
| R5 | Phase A's smoother generalisation breaks the existing PR-15 SS smoother behaviour | Low (the new function reduces to the old one in the n=1 stance case) | High (would break PR-15 dock outcomes) | Tests in Phase A.1 explicitly verify byte-equivalence to PR-15's smoother for the SS case. |
| R6 | Migration takes longer than estimated because of unknown couplings between modules | Medium | Medium | Phases B and C are the main exposure. Each phase ends with a runnable system (no big-bang merge). If a phase blows up, it can be paused and re-scoped without breaking what came before. |
| R7 | The unified planner's CoM acceleration limit (§3.4) sizes T_phase too aggressively for some anchor pairs | Medium | Low | Cfg-tunable cap; per-mass-ratio defaults. The infeasibility-triggered replan covers the worst case (over-aggressive plan → tracking error → replan). |

---

## §8  Open questions for the next engineer

These are honest unknowns. Each one has a "good enough"
default but deserves attention if the migration runs into
trouble.

### §8.1  How does the new planner pick T_phase for DS phases that DON'T need body advance?

§3.4 says "T_phase = sqrt(4·Δp_torso / a_max)" for moving
phases. For stationary DS (Δp_torso ≈ 0) the formula gives
T_phase ≈ 0. We default to `cfg.t_settle_min ≈ 0.5 s`. But
this doesn't account for the QP needing time to dissipate
post-dock impact transients.

Better: T_phase = max(T_motion, T_settle_required) where
T_settle_required is estimated from the pre-phase kinetic
energy.

### §8.2  Should the trajectory optimisation be CasADi-based (per-phase) or pyomo / IPOPT (global)?

Today's `coarse_preplanner` uses CasADi+IPOPT per-SS. The
unified planner could either:
- (a) Per-phase optimisation, sequential. Simpler. May produce
      less-optimal gaits.
- (b) Global optimisation over the whole gait. More principled.
      Slower (~10-30 s).

Phase A starts with (a); upgrade to (b) is a future enhancement.

### §8.3  How does step retry interact with the gait schedule?

If step 2 fails, do we retry step 2 with a re-planned trajectory,
or skip to terminal DS? §3.6 says "replan from current state for
remaining phases". But what if the current state is so off that
no plan exists? Need a clean error path.

### §8.4  Does the unified planner subsume the IK-fix work?

PR #15 added IK-fix work (`docs/architecture/IK_FORMULATION.md`
§9, four IK pathology fixes). These improvements live inside
`solve_phase_endpoint`. The unified planner inherits them.

### §8.5  Is the AOCS layer truly untouched?

Phase F should validate that AOCS at the new gait references
behaves correctly. There's a subtle dependency: AOCS uses
`L_com` from state estimation, but the planner now produces a
`L_com_seq` reference that NMPC tracks. If the *reference* is
poorly chosen (e.g. asks for momentum the AOCS budget can't
produce), AOCS saturates. The unified planner must respect the
AOCS budget when planning L_com_seq — open question whether
this requires AOCS-aware trajectory optimisation.

### §8.6  Is this a research project or a refactor?

The proposal is presented as a refactor (replace fragmented
modules with one unified module). But §3.4 (per-phase duration
sizing for DS) and §3.6 (replan-on-infeasibility) are *new*
behaviours that didn't exist before. The new planner is doing
new things, not just consolidating old things.

The honest framing: the unified architecture is mostly a
refactor of existing primitives, plus a small amount of new
gait-level reasoning that the current architecture has no
place for.

### §8.7  What happens to the existing tests?

PR-15's `tests/test_fk_reference_consistency.py` (E.1, E.2,
E.8, E.9, E.10) and the active-DS branch's `tests/test_active_ds.py`
(D.1–D.4) test specific primitives that survive into the
unified planner. They mostly need parameter-name updates, not
rewrites. Adversarial regression tests for the eleven removed
flags are NOT added — those flags don't exist anymore.

---

## §9  Branch and artefact lineage

```
main (post-PR-15)
  └── claude/active-ds-architecture (paused, partial-success evidence)
  └── claude/unified-planner (NEW — this proposal lands here)
        ├── Phase A: smooth_on_manifold + plan_gait + GaitTrajectory
        ├── Phase B: NMPC ↔ unified planner wiring
        ├── Phase C: QP ↔ unified planner wiring
        ├── Phase D: delete legacy modules
        ├── Phase E: T15 closed-loop validation
        ├── Phase F: T16 + step retry + replan
        └── Phase G: closing report
```

Predecessor docs:

- `docs/architecture/T15_step2_diagnosis_and_resolution.md` — the original synthesis that motivated PR #15.
- `docs/architecture/active_ds_torso_advance.md` — the active-DS proposal whose closed-loop failure (§4 of that doc, plus the Phase 4 evidence on `claude/active-ds-architecture`) demonstrated that step-2 cannot be fixed by adding more flags to the existing architecture.
- `results/M7_1pct_3step_v22_t15_fk/CLOSING_REPORT.md` — PR-15 closing.
- This document — proposes the architectural rewrite.

---

**End of proposal.**

Sleep well. The next session can start with Phase A.1 (generalise
the smoother to `smooth_on_manifold`) once you've decided to
take this on. Or it can sit for a week while you think about
whether the rewrite is the right call. Either way, this
document is the consolidated state of what we know.





