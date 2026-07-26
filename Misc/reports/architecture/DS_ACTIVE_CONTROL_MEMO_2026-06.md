# DS phase as active 6-DOF control window — architectural memo

**Date.** 2026-06-03
**Branch.** `claude/aocs-sign-fix-and-settle` (PR #21 open).
**Scope.** Whole-body control during DS (double-support) phases. Out of scope: NMPC, AOCS, mapping cascade, gait scheduler.
**Status.** Design memo; not yet implemented. Connects to long-duration scaling problem from campaign §11–§12.

---

## 1. TL;DR

During DS (both arms welded), the current architecture runs the SS task stack — torso 6D + swing EE 6D + posture — with both arms gripping. This is wrong on two counts: **the swing arm task does not exist (both are welded)**, and **the torso is over-determined by the dual-arm kinematic closure**. The result is a QP that fights the kinematic constraints, generates a persistent ~3.86° steady-state tracking error post-dock, injects reference discontinuities into the cascade at SS→DS boundaries, and treats DS as a passive hold when it could be a **6-DOF active control window**.

Replacing the DS task stack with **torso 6D only + posture in the closed-chain redundancy** turns DS into an active control regime with three high-leverage uses: (a) momentum dump that could drive the per-traversal irreversible drift toward zero, (b) posture preparation for the next SS, (c) smooth SS↔DS phase handoff that eliminates the reference discontinuity (the 2.89° step jump documented in PR #21 figures). Estimated effort: 1–2 days of code + verification.

## 2. The observation that surfaced this

Tracking diagnostic (`Misc/scripts/diag_tracking.py`, PR #21) on the pole-placement run revealed:

- `d_grip_swing` jumps to **exactly 4000 mm** the instant the last dock fires and holds there for the full 120 s settle. Root cause: `target_anchor=0` placeholder in the trailing-DS settle loop (`sim_loop.py:1929`); the swing-arm concept is stale during DS — there is no swing arm.
- `e_torso_ori` jumps from 2.28° (SS peak) to **3.86° persistent** through the entire post-dock settle, even though the robot is welded and physically stationary. Root cause: the `TorsoPlanner.set_hold(...)` reference is the dock-IK solution; the actual welded equilibrium is on the kinematic-closure manifold; they differ by 3.86° and no amount of QP tracking can close the gap.
- The reference quaternion takes a **2.89° step jump** at the dock instant. The QP chases the new target, injecting transients into the cascade that contribute to the Ḣ_FF FD-artifact spikes documented in §12.6.

These are not three independent bugs — they're symptoms of one architectural mismatch.

## 3. Kinematic decomposition

In DS, configuration variables and constraints:

| | count |
|---|---|
| Torso floating base | 6 |
| Arm A joints | 7 |
| Arm B joints | 7 |
| **Total config** | **20** |
| Arm-A gripper ↔ anchor-A weld | 6 |
| Arm-B gripper ↔ anchor-B weld | 6 |
| **Total constraints** | **12** |
| **Free DOFs** | **8** |

The 8 free DOFs decompose cleanly:

$$8 = \underbrace{6}_{\text{torso pose, within common workspace}} + \underbrace{1}_{\text{arm-A null space}} + \underbrace{1}_{\text{arm-B null space}}$$

Each 7-DOF arm subject to 6 weld constraints has 1 redundant DOF (the elbow swivel). The torso has its full 6 DOFs *within the common reachable workspace* of both anchors via their respective arm chains.

**Key consequence:** the torso is a legitimate 6D task in DS, not over-determined. The QP can command torso pose, and the kinematic closure resolves through the 2 arm-null-space DOFs (handled via posture cost or null-space damping).

## 4. Why the current architecture is wrong in DS

The current code runs `_step(..., settle_mode=True)` during DS with the same task stack as SS:

| layer | SS task | what happens in DS |
|---|---|---|
| P1 | torso 6D (angular in cooperative mode) | tracks `set_hold()` dock-IK target |
| P2 | swing EE 6D + torso linear (co-equal in cooperative) | swing arm is **welded** — task is meaningless |
| P3 | posture | active |

Three failure modes follow:

1. **Swing-task pollution.** The QP allocates effort to track a swing reference that no physical arm is following. Cost-function noise.
2. **Torso target inconsistency.** The dock-IK target lives on the kinematic-closure manifold but isn't necessarily where the dynamic settle lands; the QP commands torque to chase a target it physically cannot reach.
3. **Reference discontinuity at SS→DS.** The SS trajectory ends somewhere; the DS `set_hold()` jumps to a different IK solution. QP transient propagates as wrench injection, mapping-FD spikes, AOCS feedforward spikes.

## 5. The clean architecture: torso-only in DS

Replace the DS task stack with:

| layer | task |
|---|---|
| P1 | torso 6D (full 6D, not just angular) |
| P2 | — *(removed — no swing arm, no co-equal EE)* |
| P3 | posture in the 2-D arm-null-space redundancy |
| dynamics | unchanged; contact constraints active (welds) |
| AOCS | unchanged; sole active attitude controller |

The torso 6D task is now the only task in the priority stack, and it's well-defined within the common workspace. The 2 arm-null-space DOFs are handled by the posture term, biased toward a nominal config (the campaign §12.4 mirror-symmetric template, or per-step manipulability-optimal).

The QP is no longer fighting itself. There is no swing reference to be ignored. The torso reference is now *commandable*, not aspirational.

## 6. What DS could do once it's an active control regime

The 6 DOFs of torso control during DS aren't just "no longer wasted" — they're a control authority we currently leave on the table. Three high-value uses:

### 6.1 Momentum dump

The SS phase injects irreversible angular momentum into the structure via the swing arm's reaction wrenches. Currently the AOCS PID recovers most of it (settle in §12.4 reaches 0.012° irreversible per traversal). But conservation about structure CoM is exact: any unrecovered angular momentum becomes permanent attitude drift.

In DS we can **actively move the torso** along a trajectory that injects compensating angular momentum back into the structure body (with the AOCS wheels absorbing the net difference). Think: torso swings opposite to the last SS-injected direction, with welds re-distributing the reaction. The wheels and torso motion cooperate to drive both `ω_s → 0` and the *time-integral* of `ω_s` toward zero — eliminating the irreversible drift entirely, not just driving rates to zero.

This is RNS-like (Reaction Null Space, Nenchev–Yoshida), but applied per-DS-phase rather than within the SS swing trajectory. It's strictly easier than RNS in SS because both arms are welded and the contact wrenches have a known structure.

**If this works at full effect, the per-traversal irreversible drift could approach zero**, lifting the long-duration scaling limit from ~405 traversals (campaign §12.8) to thousands.

### 6.2 Posture preparation for next SS

Currently the next SS inherits whatever torso pose the previous DS held. Campaign §3 documented a ~4.8 mm steady-state EE offset at dock on long strides — partly because manipulability isn't optimized at the start of the swing.

DS torso steering can pre-position the torso to put the *next* swing arm at its manipulability sweet spot before releasing it. This is essentially online optimization of the gait's launch conditions, done in the slack time of DS.

### 6.3 Smooth SS↔DS phase handoff

The 2.89° reference discontinuity at SS→DS comes from the SS trajectory landing somewhere and the DS hold starting somewhere else. With a planned DS trajectory:

- DS trajectory **starts** exactly at the SS endpoint (no discontinuity at SS→DS).
- DS trajectory **ends** exactly at the next SS startpoint (no discontinuity at DS→SS).
- Trajectory in between can serve §6.1 and §6.2 purposes.

This removes one of the documented sources of cascade FD-artifact spikes (§12.6) directly.

## 7. Implementation sketch

### 7.1 WBC / QP changes (`crawlbot/solvers/wholebody_qp.py`)

The QP already supports task weights. The DS mode swaps:

- Set `alpha_swing_pos = alpha_swing_ori = 0` during DS phases (drop the swing EE task entirely).
- Keep torso 6D as P1.
- Drop the P2 cooperative-arms torso-linear/EE co-equal split (no EE to be co-equal with).
- Keep posture P3.

Probably 10–20 lines, gated by `settle_mode` or an explicit `phase == 'DS'` flag.

### 7.2 TorsoPlanner changes (`crawlbot/planning/torso_planner.py`)

Replace `set_hold(p, R, r_com)` with `set_ds_trajectory(p_start, R_start, p_end, R_end, t_duration)` — a 5th-order spline (or whatever current matching technique is used) from the SS endpoint to the next SS startpoint, with an optional intermediate waypoint for momentum dump.

For the trivial case (no momentum dump, no posture prep), this collapses to the existing `set_hold` if `p_start = p_end` and `R_start = R_end`.

### 7.3 Reachability check

Before committing to a DS trajectory, validate that the proposed torso path stays within the common workspace of both anchors. Cheap heuristic: check the dock_configuration IK at a few waypoints. If any fails, fall back to `set_hold` at the SS endpoint.

### 7.4 Sim-loop integration (`crawlbot/simulation/sim_loop.py`)

Replace the `set_hold(...)` call at the trailing-DS settle (~line 1888 area) with `set_ds_trajectory(...)`. For inter-step DS, similar replacement at the `_run_ds_passivity_loop` call site if it uses a held reference.

Drop the `target_anchor=0` placeholder bug for `d_grip_swing` logging during DS (or set it to NaN — the swing concept doesn't apply, the field shouldn't carry stale data).

### 7.5 Momentum-dump trajectory design

This is the hardest piece and may want its own follow-up:

- Compute the **net angular impulse** injected by the just-completed SS phase from the logged `Ḣ_s` (or from the structure attitude drift observed at dock).
- Design a torso trajectory whose own angular-momentum contribution **negates** that impulse, modulo what the AOCS wheels can absorb.
- The trajectory must satisfy: end velocity = 0 (for clean DS→SS handoff), kinematic reachability throughout, and joint-torque limits.

Closed-form for a simple "swing-back" motion is feasible. Optimization-based version (QP at the planner layer) is also reasonable.

## 8. Risks and known issues

- **Reachability failure**: if the DS trajectory enters infeasible workspace, QP returns infeasible. Right behavior, but the planner must catch this gracefully (fall back to `set_hold`).
- **Aggressive momentum-dump trajectories may saturate joint torques**. The current QP joint-torque bound (±20 Nm) is the gate. Conservative trajectories first; aggressive optimization later.
- **The trailing-DS settle currently runs 120 s for diagnostic purposes**. With a planned DS trajectory, the settle duration becomes a design parameter — short for inter-step DS (transition), long for final DS (full momentum dump). The CLI flag `--settle_seconds` may need to gain semantics around trajectory completion vs. dwell.
- **AOCS interaction**: if the torso is actively rotating during DS, the AOCS Ḣ_FF should NOT misinterpret that motion as a disturbance to cancel. The current FD-based feedforward might do exactly that. Either: (a) compute Ḣ_FF analytically from the planned torso trajectory (smoother by construction; cleaner separation of "intended" vs "disturbance"), or (b) zero the AOCS feedforward during DS and let only the PID feedback run.

## 9. Connection to the broader story

This memo connects four open campaign threads:

- **§11 / §12** long-duration scaling (the 0.012°/traversal drift): if DS momentum-dump works, this drops toward zero.
- **§12.6 / PR #21** FD-artifact spikes in Ḣ_FF: the SS→DS reference discontinuity is one source; smooth phase handoff eliminates it.
- **§6c** mapping cascade band-aid: less reference jitter at phase boundaries means less downstream mapping-FD noise.
- **§3** ~4.8 mm steady-state EE offset on long strides: DS posture prep could pre-position for better SS launch manipulability.

It does **not** depend on the soft-CoM cascade-consistency work, the analytical Ḣ_FF implementation, or the per-axis AOCS gain set. Those can land independently.

## 10. Proposed next steps

1. **Land this memo as a doc** (this file) — captures the design before the implementation drifts.
2. **Prototype the trivial case**: WBC task-stack swap during DS + `set_hold` retained. Verify the 3.86° persistent error goes away. ~1 day.
3. **Implement smooth phase handoff**: SS endpoint = DS startpoint; DS endpoint = next SS startpoint. Verify reference discontinuity eliminated. ~1 day.
4. **Design and prototype momentum-dump trajectory** (the high-value piece). Verify per-traversal irreversible drift drops toward zero. 2–3 days.
5. **Multi-traversal validation** with the momentum-dump DS controller. Confirm linear scaling argument from §12.8 lifts. 1 day.

Total: ~1 week of focused work for a major architectural simplification + scaling lift.

## 11. Open questions

- **Does the AOCS see the DS torso motion as a disturbance?** The current `legacy_pid_*` modes compute Ḣ_FF via FD on measured state, which would include intended DS motion. If we let the FF chase it, the AOCS would fight the planned momentum dump. Either route the planned torso accelerations into the AOCS as a feedforward-to-subtract, or gate the AOCS FF during DS.
- **What's the right cost on the 2 arm-null-space DOFs?** Mirror-symmetric template? Manipulability-optimal? Joint-limit-distance? The choice affects DS→SS handoff quality.
- **Inter-step DS vs trailing DS**: are they really the same regime, or do they want different control? Inter-step DS is short and transitional; trailing DS is open-ended hold. The trajectory design probably needs to vary.
- **Coupling with the spec §7.1 two-phase state machine**: does this change the phase-transition contract documented there? Probably refines it without breaking it.

---

**Branch / PR action.** Add to PR #21 as a follow-up note; do not implement on the PR #21 branch. The implementation deserves its own branch — possibly named `claude/ds-active-control` — once the design here is reviewed.
