# Next Session Prompt — DS↔SS Smooth Handoff for Continuous Mission Operation

> **Branch continuity:** This session continues on the existing branch
> `claude/rework-controller-tasks-0MAgl`. Do **not** create a new branch.
> All commits go there. Verify with `git branch --show-current` before
> the first commit; if you land on a different branch, `git checkout
> claude/rework-controller-tasks-0MAgl` first.

## Where we are

**Branch:** `claude/rework-controller-tasks-0MAgl` (off main, post-merge of PR #23).
**Last commit:** `fef912c` — B-probe negative result.
**Memo:** `docs/architecture/DS_REWORK_CENTROIDAL_2026-06.md` (with actuator-abstraction footnote).

**What works:**
- Single 5-step canonical traversal: 5/5 docks ≤ 4.99 mm.
- Single 10-step forward+reverse (`multi_traversal_2x.seq`): 10/10 docks, sign-symmetric h_w residual, irreversible drift 0.0015°/traversal.
- DS rework cascade (5 commits: 7991c06, df643bf, b68ff95, 13c3baa, 172accc) — trailing-DS pose drift 104° → 0.09°, joint vel 0.15 → 0.0002 rad/s, mission-scaling linear-extrapolation lifts to ~3 300 traversals (single-traversal-with-settle regime).
- Sequence-file loader (START/DOCK/DWELL) supports arbitrary gait scenarios.

**What's empirically validated:**
- `multi_traversal_10x.seq` (50 steps rapid back-to-back, no dwell): **fails at step 22**. Mechanism: h_w doesn't decay between cycles → reaches 4.0 Nms (80% of per-axis spec) → docking margin collapses.
- `multi_traversal_10x_dwell.seq` (50 steps with 30 s inter-cycle dwells): **fails at step 10**. Mechanism: pristine post-dwell state triggers DS→SS launch transient (`‖q̇‖: 0.0001 → 0.0394 rad/s` in one tick; recoil 476 mm vs 142 mm without dwell). NMPC warm-start reset ruled out as the cause (B-probe negative, commit `fb9bc9e`).

**The three mission-scaling regimes:**

| Regime | Empirical limit | Assumption |
|---|---|---|
| Linear-extrap (single + 120 s settle) | ~3 300 traversals | Each traversal gets full recovery |
| Rapid back-to-back (no dwell) | ~8 traversals | h_w accumulates → per-axis saturation |
| 30 s dwell + pristine state | ~1 traversal | DS→SS handoff transient |

---

## What's left (in priority order)

### Tier 1 — directly blocks the 3 000-traversal empirical validation

1. **Diagnose the DS→SS launch transient.** Step 10 (post-dwell, pristine state) shows a ~400× joint-velocity jump and 3× recoil at the first SS tick. Root cause unknown.
   - **Next-cheap action**: instrument the swing-trajectory queries and QP outputs at SS-start. Log `T_step` chosen by pre-planner, `p_torso_ref(t)` and `v_torso_ref(t)` over the first 0.5 s of SS, `qdd_t_qp` per tick at SS startup, captured `q_dock` target. Compare step 0 (works) vs step 10 with-dwell (fails) — the transient should reveal itself.
   - **Hypothesis to confirm/rule out**: pre-planner picks a more-aggressive trajectory from a quiescent state (full h_w budget available → faster swing → WBC can't track through welded-loop reaction). If true, fix is in the pre-planner's state-dependent T_step / a_cruise_max selection.

2. **Smooth DS↔SS phase handoff** (memo §6.3). Once the diagnosis identifies the mechanism, build the principled fix. Likely shape: replace the abrupt task-stack swap with a `torso_planner.set_ds_to_ss_transition(...)` that interpolates between the captured DS pose and the SS swing start with `v_torso = 0` at the boundary.

### Tier 2 — likely-relevant cleanups that could compound

3. **NMPC state hygiene beyond warm-start**. The B-probe only reset OSQP warm-start. There may be FD filter states (L_com, v_com previous values), wrench-FF λ_qp previous values, that need clean transition. If the diagnosis shows non-FD state biasing the first SS tick, this is the fix.

4. **F-SAT / loop-free-mapping cleanup**. Per `CLAUDE.md` Open section: F-SAT band-aid is still active; spec §6 loop-free mitigation never implemented. Over 100+ traversals the cumulative bias may degrade docking margin. Worth re-checking once Tier 1 lands.

5. **Centroidal-DS posture cycle-over-cycle drift**. The 2 arm-null-space DOFs settle to some configuration during dwell. If that configuration differs cycle-over-cycle, accumulated joint pose drift could be a mission-life limit not yet observed.

### Tier 3 — pure code hygiene (defer until Tier 1 + 2 land)

6. `wholebody_qp.py::solve()` refactor into `_build_tasks_for_phase()`. The gating matrix (`settle_mode`, `coop_mode`, `use_m2_stack`, `r_tube`, `ds_centroidal_mode`, `ds_centroidal_active`) is bushy.
7. SimConfig sprawl — consolidate DS-rework flags into a `DSReworkConfig` sub-block.
8. Doc reconciliation — `CLAUDE.md` "Active milestone" + `STACK_OVERVIEW.md` need a sync-up post-PR-23.
9. Supersede `9a112dd` commit message reasoning (the "settle_mode skips torso/CoM tasks" rationale is now resolved).

---

## Concrete next-session opening action

```bash
# Setup
bash docs/architecture/setup_env.sh

# Read the recent memo + failure artifacts
cat docs/architecture/DS_REWORK_CENTROIDAL_2026-06.md
cat results/diag_cooperative_arms_multi_traversal_10x/step_metrics.txt
cat results/diag_cooperative_arms_multi_traversal_10x_dwell/step_metrics.txt
```

**Then instrument the SS-start transient.** Suggested approach:

1. Add `_debug_ss_start_trace` to `SimulationLoop.__init__` (list).
2. In `_step()`, when entering SS for the first time (i.e., when `phase == 'SS'` and previous phase was DS), capture for the next ~50 ticks: `(t, p_torso_ref, v_torso_ref, qdd_t_qp, q_torso_actual, dq_torso_actual)`.
3. Run `multi_traversal_10x_dwell.seq` once. Compare the trace at step 0 (works) vs step 10 (fails).
4. If hypothesis confirmed (pre-planner picks aggressive trajectory from pristine state): the fix is in `crawlbot/planning/coarse_pre_planner.py` (or similar) — bound `a_cruise_max` or stretch `T_step` when initial state is unusually quiescent.

---

## Reproducibility — the failing scenarios

```bash
# Rapid back-to-back N=10 (fails at step 22 — h_w accumulation):
MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_cooperative_arms.py \
    --scenario scenarios/multi_traversal_10x.seq \
    --aocs_mode legacy_pid_numerical \
    --K_theta 36.3 --K_omega 355.4 \
    --settle_seconds 120

# With 30s inter-cycle dwells (fails at step 10 — DS→SS handoff):
MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_cooperative_arms.py \
    --scenario scenarios/multi_traversal_10x_dwell.seq \
    --aocs_mode legacy_pid_numerical \
    --K_theta 36.3 --K_omega 355.4 \
    --settle_seconds 120
```

---

## Key files / line refs for the work ahead

| File | Lines | What's there |
|---|---|---|
| `crawlbot/simulation/sim_loop.py` | ~1605–1641 | DWELL injection block (centroidal-DS settle in the middle of a scenario) |
| `crawlbot/simulation/sim_loop.py` | ~1683–1688 | SS-start weld release + NMPC warm-start reset |
| `crawlbot/simulation/sim_loop.py` | ~1610 | `_setup_torso_for_step` — captures `p_t0, R_t0` in struct frame |
| `crawlbot/solvers/wholebody_qp.py` | ~920–1010 | Centroidal-DS task block (settle_mode + ds_centroidal_active) |
| `crawlbot/planning/sequence_loader.py` | — | START/DOCK/DWELL parser |
| `scenarios/multi_traversal_10x*.seq` | — | The failing scenarios |
| `crawlbot/planning/coarse_pre_planner.py` | — | Likely home of the Tier-1.1 fix (if hypothesis confirmed) |

---

## What success looks like for next session

- DS→SS launch transient diagnosed (root cause identified, ideally with a trace).
- Either a one-line/one-config fix lands, OR a §6.3 smooth-handoff design is committed to a memo for follow-up.
- `multi_traversal_10x_dwell.seq` runs 50/50 docks.
- If 50/50 lands, kick off an N=100 or N=200 traversal run as the next mission-life data point.

Three-session estimate from here to empirical "3 000 traversals achieved or not" answer.

---

## Hard constraints carried forward

- All new behaviour gated behind config flags defaulting OFF (legacy bit-identical).
- 19/19 unit tests pass (`tests/test_reworked_qp.py` + `tests/test_aocs_physics.py`).
- No MJCF / URDF changes.
- Develop / commit / push on the active feature branch; no PR unless asked.
- Use Pinocchio struct-frame quantities (`rs.oMf_torso`, `rs.r_com`), not MJ world-frame.
- `pin == 3.9.0` (not `pinocchio` PyPI package — that's bogus).

