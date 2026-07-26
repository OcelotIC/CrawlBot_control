# M7 — EE position bisection (Track 1, minimal)

**Date:** 2026-04-19
**Scope:** Decompose the 6.7× EE position inflation in SS across three
cases. Goal: test whether the mapping is the dominant contributor.
**Input artefacts:**
- `Misc/reports/architecture/M7_TECHNICAL_LOG.md` (post-correction)
- `Misc/runs/M7_abort_diag/R1_baseline/sim_log.json` (today's v21 SS
  reference, committed in 6128db9)
- `Misc/scripts/bisect_qp_cascade.py` — extend, do not rewrite
- `Misc/scripts/test_qp_tracking_v21.py` — standalone reference

---

## 1. Intent

Close a single dock. The standalone QP achieves 24 mm EE peak; the
closed-loop SS reaches 162 mm peak and 40.8 mm at abort, 35.8 mm short
of the 5 mm dock threshold. The M7 log §2 already names the mapping as
the dominant position-inflation source. This brief tests that
hypothesis with three cases — the minimum needed to isolate the
mapping. If the hypothesis holds, we fix the mapping. If it doesn't, we
broaden.

Diagnostic decomposition, not fix task. No architectural changes. No
interpretation in the deliverable.

## 2. The three cases

All use v21 config (preplanner_a_cruise_max=0.01, planned-δ,
α_wrench=0.01, α_com_soft=0.0, 7-DOF arms, manipulability init, EE
task-consistent FF). Same SS geometry: swing arm 'b', anchor 2→3,
T_step = 7.284 s.

### Case 1 — `A_swing`: standalone, SwingPlanner EE reference

- Torso reference: **constant** at initial pose.
- EE reference: `SwingPlanner.reference_at(t)` — quintic + clearance
  bump + delayed-cosine SLERP, fully configured as in closed-loop SS.
- No NMPC, no mapping, no AOCS.
- Contact: `SINGLE_A`. Weld released manually via
  `sim._deactivate_weld('b', 2)`.

Isolates the standalone floor under the real closed-loop EE reference
shape (not the analytical septic).

### Case 2 — `B_minus`: + NMPC, torso still constant

Add to case 1:
- NMPC runs at 10 Hz. `r_com_ref` driven by
  `sim._coarse_plan.r_com_at(t - sim._coarse_plan_t0)` (the pre-planner
  trajectory). Warm-start on.
- NMPC outputs `λ_ref` and `a_com_ff` enter the QP as in closed-loop.
- **Torso reference remains constant** (bypass mapping).

Isolates NMPC's direct-to-QP contribution.

### Case 3 — `B_v21`: + mapping with planned-δ

Add to case 2:
- Mapping active with **planned-δ** (`sim._planned_arm_config(t, rs)`,
  matching sim_loop). The existing `bisect_qp_cascade.py` case B uses
  live-δ (`q_current`) — update to planned-δ for this case.

Isolates the mapping's contribution.

Case D (full sim_loop) is **not re-run**. Reuse the SS metrics from
`Misc/runs/M7_abort_diag/R1_baseline/sim_log.json` via per-phase filter.

## 3. Implementation

Extend `Misc/scripts/bisect_qp_cascade.py` with three new `--case` values:
`A_swing`, `B_minus`, `B_v21`. Keep legacy cases intact.

Each case writes `Misc/runs/M7_ee_bisection/{case}/trace.csv` with `t`,
`e_ee_pos_mm`, `tau_q_max_Nm` per QP tick. Primary metric:
`ee_pos_peak_SS` = max over `t ∈ [0, T_step]`.

## 4. Invariants

- Same initial state at t = 0 across all three cases (verify
  `||qpos - qpos_ref|| < 1e-10` at start of each).
- Same T_step = 7.284 s hard-coded.
- `A_swing`: zero NMPC/mapping/AOCS calls.
- `B_minus`: NMPC calls > 0, mapping calls = 0.
- `B_v21`: NMPC calls > 0, mapping calls > 0.
- `pytest tests/ -v` still passes.

If any invariant fails, stop and report.

## 5. Deliverable

`Misc/runs/M7_ee_bisection/summary.md`:

```
case     | description                     | ee_pos_peak_SS [mm] | ee_pos_at_T_step [mm] | tau_q_peak_SS [Nm] | Δ from prev [mm]
A_swing  | standalone, SwingPlanner EE     |                     |                       |                    | —
B_minus  | + NMPC, torso const             |                     |                       |                    |
B_v21    | + mapping (planned-δ)           |                     |                       |                    |
D        | full sim_loop SS (from R1)      |                     |                       |                    |
```

Plus one overlay plot `Misc/runs/M7_ee_bisection/overlay.png` with
`e_ee_pos_mm(t)` per case over `[0, T_step]`.

No interpretation.

## 6. Prohibitions

- No modifications to `wholebody_qp.py`, `centroidal_nmpc.py`,
  `force_estimator.py`, `com_to_torso_mapping.py`, the planners, or
  `sim_loop.py` beyond minimal wrappers to expose helpers.
- No parameter retuning.
- No additional cases beyond the three specified.
- No re-simulation of case D.
- No intermediate scratch committed.

## 7. Predictions (locked in before data)

- **P1:** `A_swing` ≈ 25–35 mm (close to the 24 mm septic standalone).
- **P2:** `B_minus` adds < 20 mm over `A_swing`.
- **P3:** `B_v21` is the dominant step, > 80 mm added.
- **P4:** D within 20 mm of `B_v21`.

If P3 holds, next task is a focused mapping-layer investigation. If
P3 fails (inflation distributed or elsewhere), the bisection is
broadened to six cases per the full version of this brief.

## 8. Validation gate

Send summary + overlay. Idriss and reviewing Claude decide next move.
