# Phase TORSO-REF-EXPORT-FIX — exported p_torso_ref is now CONTINUOUS across the whole traversal; control byte-identical

**Branch** `j2/ds-active-rework` · logging/export fix only, NO control change, NO canonical-control change ·
push-only, never merge (Idriss merges via GitHub UI).
**Fix commits:** `b619ef4` (interstep DS) + `300a364` (first trailing-settle attempt, superseded) + `b37b528`
(trailing settle, corrected guard). Verification run on the frozen `32aefaf` canonical config.

## The fix — option (b), two scoped pieces + one correction

The DS torso-reference export had **two independent discontinuity sources**, both fixed at the logging layer:

1. **Interstep DS** (`_log_ds_tick`) — was querying `reference_at(t_abs)` past the phase end, falling through
   to `_hold_reference()` = the `set_hold` pose **p_t0** (step START, a full stride behind).
   **Fix (`b619ef4`):** new additive `TorsoPlanner.reference_at_clamped(t)` (`torso_planner.py:458-478`) —
   clamps the query into `[t_start_first, t_end_last]` so past-end queries return the quintic's **terminal
   pose p_t1**; inside a phase it is exactly `reference_at` (unit-verified). `_log_ds_tick` now calls it
   (`sim_loop.py:947-954`). Mirrors the control path's own `ss_end−1e-3` cap precedent.
2. **Trailing-DS settle** (main `_step` logger) — logs `p_torso_ref_used`, which in DS carries the **live
   CoM-mapping output**: it jumped −604 mm at the SS→DS switch and ramped ~13 s to the dock-IK equilibrium.
   **Fix (`b37b528`):** when `phase=='DS' and settle_mode and not torso_planner.has_phase_at(t_log)` — i.e.
   the planner defines NO reference at that time and no torso-position task is active — log
   `reference_at_clamped(t_log)` (flat terminal hold) instead (`sim_loop.py:3536-3551`;
   `TorsoPlanner.has_phase_at` new additive helper, `torso_planner.py:451-456`).
   - **Why not the `ds_centroidal_active` flag** (the `300a364` attempt, which did NOT fire): the locked
     config runs centroidal DS in the trailing settle too — `dca.main` sets `cfg.ds_centroidal_mode=True`
     (`scripts/diag_cooperative_arms.py:352`) — so that flag is True there and cannot discriminate.
     **Phase coverage is the correct key:** the run-B DWELL's moving reference lives INSIDE an installed
     planner phase (`set_from_waypoints`, `sim_loop.py:2142-2146`) where the clamp is identity anyway ⇒
     run-B's QP-tracked moving reference is preserved automatically; the trailing settle / initial DS query
     OUTSIDE all phases ⇒ flat terminal/initial hold.

Rejected options: (a) changing `set_hold` and (c) changing `reference_at` — both are **control-consumed**
(`reference_at` feeds the SS/DS torso refs at `sim_loop.py:2856`; the hold state backs `com_reference_at`).
The chosen edits live only in `_log_ds_tick` (documented side-effect-free) and the post-solve logging block.

## VERIFY 1 — continuity: SS→DS transition, all 6 steps, BOTH runs (re-exported CSVs)

| step | C: ref_x last-SS → first-DS | jump | DS wander | U: last-SS → first-DS | jump | DS wander |
|---|---|---|---|---|---|---|
| 0 | 0.2896 → 0.2896 | **+0.00 mm** | 0.000 mm | 0.2896 → 0.2896 | **+0.00 mm** | 0.000 mm |
| 1 | 0.8205 → 0.8205 | **+0.00 mm** | 0.000 mm | 0.8408 → 0.8408 | **+0.00 mm** | 0.000 mm |
| 2 | 0.9859 → 0.9859 | **+0.00 mm** | 0.000 mm | 0.9999 → 0.9999 | **+0.00 mm** | 0.000 mm |
| 3 | 1.6053 → 1.6053 | **+0.00 mm** | 0.000 mm | 1.4774 → 1.4774 | **+0.00 mm** | 0.000 mm |
| 4 | 1.7650 → 1.7650 | **+0.00 mm** | 0.000 mm | 1.6886 → 1.6886 | **+0.00 mm** | 0.000 mm |
| 5 | 2.3764 → 2.3764 | **+0.00 mm** | 0.000 mm | 2.2414 → 2.2414 | **+0.00 mm** | 0.000 mm |

(Pre-fix: jumps of −113 to −617 mm on steps 0–4 and −604/−538 mm on step 5.) The reference now rises along
each quintic in SS, **holds exactly flat at p_t1 through the DS** (including the 20 s trailing settle), and the
next step's quintic starts from the live pose. The remaining next-SS seams (+2 to +14 mm, and −58 mm on the
step-0/2 arm-a re-anchors) are the **real tracking residuals** at re-anchoring — physical, not artifacts.

## VERIFY 2 — control byte-identical to the `32aefaf` run

Re-ran the full frozen canonical (U then C) on the final fix and diffed every control metric against the
committed pre-fix results: **n_dock, at-weld docks ×6, worst, planned/realized Ḣ_s per-axis AND per-step,
plan-at-cap counts, θ_s peak/settled, h_w per-axis/norm/overflow, drift, e_com, τ_w command max, clamp tick
counts, mjstep totals, qp_fail, nmpc_fail — IDENTICAL for both U and C** (`weights_match_frozen` true).
C: docks 6/6 at `4.02/4.89/4.99/4.97/4.95/4.62`, θ_s 0.540, e_com 0.154 — unchanged.
CSV row-level diff vs pre-fix: the only changed cells are the intended DS-row `p_torso_ref_*`/`e_torso_pos`
plus the wall-clock `nmpc_time_ms`/`qp_time_ms` columns (non-deterministic timing, phase-independent);
**all 508 SS rows differ in timing columns only.** As expected — the torso task is OFF in DS (settle-gated),
so the logged value was consumed by nothing.

## Files

| artifact | path |
|---|---|
| re-exported per-tick CSVs (continuous reference) | `results/j2_adjconv/c25_fulldiag.csv`, `u25_fulldiag.csv` (+ meta) |
| verification result JSON (byte-identical metrics) | `results/j2_adjconv/canonical2p5_result.json` |
| fix | `crawlbot/planning/torso_planner.py` (`has_phase_at`, `reference_at_clamped` — additive), `crawlbot/simulation/sim_loop.py` (`_log_ds_tick` + `_step` logging block) |

Full pytest on the final fix: running at report time; outcome appended below. **STOP** — ready for the
continuous torso figure.

---

## Addendum — full pytest on the final fix: CLEAN (identical failure set to baseline)

Full suite on the complete logging fix (`b619ef4` + `b37b528`): **219 passed, 2 failed (16:46)** — exactly the
two **pre-existing** baseline failures (`test_far_infeasible_under_tight_rate`,
`test_E7_t15_step2_dock_under_fk_mode`), the same set as pre-freeze and post-freeze. **Zero new failures**
(Rule 7 satisfied). **STOP.**
