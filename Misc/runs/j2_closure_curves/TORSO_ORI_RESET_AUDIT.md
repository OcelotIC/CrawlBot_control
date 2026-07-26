# INTERNAL — torso orientation-error reset at every DS→SS: case A (export) vs case B (controller)

Branch `j2/ds-active-rework`. **Read-only code audit — no `crawlbot/`/export change.** Resolves the one open
question from the data-level audit (the per-phase reset of `torso_ori_err_deg` in `runA_traversal.csv`):
does the re-zeroing originate in the EXPORT (A, benign) or the CONTROLLER (B, architectural)?

## VERDICT: **CASE B — the controller recaptures the torso-orientation target at each phase entry.**
The export is faithful (it copies the controller's logged error). The torso is held **per-phase-relatively**
(target := current torso orientation at each phase entry), NOT against a single global reference — so it is
free to accumulate orientation drift across the traversal. The TRUE cumulative drift is **~1.8° (final) /
~4.1° (peak)** in the structure frame — small and well-behaved, but real and unconstrained. The per-phase
resets MASK it. **No code change — this is an architecture/paper-framing discussion, per the brief.**

## Q1 — what the controller's torso-orientation task references (the A/B decider)
The whole-body torso-pose task reads its target from `self.torso_planner.reference_at(t)`:
- `crawlbot/simulation/sim_loop.py:943-948` — `tref = self.torso_planner.reference_at(t_abs); R_torso_ref =
  tref.R; R_err = R_torso_ref.T @ rs.oMf_torso.rotation; angle_err = arccos(...)`. The QP/log error is the
  geodesic angle vs this reference (also the SS path, ~:2807 `tr = self.torso_planner.reference_at(...)`).
- The reference is **RE-SET at each phase entry to the CURRENT torso orientation**, not captured once:
  - **SS entry** (`_setup_torso_for_step`): `R_t0 = rs_s.oMf_torso.rotation.copy()` (`:1411`, the live torso
    orientation at SS entry), then the end config is solved with the torso rotation **pinned to R_t0**:
    `dock_configuration_fixed_rotation(..., R_torso_fixed=R_t0, ...)` (`:1478`) ⇒ `R_t1 ≈ R_t0`, and the planner
    is loaded with this pose (`set_hold(p_t0, R_t0, ...)` `:1676`). The comment is explicit: "prefer to hold
    torso rotation at R_start ('crawl forward, don't pirouette')" (`:1416`) — i.e. hold **wherever it is now**,
    recaptured every step.
  - **DS / DWELL / terminal entry:** `set_hold(..., rs_*.oMf_torso.rotation.copy(), ...)` (`:2099-2101`,
    `:2479-2499`) — again the **current** torso rotation at DS entry.
- ⇒ the target is `R_t0(phase) = current torso orientation at that phase's entry`. At each DS→SS the reference
  is recaptured to the live (drifted) orientation, so `R_err → I` and the error snaps to ~0. **This is the
  controller, not the export.** Not a single global setpoint ⇒ **not case A.**

## Q2 — how the export computes `torso_ori_err_deg` (is the export faithful?)
`scripts/export_figure_data.py`:
- `:136` — `e_to = np.asarray(sl['e_torso_ori'], float)` — it **reads the controller's logged error
  verbatim** (the `angle_err` of sim_loop:948/the SS analogue), it does **not** recompute against any
  export-side reference.
- `:264` — `('torso_ori_err_deg', lambda i: f'{e_to[i]:.6f}')` — written through unchanged.
- ⇒ **the export is faithful.** The per-phase reset in the CSV is exactly the controller's error vs its
  per-phase-recaptured reference. (The first-tick 5.157° is the step=−1 init offset before the first capture.)

## Q3 — TRUE cumulative drift vs a SINGLE fixed t=0 reference (locked-config run, 1184 ticks, 60.9 s)
Geodesic angle of the torso orientation vs its **t=0** value (single fixed reference), per tick:

| metric | peak | final |
|---|---|---|
| **structure-frame** torso drift (`R_structᵀ·R_torso` vs t=0) — torso held on the platform | **4.118°** | **1.803°** |
| **world-frame** torso drift (`R_torso` vs t=0) — incl. structure rotation | 4.518° | 1.801° |
| per-phase-RESET metric `e_torso_ori` (the masked one) | 5.157° (=init offset; ≤1.25° within-phase after) | — |
| context: structure attitude ‖θ_s‖ (AOCS-held platform) | 0.607° | 0.098° |

Block-end structure-frame drift grows ~monotonically across the 5 steps: 0 → 0.66 → 0.45 → 0.58 → 0.73 →
1.16 → 1.45 → 1.34 → 1.43 → 1.59 → **1.80°**. So the torso accumulates ~1.8° of orientation drift over the
traversal (peaking ~4.1° during swing transients) — **a real accumulation that the per-phase resets hide**
(the within-phase error never exceeds ~1.25°). The structure itself is well-held (θ_s ≤ 0.6°), so this is the
**torso drifting relative to its own start**, not the platform.

## Conclusion (for the orientation-panel decision)
- **Case B confirmed.** The controller does not constrain torso orientation to a global reference; it
  re-references the current orientation at every phase entry (`R_torso_fixed = R_t0`, recaptured per step).
  The export faithfully reports that per-phase error.
- **Magnitude:** the unconstrained cumulative drift is **small (~1.8° final, ~4.1° peak)** — the "torso
  stabilized" claim is *mostly* true (well-behaved, bounded, no runaway) but is a **per-phase relative hold,
  not a global hold**, and it does accumulate ~1.8° over five steps.
- **fig6 orientation panel:** the per-phase-reset `torso_ori_err_deg` is **misleading for the global-stability
  claim** (it masks the accumulation). The honest figure is the **cumulative drift vs a single t=0 reference**
  (~1.8° final / ~4.1° peak). This is a paper-framing / architecture choice — **not an export bug** — so per
  the brief: reported, nothing changed.

**No code change, no merge, no PR.** (Read-only; the locked-config run used for Q3 is reproducible and not committed.)
