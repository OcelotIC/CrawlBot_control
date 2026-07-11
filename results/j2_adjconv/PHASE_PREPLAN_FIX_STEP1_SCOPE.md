# Phase PREPLAN-FIX · STEP 1 (STOP-GATE 1) — blast-radius scope of correcting the pre-planner rate constraint

READ + trace only, **no code change, no run**, `j2/ds-active-rework`. Goal: before applying the
CSTR-ORBITAL fix (make the pre-planner cap the about-origin `Ḣ_s`, matching the NMPC), determine
whether it changes **T_step** (full re-baseline) or only the plan shape (lighter).

## Headline verdict
- **T_step is DECOUPLED from the rate-cap quantity** — it is derived from the **momentum box `h_max`**,
  not `τ_w_max` and not the `L̇_com`-vs-`Ḣ_s` choice. Correcting the constraint **does not change T_step**.
- **The fix is surgical: `coarse_preplanner.py:342` only** (drop `- rk`). The state ODE `:317` MUST stay
  centroidal (its `L` state feeds the h_w box, the cost, and the conservation constant `c`).
- **The correction is almost certainly NOT slack** at non-zero standoff → the CoM **reference shape**,
  planned `Ḣ_s`, and dock gaps will **shift**; **T_step / phase schedule stay identical**.
  ⇒ **MEDIUM re-baseline** (same timing, re-measured reference/numbers), *conditional on feasibility*
  which only the Step-2 solve can confirm.

## 1. T_step derivation — BUDGET-DECOUPLED
`sim_loop.py:1886-1890` (`_run_preplanner`):
```python
h_max = np.asarray(cfg.h_max_tight).reshape(3)          # the MOMENTUM (wheel-storage) box
lever_arm = 1.0
v_max = min(|h_max|) / m / lever_arm                     # from h_max, NOT tau_w_max
distance = ||r_com_goal - r_com_0||
T_step_guess = max(0.5, distance / v_max)
```
- T_step is a function of **`h_max` (momentum box)** and the CoM transfer distance — it **never**
  references `τ_w_max` or the rate-constraint moment-arm (`L̇_com` vs `Ḣ_s`).
- It is passed as a **fixed parameter** `T_step=T_step_guess` (`:1899`); the pre-planner has **no T_step
  decision variable** (`p_dt = T_step/M` is a parameter, `coarse_preplanner.py:306`) and **echoes it**
  in `result.T_step`.
- **No retry loop:** a failed NLP solve **skips the step**, no heuristic fallback (`:1850-1852`).
⇒ Correcting `:342` cannot change T_step. (The prior T_step-frontier work coupled T_step to `h_max`,
not to the rate quantity — consistent.)

## 2. Downstream dependency list (what the constraint-quantity change perturbs)
The rate constraint `:342` bounds a function of the **controls `(f, τ)`** only. Changing its moment arm
from `(r_C − r_com)` (about CoM) to `r_C` (about origin) perturbs:
- **Directly:** the feasible `(f, τ)` set at each knot (the corrected form is *tighter* at standoff — it
  adds the orbital term `r_com×f`).
- **`f` → `v_dot = f/m` → the `r_com`/`v_com` trajectory** (`coarse_preplanner.py:315-316`) = the plan's
  CoM reference (`r_com_at`/`v_com_at`), pinned at the ends by the **hard** terminal `X[0:3,M]=r_goal`
  (`:357`) + fixed T_step, and soft terminal `v_N,L_N≈0` (`:362-363`).
- **NOT perturbed (must stay centroidal — do NOT touch `:317`):** the `L` **state** integrates the
  centroidal `L̇_com` (`:317`) and feeds:
  - the **h_w storage box** `h_w = c − L − r_com×mv` (`:374`), which requires the *centroidal* `L_com`;
  - the **cost** `w_L‖L‖²` and terminal `w_L_terminal‖L_N‖²` (`:385,389`);
  - the **conservation constant** `c = h_w0 + L_com0 + r_com0×mv0` (`sim_loop:1875`).
  Changing `:317` to about-origin would silently corrupt all three. ⇒ **fix `:342` only.**
- **Downstream of the pre-planner output:** `r_com_at`/`v_com_at` → NMPC `r_com_ref`/`v_com_ref`
  (`sim_loop:2648-2657`, the tracking reference, PREPLAN-ROLE) → torso ref → QP → realized motion.
- **T_step → scheduler SS phase** (`sim_loop:1398`) — **unchanged** (decoupled).

## 3. Identical or different after the fix?
- **T_step / phase schedule:** **IDENTICAL** (decoupled, §1).
- **Reference `r_com`/`v_com`, planned `Ḣ_s`, dock gaps:** **LIKELY DIFFERENT.** The correction is
  *not* slack: the current canonical solution chooses `τ` to keep the **centroidal** rate small
  (`L̇_com ≈ O(1) N·m`, Phase-1h R1), which leaves the **orbital** term `r_com×f = m·(r_com×a_com)`
  uncancelled in the about-origin quantity. Order of magnitude: `m≈71`, `|a_com|≈4Δr/T² ≈ 0.10 m/s²`
  (`Δr≈0.195 m`, `T≈2.78 s`), `|r_com| ≈ 0.9→2.4 m` across the traversal ⇒ orbital `≈ 6→17 N·m`, well
  over the `τ_w_max = 5` cap. Corroboration: on C the NMPC's *realized* about-origin `Ḣ_s` **saturates
  at exactly 5.00** (ABL-HDOT-2) — the tracked motion's about-origin demand is already *at the cap*, so
  a pre-planner that constrained `Ḣ_s ≤ 5` would re-shape its plan to that boundary (different `τ`, a
  swinging centroidal `L` state, and likely a slightly slower/re-shaped `r_com`/`v_com`). It would be
  *identical* only if about-origin `Ḣ_s` happened to stay < 5 everywhere — unlikely given the above.
- **Feasibility:** the orbital term (≤~17 N·m) is within `τ_max = 20 N·m`, so `τ` can most likely absorb
  it → feasible but with larger `τ` (higher `w_u‖τ‖²`) and a larger centroidal `L` excursion (fought by
  `w_L‖L‖²`). **Risk:** with no T_step retry, an infeasible high-standoff step is *skipped* (run breaks);
  Step 2 must confirm 6/6 steps stay feasible.

## Re-baseline verdict
**MEDIUM — not a timing re-baseline, but expect a reference/number shift.**
- Timing (`T_step`), phase schedule, and the h_w-box/cost/conservation machinery: **unchanged.**
- The plan's `τ` profile and centroidal `L` trajectory: **change** (now nulling the about-origin rate).
- The `r_com`/`v_com` reference, planned `Ḣ_s`, and per-step dock gaps: **likely shift** and must be
  re-measured → a new canonical baseline for those figures (but the *same* T_step/schedule).
- **Only the Step-2 corrected solve can turn "likely" into the exact outcome** (identical / re-shaped-
  feasible / infeasible-at-a-step). If any step goes infeasible, the fix's blast radius grows (needs a
  T_step-retry or an `h_max` bump) — flagged.

## Proposed surgical diff (SHOWN, uncommitted — awaiting Idriss GO)
```diff
--- a/crawlbot/planning/coarse_preplanner.py
@@ ~341-343 (inside the multiple-shooting loop) @@
-            rk = xk[0:3]
-            L_dot = ca.cross(p_r_C - rk, fk) + tauk
-            opti.subject_to(opti.bounded(-cfg.tau_w_max, L_dot, cfg.tau_w_max))
+            # About-origin Ḣ_s = r_C×f + τ (moment arm about the STRUCTURE origin;
+            # orbital r_com×f included). Matches the NMPC (centroidal_nmpc.py:279).
+            H_dot_s = ca.cross(p_r_C, fk) + tauk
+            opti.subject_to(opti.bounded(-cfg.tau_w_max, H_dot_s, cfg.tau_w_max))
```
Only the **constraint** moment arm changes (`p_r_C - rk` → `p_r_C`). The ODE `:317` and everything
else stay centroidal. STOP-GATE 1 — await GO before Step 2 (apply + re-run canonical C & U).
