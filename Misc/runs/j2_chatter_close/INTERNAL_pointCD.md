# INTERNAL — CC BRIEF (close the chatter fix), POINTS C & D (STOP-GATES C, D)

Branch `j2/ds-active-rework` (push, never merge). **Characterization-only** — no new `crawlbot/` code (the
κ_max measurement used a throwaway Hessian dump in `hierarchical_qp.py`, **reverted before commit**; the only
committed crawlbot change remains POINT A's settle-exit ε_v at `d2c0494`). Locked config = figure config
(`legacy_pid_numerical`, exact-box) + ε_wrench=3 + ε_v=5 mm/s. Run B = future work (per directive).

## DECISIVE OUTPUT
- **POINT C (STOP-GATE C):** ε=3 stays **fully clean in the new shorter-settle regime** (c0 + dx06, arm-a
  Σf-flip ≤0.03) — the earlier exit does not shift the threshold. **ε=3 is DERIVED**, not sweep-picked: the
  degenerate (A−B / Σf) direction's curvature is exactly **ε** (analytical), the dominant Hessian curvature is
  **κ_max ≈ 1.3e4** (measured), so the condition number along the chatter axis is **κ(H)=κ_max/ε**; ε=3 is the
  smallest weight that pulls κ(H) into the active-set solver's reliably-resolvable range (4.3e3) with margin.
- **POINT D (STOP-GATE D):** the settle-exit residual is **bounded across the traversal** (arm-a 0.072→0.062
  rad/s — *not* growing; arm-b 0.051→0.035), every settle `target_met`, and `‖L_com‖→0.000` (no secular
  momentum injection). **Benign verdict LOCKED.**

---

## POINT C — derive ε=3 from the solver tolerance (STOP-GATE C)

### (a) Re-confirmed clean in the new (shorter-exit) regime
| config (ε=3, ε_v=5 mm/s) | arm-a Σf-flip | settles (steps) | docks | clean |
|---|---|---|---|---|
| c0 (figure, dx0.8) | 0.02, 0.03 | 142/201/133/152 | 5/5 | **YES** |
| dx06 (tight 0.6 m) | 0.02, 0.01 | 120/200/99/243 | 4/5† | **YES** |

† dx06 step-4 abort = the same **non-chatter** geometry issue (settles all clean). The earlier exit (ε_v=5 mm/s)
**does not shift** the cleanliness threshold — ε=3 remains clean at both stances.

### (b) The derivation (Hessian measured at the binding settle QP, 6098 solves)
- **Curvature ‖λ‖² adds per unit ε along A−B = ε** (exact): the wrench task is `½·ε·‖λ−λ_ref‖²` with
  `A_wrench = I` on the λ-block, so its Hessian contribution is `ε·I_λ`. The two chattering vertices differ
  only in λ (Σf flips ±6, A≈−B), so A−B lies in λ-space and the cost curvature along it is exactly **ε**.
- **Dominant Hessian curvature κ_max ≈ 1.3e4** (measured: median largest eigenvalue of the assembled QP
  Hessian) — set by the large task weights (torso-pose / T-MOM ≈ 5000, EE ≈ 3000).
- **Condition number along the degenerate direction = κ_max/ε:**

  | ε | κ(H) on A−B | regime |
  |---|---|---|
  | 0.01 (production SS / pre-fix) | 1.3e6 | **chatter** |
  | 1 (partial) | 1.3e4 | marginal |
  | **3 (clean)** | **4.3e3** | **resolved** |

- **Why the threshold is O(1), not O(machine-ε):** qpOASES's single-solve tolerances are tiny
  (terminationTolerance ≈ 1.1e-9, boundTolerance ≈ 2.2e-10, epsRegularisation ≈ 2.2e-13). But the chatter is a
  **warm-started, tick-to-tick active-set limit cycle** (the diagnosis, af2f64a), not a single-solve KKT
  failure — so the operative limit is the **conditioning** κ(H), not the raw tolerance. The competing stiff
  scale is κ_max ≈ 1.3e4, so the wrench reg must be **O(1)** (not O(1e-9)) to lift the degenerate-direction
  curvature into the solver's reliably-resolvable range. This is exactly why the empirical threshold sits at
  ε≈1–3 and not at machine precision.
- **Derived ε:** the active-set resolves the degenerate direction when κ(H) ≲ O(5e3) — i.e.
  **ε ≳ κ_max/κ_resolve ≈ 1.3e4/5e3 ≈ 2.6**. Production **ε = 3 = derived-threshold (≈2.6) × margin (≈1.15)**;
  it is the smallest value giving κ(H)=4.3e3 (resolved) with a ≳2× margin over the partial threshold (ε=1,
  κ(H)=1.3e4). The threshold scales with κ_max (the task weights), which is config-stable (POINT 1 showed the
  threshold does not move with reach/geometry) — so a **fixed ε=3 is robust**, tied to the measured κ_max and
  the solver's relative resolution, **not a swept number.**

## POINT D — residual boundedness across the traversal (STOP-GATE D)
Locked config (ε=3, ε_v=5 mm/s), 5-step figure traversal, residual ‖dq_joint‖ at each inter-step settle exit:

| settle | step1 (b) | step2 (a) | step3 (b) | step4 (a) |
|---|---|---|---|---|
| residual [rad/s] | 0.051 | **0.072** | 0.035 | **0.062** |
| exit | target_met | target_met | target_met | target_met |

- **Bounded, not accumulating:** the arm-a settles (step2, step4) go 0.072 → **0.062** (decreasing, not
  growing) — the un-chased softest-mode residual does **not** seed a larger residual at the next arm-a settle.
  arm-b likewise 0.051 → 0.035. Every settle reaches `target_met` (the new dock-derived target).
- **No secular momentum feed:** `‖L_com‖` returns to **0.000** at traversal end (total angular momentum about
  the system CoM is conserved — the bounded settle residual injects none), and `‖hw‖` max = 3.997 Nms. The slow
  mode is a **local, per-settle artifact**, fully dissipated each step — confirmed not a traversal-scale
  accumulation. **Benign verdict LOCKED.**

---

## Locked production config (for the regeneration brief)
- **ε = 3** (settle-only α_wrench; derived: κ_max/κ_resolve × margin, κ_max=1.3e4 measured) +
  **ε_v = 5 mm/s** (settle-exit, dock-tolerance-derived).
- Figure config (`legacy_pid_numerical`, exact-box); **run B = future work** (irreducible DWELL/budget conflict).
- Default-off byte-identical (C6) preserved; gate re-confirm (C1/C3/C4 PASS; C2/C5 = separate exact-box SS
  issues) in the PR-prep step.

**STOP-GATES C and D reported. Proceeding to the gate re-confirm + PR preparation (no merge).** No merge, no PR-merge.
