# Phase REG-DIAG — the QP regularization ε is an INERT null-space filler, not a dock-precision limiter

**Branch** `j2/ds-active-rework` · measurement sweep, NO canonical change · pushed, never merged.
Data: `results/j2_adjconv/reg_sweep.json`; scripts `scripts/diag_regdiag_{run_one,sweep,extract}.py`
(monkeypatch forces `HierarchicalQP.regularization=ε`; no `crawlbot/` change). Each ε = one full canonical
C run (**EE=3000 baseline, ε isolated — NOT combined with the EE6000 change**). Solver: qpOASES.

**Headline:** ε is a **null-space filler with no effect on dock precision**. Docks are **bit-identical across
ε ∈ [1e-12, 1e-6]** (six orders), and κ(H) does **not** blow up as ε→0 — because the task-only Hessian
`H_LS = Σ αᵢ AᵢᵀAᵢ` is **already positive-definite (min λ_min = 0.01)** from the 0.01-weighted regulariser
tasks (accel-reg, wrench-track). ε is redundant insurance; the shipped 1e-6 sits 4 orders below λ_min.

---

## Dock / θ_s / Ḣ_s vs ε (full canonical C, 6 steps)

| ε | 6 SS-docks [mm] (s0…s5) | worst | Δ vs baseline | θ_s pk | dock/QP |
|---|---|---|---|---|---|
| 1e-4 | 4.940 4.405 4.904 4.436 4.045 **4.589** | 4.940 | **0.411** (s5 only) | 0.5936 | 6dk / 0 fail |
| **1e-6** (shipped) | 4.940 4.405 4.904 4.436 4.045 5.000 | 5.000 | 0.00000 | 0.5936 | 6dk / 0 fail |
| 1e-8 | 4.940 4.405 4.904 4.436 4.045 5.000 | 5.000 | 0.00000 | 0.5936 | 6dk / 0 fail |
| 1e-10 | 4.940 4.405 4.904 4.436 4.045 5.000 | 5.000 | 0.00000 | 0.5936 | 6dk / 0 fail |
| 1e-12 | 4.940 4.405 4.904 4.436 4.045 5.000 | 5.000 | 0.00000 | 0.5936 | 6dk / 0 fail |

- **Docks are BIT-IDENTICAL for ε ∈ [1e-12, 1e-6]** (Δ vs committed baseline `figC_sw_s5_x1` = 0.00000).
  ε=1e-6 exactly reproduces the baseline → the harness is faithful. **The ~5 mm floor is NOT from ε.**
- The **only** ε that moves any dock is 1e-4 — and it shifts **only step 5** (5.000→4.589). ε=1e-4 is just
  2 orders below λ_min=0.01, so the reg begins to perturb the least-constrained (near-null) DOFs. This is
  **reg contamination onset, not a precision lever**: it is an uncontrolled nudge of the terminal
  tracking-limited step, it leaves the real floor untouched (worst is still 4.904/4.940, steps 0/2), and it
  bounds ε from **above** (ε must stay ≪ λ_min to remain inert).
- **θ_s = 0.5936 for every ε** (identical). **Realized |Ḣ_s|pk is bit-identical for every ε** (s0 4.817,
  s1 1.923, s2 5.000, s3 1.940, s4 5.000, s5 2.438 — same to 3 dp even at 1e-4). The momentum-management
  result is ε-invariant.

### Step-2 (momentum-saturated) floor is ε-invariant — confirmed
| ε | 1e-4 | 1e-6 | 1e-8 | 1e-10 | 1e-12 |
|---|---|---|---|---|---|
| step-2 dock [mm] | 4.9041 | 4.9039 | 4.9039 | 4.9039 | 4.9039 |
| step-2 Ḣ_s | 5.000 | 5.000 | 5.000 | 5.000 | 5.000 |

Step 2 moves **0.0002 mm** even at ε=1e-4 (where step 5 moved 0.41 mm). It is pinned at ~4.90 mm with
Ḣ_s saturated at the 5.0 momentum-rate cap — **ε does not touch the momentum-saturated floor** (as
predicted; cross-checks DOCK-CAUSE / EE-RATIO-GLOBAL). The step-5 sensitivity at 1e-4 is precisely because
step 5 is the *tracking-limited* step with Ḣ_s headroom (2.44) — its near-null solution is perturbable,
the momentum-capped one is not.

---

## Conditioning κ(H) vs ε + well-posedness floor

κ(H) = (λ_max(H_LS)+ε)/(λ_min(H_LS)+ε), measured at representative ticks; **min λ_min(H_LS) over the whole
run = 0.0100 for every ε** (H_LS excludes reg, so this is ε-independent — a cross-check the monkeypatch is
clean).

| ε | κ_DS | κ_SS_mid | κ_SS_dock | min λ_min(H_LS) |
|---|---|---|---|---|
| 1e-4 | 9.901e5 | 3.930e6 | 3.607e6 | 0.0100 |
| 1e-6 | 9.999e5 | 3.969e6 | 3.643e6 | 0.0100 |
| 1e-8 | 1.000e6 | 3.970e6 | 3.643e6 | 0.0100 |
| 1e-10 | 1.000e6 | 3.970e6 | 3.643e6 | 0.0100 |
| 1e-12 | 1.000e6 | 3.970e6 | 3.643e6 | 0.0100 |

- **κ does NOT blow up as ε→0.** For ε ≤ 1e-6, κ is **constant** (3.643e6 SS-dock, 1.000e6 DS) — it has
  converged to λ_max/λ_min. As ε→0, κ → λ_max/λ_min = **finite**, because λ_min(H_LS)=0.01 ≫ ε. There is no
  near-singularity anywhere down to 1e-12; **0 QP failures at every ε** (incl. 1e-12), solver clean.
- The only ε that changes κ is 1e-4, which *lowers* it slightly (3.643e6→3.607e6) because ε=1e-4 is 1 % of
  λ_min and lifts the denominator — i.e. more reg helps κ a hair but starts perturbing the solution (the
  step-5 shift above). This is the trade the reg makes; it is **negligible at 1e-6**.

**Well-posedness floor ε_min ≈ 0 for this task stack.** `H_LS` is positive-definite on its own
(λ_min = 0.01 everywhere in the run), supplied by the always-on 0.01-weighted **accel-reg** (on all q̈) and
**wrench-track** (on all λ) tasks. The 1e-6 reg is 4 orders below that — it never sets λ_min, never binds.
There is **no lower ε in the swept range that makes H ill-posed**; ε could go to 0 without singularity here
(not tested per the brief). Practically, any ε ≪ λ_min=0.01 (say ≤ 1e-5) is inert and solver-clean; the
shipped **1e-6 is a safe, comfortable choice** (inert for precision, redundant for conditioning).

---

## Verdict

1. **ε is INERT for dock precision.** Docks bit-identical across ε ∈ [1e-12, 1e-6] (Δ=0.00000 vs baseline).
   The ~5 mm dock floor is **not** an ε artifact — ε does not pull the EE solution in the well-posed range.
2. **ε is a null-space filler, not a precision limiter.** It only matters in directions where `H_LS` is
   rank-deficient — but `H_LS` is never rank-deficient here (λ_min=0.01, from the 0.01 regulariser tasks),
   so ε does nothing in [1e-12, 1e-6].
3. **Step-2 (momentum-saturated) floor is ε-invariant** (4.9039 mm flat, Ḣ_s=5.00) — the floor is the
   momentum-rate cap + swing-b kinematics, not the reg. Confirmed.
4. **κ does not blow up as ε→0** (constant λ_max/λ_min; H_LS already PD). **ε_min ≈ 0**; the well-posedness
   is provided by the task weights, not ε. All ε down to 1e-12: 0 QP failures.
5. **Upper bound on ε:** at ε=1e-4 (2 orders below λ_min) the reg **begins to contaminate** the solution
   (step-5 dock shifts 0.41 mm — a spurious near-null perturbation, not a gain, leaving the real 4.9 mm
   floor intact). So the inert window is roughly ε ∈ (0, ~1e-5]; **1e-6 sits safely inside it.**

**Conclusion: leave ε at 1e-6.** It is inert for the dock (no precision to reclaim by lowering it) and the
Hessian is well-posed independent of it. The dock floor is momentum/kinematic (steps 0/2/swing-b), not a
regularization artifact — consistent with DOCK-CAUSE, EE-RATIO-GLOBAL, and STRICT-HIERARCHY-AUDIT.

---

## Deliverable (STOP-GATE)
- **Dock-vs-ε table** (6 docks + θ_s + Ḣ_s + κ(H) + solver status per ε): above + `reg_sweep.json`.
- **Verdict:** ε **inert** for precision — docks bit-identical over [1e-12, 1e-6]; the only movement is a
  spurious step-5 nudge at 1e-4 (reg contamination onset, not a gain).
- **ε_min well-posedness floor ≈ 0:** min λ_min(H_LS)=0.01 everywhere; H_LS is PD without ε; 0 QP failures
  down to 1e-12. Shipped 1e-6 is 4 orders below λ_min — safe and inert.
- **Step-2 floor ε-invariant:** 4.9039 mm flat (Δ 0.0002 mm even at 1e-4), Ḣ_s=5.00 — momentum-capped,
  not reg-limited.

NO canonical change. `crawlbot/` untouched. Raw runs (`figC_reg_*`) gitignored. **STOP for cross-check.**
