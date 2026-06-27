# INTERNAL — J1 closing: Lemma 2 (centroidal reduction identity) validated in SS

**Goal:** the empirical validation J1 originally targeted (A.4) but could never get cleanly in DS
(low SNR, quiescent rates) — does the centroidal **reduction identity** (Lemma 2, §B) hold in the
**plant**? Tested in **SS** on the **Fix-A canonical** (`main = ae0673e`, the conserving plant).

**Identity (Lemma 2, §B; gravity off, single SS weld), inertial frame about O_s = structure CoM:**
> Ḣ_R/Os = −τ_contact  ⇔  Ḣ_R/Os = +w_weld/Os  (Newton–Euler for the robot subsystem: the rate of
> the robot's angular momentum about O_s equals the net external — i.e. weld — wrench on it about O_s).

**Mode:** non-perturbing instrumented read, Part-1 pattern (gated `log_hifreq_ss`, SS-only, per
QP-substep @100 Hz). **No mutation of `main`, no PR, no commit.** **Tooling:** instrument in
`crawlbot/simulation/{logging.py, sim_loop.py}` (`_log_lemma2_ss`); `scripts/{run_lemma2_ss.sh,
audit_lemma2_a0.py, audit_lemma2.py, plot_lemma2.py}`. Run: `results/lemma2_ss/`. Figure:
`lemma2_plots/lemma2_reduction_identity.png`.

---

## ✅ RESULT — Lemma 2 holds in the plant: ε ≈ 1 % of ‖Ḣ_R/Os‖, unstructured; the m_i moment is required

Both sides computed **MuJoCo-direct in the inertial/world frame about O_s** (the A′ lesson: the logged
`L_com` is structure-relative and was NOT used). LHS `Ḣ_R/Os` = central-FD of the robot-subtree
(torso+arms) angular momentum about O_s; RHS `w_weld/Os` = the **plant-realized** weld wrench (from
`qfrc_constraint`, not `lambda_qp`) reduced about O_s. Over the 3 highest-rate SS swings:

| swing | t [s] | ‖Ḣ_R/Os‖ mean [N·m] | **ε/‖Ḣ‖ (with m_i)** | ε/‖Ḣ‖ (without m_i) |
|---|---|---|---|---|
| 0 (binding) | 26.9–30.4 | 5.21 | **0.013** | 0.90 |
| 1 (binding) | 13.9–17.2 | 4.63 | **0.011** | 1.15 |
| 2 (binding) | 0.1–3.3 | 3.57 | **0.010** | 1.60 |

- **ε ≈ 1 % of ‖Ḣ_R/Os‖** — Lemma 2 is satisfied to finite-difference + measurement tolerance.
- **Sign:** `Ḣ_R/Os = +w_weld/Os` (the opposite sign gives ε/‖Ḣ‖ ≈ 2.0). This is Newton–Euler for the
  robot subsystem; equivalently `Ḣ_R/Os + τ_contact = 0` with `τ_contact = −w_weld` (the brief's
  structure-side convention).
- **The 6-DOF weld moment m_i is mandatory:** dropping it sends ε to **90–160 %** of ‖Ḣ‖. The weld
  transmits a real pure contact moment (mean |m| ≈ 2.6 N·m vs |f| ≈ 3.4 N) — the §B/J0 6-DOF claim,
  shown empirically.

## 0. Non-perturbation (instrument is read-only)

`lemma2_ss` vs canonical `fixA_gate`, all shared fields: **72 fields, worst |Δ| = 0.000e+00 →
BIT-IDENTICAL.** `mj_subtreeVel`/`mj_jacSite` only read `mj_data`; the trajectory is the canonical
Fix-A run. The instrument adds 8 SS-only hifreq fields; everything else is untouched.

## a. Weld-wrench reconstruction is exact (self-check)

The plant weld wrench is recovered from the realized `qfrc_constraint` via the world relative-site weld
Jacobian `J = [J_grip − J_anchor]`, `λ = lstsq(Jᵀ, qfrc_constraint) = [f; m]`. Per SS swing:
**`nefc = 6/6`** (the stance weld is the *only* active constraint — no joint limits/contacts) and the
reconstruction residual `‖Jᵀλ − qfrc_constraint‖/‖·‖ ≈ 1e-16` (machine precision). So `(f, m)` is the
exact plant weld wrench, not a fit.

## b/c. Residual is small AND unstructured; window-integral closes

Best swing (z-dominant, the crawl axis), per-axis residual vs rate:

| axis | mean‖Ḣ‖ | mean‖resid‖ | corr(resid, Ḣ) |
|---|---|---|---|
| x | 0.78 | 0.003 (0.4 %) | +0.15 |
| y | 1.89 | 0.029 (1.5 %) | +0.05 |
| z | 4.70 | 0.062 (1.3 %) | +0.02 |

The residual is **uncorrelated with the rate** (corr ≈ 0.02–0.15) → it is FD/measurement noise, **not a
structured frame/reduction-point error** (the failure mode A′ caught and this pipeline avoids by staying
fully inertial-about-O_s). **Window-integral cross-check:** ∫ w_weld/Os dt over the swing =
[−0.084, −0.209, 0.298] vs the realized ΔH_R/Os = [−0.085, −0.205, 0.295], ‖diff‖ = 0.0052 (**1.4 %**) —
the rate test and the accumulated test agree.

## d. J3 byproduct (noted, not chased)

Plant-realized weld wrench |f| ≈ 3.4 N, |m| ≈ 2.6 N·m (SS mean) vs `lambda_qp` (QP *command*, 12-vec
struct frame) ‖·‖ ≈ 5.7. The gap between the QP-commanded and plant-realized contact wrench is the
**ε_wrench** that **J3** diagnoses — observable here but **out of scope**. The Lemma-2 test used the
plant-realized wrench throughout (testing against `lambda_qp` would be circular — the QP enforces
`L̇_com = M_λ·λ` by construction).

## Verdict — A.4 / Lemma 2 closed for J1

The centroidal reduction identity **holds in the plant** (ε ≈ 1 %, unstructured, m_i-required,
integral-consistent), validated in the high-rate SS regime on the conserving Fix-A canonical. This is
the clean empirical close J1 could not get in the quiescent DS windows. Contact-count invariance (J0):
the identity validated at n = 1 (single SS stance weld) is the same one that governs n = 2 (DS). No
mutation, no commit (awaiting Idriss's direction).
