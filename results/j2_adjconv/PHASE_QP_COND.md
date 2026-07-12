# Phase QP-COND — a common-factor weight rescale CANNOT reduce κ(H); the premise is scale-invariant

**Branch** `j2/ds-active-rework` · read + measure only, NO canonical change · pushed, never merged.
Data: `results/j2_adjconv/qpcond_measure.json`; scripts `scripts/diag_qpcond_measure.py`
(instrumented 1-step canonical, monkeypatch capture — nothing committed to `crawlbot/`) and
`scripts/diag_qpcond_confirm.py` (closed-loop 6-step confirmation). Solver: **qpOASES** (CasADi conic).

**Headline:** κ(H) is genuinely large (DS ≈ 1.0e6, SS ≈ 3.6e6) — but it is set **entirely by the weight
span**, and the condition number of a weighted-LS Hessian is **invariant under a common-factor rescale**
(κ(cH)=κ(H)). So the requested change keeps the canonical numerically identical (true) but delivers
**zero** conditioning benefit (the goal fails). The only scale-dependent term is an absolute `1e-6·I`
regularization, which only affects κ by dominating the problem — i.e. by breaking the solution.

---

## STEP 0 — baseline conditioning + weight-vs-Jacobian decomposition

Full weight list of the canonical SS docking QP (`ss_two_task_mode`, `weight_ratio=1` ⇒ every task sums
at face-value α; `wholebody_qp.py:678-722`, `:1189-1290`; Hessian assembly `hierarchical_qp.py:261-276`):

| task | Jacobian | α (SS canonical) | priority |
|---|---|---|---|
| hw-slack (M5) | slack | 10 000 | 1 |
| torso-pose 6-D | J_torso | **24 000** (max) | 2 |
| momentum (lin CMM) | J_com | 5 000 | 2 |
| swing-EE 6-D | J_ee | 3 000 | 2 |
| posture | I(nq) | 20 | 3 |
| wrench-track | I(λ) | **0.01** (min) | 4 |
| torque-min | I(τ) | 1 | 5 |
| accel-reg | I | **0.01** (min) | 6 |

**SS weight span = 24 000 / 0.01 = 2.4e6.** DS (settle) span = 10 000 / 0.01 = 1.0e6 (settle-damping
1 000, hw-slack 10 000, wrench 3, torque 1, accel-reg 0.01).

κ(H) measured on canonical C (H = Σ αᵢ AᵢᵀAᵢ + 1e-6 I):

| tick | n_eq | κ(H, +1e-6 reg) | κ(H_LS, no reg) | **κ(flat, weights=1 = Jacobian-only)** | weight-span inflation |
|---|---|---|---|---|---|
| DS-settle (nc=2) | 32 | 9.999e5 | 1.000e6 | **2.00** | ×499 950 |
| SS-mid-swing (nc=1) | 26 | 3.627e6 | 3.628e6 | **8.07** | ×449 377 |
| SS-dock-approach (nc=1) | 26 | 3.555e6 | 3.555e6 | **8.14** | ×436 679 |

Trajectory over the whole run: DS κ ≈ **9.999e5** (flat, 402 ticks); SS κ ∈ **[3.545e6, 3.751e6]** (64 ticks).

**Decomposition verdict:** the task **Jacobians are near-perfectly conditioned (κ_flat = 2–8)**. The
weights inflate this baseline by **×450 000** up to κ(H) ≈ 1e6–3.6e6, which sits at the **same order as
the weight span** (κ_H/span = 1.00 DS, 1.48 SS). Conditioning is **~100 % a weight-ratio problem, ~0 % a
Jacobian problem.** The 1e-6 reg is currently negligible (κ with vs without reg differ < 0.01 %) — it does
not set λ_min (the smallest always-on task weight, 0.01, does).

---

## STEP 1/2 — the rescale test: κ-invariant, solution-invariant (to solver tol), scale-term flagged

Dividing **every** task weight by a common divisor D preserves all ratios exactly (torso-pose:EE stays
8:1, momentum:EE 1.67:1, EE:posture 150:1, …). Per-tick re-solve, original vs rescaled, on the captured
canonical Hessians. Two variants: **(a) reg FIXED at 1e-6** (= exactly what editing the config task
weights does — `regularization` is a separate absolute constructor arg, `hierarchical_qp.py:98,275`) and
**(b) reg scaled by 1/D too** (= pure scalar H→H/D, the mathematical invariance baseline).

**SS-dock-approach tick** (κ₀ = 3.5548e6), max‖Δx*‖ vs baseline:

| D (24000→) | κ (a: reg fixed) | ‖Δx*‖ (a) | κ (b: pure cH) | ‖Δx*‖ (b) |
|---|---|---|---|---|
| 24 (→1000) | 3.5467e6 (−0.2 %) | 1.7e-5 | **3.5548e6 (identical)** | 2.9e-15 |
| 240 (→100) | 3.4719e6 (−2.3 %) | 1.8e-4 | **3.5548e6 (identical)** | 1.0e-15 |
| 1e6 (→0.024) | 3.5201e4 (−99 %) | **5.0e-1** | **3.5548e6 (identical)** | 1.4e-15 |

(DS + SS-mid ticks identical story; full table in `qpcond_measure.json`.)

**Read-out:**
- **(b) pure cH: κ IDENTICAL at every D, ‖Δx*‖ ~1e-15 (machine precision).** This is the mathematical
  fact κ(cH)=κ(H) and x*(cH,cg)=x*(H,g), confirmed empirically. **A common-factor rescale that also scales
  the reg is bit-identical AND κ-identical.** There is nothing to gain: κ does not move.
- **(a) reg-fixed (the realistic config edit): κ barely moves for any behaviour-preserving D** (−0.2 % at
  D=24, −2.3 % at D=240). κ only collapses at D=1e6 — where the fixed 1e-6 reg finally dominates λ_min —
  and there **‖Δx*‖ = 0.5 (the solution is destroyed).** So the *only* way scaling changes κ is by making
  the problem reg-dominated, i.e. by changing the behaviour.
- **Scale-dependent term FOUND (as the prompt anticipated):** `HierarchicalQP.regularization = 1e-6·I`
  (absolute Tikhonov, `hierarchical_qp.py:98`, added at `:275`). It is the sole reason variant (a)
  deviates from exact invariance (‖Δx*‖ grows 1.7e-5→1.8e-4→0.5 as D rises and the fixed reg's relative
  weight climbs). The qpOASES backend itself is scale-clean (variant b ‖Δx*‖ ~1e-15 across 6 orders of D;
  qpOASES uses nWSR/CPUtime budgets, not absolute cost tolerances that would bite here).

### Closed-loop confirmation (full 6-step canonical C, realistic rescale: all task weights ÷24, reg fixed)

`scripts/diag_qpcond_confirm.py 24 0 6` — divides EVERY task weight by 24 (24 000→1 000, ratios exact),
reg left at 1e-6, full traversal. SS-phase closest-approach dock (C5 metric) vs committed baseline
`results/figC_sw_s5_x1`:

| step | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| baseline SS-dock [mm] | 4.9401 | 4.4054 | 4.9039 | 4.4358 | 4.0453 | 4.9999 |
| **rescaled ÷24 SS-dock [mm]** | **4.9401** | **4.4054** | **4.9040** | **4.4358** | **4.0454** | **5.0000** |
| \|Δ\| [mm] | 0.0000 | 0.0000 | 0.0001 | 0.0000 | 0.0001 | 0.0001 |
| T_step [s] (both, bit-identical) | 2.775 | 8.399 | 2.868 | 7.980 | 3.056 | 6.112 |

**All 6 docks match to ≤ 0.0001 mm (0.1 µm); T_steps bit-identical.** The 0.1 µm residuals on steps 2/4/5
are the accumulated fixed-reg per-tick perturbation (≤7e-5 per solve) — utterly negligible, far below the
5 mm gate. Canonical is numerically identical, as predicted. (κ over this run: unchanged from baseline —
DS 9.999e5, SS 3.55e6 — since κ(cH)=κ(H).)

---

## Verdict

1. **κ(H) is real and large (DS ~1.0e6, SS ~3.6e6), but it is the weight span, not the Jacobians**
   (κ_Jacobian-only = 2–8; span inflation ×450 000). It is the *intended* cost of encoding the task
   hierarchy as weight ratios (primary tasks 3 000–24 000 vs tiny regularizers 0.01) in a single weighted
   QP with `weight_ratio=1` and no null-space projection.
2. **A common-factor weight rescale cannot reduce κ.** κ(cH)=κ(H) exactly — proven analytically and
   confirmed empirically (variant b: κ identical at every D to the last digit). The premise
   ("common-factor rescale tightens conditioning") is **mathematically false**: κ depends on the weight
   *span/ratios*, which a common factor leaves unchanged, not on the absolute magnitude.
3. **The rescale IS behaviour-safe** (variant b bit-identical; variant a identical to solver tolerance for
   sane D) — so it does no harm — but it buys **nothing**. There is no reason to apply it.
4. **Scale-dependent term:** the absolute `1e-6·I` reg (`hierarchical_qp.py:98,275`). If one ever *did*
   rescale weights, the reg must be scaled by the same factor to preserve exact invariance; leaving it
   fixed introduces a small (∝ relative-reg-growth) perturbation and, at extreme D, changes the solution.
5. **The only levers that actually lower κ are behaviour-changing or solver-level, NOT a weight rescale:**
   (i) compress the weight *ratios* (loses the priority separation — changes behaviour); (ii) switch to the
   strict/null-space hierarchy (`method='strict'` / the M2 null-space stack, which achieves priority
   isolation geometrically at `weight_ratio=1` and needs no large span); or (iii) diagonal preconditioning
   x = S·y (a similarity transform that lowers κ(SᵀHS) while preserving x*). None is a common-factor
   weight rescale. **Recommend NOT pursuing the rescale; if conditioning is a genuine concern, (ii)/(iii)
   are the defensible routes — behaviour-changing, so GATED on Idriss.**

---

## Deliverable (STOP-GATE)
- **Full weight list + span** (§Step 0): 8 tasks, max 24 000 (torso-pose) / min 0.01 (wrench, accel-reg) →
  span 2.4e6 (SS), 1.0e6 (DS).
- **Common divisor + ratio table:** dividing all weights by D preserves every ratio by construction
  (torso-pose:EE 8:1, momentum:EE 1.67:1 unchanged) — ratio table **identical before/after**.
- **κ(H) before/after** (§Step 1/2): pure-cH **identical** at all D; reg-fixed −0.2 % (D=24) / −2.3 %
  (D=240) — **not materially lower**; only −99 % at D=1e6 where the solution breaks.
- **Canonical C identical:** per-tick ‖Δx*‖ ~1e-15 (pure cH) / ≤7e-5 (reg-fixed, sane D); closed-loop
  6-dock table (above) confirms.
- **Canonical U:** not re-run separately — the invariance κ(cH)=κ(H) and x*(cH,cg)=x*(H,g) is a property
  of scalar multiplication, **independent of the task weights / config**, so it holds identically for U
  (rate-off) as for C. A U re-run would only re-demonstrate the same solver-level fact; given the verdict
  is a no-op, it adds no information. (If desired, `diag_qpcond_confirm.py` runs against any config.)
- **Scale-dependent term flagged:** absolute `regularization=1e-6·I` (`hierarchical_qp.py:98,275`).
- **Verdict: the common-factor rescale is a NO-OP for conditioning (κ scale-invariant). Do not apply.**

NO canonical change. `crawlbot/` untouched. Scratch runs gitignored. **STOP for cross-check.**
