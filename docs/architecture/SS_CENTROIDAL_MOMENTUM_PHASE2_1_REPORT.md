# SS Centroidal-Momentum Task — Phase 2.1 Report (single-step swing, α_mom sweep)

Branch `feat/ss-centroidal-momentum-task`. Memo §4 Phase 2.1 governs. Driver:
`scripts/diag_cooperative_arms.py --n-steps 1` (canonical step 1: start_a=start_b=2,
real NMPC, MuJoCo plant, cooperative-arms stack). Reporting: `scripts/report_phase2_1.py`
(h_w in per-axis **∞-norm**; τ_w-sat at the **100 Hz** QP rate from `log_hifreq_ss`;
`metrics.py` untouched).

---

## 1. Verdict

**T-MOM works under swing, the ratio question is RESOLVED BY WEIGHT, and the architecture
question is NOT triggered.** Variant A at the shipping `ss_alpha_mom=500` **fails to dock**
(47 mm timeout); raising α_mom to **5000 restores docking (2.79 mm)** with torso-angular
**better than baseline** (0.76° vs 0.98°), and torso-angular does **not** degrade as α_mom
rises — it *improves* monotonically (8.5°→0.67°), so the feared strict-P1-vs-P2 trade-off
does not materialise. **Variant B** (weak torso-linear regulariser) **docks at all three
weights including 500** (3.73 mm) — the regulariser rescues the low-authority regime — and
converges to Variant A at the working weight (B@5000 2.83 mm ≈ A@5000 2.79 mm). **Neither
variant is killed; the working point is α_mom ≈ 5000** (tightest dock, lowest jitter).

---

## 2. Preconditions

- Branch `feat/ss-centroidal-momentum-task`; runs tagged from clean HEAD `f43bec3`
  (the 100 Hz-logging infra commit).
- **Flag-OFF bit-identical:** the Phase-2.1 instrumentation (`cfg.log_hifreq_ss`,
  `SimLog.*_ss_hifreq`, the gated SS append) is **append-only and gated**; the shipping
  default (`log_hifreq_ss=False`, `ss_centroidal_momentum_task=False`) runs none of it, so
  the control trajectory + 10 Hz log + `postproc_F3F4.csv` are bit-identical by
  construction. The T-MOM control code is unchanged since `7f3d091` (Phase-1 proven point).
- **100 Hz SS logging validated:** `t_ss_hifreq` = 1080 samples at exactly 0.0100 s
  (100 Hz), 10× the 108 SS 10 Hz ticks.

---

## 3. Metrics — single step, vs flag-OFF baseline

| metric | baseline | A@500 | A@5000 | A@30000 | B@500 | B@5000 | B@30000 |
|---|---|---|---|---|---|---|---|
| docked | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ |
| d_final [mm] | 1.85 | 47.42 | 2.79 | 3.58 | 3.73 | 2.83 | 3.60 |
| torso-ori peak [deg] | 0.98 | 8.51 | 0.76 | 0.67 | 1.05 | 0.75 | 0.66 |
| torso jitter pk-pk x [mm] | 26.3 | 220.7 | 37.8 | 54.6 | 39.5 | 38.3 | 54.7 |
| h_w peak ∞-norm [N·m·s] | 0.83 | 1.07 | 1.41 | 1.63 | 0.74 | 1.41 | 1.62 |
| τ_w-sat @100 Hz [%] | 0.00 | 0.93 | 0.00 | 0.00 | 2.86 | 0.00 | 0.00 |
| joint-τ peak [N·m] (±20) | 3.27 | 12.77 | 3.18 | 3.50 | 5.44 | 3.18 | 3.53 |
| QP inner-loop p50/p99 [ms] | 82/90 | 85/97 | 79/92 | 67/74 | 78/92 | 70/80 | 70/77 |
| realized/planned Ḣ_s peak [N·m] | 1.97/1.96 | 3.38/2.47 | 2.28/2.26 | 2.31/2.28 | 1.95/1.93 | 2.27/2.25 | 2.30/2.27 |

(Full machine-generated table + plots: `Misc/runs/phase2_1_report/`. h_w 2-norm for the
baseline = 1.37 vs ∞-norm 0.83 — ∞-norm used throughout per brief.)

---

## 4. Ratio read (the central Phase-2.1 question)

| α_mom | docked | d_final | torso-ori peak |
|---|---|---|---|
| 500 (shipping) | ✗ timeout | 47.4 mm | 8.51° |
| 5000 | ✓ | 2.79 mm | 0.76° |
| 30000 | ✓ | 3.58 mm | 0.67° |

**Does raising α_mom restore CoM authority under swing? — YES.** Docking goes from
fail → 2.79 mm → 3.58 mm. **At what cost to torso-angular? — NONE; it improves.**
torso-ori peak goes 8.51° → 0.76° → 0.67° (monotonically down, ending *below* the 0.98°
baseline). h_w∞ rises 1.07 → 1.41 → 1.63 (more CoM authority spends more wheel momentum,
all well under the 5 N·m·s budget). **Sweet spot ≈ 5000** (tightest dock 2.79 mm, lowest
jitter 38 mm; 30000 docks slightly looser at 3.58 mm with rising jitter 55 mm + h_w 1.63).

**→ The sweep RESOLVES the ratio question. The strict-P1-vs-strong-weighted-P2 architecture
question is NOT triggered** (no torso-angular degradation as α_mom rises). Per memo §4
Phase 2.1, that question is therefore not raised.

---

## 5. Per-variant cascade-bisection diagnosis — A@500 failure

A@500 is the **only** misbehaviour. Signature: dock timeout at 47 mm, torso jitter **221 mm**,
torso-ori **8.5°**, joint-τ peak **12.8 N·m** (vs ~3.2 for the working runs), τ_w-sat 0.93%.
Diagnosis: **under-authority, exactly as Phase 2.0 predicted.** At α_mom=500
(= `alpha_torso_ang` under `weight_ratio=1`) the T-MOM CoM task cannot assert enough
authority under swing; the (unconstrained, Variant-A) torso linear drifts ~221 mm, the swing
cannot be coordinated to the dock, and joints fight the instability (12.8 N·m). This is **not
a structural fault** — it is the weight-balance the memo flagged ("shipping 500 balanced
torso-linear vs EE, but T-MOM is a different-nature task"). Raising α_mom (the sweep, i.e.
the sanctioned weight exploration — not reweighting-to-mask) fixes it. No other run misbehaves.

---

## 6. F-SAT confirmation (brief requirement)

F-SAT telemetry shows clips in every run (~9 mm max clip) **but under Variant A its output is
inactive on control**: F-SAT clips the CoMToTorsoMapping's torso-**linear** reference, and in
Variant A (`ss_alpha_tl_weak=0`) the torso-linear P2 channel is **removed** (T-MOM replaces
it) — the clipped reference feeds no QP task (`_coop_A_lin` is not added; the null-space
projector and EE feedforward use the angular rows only). So F-SAT executes but is
**computed-but-unused** under the flag (a cleanup opportunity; per brief, no removal this
phase). The baseline genuinely uses F-SAT (torso-linear active). The 221 mm torso jitter at
A@500 is the *free torso-position outcome*, not an F-SAT'd reference.

---

## 7. Reported-not-acted-on numbers

- **τ_w-sat at 100 Hz (paper cadence question):** baseline **0.00 %**, A@5000 **0.00 %**,
  A@30000 **0.00 %**; only the failing A@500 saturates (0.93 %). So at the true QP rate the
  working single step does not saturate the wheels. Logged + reported; **not acted on** (the
  separate paper τ_w-sat reconciliation is out of scope here).
- **h_w norms:** baseline ∞-norm 0.83 vs 2-norm 1.37 (reported once); ∞-norm used throughout.

---

## 8. Caveats (data honesty)

- **CoM-tracking-error metric (~100–140 mm peak) is NOT a variant discriminator** — it is
  ~130 mm even for the docking baseline (which tracks the torso via the mapping, *not* CoM
  directly), and it is a sustained gap, not a startup transient. It reflects the
  CoM-vs-NMPC-plan offset, similar across all runs. The operative outcomes are **dock margin**
  and **torso-ori**, which discriminate cleanly.
- **QP "solve time" = the 100 Hz inner-loop wall time** (10 QP solves + 10 physics steps +
  AOCS + GMO per NMPC step), ~70–90 ms; per-QP is a fraction. Environment-dependent (this
  container, osmesa disabled). Not a control concern; reported for completeness.

---

## 9. Variant B (weak torso-linear regulariser, `ss_alpha_tl_weak=50`)

**Variant B docks at ALL three weights, including α_mom=500** (B@500 **3.73 mm**, where
A@500 failed at 47 mm). The weak torso-linear regulariser **rescues the low-authority
regime**: at α_mom=500 it cuts torso jitter 221 → 40 mm, torso-ori 8.5° → 1.05°, and
joint-τ peak 12.8 → 5.4 N·m vs Variant A — i.e. it supplies the floor of torso-linear
authority that pure T-MOM lacks at 500. At the working weight the regulariser becomes
negligible: A@5000 (2.79 mm) ≈ B@5000 (2.83 mm), torso-ori 0.76° ≈ 0.75°, jitter 38 ≈ 38 mm;
likewise at 30000. The only B cost is a slightly higher τ_w-sat at 500 (2.86 % vs A 0.93 %),
moot since A@500 doesn't dock. **Neither variant is killed.** Both dock at α_mom≈5000 with
torso-ori parity-or-better than baseline; **Variant B is additionally robust at low weight.**

---

## 10. Carry-forwards / open items (for Phase 3 — do not act here)

1. **α_mom ≈ 5000 is the working point** (both variants), not the shipping 500. Phase 3
   (5-step) should run at the swept working weight, not inherit 500.
2. **Variant choice deferred to the Phase-3 gate** (per memo): both survive. Variant B is
   strictly more robust at low α_mom and equivalent at the working weight, at a small τ_w-sat
   cost only in the (non-docking) A@500 regime — a point in B's favour, but not a kill of A.
3. Architecture question (strict-P1 vs strong-weighted-P2): **not triggered** — torso-angular
   improved with α_mom in both variants. Closed for the single step; re-watch under multi-step.
4. F-SAT computed-but-unused under the flag → post-submission cleanup (no removal this phase).
5. No Phase-3 gate evaluated; no 5-step run (out of scope).
