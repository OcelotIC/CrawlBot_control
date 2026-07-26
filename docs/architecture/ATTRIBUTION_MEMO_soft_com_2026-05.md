# Attribution memo — the step-2 QP↔NMPC wrench mismatch (2026-05-28)

**Question.** The campaign (`CAMPAIGN_5STEP_TRAVERSAL_2026-05.md` §6/§7)
attributes the **10.8× QP-vs-NMPC contact-force mismatch** at the step-2
peak (`t≈16.63`, 1% canonical run) to the **soft-CoM residual being OFF**
(`α_com_soft=0`) and proposes re-engaging it. Before tuning that knob,
**which mechanism actually produces the excess wrench?** Soft-CoM can only
help if the answer is "the QP is choosing a momentum-inconsistent point in
the torso null space" — it is *constitutively unable* to touch internal
force, and is *the wrong tool* if the reference itself is wrong.

All work is **investigation only** (one standalone analysis script + one
config-flag re-run; no change to any control code). Canonical baseline:
`Misc/runs/diag_cooperative_arms/sim_log.json`.

---

## Falsifier table

| # | Hypothesis | Test | Result | Verdict |
|---|---|---|---|---|
| H1 | **Internal/squeeze force** — two contacts simultaneously large & opposing, living in the null space of the 12→6 centroidal map (soft-CoM cannot touch it). | λ decomposition at the peak (`scripts/diag_lambda_decomp_step2.py`, pure analysis). | **Contact-2 force ≡ 0 N across all 89 step-2 SS ticks** (max 0.000 N); only the single stance weld is active. A single 6D contact has **no internal null space** → internal force = 0 by construction. `|f1|=62.2 N`, independently cross-checked by `m·\|a_com\|=45.8 N` (CoM is genuinely accelerating). | **REFUTED.** The 62 N is **net** wrench, not internal. |
| H2 | **Arm angular-momentum** — the swing arm's rapid motion on the longest stride dumps centroidal angular momentum the point-mass NMPC never modelled (→ CMM-feedforward). | Compare logged `L_dot` (actual ḣ_com) at the peak against the contact moment and the 5 Nm budget (pure analysis). | **`\|L_dot\|=5.40 Nm` at the peak — within the 5 Nm budget (1.1×).** The 36 Nm contact *moment* is mostly the lever-arm reaction (r×f) of the anomalous 62 N *force*, not an angular-momentum dump. Angular momentum is **not** anomalous. | **REFUTED as the primary driver.** Arm angular-momentum is in budget; the anomaly is **linear** (CoM-acceleration excess). |
| H3 | **Mapping / F-SAT debt** — the world-frame `δ(q_current)` feedback + F-SAT rate-clamp (clips 49.6% of SS ticks) injects jerky base-acceleration demands the QP chases, inflating the contact *force*. | (a) Re-run 1% canonical with `use_local_delta_mapping=True` (`scripts/diag_loopfree_mapping_step2.py`). (b) Pure-analysis spike statistics on the canonical log. | **(a) Loop-free flag-flip is BLOCKED: it times out at *step 0*** (d=30.9 mm, ori=5.67° at the gate; torso tracking fine at e_ori≤4.5°/e_pos≤3.2 mm — the swing EE just won't close). 0/5 docks, never reaches step 2 → the *direct* test is impossible (loop-free's own unmitigated §3 drift; §6 mitigation absent). **(b) But the canonical log itself shows the peak is a transient jerk:** over step-2 SS `\|f1\|` median **2.2 N**, p95 **10.3 N**, peak **62.2 N** (peak ≈ **28× median**), while `v_com` tracks `v_com_ref` at **1.04×** (CoM *velocity* on-plan). A lone high-frequency spike with on-plan velocity is the **F-SAT/δ(q_current) jitter signature**, not a sustained authority shortfall. | **Mechanism SUPPORTED (by route b); fix BLOCKED (route a).** The peak is mapping jitter. The loop-free *cure* is not a free swap — it regresses docking at step 0. |

---

## Standing facts (independent of H3)

- **The peak wrench is net, single-contact, and real.** `|f1|=62.2 N`
  through the stance weld; `m·|a_com|≈46 N` confirms the CoM is being
  accelerated at ~0.65 m/s² — ~8× the NMPC's planned 0.08 m/s²
  (`|f|_NMPC=5.76 N`). The NMPC moment is *saturated* at its preplanner
  cap (`|τ|_NMPC=8.00 Nm`).
- **The anomaly is linear, not angular.** `|L_dot|=5.4 Nm` (in budget);
  the large contact moment is the r×f reaction of the linear force.
- **Soft-CoM cannot be the right primary fix here, regardless of H3:**
  - It is feedback that enforces tracking of `a_com_des =
    a_com_ff(NMPC)+PD` (`wholebody_qp.py:578`). The point-mass NMPC has
    **no arm-momentum term**; soft-CoM and a CMM-feedforward into the
    NMPC are duals, and **feedback against a wrong reference cannot
    win**. (The campaign's prior `M5_alpha_sweep` "evidence" is invalid
    anyway — it ran `cooperative_arms_mode=False`, the wrong stack;
    `CAMPAIGN…md` §7c.)
  - Its projection basis collapses to `null(torso-angular)∩null(EE)` in
    cooperative mode, so it would land in the **torso-linear** subspace
    — i.e. compete with a co-equal P2 task (hierarchy redesign, not a
    knob; §7c).

---

## Verdict

Mapping to the user's three buckets {CMM-feedforward, loop-free-mapping,
internal-force / different problem}:

- **internal-force → RULED OUT.** Single active contact at the peak; no
  null space to hide internal force in (H1).
- **CMM-feedforward → NOT INDICATED.** The thing CMM-feedforward would
  correct — centroidal *momentum* the point-mass NMPC mis-budgets — is
  **in budget**: `|L_dot|=5.4 N·m` (≤5 N·m limit) and `v_com` tracks the
  plan at 1.04× (H2). The anomaly is not a sustained momentum/inertia
  shortfall, so a richer NMPC inertia term would not address it.
- **mapping (loop-free) → this is the mechanism, but the cure is
  blocked.** The 62 N peak is a **transient jerk** (28× the SS-median
  `|f1|`, velocity on-plan) — the F-SAT/δ(q_current) signature (H3b).
  But flipping `use_local_delta_mapping=True` **regresses to a step-0
  dock timeout** (H3a): the principled loop-free mapping cannot be used
  as-is because the spec §3 constrained-dynamic-singularity drift it
  exposes has **no §6 mitigation implemented**.

**Bottom line: an `α_com_soft` sweep is the wrong next action — in every
branch.** Soft-CoM enforces tracking of the NMPC CoM plan, which `v_com`
*already* follows (1.04×); it cannot suppress a high-frequency mapping
jerk, and there is no internal force or momentum-consistency error for it
to correct. The campaign's "10.8× → soft-CoM-off is the root cause"
headline conflates a **smooth NMPC plan vs a single QP jitter spike**.

**Recommended next (await approval — not done here):**
1. **Attack the F-SAT/δ(q_current) jitter at its source**, or implement
   the **spec §6 loop-free mitigation** (condition-number monitor +
   damped null-space) so the principled mapping can replace the band-aid
   without the step-0 regression. This is the lever actually attached to
   the peak.
2. *(optional confirm)* a **frozen-swing-arm** run (swing-only
   `_diag_lock_*` toggle, default-off) to confirm the jitter is
   swing-arm-driven via δ(q). Deferred here: H2 already bounds the
   angular contribution and H3b already identifies the jitter signature,
   so this is corroboration, not a decision input.

The campaign-doc soft-CoM attribution (§6/§7) has been annotated with
these caveats (§7c); `STACK_OVERVIEW.md` §2.6/§5 updated likewise.

---

## Probes 1 & 2 — de-risking a CMM-interface (2026-05-28)

To decide whether to replace the `δ(q)` position-mapping with a
**centroidal-momentum QP task** (`A(q)q̈+Ȧq̇ = ḣ_ref`, tracking the NMPC's
native momentum), two probes were run. Both came back as **honest
negatives that weaken the case for any 1% architecture change**:

- **Probe 1 (reference smoothness, `scripts/probe1_reference_smoothness.py`).**
  The premise — "the QP chases a jittery δ-mapped reference" — is **NOT
  supported**. The *tracked* reference `p_torso_ref` (post-F-SAT) is
  smooth (per-tick |Δ| max 21 mm, max/median **3.4×**); the NMPC's
  `r_com_ref` is the rough one (10 Hz ZOH staircase, 76–205 mm jumps,
  max/median 32×). **At the f1 peak, Δp_torso_ref = 3.5 mm (35th pct) —
  no reference jerk.** F-SAT already delivers a smooth reference, so a
  CMM task would not be tracking a *smoother* signal (both face the same
  10 Hz NMPC staircase).
- **Probe 2 (constrained conditioning, `scripts/probe2_cmm_conditioning.py`).**
  The §3-singularity hypothesis for the spike is **REFUTED**. At the
  f1-peak bracket, `σ_min(stance arm)=0.18–0.20` and
  `σ_min(CoM|constrained)=0.29` — **above the traversal average**, not a
  dip. Worst conditioning is step-1 release (0.056) and the dock configs
  (~0.13), none near-singular, none at the peak. **Consequence:** (i) a
  CMM-CoM task would be well-conditioned throughout the 1% traversal — it
  would *not* inherit a §3 wall; (ii) but the loop-free step-0 failure
  (H3a) is **NOT** explained by §3 conditioning (step-0 σ_min is mid-pack
  0.167) — so the memo's earlier "loop-free blocked by §3" attribution is
  **shaky** and the step-0 timeout likely has another cause (candidate: a
  bug/scaling in `compute_delta_local`, not a genuine singularity).

**Revised bottom line (1% baseline).** The step-2 "10.8× mismatch" is a
**2-tick transient** (t=16.63–16.73; `|f1|>40 N` on **1/508 ticks**,
`τ_joint=20 Nm` on the same **2 ticks**; `qp_ok`/`nmpc_ok` True; median SS
contact force **0.00 N**; `v_com` on-plan throughout) — preceded by a
single **128 ms** NMPC solve (vs ~30 ms). It is a 10 Hz↔100 Hz NMPC↔QP
coordination hiccup, **not** a mapping-jitter, momentum, conditioning, or
authority deficit. **The 1% paper baseline is healthy** (5/5 docks, smooth
tracked references, well-conditioned). A CMM-interface refactor is
**conditioning-safe but weakly motivated**: it is architectural
*defensibility* (replace the F-SAT band-aid — which clips ~49.6% of raw δ
increments — with a principled, citable centroidal coupling), **not a fix
for a measured failure**. It would not remove the 2-tick transient.

## Artifacts

- `scripts/diag_lambda_decomp_step2.py` → `Misc/runs/diag_attribution/lambda_decomp/{metrics.json, lambda_decomp.png}`
- `scripts/diag_loopfree_mapping_step2.py` → `Misc/runs/diag_attribution/loopfree_mapping/{sim_log.json, compare.json}`
- `scripts/probe1_reference_smoothness.py` → `Misc/runs/diag_attribution/probe1_reference_smoothness/{metrics.json, probe1.png}`
- `scripts/probe2_cmm_conditioning.py` → `Misc/runs/diag_attribution/probe2_cmm_conditioning/{metrics.json, probe2.png}`
