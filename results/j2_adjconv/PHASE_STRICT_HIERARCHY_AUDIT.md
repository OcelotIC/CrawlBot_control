# Phase STRICT-HIERARCHY-AUDIT — feasibility & risk of switching the SS WBC to a strict hierarchy

**Branch** `j2/ds-active-rework` · READ + ANALYSIS ONLY, no code change, no run, no canonical touch ·
pushed, never merged. All claims are code-facts (file:line) or reasoning from prior committed measurements
(DOCK-CAUSE, EE-RATIO-GLOBAL, QP-COND).

**Bottom line:** A true strict hierarchy is **NOT a config flag** — it is a control-architecture change
(days of work on untested code + full re-baseline), and — critically — **it would almost certainly NOT
clear the swing-b dock floor**, because that floor coincides with the **hard momentum-rate cap** (Ḣ_s=5.00,
enforced identically in weighted and strict) and arm-reach/Jacobian conditioning, not the weighting. The
weighted hierarchy is a standard, publishable choice. **Recommend: keep weighted; do not switch for this
paper.**

---

## AUDIT 1 — what EXISTS in the code

There are **three** distinct mechanisms; none is "the canonical SS path running a strict hierarchy."

**(a) Generic strict cascade `HierarchicalQP._solve_strict` (`hierarchical_qp.py:297-373`).**
- **Algorithmically complete**: lexicographic cascade — at each priority level solves a QP subject to the
  user inequality constraints + bounds + the *frozen* higher-priority task residuals (`A_i x = A_i x*`,
  :359-370), then freezes this level for the next. Returns failure if any level's QP fails (:344-354).
- **Never used by the WBC.** The WBC builds `HierarchicalQP(method=cfg.method)` (`wholebody_qp.py:449`) and
  `WholeBodyQPConfig.method` defaults to `'weighted'` (`:67`); **no code anywhere sets `method='strict'`**
  (grep: the string only appears inside `_solve_strict`'s own return tag). So the strict path is reachable
  by config but **dead** in practice.
- **ZERO test coverage** (grep `tests/` for `_solve_strict` / `method='strict'` → none). Untested on any
  problem, let alone the 52-var WBC.
- **Same-priority tasks are merged into one weighted block** (`:327-331`) — so distinct integer priorities
  are required to actually separate tasks (see AUDIT 2).

**(b) M2 null-space-projected *weighted* stack (`use_m2_stack`/`ee_null_space`, `wholebody_qp.py:866-951,
1010-1088`).** This is **NOT** the strict cascade — it is the De-Lasa-style null-space-projected weighted
QP: it computes `N_torso = I − A_torso⁺A_torso` (`:879-881`) and projects the EE task into it
(`A_ee_proj = A_ee @ N_torso`, `:945`), then adds it as a **weighted** task (still `α_ee`, `:947`); posture
is projected against `[A_torso; A_ee]` (`:1073-1083`). 
- **Partial task set:** handles torso (P1) + EE (P2, null-space) + posture (P3) + soft-CoM (P4). It has
  **NO momentum task** — the canonical linear-CMM momentum task (`ss_alpha_mom`) lives only in the
  *weighted* `_two_task`/`ss_centroidal_momentum_task` paths and has **never been combined with the
  null-space projection**.
- **Tested** (`tests/test_reworked_qp.py`: T7 torso+EE null-space tracking, soft-CoM RMS, residual, energy
  decay) — but note `test_residual_small` `pytest.skip`s when the QP fails (`:418`), i.e. success is not
  asserted at every step.
- **Dormant in canonical.** The M7 config sets `use_m2_stack=True` (`scripts/run_m7_single_step.py:40`),
  but the canonical run uses `ss_two_task_mode=True`, and the entire M2 block is gated by `not _two_task`
  (`:725,738,895,1010,1037`). With `_two_task` true, `torso_task_active` is False → `A_torso` stays None →
  `N_torso` is never even formed (`:879`). **The null-space machinery does not execute in the canonical
  docking run.**

**(c) Weighted approx of strict via `weight_ratio` (`hierarchical_qp.py:266-267`).** `w_p = weight_ratio^(p−1)`
divides each priority's weight; large `weight_ratio` + distinct priorities ≈ strict. Canonical uses
`weight_ratio=1` (priorities ignored). This route reintroduces weight tuning AND the QP-COND conditioning
blow-up (κ ≈ weight span), so it is not "removing tuning."

**Priority order:** hard-coded as integer `priority=` args in the `add_task` calls (`_two_task`:
hw-slack P1, momentum/torso-pose/EE **all P2**, posture P3, wrench P4, torque P5, accel P6). **Not
configurable** — no priority-order config field. Under the canonical `_two_task` block the three main tasks
share priority 2, so even `method='strict'` would solve momentum+torso+EE as **one weighted block**, not a
strict order.

---

## AUDIT 2 — engineering effort to activate strict for the canonical SS path

**Not flag-only.** Setting `cfg.method='strict'` alone does NOT produce a momentum≻torso≻EE hierarchy — it
would cascade `{hw-slack} ≻ {momentum+torso-pose+EE, still weighted together} ≻ {posture} ≻ …`, because
those three are all priority 2. Specific gaps to fill for a genuine strict SS controller:

1. **Define & assign a distinct priority order** in a new SS strict task block (code): choose and hard-code
   e.g. hw-slack ≻ torso-ang ≻ momentum ≻ EE ≻ torso-lin ≻ posture. This is a *design* decision (which
   task dominates when they conflict) that the weighted sum currently sidesteps.
2. **Wire a route** (`--ss-strict` flag → `cfg.method='strict'` + the new priority block) into
   `_build_qp`/`_step` (code).
3. **Feasibility hardening.** The cascade freezes each level's *full* residual as a hard equality
   (`:359-370`). Freezing a 6-D torso task removes 6 of the 14 free DOF as equalities; adding momentum (3)
   + EE (6) as successive hard equalities on top of the dynamics + contact + **momentum-box** + torque
   bounds risks **infeasibility** where the weighted QP is feasible. `_solve_strict` returns `success=False`
   and `x_opt` = last level's solution on failure — a WBC-facing failure path that has **never run** and is
   not handled downstream. Needs a feasibility/slack strategy per level.
4. **Performance.** Strict solves **one QP per priority level** (≈5–6 QPs/tick) vs one weighted QP. At the
   100 Hz WBC rate with the qpOASES CPUtime=5 ms budget (`hierarchical_qp.py:501`), 6× the solves may blow
   the tick budget — needs profiling.
5. **Test the cascade** from zero (no existing coverage).

**Estimate: DAYS (not flag-only, not hours), on untested code.** Closer to a Chemin-2/3 sub-project than a
switch. The M2 null-space stack (b) is the more-mature partial alternative, but it also needs the momentum
task added under projection + a full re-validation, so it is not a shortcut either.

---

## AUDIT 3 — scientific / behavioural risk

**(i) Re-baseline scope: full.** Strict yields a *different solution* at every tick → **every** canonical
number shifts (h_w, θ_s, all 6 docks, realized Ḣ_s, torso residual, settle) and **every paper figure**
built on the canonical run must be regenerated and re-validated (the C1–C6 gate, the momentum-management
Ḣ_s result, the dock table). Magnitude is unknowable without running — which this phase forbids. This alone
is a large, risky change late in the paper.

**(ii) Does strict clear the swing-b dock floor? — Almost certainly NOT.** Reasoned from committed data:
- The binding step is **step 2 (swing-b @ anchor 4)**, and EE-RATIO-GLOBAL showed its dock is
  **ratio-invariant** (4.904 mm at 8:1 = 4.904 at 4:1, worsening off-baseline). A quantity that does not
  move with the torso:EE weight is **not weight-limited** — so re-arranging the *same* tasks into a strict
  order (which only changes how weight conflicts are resolved) has no lever on it.
- Step 2's realized **|Ḣ_s| = 5.00 — saturated at the τ_w_max = 5 momentum-rate cap** (DOCK-CAUSE / TSTEP).
  The momentum-box is a **hard inequality** (`wholebody_qp.py:558-580`) passed to **every** cascade level
  (`hierarchical_qp.py:338-342`). A strict hierarchy makes the EE task exact **only within the feasible set
  defined by the same constraints** — it **cannot exceed the momentum-rate cap**. On a momentum-saturated
  step, the EE approach is already spending the full momentum budget; promoting EE to top priority unlocks
  no additional budget. → **Same ~4.9 mm wall.**
- The residual↔reach coupling (corr = **−0.68**, DOCK-CAUSE) is an **arm-configuration / Jacobian**
  effect: strict makes EE exact in the null space of higher-priority tasks *subject to the same J_ee*, so a
  rank/conditioning-limited approach direction is **not** cleared by re-prioritization.
- Nuance: on the **non-saturated, tracking-limited** steps (e.g. step 5, Ḣ_s headroom 2.44; the long-reach
  swing-a steps), strict *might* reduce residual by making EE exact — but those are **not** the worst
  dockers. The worst dock (the swing-b, momentum-saturated floor) is the one strict cannot move. So strict
  would **not** deliver all-6 < 4.8 mm; it would hit the same physical wall EE-RATIO-GLOBAL hit.

**(iii) Dormant-bug risk: high.** `_solve_strict` has never executed in this project; the M2 null-space
stack executes in unit tests but is dormant in the canonical closed loop and its failure path `pytest.skip`s
rather than asserts. Activating either on the full 52-var SS problem risks silent infeasibility, tick-budget
overrun, or a wrong-but-plausible solution — exactly the class of bug that is expensive to catch late.

---

## AUDIT 4 — necessary or optional for the paper

**The weighted hierarchy is defensible as published.** Weighted multi-objective / weighted null-space
inverse-dynamics QP is a standard, widely-cited control architecture (De Lasa & Hertzmann 2010; Salini et
al. 2011; Bouyarmane & Kheddar; Feng et al.), as is strict HQP (Kanoun et al. 2011; Escande et al. 2014).
Using weighted-LS here is a legitimate, conventional choice — **there is no reviewer-fatal flaw** in it. The
dock margin (≈4.9 mm vs the 5 mm gate) is a *performance* number, not an architectural defect, and the
momentum-management result (Ḣ_s capped at h_max) is independent of weighted-vs-strict (it is a hard
constraint).

**Framing the decision:**
| | Weighted (current) | Strict hierarchy |
|---|---|---|
| Tuning | weight ratios to set (torso:EE, θ_s collateral) | none (priority order instead) |
| Conditioning | κ ≈ weight span ~3.6e6 (QP-COND; behaviour-safe) | better κ per level, but 5–6 QPs/tick |
| Maturity | canonical, validated, all figures built on it | `_solve_strict` untested; M2 stack partial/dormant |
| Effort | 0 (status quo) | DAYS: priorities + wiring + feasibility + tests |
| Re-baseline | none | **full** (every figure + gate) |
| Fixes the dock? | no (swing-b floor) | **also no** (momentum-cap / Jacobian floor, AUDIT 3) |
| Publishability | standard, defensible | cleaner narrative ("no weights") |

Because strict is a control-architecture change that (a) requires a full re-baseline, (b) runs on untested
code, and (c) — per AUDIT 3 — **does not actually fix the dock floor it would be motivated by**, it is
**OPTIONAL and not worthwhile for this paper**. It is a Chemin-2/3 refactor whose main payoff (removing
weight tuning) is a narrative nicety, not a result-enabling change.

---

## Deliverable (STOP-GATE)
1. **What exists:** strict cascade `_solve_strict` = **complete but unused + untested** (WBC never sets
   `method='strict'`); M2 null-space stack = **tested but partial (no momentum task) + dormant** in
   canonical (`not _two_task` gate); priority order **hard-coded**, and the three main tasks share P2 so
   `method='strict'` alone would not separate them.
2. **Effort: DAYS (not flag/hours)** — gaps: assign distinct priorities (design), wire a route, harden
   cascade feasibility (freezing full residuals over the hard momentum box risks infeasibility), profile
   5–6 QPs/tick vs the 5 ms budget, test from zero.
3. **Risk:** full re-baseline of every canonical number + figure; **strict would NOT clear the swing-b floor**
   (it is momentum-rate-cap-saturated Ḣ_s=5.00 + reach/Jacobian-limited, both identical under strict);
   high dormant-bug risk on never-run code.
4. **Necessary vs optional:** weighted is a standard, publishable, defensible architecture with no
   reviewer-fatal flaw → strict is **OPTIONAL**. Given it would not fix the dock and costs a full
   re-baseline on untested code, **recommend NOT switching for this paper.**

NO code change, NO run, `crawlbot/` untouched. **STOP for decision.**
