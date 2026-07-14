# Phase COPRIORITY-1000-FINAL — Idriss's 1:1:1 design is INFEASIBLE (step-0 dock timeout)

**Branch** `j2/ds-active-rework` · measurement, NO canonical commit · pushed, never merged.
Data: `results/j2_adjconv/copri1000_result.json`; script `scripts/diag_copriority1000_run.py` (full vector
forced via monkeypatch + κ capture). Raw run `figC_copri1000` (gitignored).

**Verdict:** the co-priority 1:1:1-at-1000 design **times out on step 0** (min d 6.9 mm, never < 5 mm) →
**infeasible**. Both predicted risks materialized: the **torso-floor timeout** (torso 1000 ≪ attitude-hold
floor) AND **momentum de-saturation** (5000→1000 dropped realized SS Ḣ_s to 2.3). The conditioning goal is
met handsomely (κ 1e4, 360× better), but the controller does not dock. Per instruction: reported, **not
patched**.

## Weight vector as applied (confirmed exactly)
| task | weight | | ratio |
|---|---|---|---|
| hw-slack | 10 000 | torso : momentum : EE | **1 : 1 : 1** (co-priority) |
| torso-pose | 1 000 | hw-slack : main | 10 : 1 |
| momentum | 1 000 | main : posture | 50 : 1 |
| swing-EE | 1 000 | main : regularizer-floor | 1 000 : 1 |
| posture | 20 | posture : floor | 20 : 1 |
| torque-min | 1 | **span** | 10 000 / 1 = **1e4** |
| wrench-track | 1 | | |
| accel-reg | 1 | | |
| Tikhonov ε | 1e-6 | | |

## Measurements
| # | metric | result | pass/fail |
|---|---|---|---|
| 1 | **FEASIBILITY** | **DOCK_TIMEOUT step 0** (min d 6.87 mm, never <5 mm), traversal aborts. 0/6 docks. nmpc_fail=10, qp_fail=0. | **FAIL (a)** |
| 2 | **SS-swing saturation** | **GONE** — realized SS Ḣ_s (step 0) = **2.31** (steps 1–5 never ran). Planned Ḣ_s = 5.00 (NMPC still plans to saturate) but the WBC realizes only 2.31. Momentum 1000 de-saturates the swing (like userw2's 400; 5000 saturates). | **FAIL (b)** |
| 3 | **at-weld docks ×6** | none (timeout); worst/margin N/A | **FAIL (d)** |
| 4 | **e_com peak** | 0.1034 (step 0 only) — loosened from canonical 0.095 toward userw2 0.137, consistent with momentum ↓ to 1000 | ~ (partial) |
| 5 | **θ_s / Ḣ_s** | θ_s pk 0.167 / settled 0.068 (abort artifacts — only partial step 0). realized 2.31 vs planned 5.00 (WBC under-realizes the plan on the doomed step). | — |
| 6 | **κ_SS** | **1.000e4** — exactly the span; λ_min(H_LS) = **1.0** (from accel-reg/wrench = 1). **360× below canonical 3.6e6.** | **PASS (e)** |
| 7 | **solver / h_w** | qp_fail 0, nmpc_fail 10 (NMPC struggling on the failed step 0); h_w peak 2.31 Nms (< ±5 box; hw-slack 10000 not stressed since only step 0 ran) | — |

## Pass/fail summary
- (a) feasible 6/6 — **FAIL** (timeout step 0)
- (b) swing-saturation intact — **FAIL** (gone; momentum 1000 de-saturated)
- (c) CoM tight — partial (0.103, slightly loose; N/A under abort)
- (d) dock < 5 mm at-weld with margin — **FAIL** (no docks)
- (e) κ materially below canonical — **PASS** (1e4 vs 3.6e6, 360×)

## Which predicted risk materialized, and the likely-responsible weight
**Both.** (1) The **torso-floor timeout** fired — step 0 never closed below 6.9 mm. (2) The **momentum
de-saturation** fired — realized SS Ḣ_s 2.31, not ~5.

**Likely-responsible weight: torso must OUT-weight momentum for the docked-arm attitude hold, and here it
does not.** The discriminating evidence across the arc:
- userw2 #1 (**torso 1000**, momentum **400** → torso:mom = 2.5:1) **docked 6/6**.
- userw2 #2 (torso 2000, momentum 400 → 5:1) **docked with margin**.
- this run (**torso 1000, momentum 1000 → 1:1**) **times out**.
- HIER-SWEEP (momentum 5000) put the torso floor at ~12000 (torso:mom ≥ 2.4:1 to dock).

So the failure is the **torso : momentum ratio at 1:1** — at co-equal weight the momentum task steals the
attitude DOFs the docked stance arm needs, and the torso can no longer hold, so the swing never converges
(step-0 timeout). The **fix (NOT applied): raise torso above momentum** (userw2 #2's torso 2000 docked) or
keep momentum < torso. Co-priority *including momentum* (torso = momentum) is the specific thing that breaks;
torso = EE co-priority is fine (HIER-SWEEP reached 1:1 torso:EE) as long as **torso > momentum**.

## Note — the conditioning lever DID work
κ collapsed to **1.00e4** (from 3.6e6) because raising the regularizer floor (accel-reg/wrench 0.01 → 1)
lifted λ_min(H_LS) from 0.01 to **1.0**, and the span (hw-slack 10000 / floor 1) directly sets κ = 1e4.
This confirms QP-COND (κ = span) and shows **the floor-raise is the correct, effective κ lever** — worth
keeping in any *feasible* redesign (it cost nothing here except that the design was infeasible for the
unrelated torso:momentum reason).

## Deliverable (STOP-GATE)
Full vector + span 1e4 + ratios (above, 1:1:1 torso:mom:EE); **feasibility = FAIL, DOCK_TIMEOUT step 0**
(min d 6.87 mm); SS-saturation **GONE** (realized 2.31, planned 5.00); at-weld docks **none**; e_com 0.103;
θ_s pk 0.167 (abort); **κ_SS 1.00e4** (λ_min 1.0); h_w peak 2.31; qp_fail 0 / nmpc_fail 10. Risks
materialized: **both** (torso-floor timeout + momentum de-saturation); responsible: **torso : momentum = 1:1**
(torso must out-weight momentum). NOT patched. `crawlbot/` untouched. **STOP for cross-check.**

---

## Addendum — momentum 2000 variant (torso 1000, EE 1000, momentum 2000 → torso:mom = 1:2)
Data: `results/j2_adjconv/copri_m2000_result.json`. Raised momentum ABOVE torso to discriminate two
hypotheses for the 1:1 timeout: **H1** — the torso:momentum ratio (raising momentum should worsen it);
**H2** — the *low absolute* momentum / de-saturation (raising momentum should restore swing authority and dock).

**Result: near-IDENTICAL timeout — momentum is not the lever.**
| metric | mom 1000 (1:1) | mom 2000 (1:2) |
|---|---|---|
| feasibility | TIMEOUT step 0, min d **6.87** mm | TIMEOUT step 0, min d **6.86** mm |
| SS Ḣ_s realized (step 0) | 2.31 | 2.33 |
| e_com pk | 0.1034 | 0.1034 |
| θ_s pk / settled | 0.167 / 0.068 | 0.167 / 0.068 |
| κ_SS | 1.00e4 | 1.00e4 |
| h_w peak | 2.31 | 2.30 |
| nmpc_fail | 10 | 10 |

- **H2 DISPROVED:** raising momentum 1000→2000 does **nothing** — the step-0 failure is **insensitive to the
  momentum weight** (bit-for-bit the same de-saturated swing, min d 6.86 vs 6.87). The timeout is not the low
  absolute momentum.
- **H1 refined & consistent:** torso 1000 is below the attitude-hold threshold, and once past it the failure
  is momentum-independent (mom 1000 = 2000). The docking runs had torso ≥ 2000 (userw2 #2) **or** momentum
  ≤ 400 with torso 1000 (userw2 #1); at torso 1000 with momentum ≥ 1000 it fails.
- **The lever is TORSO, not momentum.** Caveat: this run does not isolate torso from the other
  copri-stack weights that also differ from the docking userw2 configs (hw-slack 10000 vs 800, torque 1
  vs 5, ε 1e-6 vs 1e-4). The clean isolation test (NOT run — awaiting direction): **torso 2000 @ momentum
  1000, EE 1000** on this stack — if it docks, torso is confirmed the lever; if it still times out, the
  suspect shifts to hw-slack 10000 / the floor-raise interaction. NOT patched. **STOP for cross-check.**

---

## Addendum 2 — torso 2000 isolation (torso 2000, momentum 1000, EE 1000 → 2:1:1): torso is ALSO ruled out
Data: `results/j2_adjconv/copri_t2000_result.json`. Raised torso above momentum to test the "torso is the
lever" reading from Addendum 1.

**Result: STILL TIMES OUT — torso is not the lever either.** Three copri-stack runs, all near-identical:
| copri-stack run | torso:mom | feasibility | SS Ḣ_s | min d @ abort |
|---|---|---|---|---|
| torso 1000, mom 1000 | 1:1 | TIMEOUT step 0 | 2.31 | 6.87 mm |
| torso 1000, mom 2000 | 1:2 | TIMEOUT step 0 | 2.33 | 6.86 mm |
| **torso 2000, mom 1000** | **2:1** | **TIMEOUT step 0** | 2.51 | **6.87 mm** |

(e_com ~0.10, θ_s ~0.167, κ 1.00e4, h_w ~2.4, nmpc_fail 10 — all three essentially identical.)

**Honest correction: both my attributions were wrong.** The copri-stack failure is **insensitive to BOTH
torso and momentum** across 1000↔2000 — raising either does nothing (Addendum 1's "torso must out-weight
momentum" is disproved by the 2:1 run timing out the same as 1:1 and 1:2). The lever is **not** the
torso:momentum weights.

**Where the lever actually is.** userw2 #2 **docks** with the SAME torso 2000 / EE 1000 but differs from
this failing copri stack in exactly four weights:
| weight | userw2 #2 (docks) | copri (fails) |
|---|---|---|
| momentum | **400** | 1000 |
| hw-slack | **800** | **10000** |
| torque-min | 5 | 1 |
| ε | 1e-4 | 1e-6 |

Since torso/EE are held identical (2000/1000) and one docks while the other times out, the lever is among
**{momentum≤~400 threshold, hw-slack 800 vs 10000, torque, ε}** — NOT torso, NOT momentum-in-[1000,2000].
The strongest single suspect is **hw-slack = 10000** (12.5× userw2's 800, and the *largest* weight in the
copri stack — it can dominate the QP), or the **momentum threshold** (userw2 docks at 400; every copri
failure used ≥1000). **Clean discriminator (NOT run — awaiting direction): copri stack @ momentum 400**
(torso 2000, mom 400, EE 1000, hw 10000, floor 1, ε 1e-6) — if it docks, momentum-must-be-low (~400) is the
lever; if it still times out, the culprit is **hw-slack 10000** (or torque/ε). NOT patched. **STOP.**

---

## Addendum 3 — momentum 400 discriminator (torso 2000, mom 400, EE 1000, hw 10000): momentum ruled out too → hw-slack is the suspect
Data: `results/j2_adjconv/copri_m400_result.json`. Dropped momentum to userw2's 400 (holding the copri
stack) to test the momentum-threshold vs hw-slack.

**Result: STILL TIMES OUT** — step 0, min d **6.87 mm**, realized SS Ḣ_s 2.50, all metrics identical to the
other three copri runs. **Momentum-must-be-low (~400) is NOT the lever.** Four copri-stack runs now, every
one timing out at min d ~6.87, **insensitive to torso (1000/2000) AND momentum (400/1000/2000)**.

**By elimination, the culprit is in {hw-slack, torque, ε}.** This run (fails) vs userw2 #2 (**docks**) share
torso 2000 / mom 400 / EE 1000 / wrench 1 / accel-reg 1 / posture 20 exactly, and differ ONLY in:
| weight | userw2 #2 (docks) | copri m400 (fails) |
|---|---|---|
| hw-slack | **800** | **10000** |
| torque-min | 5 | 1 |
| ε | 1e-4 | 1e-6 |
ε is inert at floor=1 (REG-DIAG: ε ≪ λ_min=1 → no effect); torque 1↔5 is a weak τ-regularizer. **⇒ hw-slack
10000 is the prime suspect.**

**Mechanism (coherent with ALL data):** hw-slack (momentum-box slack penalty, priority 1) is the **largest
weight in the copri stack (10000 > torso 2000)** → it **dominates** the QP. In *every docking* config hw-slack
sits **below** the main tasks: userw2 #2 (800 < torso 2000), canonical (10000 < torso 24000). So hw-slack
10000 is not intrinsically bad (canonical uses it) — it breaks docking specifically when it **out-weights the
tracking tasks** (the low-weight copri stack). **Decisive test (NOT run — awaiting direction): hw-slack → 800**
on the copri stack (torso 2000, mom 400, EE 1000, hw **800**, floor 1, ε 1e-6) — if it docks, hw-slack-must-be-
below-the-main-tasks is confirmed, and the feasible + κ-good recipe is "keep the floor-raise (κ 1e4) but keep
hw-slack under the tracking weights." NOT patched. **STOP.**

---

## Addendum 4 — hw-slack 800 test: hw-slack ruled out too → the difference is now just {torque, ε}
Data: `results/j2_adjconv/copri_hw800_result.json`. Dropped hw-slack to userw2's 800.

**Result: STILL TIMES OUT** — step 0, min d **6.87 mm**, all metrics identical (κ dropped to 6.78e3, so the
hw-slack override *did* take effect — harness confirmed working). **hw-slack is NOT the culprit.** Five
copri-stack runs now, every one timing out, insensitive to torso, momentum, AND hw-slack.

**The failing copri hw800 now differs from the DOCKING userw2 #2 in EXACTLY TWO weights** (verified against
`userweights_result2.json`; harnesses structurally identical — both apply the weight vector via
`WholeBodyQP.__init__` and ε via `HierarchicalQP._solve_weighted`):
| weight | userw2 #2 (docks 6/6, worst 4.59) | copri hw800 (TIMEOUT) |
|---|---|---|
| **torque-min** | **5** | **1** |
| **ε** | **1e-4** | **1e-6** |
(identical: torso 2000, hw-slack 800, momentum 400, EE 1000, posture 20, wrench 1, accel-reg 1)

**ε is inert here** — λ_min(H_LS) = 1.0 (floor-raise), so ε 1e-4 and 1e-6 are both ≥ 4 orders below λ_min ⇒
H is ~identical (REG-DIAG: ε ≪ λ_min → no effect). **⇒ the last suspect is `alpha_torque` (1 vs 5)** — a 5×
change in the joint-torque regularizer is the only non-inert difference left between docking and timeout.
(Counter-intuitive — a minor τ-regularizer as the dock/timeout switch — so it must be *tested*, not asserted;
prior confident attributions were wrong 3×.) **Decisive test (NOT run — awaiting direction): copri hw800 +
torque 5** (torso 2000, hw 800, mom 400, EE 1000, torque **5**, floor 1, ε 1e-6). If it docks, `alpha_torque`
is the switch AND we have a **feasible + well-conditioned** recipe (κ 6.78e3, docks); if it still times out,
the effect is the ε 1e-4 vs 1e-6 (surprising, would contradict the λ_min=1 inertness) — reproduce userw2 #2
exactly to close. NOT patched. **STOP.**

---

## Addendum 5 — SOLVED: `alpha_torque` (joint-torque regularizer) is the switch
Data: `results/j2_adjconv/copri_tq5_result.json`. Ran the copri stack with **only** `alpha_torque` 1 → 5.

**Result: DOCKS 6/6.** torque 1 → step-0 TIMEOUT; **torque 5 → 6/6 dock** — nothing else changed. So after
five red-herring runs (torso, momentum, hw-slack all ruled out), the single weight that flips
timeout → dock is the **lowest-priority joint-torque regulariser `alpha_torque`**.

| metric | torque 1 (fails) | **torque 5 (docks)** |
|---|---|---|
| feasibility | TIMEOUT step 0 | **6/6 dock** |
| at-weld docks [mm] | none | 2.56 4.59 **4.89** 4.39 2.49 4.49 → worst **4.89**, margin **0.11** |
| SS-saturation | gone | gone (realized ≤2.5) |
| θ_s pk / settled | 0.167 (abort) | 0.432 / 0.424 |
| e_com pk | 0.10 (abort) | 0.137 |
| κ_SS | 6.78e3 | **7.61e3** (530× below canonical 3.6e6) |
| h_w peak | 2.47 (abort) | 4.12 (< ±5 box) |

- **Reproduces userw2 #2 to 0.01 mm** (at-weld `2.56/4.59/4.89/4.39/2.49/4.49` vs userw2's
  `2.57/4.59/4.89/4.39/2.49/4.49`; only step 0 differs by 0.01 mm) — with ε 1e-6 here vs userw2's 1e-4. **⇒ ε
  is confirmed inert** (as REG-DIAG predicted at λ_min=1) and `alpha_torque` is the sole switch.
- **NMPC-fail caveat defused:** the 66 % `nmpc_fail` is **shared** with userw2 #2 (identical 1232/1860) and
  canonical is 51 % — it is dominated by **intentional DS-phase NMPC bypass** (`nmpc_ok=False` = "not run; not
  a failure", `sim_loop.py:1114`), not solver failures. Pre-existing, not introduced here.

### Mechanism / design rule (defensible)
The regulariser tier is {torque, wrench, accel-reg}. Canonical: torque **1** vs floor **0.01** ⇒ torque is
**100×** the acceleration/wrench floor. The copri stack raised the floor to **1** (for conditioning, κ 1e4),
which dropped torque:floor to **1:1** (torque 1) — the torque-min lost its dominance over the co-equal
accel-reg, the redundant-DOF resolution degraded, and the swing never closed (step-0 timeout). Raising torque
to **5** restores torque:floor to **5:1** — enough to dock. **Rule: when you raise the regulariser floor for
conditioning, raise `alpha_torque` with it (keep it ≳5× the floor).**

### The payoff — a feasible + well-conditioned recipe
`torso 2000, hw-slack 800, momentum 400, EE 1000, posture 20, torque 5, wrench 1, accel-reg 1, ε 1e-6`
**docks 6/6** (weld-worst 4.89 mm, margin 0.11) with **κ_SS 7.6e3 — 530× better than canonical 3.6e6** (the
floor-raise conditioning win, preserved). This is essentially userw2 #2 (ε inert). Trades: SS-saturation gone
(momentum 400), e_com loosened (0.137 vs 0.095), θ_s 0.43 (peak better, settled worse), slower.

**Honest note:** my COPRIORITY attributions were wrong on torso, momentum, AND hw-slack before disciplined
elimination landed on `alpha_torque`. Lesson banked: test each weight, don't assert.

**Open (NOT run):** does the *true* co-priority 1:1:1 (torso=mom=EE=1000) dock with torque 5? Only the torso
2000 / mom 400 operating point was proven; the 1:1:1 point may still hit the momentum-co-priority issue.
NOT patched. `crawlbot/` untouched. **STOP for cross-check.**

---

## Addendum 6 — low co-priority (torso=EE=300): docks 2/6, dies at step 2 on **EE authority** (not torso)
Data: `results/j2_adjconv/copri_h1000m500t300_result.json`. Idriss's vector: **hw-slack 1000, momentum 500,
torso 300, EE 300** (torso:EE = **1:1** true co-priority, low absolute), carrying torque 5 / wrench 1 /
accel-reg 1 / posture 20 / ε 1e-6 forward from the Addendum-5 docking config. Span 1000. Raw run
`figC_copri_h1000m500t300` (gitignored).

**Verdict: does NOT complete — docks steps 0–1, then TIMEOUT at step 2** (min d **8.50 mm**, never < 5 mm).

| step | outcome | d [mm] | ori [°] | EE err [mm] |
|---|---|---|---|---|
| 0 | **DOCK** | 4.59 | 0.38 | 27.4 |
| 1 | **DOCK** | 4.59 | 0.17 | 33.9 |
| 2 | **DOCK_TIMEOUT** | **8.50** | 0.11 | **46.1** |

| metric | value | vs Addendum-5 docking config |
|---|---|---|
| feasibility | **2/6** (timeout step 2) | 6/6 dock |
| at-weld docks [mm] | 4.59, 4.59, — | 2.56 4.59 4.89 4.39 2.49 4.49 |
| κ_SS | **2.11e3** (best measured) | 7.61e3 |
| SS-saturation | gone (realized ≤ 2.38) | gone |
| θ_s pk / settled | 0.472 / 0.176 | 0.432 / 0.424 |
| e_com pk | 0.154 | 0.137 |
| h_w peak | 3.31 (< ±5) | 4.12 |
| qp_fail | 0 | 0 |

### Root cause — EE authority, NOT torso pose
- **torso 300 is fine.** Steps 0–1 docked with orientation error **0.38° / 0.17°** (gate is 5°) — a 300-weight
  torso task held the pose with 13× margin. So torso can floor far lower than the ≥1000 used in every prior
  docking config; **300 is not the binding constraint here.**
- **EE 300 is the killer.** The swing-EE tracking error **grows** across steps (27.4 → 33.9 → **46.1 mm**) and
  step 2 (arm-a, the hardest reach) stalls at **8.50 mm > 5 mm gate**. Dropping EE 1000 → 300 removed the
  authority the swing arm needs to close the last few mm — exactly the mechanism **Phase DOCK-CAUSE** isolated
  (dock outcome = 100 % WBC EE-tracking residual). **EE is the dock lever; it needs ≳ 1000.**

### What this pins down
- **Co-priority direction confirmed asymmetric.** torso:EE = 1:1 is fine *if* the common level is high enough
  for EE — at 1000/1000 (Addendum 1) it timed out on momentum; at 300/300 it times out on EE. The floor is set
  by **EE**, not torso: torso 300 docks-2, EE 300 does not dock-6.
- **Conditioning is now excellent and free** — κ_SS 2.11e3 (1700× below canonical 3.6e6), the best of the whole
  sweep, purely from the span-1000 vector. But conditioning ≠ docking: this is the best-conditioned *and*
  a non-docking config, a clean reminder that κ and dock precision are independent axes (as QP-COND showed).
- **torque 5 was not enough to rescue it** — the Addendum-5 rule (torque ≳ 5× floor) is necessary but not
  sufficient; it fixes the redundancy-resolution timeout, not the EE-reach deficit. Here torque:floor = 5:1
  (correct) yet EE still starved.

### Recipe standing (unchanged best): Addendum-5 config
`torso 2000, hw-slack 800, momentum 400, EE 1000, posture 20, torque 5, wrench 1, accel-reg 1, ε 1e-6` remains
the only **feasible + well-conditioned** vector (6/6 dock, κ 7.6e3). If the goal is lower absolute weights /
better κ while keeping the dock, **raise EE back toward 1000** (torso may stay ≤ 300 — it has margin); e.g.
torso 300 / EE 1000 / mom 500 / hw 1000 would test whether the torso-floor really is that low with EE restored.

`crawlbot/` untouched. Measurement only. **STOP.**

---

## Addendum 7 — EE-restore (torso 300, **EE 1000**): docks 4/6, step-4 **0.02 mm near-miss**
Data: `results/j2_adjconv/copri_t300e1000_result.json`. Raised swing-EE **300 → 1000** (the dock lever from
Addendum 6), kept torso at the low **300**, momentum 500, hw-slack 1000; torque 5 / wrench 1 / accel-reg 1 /
posture 20 / ε 1e-6. Span 1000. Raw run `figC_copri_t300e1000` (gitignored).

**Verdict: docks 0–3, then step-4 TIMEOUT at min d = 5.02 mm — misses the 5 mm gate by 0.02 mm.** Restoring EE
did exactly what Addendum 6 predicted: the gross reach deficit vanished (step 2: **8.50 mm timeout → 4.92 mm
dock**) and the failure moved to a razor-thin near-miss at the end.

| step | outcome | d [mm] | ori [°] | EE err [mm] | (Add-6, EE 300) |
|---|---|---|---|---|---|
| 0 | **DOCK** | 3.50 | 0.20 | 42.5 | 4.59 |
| 1 | **DOCK** | 4.73 | 0.06 | 22.0 | 4.59 |
| 2 | **DOCK** | 4.92 | 0.04 | 49.1 | **timeout 8.50** |
| 3 | **DOCK** | 4.97 | 0.08 | 20.5 | — |
| 4 | **DOCK_TIMEOUT** | **5.02** | 0.03 | 33.5 | — |

| metric | Add-7 (torso 300, EE 1000) | Add-5 dock (torso 2000, EE 1000) | Add-6 (torso 300, EE 300) |
|---|---|---|---|
| feasibility | **4/6** (step-4 by 0.02 mm) | 6/6 | 2/6 (step-2 by 3.5 mm) |
| at-weld docks [mm] | 3.50 4.73 4.92 4.97 — | 2.56 4.59 4.89 4.39 2.49 4.49 | 4.59 4.59 — |
| SS-saturation | **PARTIAL** (Ḣ_s pk 3.61) | gone (≤ 2.5) | gone |
| θ_s settled | **0.083** (excellent) | 0.424 | 0.176 |
| e_com pk | **0.164** (highest yet) | 0.137 | 0.154 |
| κ_SS | 6.33e3 | 7.61e3 | 2.11e3 |
| h_w peak | 3.41 | 4.12 | 3.31 |

### Reading — torso 300 is NOT the residual; momentum 500 is the suspect
Two things changed from the Add-5 6/6-dock config: torso 2000 → 300 **and** momentum 400 → 500. The data
points away from torso and toward momentum:
- **torso 300 held attitude beautifully** — θ_s settled **0.083**, the best of the whole sweep (5× tighter than
  Add-5's 0.424), and every docked step had ori ≤ 0.20°. A 300-weight torso task is clearly not starved. So
  lowering torso 2000 → 300 is **not** what cost the last 0.02 mm.
- **momentum 500 reintroduced partial saturation** — realized Ḣ_s peaks at **3.61** (step 2), vs Add-5's
  fully-desaturated ≤ 2.5. This is the exact **USERW2 lever**: momentum-task weight drives SS saturation, and
  saturation competes with the swing reach. It also shows up as the **highest e_com of the sweep (0.164)** and
  as docks that **creep monotonically toward the gate** (3.50 → 4.73 → 4.92 → 4.97 → 5.02) — a secular
  CoM/momentum drift accumulating across steps, not a per-step pose error.

### Decisive next test (not run — awaiting direction)
**torso 300 / EE 1000 / momentum 400 / hw 1000** — drop momentum to the Add-5 value that docked 6/6 with
saturation GONE, holding torso at 300 and EE at 1000. If it docks 6/6, we've isolated momentum as the last
0.02 mm *and* obtained a **low-torso (300) recipe** — lighter and comparably conditioned to Add-5. Caveat: the
two-variable change here means only the mom→400 run fully isolates it, but the θ_s=0.083 evidence already makes
torso the unlikely culprit. `crawlbot/` untouched. Measurement only. **STOP.**

---

## Addendum 8 — REFUTED (momentum) → ISOLATED (torso). mom 500→400 changes nothing; **torso is the lever**
Data: `results/j2_adjconv/copri_t300e1000m400_result.json`. Single-variable test of Addendum 7's momentum
hypothesis: drop momentum **500 → 400** (Add-5's desaturating value), holding torso 300 / EE 1000 / hw 1000.

**Result: momentum hypothesis REFUTED.** Still **4/6**, step 4 still misses at **5.02 mm**, realized Ḣ_s
**unchanged at 3.60** (was 3.61 at mom 500). Dropping the momentum weight did **not** desaturate the swing and
did **not** recover the 0.02 mm. My Add-7 reading ("momentum 500 is the suspect") was wrong — the same
assert-before-testing error the COPRIORITY hunt has punished repeatedly. The disciplined single-variable run
corrected it.

### The isolation — three runs, one lever
| config | docks | Ḣ_s pk | sat | e_com | θ_s settled | step-4 |
|---|---|---|---|---|---|---|
| Add-5  torso **2000** EE1000 mom400 (hw800) | **6/6** | 2.50 | GONE | 0.137 | 0.424 | 2.49 dock |
| Add-7  torso 300 EE1000 mom **500** (hw1000) | 4/6 | 3.61 | PARTIAL | 0.164 | 0.083 | **MISS 5.02** |
| **This** torso 300 EE1000 mom **400** (hw1000) | 4/6 | 3.60 | PARTIAL | 0.164 | 0.084 | **MISS 5.02** |

- **mom 500 vs 400 @ torso 300** (rows 2 vs 3): identical outcome — momentum weight is **inert** here.
- **torso 2000 vs 300 @ mom 400, EE 1000** (rows 1 vs 3): the *only* effective difference (hw-slack 800 vs
  1000 is inert — priority-1 at weight_ratio=1, box inactive at h_w 3.4 < 5). It flips **6/6 → 4/6**, drives
  Ḣ_s **2.5 → 3.6**, and tips step 4 over the gate. **⇒ torso is the dock lever at EE 1000.**

### Mechanism — transient vs settled (why low torso hurts the dock but not the final attitude)
A subtlety: torso 300 gives a **better settled** attitude (θ_s 0.084 vs torso-2000's 0.424) yet **docks worse**.
Resolution: a strong torso task damps the **base/momentum transient during the swing** (holds the base firm →
realized Ḣ_s stays ≤ 2.5), which is what lets the swing EE close the last mm. A weak torso task still converges
to a level attitude *at settle*, but during the swing the base moves more (Ḣ_s 3.6, e_com 0.164), and the dock
gate is set by the **swing transient**, not the settled pose. So torso authority buys dock precision through
the transient — invisible in the settled θ_s.

### Corrected picture & next test
The dock lever at EE 1000 is **torso**, with a 6/6 floor somewhere in **(300, 2000]**: torso 2000 docks 6/6
(Add-5), torso 300 misses step 4 by 0.02 mm. Momentum 400↔500 and hw-slack 800↔1000 are both inert.
**Next: bisect the torso floor — torso 1000 / EE 1000 / mom 400 / hw 1000.** If 6/6, the floor is ≤ 1000 (a
lighter recipe than Add-5's 2000) and we push lower; if it misses, the floor is in (1000, 2000]. `crawlbot/`
untouched. Measurement only. **STOP.**
