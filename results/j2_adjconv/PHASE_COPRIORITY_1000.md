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
