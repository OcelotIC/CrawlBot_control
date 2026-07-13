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
