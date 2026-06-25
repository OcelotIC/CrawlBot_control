# Phase 2.1 (reformulated) — two-task integration, single-step swing

Branch `feat/ss-centroidal-momentum-task`. Construction commit `d526bf0` (two-task stack,
default OFF). Driver `diag_cooperative_arms.py --ss-two-task --n-steps 1` (canonical step 1,
real NMPC + MuJoCo). Analysis `scripts/report_phase2_1_twotask.py`. h_w in ∞-norm; τ_w-sat at
100 Hz; metrics.py untouched.

## Verdict

The two-task stack (T-MOM + 6-D torso-pose, fully weighted, raw-quintic reference, no
δ-mapping) **works and brings the torso to its planned pose**. Torso arrival is a **tunable
cursor, NOT a structural wall** — the three tasks co-realise on 20 DOF. The operating window is
bounded by **EE-yield** (over-weighting momentum → swing under-docks) and **τ_w-envelope
stress** (over-weighting torso-pose → wheels saturate); **balanced ss_alpha_mom:alpha_torso_pose
≈ 5000:5000 is the working point** (docks 4.95 mm, torso 14 mm = 11 % of travel, envelope kept).
F-SAT is genuinely OFF (0 calls). Test 6 (hierarchy) PASSES at the intended weighting.

## Step A — 14 mm judged against the torso's OWN travel (the gate)

Geometric torso displacement over SS, `p_t1 − p_t0` (FK-torso @ q_start → @ q_end) =
`[113.0, 54.2, 7.4] mm`, **|disp| = 125.5 mm** (mostly +x forward). Arrival error as a fraction:

| run | torso arrival vs geometric p_t1 | % of the 125.5 mm travel |
|---|---|---|
| baseline (δ-mapping) | 127.1 mm | **101 %** — torso barely moves to the IK target, yet docks |
| T-MOM-only (v1) | 99.6 mm | 79 % |
| **two-task balanced** | **14.3 mm** | **11 %** |
| two-task torso-pose-dom | 4.2 mm | 3 % |

**Reframe (corrects the guidance-audit's "baseline on-target 1.69 mm"):** that 1.69 mm was vs the
*mapping* reference, which is itself ~125 mm off the *geometric* IK target. Against the geometric
target the δ-mapping never moves the torso there (101 %); **the two-task (raw quintic) brings it
89 % of the way (11 % residual)**. So 14 mm is a small fraction of a large move → good.

## Sweep — cursor vs wall (ratio = torso-pose / momentum weight)

| metric | baseline | TMOM-only | M-dom 30k:0.5k | **bal 5k:5k** | TP-dom 0.5k:30k |
|---|---|---|---|---|---|
| docked | ✓ 1.85 | ✓ 2.79 | ✗ **6.99 (timeout)** | ✓ **4.95** | ✓ 4.97 |
| torso arrival vs p_t1 [mm] | 127 | 100 | 5.5* | **14.3** | 4.2 |
| torso-pose track pk [mm] | 10.3 | 10.4 | 85.4 | 22.2 | 5.2 |
| torso-ori track pk [deg] | 0.98 | 0.76 | 0.73 | 0.05 | 0.03 |
| CoM(momentum) track pk [mm] | 131 | 127 | 108 | 92 | 83 |
| EE dock min [mm] | 0 | 0 | 0 | 0 | 0 |
| **τ_w-sat @100 Hz [%]** | 0 | 0 | 0 | **1.3** | **23.8** |
| joint-τ peak [N·m] (±20) | 3.3 | 3.2 | 4.9 | 8.2 | 9.7 |
| h_w peak ∞-norm | 0.83 | 1.41 | 1.79 | 2.22 | 2.25 |
| QP inner-loop p50/p99 [ms] | 82/90 | 79/92 | 77/89 | 77/82 | 85/105 |

(*M-dom arrival is at the timeout instant, not a clean dock.)

**It is a CURSOR, not a wall.** Toward torso-pose-dominant the arrival error **falls** (14.3 → 4.2
mm) and torso-pose tracking tightens (22 → 5 mm) — the 6+6+6 tasks DO co-realise; there is no
18-on-20 over-constraint pinning the torso. **But the cost transfers to the τ_w envelope:**
τ_w-sat climbs 1.3 → 23.8 % and joint-τ 8.2 → 9.7 N·m as torso-pose is pushed. The opposite bound
appears under momentum-dominance: **M-dom under-docks (6.99 mm) because the over-strong momentum
(30k) starves the fixed-weight swing-EE — the EE yields** (torso ori is a perfect 0.005°, the EE
is what misses). So the operating window is bounded by **EE-yield (momentum too high)** and
**envelope-stress (torso-pose too high)**, with **balanced 5k:5k** in the middle: docks 4.95 mm,
14 mm arrival, τ_w-sat only 1.3 %, joint-τ 8.2 N·m.

## Two-regime frame (under-exercised here, as predicted — deferred to Phase 3)

Planned ‖Ḣ_s‖∞ peaks 0.6–2.3 N·m < 5 and `envelope_binding_pct = 0 %` for ALL runs — the **NMPC
plan never binds the envelope** on this single step, so we are in the **consistent regime**: r_com*
≈ the quintic projection, the two references agree, and the torso reaches p_t1 (11 % residual). The
TP-dom τ_w-sat (24 %) is **execution cost** of over-tight torso tracking, NOT plan-envelope binding
(realized τ_w saturates while planned Ḣ_s stays < 5). Binding-regime validation (where the NMPC
deviates r_com* from the quintic) requires the multi-step traversal → **deferred to Phase 3**.

## Test 6 — hierarchy direction (PASS at the intended weighting)

At the intended hierarchy (momentum ≈ torso-pose, EE high — the balanced point), **torso-pose
yields the residual (14 mm) and the envelope is KEPT** (τ_w-sat 1.3 %). Correct direction → PASS.
The TP-dom point (deliberately inverted, torso-pose ≫ momentum) shows the envelope getting stressed
(24 %) — which is exactly *why* the starting hierarchy puts momentum at/above torso-pose; it
validates the ordering rather than contradicting it. No STOP condition (the envelope does not yield
at the intended weighting).

## Per-run cascade diagnosis

- **M-dom (30k:0.5k) — under-dock 6.99 mm:** momentum over-prioritised vs the fixed-weight EE →
  the swing EE yields and misses the 5 mm gate (torso/ori themselves fine). Diagnosis: keep
  momentum from dwarfing EE; this is the lower bound of the window, not a structural fault.
- **TP-dom (0.5k:30k) — envelope stress 24 %:** torso-pose over-prioritised → the AOCS saturates
  the wheels to hold attitude against the aggressive torso motion. Upper bound of the window.
- **bal (5k:5k):** no misbehaviour; the working point.

## Confirmations / logged-not-acted

- **F-SAT OFF (genuinely): 0 saturator calls** in two-task mode (the raw quintic bypasses the
  CoMToTorsoMapping → F-SAT is not called, not merely computed-but-unused).
- **τ_w-sat @100 Hz** (paper-cadence question): baseline/TMOM-only/M-dom 0 %, **bal 1.3 %**, TP-dom
  23.8 %. Logged + reported; not acted on.
- **Bit-identical-OFF:** preserved by construction (all two-task gates are `… and not _two_task`,
  no-op when OFF); `test_reworked_qp` 8/8 pass.

## Carry-forwards to Phase 3 (do not act here)

1. **Working point: balanced ss_alpha_mom:alpha_torso_pose ≈ 5000:5000** (EE high). Phase 3 runs
   the 5-step traversal at this ratio, not the extremes.
2. Torso arrival is tunable (cursor); the trade-off is arrival-tightness vs τ_w-envelope. Phase 3
   (multi-step) will exercise the **envelope-binding regime** the single step did not — re-measure
   the two-regime torso read there.
3. No architecture change indicated (tasks co-realise; hierarchy direction correct). The
   review-session knobs, if tighter arrival is wanted: torso-pose → position + weak-orientation, or
   momentum-linear de-weight — named, not implemented (per addendum Step C).

## 11. SS QP weights exposed as CLI flags (behavior-preserving)

`diag_cooperative_arms.py` now exposes the SS-stack weights: `--ss-alpha-mom`,
`--alpha-torso-pose` (alias `--ss-alpha-torso-pose`), `--ss-alpha-ee`, `--ss-alpha-posture`,
`--ss-alpha-wrench` (all `type=float`). **Each default equals the committed config value**
(ss_alpha_mom=500, alpha_torso_pose=1000, ss_alpha_ee=3000, ss_alpha_posture=20,
ss_alpha_wrench=0.01); `ss_alpha_torso_ang/lin` are NOT exposed (dead in two-task mode).
**Bit-identical verified:** the explicit balanced invocation
(`--ss-two-task --ss-alpha-mom 5000 --alpha-torso-pose 5000 --n-steps 1`) reproduces `20e6031`
exactly — all physical log arrays Δ=0, docks 4.95 mm, only wall-clock timers differ. The change
is **diag-only** (no `crawlbot/` edit) so flag-OFF + `test_reworked_qp` are unaffected.

**⚑ Discrepancy flagged (brief vs config):** the brief stated the working point
(ss_alpha_mom=5000, alpha_torso_pose=5000) as the *config default* and expected a NO-flag
two-task run to reproduce the balanced run. The committed config defaults are actually
**500/1000** — the 5k:5k working point was set by *explicit flags* in the sweep, not the default.
Per the brief's overriding constraint ("MUST NOT change the working point or any default
behavior" / "changes no default"), the mom/torso-pose defaults were **kept at 500/1000**; a
no-flag two-task run is therefore 500/1000 (unchanged), NOT the working point. Promoting them to
5k:5k (so no-flag = working point) is a one-line default change, deliberately NOT made here —
flagged for the review session.
