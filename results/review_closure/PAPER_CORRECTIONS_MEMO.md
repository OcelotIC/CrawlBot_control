# Paper-facing corrections — consolidated

**Review-Closure Bloc 2, closing deliverable.** Every item below is a place
where a sentence the paper makes (or is about to make) diverges from what the
code does or what the artifacts measure. Each carries the measurement, the
evidence path, and the narrowest wording the data supports.

Ordered by consequence, not by phase. **Nothing here is a recommendation to
change the science — only to state it accurately.**

Branch `claude/review-closure-bloc-2-uwu1x7`. All measurements reproduce from
committed artifacts; every phase report is under `results/review_closure/`.

---

## 1. The attitude benefit is the actuator clip's, not the planner's

**Severity: highest. This changes a headline claim.**

**Where it comes from.** The published pair (`c25` vs `u25`) shows θ_s peak
0.3668° managed vs 0.8949° unmanaged and attributes the 2.4× to the momentum
envelope. But `u25` moves **one kwarg into three consumers**: the NMPC rate
constraint, the whole-body QP envelope box, **and the AOCS output clip**
(`C4_ABLATION.md` §1, measured field-by-field over 131 `SimConfig` fields).

**The measurement.** C4 ran the arm the published pair does not contain —
planner constraints fully removed, actuator clip **held** at ±2.5:

| configuration | θ_s peak, traversal window |
|---|---:|
| `u25` (planner **and** clip lifted) | 0.8949° |
| **`none`** (planner lifted, clip held) | **0.3664°** |
| `full` = canonical | **0.3668°** |

Removing the planner's entire envelope constraint set changes the attitude peak
by **0.1 %, in the unconstrained arm's favour**.

**What the constraint does buy**, from the same three arms: saturation drops
from 10.15 % to 4.57 % of ticks, and the planned ‖L̇_s‖∞ from 12.741 N·m
(5.1× the envelope) to 2.500 on all six steps. It also *increases* peak storage
use, 3.505 → 4.102 N·m·s.

**Narrowest supported wording.** The envelope keeps the *planner's demand*
inside the actuator's authority — planned ‖L̇_s‖∞ at the bound on every step
against 5.1× without it, and half the saturated cycles. Attitude accuracy is
delivered by the reaction-wheel clip and the AOCS, and is essentially unchanged
by the planner constraint at this operating point.

**Evidence**: `results/review_closure/c4_ablation/C4_ABLATION.md` §3,
`c4_metrics.json`, `c4_config_diff.py`.

---

## 2. The traversal-time cost is capture refusals, not a slower plan

**Severity: high. A sentence shipped in r17 depends on it.**

r17 states the managed traversal takes 84.5 s against 75.0 s and attributes the
difference to the envelope constraint. Both durations reproduce (84.53 / 74.98).
The attribution does not.

**Decomposition** against the logged `preplanner_T_steps` (C1.6):

| component | Δ | share |
|---|---:|---:|
| longer **planned swings** | ≈ +0.3 s | **3 %** |
| longer **post-swing convergence** | ≈ +8.4 s | **88 %** |
| settle | +0.85 s | 9 % |

**Mechanism**, from `dock_gate_trace`: on steps 2 and 4 the capture gate refused
approaches whose position and orientation criteria were both already satisfied,
because the weld-relative twist exceeded `eps_twist = 0.05`. C4 reproduced this
in the clean three-way design: **steps 2 and 4 account for 8.5 s of the 8.6 s
(99 %)**, and the unconstrained arm has **zero** pose-valid refusals.

**Narrowest supported wording.** The envelope-limited wrench cannot arrest the
swing as cleanly, so the managed run arrives with ~10× more weld-relative twist
and the capture gate refuses the first approach on two of six steps; the cost is
re-convergence time, not a slower planned trajectory.

**Evidence**: `C1_EXACTNESS.md` §C1.6, `C4_ABLATION.md` §2.

---

## 3. The reported worst dock is a gate artifact — with a caveat

The canonical's worst at-weld dock is 4.990 mm, margin 0.01 mm against the 5 mm
capture radius. On the same step the gripper was at **4.334 mm with both pose
criteria satisfied** at t = 21.90 s, and was refused on twist. Step 4 likewise
reached 3.326 mm and is reported at 4.950 mm.

**The caveat, which C3.5 added and which cuts the other way.** Accepted twists
cluster at 0.005–0.020 and refused ones at 0.051–0.061, with an **empty
factor-2.5 gap**. `eps_twist = 0.05` sits in that gap: any value in roughly
(0.021, 0.050) gives identical results. The bound is **not** arbitrary within a
broad range, even though it was never swept.

But the two consequences have very different sensitivities:

- **dock precision** — recovering step 2's 4.334 mm needs `eps_twist ≥ 0.0606`,
  **+21 %**;
- **traversal time** — step 4's refusal misses by **1.2 %**; a bound of 0.051
  recovers **7.0 s of the 8.5 s** while leaving the worst dock unchanged.

**Narrowest supported wording.** Report the at-weld distance as the capture
metric (Rule 10) and state `eps_twist` in the same table, noting that ~82 % of
the timing penalty rests on a 1.2 % margin against an unswept constant. Do not
present the 0.01 mm margin as a tracking limit — it is where the gate let the
arm stop, not where it could stop.

**Evidence**: `C3_1_C3_5_REPORT.md` §C3.5, `C1_EXACTNESS.md` §C1.6.

---

## 4. AOCS law: three sign errors and one wrong error map

**Severity: notational, but the equation as printed is not the implemented law.**

The paper writes
`τ_w = τ_ff − K_θ e_θ − K_ω ω_s − K_d ω̇_s + K_hw(sat(h_w) − h_w)`.

The code is `pid_term = **+** K_θ·θ_s **+** K_ω·ω_s **+** K_d·ω̇_s`, added with
`+`. The positive sign is deliberate and derived in-source: τ_w on the wheels
produces −τ_w on the structure, so driving `θ_s → 0` needs a positive
contribution. With the code's own definitions of `θ_s` and `ω_s`, all three are
`+`.

Separately, `θ_s` is the **Lee–McClamroch** error `½ vee(R_err − R_errᵀ)`, not
`log3(R_err)`. The call-site comment is explicit; the function docstring still
claims `log3`. At ≤ 0.54° the two agree to better than 1e-5 relative.

**Narrowest supported wording.** Either flip the three signs, or define
`e_θ ≜ −θ_s` and `ω ≜ −ω_s` explicitly. Print the error map as
`½(R_refᵀR − RᵀR_ref)^∨` if it is printed at all.

**Evidence**: `C1_EXACTNESS.md` §C1.4.

---

## 5. The orientation-error frame in the task composition

The torso and swing-EE tasks use `log3(RᵀR_ref)` — a **body-frame** rotation
vector — while the Jacobians are `LOCAL_WORLD_ALIGNED`, whose angular rows
produce world-aligned acceleration. The frame-consistent form is
`log3(R_ref Rᵀ) = R·log3(RᵀR_ref)`.

Magnitudes are identical; directions differ by the frame's own attitude. The
torso is **not** near identity — measured 3.94–6.13° across the run — so the
angular PD term is driven in a direction wrong by **6.9–10.7 %**, bounded at
**0.13°** of commanded error. The swing-EE effect is negligible (≤ 2.3 %).

**Narrowest supported wording.** State the frame. If §VI-D.5/6 prints the
world-frame form, it describes something the code does not do; printing
`e_R = log(RᵀR_ref)` with "expressed in the body frame" is a one-line fix.

Two related facts worth a sentence: the torso **orientation** reference is
constant to 7e-18 across all 2077 ticks (a pure 5.157° yaw held for the whole
traversal), so the angular feedforward is identically zero and the SLERP runs
between identical endpoints. And the torso gains are `K_p = 3, K_d = 2.5` —
the paper is right; the `SimConfig` default of 6.0/5.0 is the trap.

**Evidence**: `C1_EXACTNESS.md` §C1.2, `c1_jacobian_probe.json`.

---

## 6. The 4.1 % clip fraction reproduces under no convention

Measured from `aocs_tau_w_preclip` against the ±2.5 cap:

| convention | value |
|---|---:|
| any-axis per tick, **traversal window** | **5.064 %** |
| any-axis per tick, full log | **4.574 %** |
| axis-sample, traversal | 1.759 % |
| axis-sample, full log | 1.589 % |
| *paper as cited* | *4.1 %* |

The companion "368 / 51 448 plant clamps" is at a third cadence: 51 448 is
neither the tick count (2077), the axis-sample count (6231), nor ticks × 10
sub-steps (20 770). Its denominator is recorded nowhere in the repository.

**Narrowest supported wording.** Restate from these channels with the
convention named. Saturation is concentrated on **z** (4.0 % of traversal
ticks) and is more frequent in the inter-step settles than in the swings
(5.48 % vs 3.94 %) — the opposite of what the text assumed — with the terminal
settle never saturating.

**Evidence**: `C3_1_C3_5_REPORT.md` §C3.1, `C2_1_C2_3_INSTRUMENTATION.md` §4.

---

## 7. OPEN — a 34x wheel-momentum bookkeeping gap, unresolved

**Severity: potentially high, and unresolved. Flagged, not closed.**

On an instrumented 900 s settle, the integral of the commanded wheel torque is
**+2.884 N·m·s** while the logged wheel momentum changes by **+0.084 N·m·s** —
a **34.4× gap** on the z axis. Aliasing, joint armature and wheel damping are
all eliminated by measurement; channel indexing was verified correct. The
attitude loop itself converges normally (τ_w, h_w and θ_s all decay with a
~320 s time constant).

**Why it matters for the paper: the ±5 N·m·s storage claim rests on this
channel.** C4 reports the canonical peaking at 82.0 % of the box, and Gate D
turns on that margin. Either the wheel absorbs momentum the channel does not
see (margin overstated), or the commanded torque is not applied (the AOCS is
weaker than modelled and attitude is delivered by something else). The data does
not yet distinguish them.

**Do not quote the storage margin as settled until this is closed.** Closing it
is one short probe: read `actuator_force` for the three wheel actuators against
`ctrl`, and integrate the wheel's *absolute* angular momentum rather than the
joint-relative `I_w·qvel`.

**Evidence**: `C3_3_THETA_Y_AUDIT.md` Addendum, `c3_3_settle900.json`.

---

## 8. Smaller items, each a single sentence

| item | correction | evidence |
|---|---|---|
| **Solver timings** | Wall-clock moves ~25 % between machines while IPOPT iteration counts are byte-identical. Any timing table needs a hardware row; the committed artifacts' machine is unrecorded. Prefer iterations: the rate constraint costs **+56.8 %** per solve, storage **+0.36 %** | `C3_4_SOLVER_STATS.md` §3, `C4_ABLATION.md` §5 |
| **`qp_time_ms`** | Not a QP solve time — it times the whole WBC block (10 solves + Pinocchio + AOCS + `mj_step`). True per-solve median is **5.641 ms**; the QP is ~71 % of the block | `C3_4_SOLVER_STATS.md` §0 |
| **"0 QP failures"** | Was never a measurement: `error_on_fail: False` means an infeasible QP returns instead of raising, and `qp_ok` is hardcoded True on 1368 of 2077 ticks. Now measured properly: **0 failures over all 8458 solves** | `C2_2_SOLVER_INSTRUMENTATION.md` §3 |
| **Real-time** | Not supported. Controller-only QP cost exceeds the 10 ms budget on **1.06 %** of ticks (0.59 % in SS), with one unexplained 98 ms outlier | `C2_2_SOLVER_INSTRUMENTATION.md` §5 |
| **θ_s convention** | Traversal-window and full-log peaks differ (0.3668 vs 0.5346) because attitude keeps rising through the terminal settle. Label which is quoted | `C4_ABLATION.md` §2 |
| **Structure inertia (Table II)** | mass 7110 kg, `fullinertia 597/1493/1777` body-frame, radius of gyration 0.29/0.46/0.50 m. State it as a *declared* parameter — the same file draws a 4.8 m beam whose uniform inertia would be ~9× larger | `C1_EXACTNESS.md` §C1.7 |
| **Storage constraint** | Inactive at the 1 % ratio: `‖h_w‖` peaks at **82.0 %** of the box and never leaves it; removing it changes solver iterations by 0.36 % and nothing physical | `C4_ABLATION.md` §4 |
| **θ_y plateau** | Expected behaviour, not a bug: the AOCS has no integrator, so a P attitude law parks at a steady-state offset against a persistent welded-loop couple. On y the feedforward outvotes the attitude term **15.4:1** (vs 3.9 on x, 0.96 on z) because a residual net z-force at the welds torques about y on a **2.4 m lever**. State `K_θ = 1.0 N·m/rad` if the plateau is discussed | `C3_3_THETA_Y_AUDIT.md` |
| **Conservation residual (Fig. 3)** | ‖L_total‖ ≤ **1.4937e-03** N·m·s, injected at the six weld events and **flat thereafter** (0.0000e+00 drift over 879 s). Not integrator round-off — the swing legs carry 22× the CoM path yet inject 297× less | `C3_2` §(e) |

---

## What is *not* a correction

Confirmed exactly as the paper states, and worth not re-litigating:

- the QP is fully **weighted** with no null-space projection, `weight_ratio = 1`;
- the Add-5 weights — torso 2000, EE 1000, momentum 400, torque-min 5;
- the momentum task uses the **linear CMM rows** `A_{G,v}` (verified to 2.2e-16)
  and `L_com` is constrained, never tracked, by any Stage-2 task;
- the task Jacobian block structure, with both zero blocks exactly zero;
- **no integral term** in the AOCS;
- the anti-windup term is identically zero over the whole run — now measured
  per-tick rather than inferred;
- torso gains `K_p = 3`, `K_d = 2.5`;
- 6/6 docks, and the `full` ablation arm reproduces the committed canonical
  byte-identically (132 928 fields).

Two weight-table omissions rather than errors: `w_hw_slack = 800` is a cost term
numerically above the momentum task (though inert here), and `alpha_reg = 1` is
the floor that `alpha_torque = 5` is measured against.
