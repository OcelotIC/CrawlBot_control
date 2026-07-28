# C3.3 — θ_y micro-audit

**Review-Closure Bloc 2, Phase C3.3.** Read-only, no runs. Parent **`c4eb1f1`**.
Env exact-matches `gate/environment.lock`. Artifacts: `t4b_trace_900s.csv`
(4823 samples / 964 s), `results/review_closure/c2/c25_c2_fulldiag.csv` (the
canonical with the C2.3 AOCS decomposition).

---

## 0. The gate the brief set — and why it opened

The brief made C3.3 conditional: *"the conservation residual, left as
unregulated rotation, would accumulate ~0.04° over the same 900 s window… If
C3.2(e) shows the residual can account for it, stop there."*

**It cannot.** C3.2(e) established the residual is injected at six discrete weld
events and is then **exactly flat — 0.0000e+00 drift over 879 s**. There is no
time-accumulating rotational floor of the hypothesised kind, so none of the
plateau is explained away by it. The audit proceeds.

---

## 1. The plateau, characterised

Quiescent tail, t ≥ 120 s (4223 samples):

| axis | mean θ_s [deg] | std | drift over 900 s | mean ω_s [deg/s] | mean h_w [N·m·s] | mean τ_w [mN·m] |
|---|---:|---:|---:|---:|---:|---:|
| x | +0.02242 | 1.6e-2 | −5.7e-2 | −7.1e-5 | +0.00255 | +0.3049 |
| **y** | **+0.03214** | 1.5e-2 | +5.1e-2 | +6.2e-5 | −0.00149 | **−0.4035** |
| z | +0.19663 | 1.5e-1 | −5.2e-1 | −6.4e-4 | +0.02366 | +3.0921 |

The y plateau is real and quasi-static. Note z is 6× larger — y is not the
biggest offset, it is the *anomalous* one, for the reason in §2.

---

## 2. The finding: on y the attitude term is not merely weak, it is **outvoted**

The AOCS has **no integral term** (C1.4, confirmed). A pure proportional
attitude law therefore parks at a steady-state offset against any persistent
disturbance couple. In the quiescent tail ω_s ≈ 0 and anti-windup ≡ 0, so the
commanded torque should be dominated by `K_θ·θ_s` with `K_θ = 1.0 N·m/rad`.

| axis | `K_θ·θ_s` [mN·m] | measured τ_w [mN·m] | ratio |
|---|---:|---:|---:|
| x | +0.3912 | +0.3049 | **+0.779** |
| **y** | **+0.5610** | **−0.4035** | **−0.719** |
| z | +3.4319 | +3.0921 | **+0.901** |

**x and z sit at 0.78–0.90 — consistent with the attitude term plus small
companions. y is −0.72: the same magnitude, the opposite sign.** The commanded
wheel torque on y opposes what the attitude term asks for. That is the "bias in
the y error path" the standing hypothesis predicted, and it is a sign
inversion in the *net*, not a gain error.

### What is doing the outvoting

From the instrumented canonical terminal settle (201 ticks, the only committed
data carrying the C2.3 decomposition):

| axis | θ_s [deg] | `tau_att_p` | `tau_rate_d` | `tau_accel_d` | **`tau_ff`** | total |
|---|---:|---:|---:|---:|---:|---:|
| x | +0.07116 | +1.2385 | +0.1229 | −0.2336 | +4.7969 | +5.9248 |
| **y** | +0.06562 | +1.1509 | −5.8066 | +0.7215 | **+17.6795** | +13.7453 |
| z | +0.40174 | +7.0077 | +15.8526 | −0.3726 | −6.7333 | +15.7545 |

*(mN·m, means over the settle)*

**`|tau_ff| / |tau_att_p|` is 3.9 on x, 0.96 on z, and 15.4 on y.** The y-axis
attitude authority is swamped by the feedforward by an order of magnitude more
than either other axis. On z the two are comparable, which is why z — despite a
6× larger offset — behaves as a well-posed P-control steady state and y does
not.

### Why y, geometrically

At the final welded pose both welds sit at **x = 1.2 m**, y = ±0.3 m. The couple
a *net* contact force produces about each axis:

| net force | resulting couple | dominant axis | lever |
|---|---|---|---:|
| f_x | [0, 0.05, 0] | y | 0.05 m |
| f_y | [−0.05, 0, 2.4] | z | 2.4 m |
| **f_z** | **[0, −2.4, 0]** | **y** | **2.4 m** |

**A residual net z-force at the welds torques about y with a 2.4 m lever** — the
longest lever available to a contact-force residual, tied with f_y→z. Against
that, the attitude term's authority at a 0.03° offset is ~0.6 mN·m, fixed by
`K_θ = 1.0`. y is structurally the axis where a small residual contact force
most easily overwhelms attitude control.

**No bug is required to explain the plateau.** It is the steady state of a
proportional attitude law with no integrator, sitting against a persistent
welded-loop couple that the geometry amplifies onto y. The three candidate
mechanisms the brief listed — y-reference construction, quaternion→error mapping
on y, sign conventions — are **not** needed, and I found no evidence for any of
them: the error map is the same `½vee(R_err − R_errᵀ)` on all three axes, with
no axis-specific branch.

---

## 3. ⚠ One anomaly I could not close: h_w,y does not ramp

The brief flagged it and it survives the audit. A sustained wheel torque must
integrate into wheel momentum — `dh_w/dt = τ_w` — but over the tail:

| axis | mean τ_w [mN·m] | measured `dh_w/dt` [mN·m] | ratio |
|---|---:|---:|---:|
| x | +0.3049 | −0.00376 | −81× |
| y | −0.4035 | +0.00564 | −72× |
| z | +3.0921 | −0.06868 | −45× |

**All three axes show the same 45–81× shortfall, with the sign reversed.** This
is not y-specific, which is itself informative: the brief framed "h_w,y not
ramping" as evidence for a y-axis bias, and it is not — it is a system-wide
inconsistency between the *logged commanded* τ_w and the *realized* wheel
momentum rate.

Candidate explanations, none verified here:

- the logged `tau_w` is the command, and the applied torque differs (the plant
  clips at ±2.5 N·m, far above these values, so clipping is not it);
- wheel joint damping (`damping="1e-4"` in the MJCF) opposes it — but at
  ω_w = h_w/I_w ≈ 0.15 rad/s that is ~0.015 mN·m, ~4 % of the gap, not enough;
- the mean is taken over a window where τ_w oscillates about a small offset and
  the ramp is genuinely below the numerical resolution of the h_w channel.

**This is the right next question and it is a measurement, not an inference** —
it needs the C2.3 decomposition and the wheel state logged together over a long
settle, which no committed artifact has (T4b predates the instrumentation). One
900 s run with `--solver-diag` would settle it. Out of scope for a read-only
phase; flagged.

---

## 4. Honest limitation

`tau_ff` in DS should be the welded-loop couple `−Σ(r_Ci×f_i + τ_i)`, i.e. the
negative of the realized `Ḣ_s`. Checked: they agree on x (4.797 vs 4.797) and
approximately on y (17.68 vs 16.79) but **not on z** (−6.73 vs +0.99), max
discrepancy 0.205 N·m. The exporter reconstructs `Hdot_s_realized` from
hard-coded per-step stance/anchor tables, whereas the AOCS uses the scheduler's
live anchors; for the terminal settle those assignments need not agree. So the
§2 argument rests on `tau_ff` as logged by the controller — which is the
authoritative one — and **not** on the exporter's reconstruction. Worth
reconciling, but it does not affect the conclusion.

---

## STOP

**Verdict: the θ_y plateau is a proportional-control steady state, not a
control-path bug.** The attitude term is outvoted on y by a persistent
feedforward couple (15.4:1, vs 3.9 on x and 0.96 on z), amplified onto y by a
2.4 m lever from any residual net z-force at the welds. The sign inversion in
the net torque is the symptom of that, not of an error-path defect.

Two consequences for the paper, if θ_y is discussed at all:

1. The plateau is expected behaviour for an integrator-free attitude law and
   should be described as such, with `K_θ = 1.0 N·m/rad` stated — not as a
   residual drift or an unexplained bias.
2. Raising `K_θ` would shrink it proportionally, but §2 of
   `force_estimator.md` notes the attitude term is **momentum**-bound, not
   torque-bound, so that is a trade against desaturation headroom rather than a
   free improvement. Not a change I would make without its own phase.

One open item handed on: the 45–81× τ_w-vs-`dh_w/dt` inconsistency (§3), which
is system-wide, not y-specific, and needs one instrumented long settle to close.

---

# ADDENDUM — §3 closed by an instrumented 900 s settle

Run: `results/review_closure/c3/settle900/`, produced by
`c3_3_run_settle900.py` (canonical kwargs verbatim, `settle_seconds` 20 → 900,
nothing else changed). Traversal reproduces the canonical: **6/6 docks**,
4.02 / 4.89 / 4.99 / 4.97 / 4.95 / 4.62 mm. 8699 settle ticks at 10 Hz with the
full C2.x instrumentation. Analysis: `c3_3_settle900_analyse.py` →
`c3_3_settle900.json`.

## The original framing was partly an artifact — and a bigger problem survives

**§3 overstated the mismatch by treating a decaying transient as a steady
state.** Over the settle, τ_w, h_w and θ_s all decay exponentially with the
*same* time constant (**323.7 / 323.1 / 301.9 s**), so a window mean and a
linear `dh_w/dt` fit are both meaningless. The control loop is working: the
attitude returns to zero as designed.

Three of the four hypotheses are eliminated outright:

| | test | verdict |
|---|---|---|
| **H1** aliasing | full-rate mean vs 0.2 s subsample differs by 0.001 mN·m (0.3 %) | **rejected** |
| **H2** armature | correcting `I = 0.01 → 0.02` moves the ratio 0.011 → 0.023 | **rejected** (factor 2, not 34) |
| **H3** damping | explains 2.8–7.9 % of τ_w; a sustained τ_w,z would need ω_w = 33.2 rad/s, actual 2.31 | **rejected** |

## What survives: a 34× momentum-bookkeeping gap

The clean test is the **integral**, which is immune to the decay:

| quantity | value |
|---|---:|
| ∫ commanded τ_w,z dt over the settle | **+2.884 N·m·s** |
| actual Δh_w,z over the same window | **+0.084 N·m·s** |
| ratio | **0.029 — a 34.4× gap** |
| ω_w,z implied if the wheel had received it | **144 rad/s** |
| actual final ω_w,z | **0.412 rad/s** |

Per-tick, `Δh_w` and `τ_w·dt` correlate at only 0.478 with a ratio of 0.029, so
this is not a windowing effect either.

Both channels were checked for indexing: `hw_physical = rwa_I_w · qvel[6:9]` and
the wheel actuators are `ctrl[14:17]`; the MJCF orders the structure freejoint
(qvel 0:6) then `rw_x/y/z` (6:9), and the actuator list places `act_rw_*` after
the 14 arm motors. Both are correct.

## Why this matters more than the θ_y question it came from

**The ±5 N·m·s storage claim rests on `hw_physical`.** C4 reports the canonical
peaking at 82.0 % of the box, and Gate D turns on exactly that margin. If the
wheel is absorbing momentum the logged channel does not see, the margin is
overstated; if the commanded torque is not being applied, the AOCS is weaker
than modelled and the attitude performance is being delivered by something else.
Both readings are consequential and the data does not yet distinguish them.

**Explicitly not resolved here.** The remaining candidates need a targeted probe
rather than another traversal: read MuJoCo's `actuator_force` for the three
wheel actuators and compare against `ctrl`; and integrate the wheel's *absolute*
angular momentum (including the structure's rotation carrying the wheel) rather
than the joint-relative `I_w·qvel`. That is one short instrumented run and a
direct MuJoCo state read — a phase of its own, and one I would not fold into a
θ_y audit.

**Revised status of §3:** the "45–81×, all three axes" figure is withdrawn — it
was a mean over a decaying transient. The defensible statement is a **34.4×
integral discrepancy on the z axis** in a run where the attitude loop
demonstrably converges, with aliasing, armature and damping eliminated as
explanations.
