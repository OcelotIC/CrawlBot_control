# The h_w box bounds a quantity that can be off by more than its own margin

**Branch** `claude/com-gain-semantics-audit-j0u6yr`
**Question taken on:** the wheels do not store the robot's momentum directly —
`h_w` is the integral of `τ_w`, the moment that rejects the contact wrench. The
NMPC box constrains an *inferred* `h_w`, so how accurate is the inference?

**Answer: on the z axis, the reconstruction's frozen-platform error over one
horizon (1.27 N·m·s) exceeds the entire remaining box margin (1.19 N·m·s).**
Measured by `scripts/audit_nmpc_conservation_drag.py`.

---

## 1. What the spec actually says

Spec §3.5 gives the **exact** law (its words: *"This is the exact expression — no
approximations"*):

$$\mathbf{h}_w^s = \mathbf{R}_s^\top\mathbf{L}_0 - (\mathbf{I}_s + \mathbf{I}_\text{robot}^{O_s,s})\boldsymbol{\omega}_s^s - m\,\mathbf{r}_\text{com}^s\times\mathbf{v}_s^s - \mathbf{L}_\text{com}^{s,\text{rel}} - \mathbf{r}_\text{com}^s\times m\,\mathbf{v}_\text{com}^{s,\text{rel}}$$

and then two reductions:

**Option A — freeze the platform.** `R_s`, `ω_s`, `v_s`, `I_s` are held at their
t=0 values over the horizon. The measured constant is
`c = h_w0 + L_robot^{in,0}`, which equals `R_s^T L_0 − I_s ω_s0`. So `I_s ω_s`
is *absorbed at t=0*, not carried. Residual error: **`I_s·(ω_s(k) − ω_s(0))`**.

**Option B — additionally drop the robot drag terms.** This is what
`compute_c_simple` implements. Its extra error over A is
`m·(r_com(k) − r_com(0)) × v_s0`, which the spec names `ε_drag` and for which it
prescribes:

> This can be compensated by tightening the box: `h_max' = h_max − ε_drag·1`

So the code is a **faithful implementation of Option B** — the docstring's
"cancel algebraically" claim is correct, and my earlier suspicion that `I_s·ω_s`
was simply forgotten was wrong. It is absorbed into `c` by construction.

## 2. Finding 1 — the prescribed tightening was never applied

```
physical envelope hw_max      = [5.0, 5.0, 5.0]
box actually used h_max_tight = [5.0, 5.0, 5.0]
tightening applied            = [0.0, 0.0, 0.0]
```

`h_max_tight` equals the **full physical envelope**. The allowance the spec
requires for the Option-B neglect is not paid.

**But it does not matter here**, because ε_drag is tiny:

```
max |Δr_com| over one 2.0 s horizon = 0.2443 m
|v_s| <~ 21.07 mm / 77 s            = 0.272 mm/s
eps_drag <~ 71.1 * 0.2443 * 2.72e-4 = 4.7e-3 Nms
```

4.7 mN·m·s against a 5 N·m·s box. So the missing tightening is harmless — for
the term it was meant to cover.

## 3. Finding 2 — the term that *does* matter has no allowance at all

The frozen-platform residual `I_s·Δω_s` is present in **Option A and Option B
alike**. It is not what `ε_drag` covers, and the spec prescribes nothing for it.
Measured over one 2.0 s horizon on the canonical run:

| axis | max \|I_s·Δω_s\| [Nms] | box margin [Nms] | **% of margin** |
|---|---|---|---|
| x | 1.4446 | 4.4172 | 32.7 % |
| y | 1.9037 | 2.6612 | 71.5 % |
| **z** | **1.2663** | **1.1854** | **106.8 %** |

(structure inertia diag `[1777, 1493, 597]` kg·m²; box margin = `h_max_tight`
minus the realized per-axis `|h_w|` peak.)

**On z the reconstruction error over one horizon is larger than the whole
remaining margin.** So a hard constraint at ±5 N·m·s is bounding a quantity that
may sit more than a margin's width away from the wheels' true state — on the
same axis that is at the torque cap 14 % of the time and has only 0.47 s of
saturated authority in hand.

## 4. Why the spec's justification does not transfer

The spec justifies dropping the drag terms for:

> *"SpaceServicer on a realistic structure (mass ratio ~14 %, ‖ω_s‖ ≲ 0.05 rad/s,
> ‖v_s‖ ≲ 0.02 m/s)"*

The canonical runs at **mass ratio 0.01**, not 0.14. That cuts both ways and the
net is adverse:

- `ω_s` is *far* smaller than assumed — 1.34 mrad/s peak vs the 0.05 rad/s bound,
  37× better. The spec bounded **ω_s**.
- but `I_s` is *far* larger — a 7110 kg structure with inertia to 1777 kg·m².
  The spec never bounded **`I_s·ω_s`**, which is the product that enters the law.

At 14 % mass ratio the structure would be ~500 kg and `I_s·Δω_s` would indeed be
negligible. At 1 % it is O(1) N·m·s. **The neglect analysis was done for a
different vehicle than the one being simulated.**

## 5. What this does and does not mean

**Does not mean the run is wrong.** The realized `h_w` peaks at 3.815 against a
±5 envelope and the wheels never saturate in momentum; the traversal is sound.
The box is not producing bad behaviour.

**Does mean the box's guarantee is weaker than it appears.** It is a hard
per-axis constraint on `ĥ_w`, an estimate whose horizon-scale error is
comparable to — on z, larger than — the margin it defends. Quoting it as an
envelope *guarantee* over-claims. It is better described as a hard constraint on
a good-but-not-tight estimate.

**Does not mean the error accumulates over the traversal.** `h_w` is a
within-step *excursion*, not a wind-up: net drift after six steps is x +0.001,
y +0.098, z −0.239 N·m·s against a z peak of −3.815 (`NMPC_F2_RWA_BOX.md` §2.4).
The residual `I_s·Δω_s` is likewise measured over **one horizon**, so it is a
transient error on a transient quantity — the estimate does not walk away over
the run. What is at stake is the accuracy of the box inside one long push, which
is the only regime the box governs.

## 5.1 The error and the binding are driven by the same quantity

This is the part that constrains how the box can be demonstrated. `Ḣ_s` on z is
**already clipped** — its 95th percentile *is* the 2.5 N·m cap, 302 of 1967
ticks at it. So making the box bind cannot work through amplitude; it works
through the **duration** of saturation, which `h_w` integrates (z would need
2.00 s continuous against today's 1.53 s, +31 %).

But every second of saturation is a second the wheels do **not** fully reject
the applied moment, so the excess spins the structure: `ω_s` grows, and with it
`I_s·Δω_s` — this section's error term. Pushing the robot further from the
structure CoM to force the box to bite would raise the reconstruction error on
the same axis, by the same mechanism, at the same time.

**Consequence:** the export gaps in §7 are not a nicety, they are a
precondition. Without `ĥ_w − h_w` logged per tick, a run in which the box
appears to bind cannot be distinguished from one in which the estimator has
drifted into the constraint.

## 6. Options, in increasing cost

1. **Pay the allowance the spec already asks for, sized correctly.** Set
   `h_max_tight = hw_max − ε` with ε covering `I_s·Δω_s`, not just `ε_drag` —
   here ε ≈ 1.9 N·m·s, so `h_max_tight ≈ 3.1`. Cheapest, one number, and it
   makes the constraint honest. Cost: it would start binding (z has only 1.19
   N·m·s of margin today), so it is a real behavioural change, not free.
2. **Carry `I_s·ω_s` per knot instead of freezing it** — the spec's Option A
   upgrade path. Needs `ω_s` predicted over the horizon, which the NMPC does not
   model (the structure is deliberately outside its state).
3. **Leave it and document it.** Defensible while `ω_s` stays this small, but
   the number should be in the paper's §V rather than implied away.

**Recommendation: (1), with the ε chosen from a measured `I_s·Δω_s` sweep rather
than a single run** — and explicitly *not* as a silent retune, since it changes
the binding set. This is Idriss's call, not mine.

## 7. What would make this checkable in future

`v_s` (platform linear velocity) is **not in the fulldiag export** — §2's ε_drag
had to be bounded from the structure-drift telemetry rather than measured. Adding
`v_struct_*` and the reconstruction residual `ĥ_w − h_w` as export channels would
turn this whole analysis into a per-run diagnostic instead of a one-off study.
