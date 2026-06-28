# INTERNAL — J2 envelope audit: is Piste A QP-formulable? (`ae0673e`, READ-ONLY)

**Question.** Mechanism (3) for the moving-reference reconciliation = *allow positive work up to what the
envelope permits.* Piste A realizes it by replacing the passivity inequality's zero RHS with a parametric
work-budget:
```
dqᵀ τ_q + 2α T_kin ≤ W_budget        (vs the current  … ≤ 0)
```
Piste A is a clean contribution **iff** this is a linear (or SOC) QP constraint with `W_budget` a **pre-solve
parameter**. This audit decides that.

**VERDICT: YES — Piste A is formulable as a LINEAR QP row.** `dqᵀτ_q + 2α T_kin` is already linear in the
decision variable `τ_q` (it *is* the existing passivity row, "Linear in τ_q only", `wholebody_qp.py:546`);
and a **planned envelope margin** is available as a pre-solve scalar from the NMPC's `lambda_ref` (the NMPC
solves *before* the QP and its plan already respects the envelope). `τ_q` (passivity) and `λ` (envelope) are
**distinct decision variables coupled only by the linear equation of motion — no bilinear product** — so
convexity is preserved. Reproducer `scripts/audit_j2_envelope.py`: **17/17**.

**Two flags carried into the spec (neither breaks formulability):**
- **FLAG 1 (units):** the envelope margin is a **torque** [N·m]; the passivity LHS is **power** [W]. So
  `W_budget` cannot *be* the margin — it is a power derived from it via a units gain [1/s]. A pre-solve
  scalar; does **not** affect linearity, but the spec must define the gain (a modeling choice).
- **FLAG 2 (proxy mismatch):** the QP's *own* envelope box uses the `|L̇_com|` proxy (lever-from-robot-CoM)
  that the **NMPC explicitly abandoned** as "wrong quantity at non-zero standoff." Derive `W_budget` from the
  **NMPC-exact `Ḣ_s(lambda_ref)`** margin (about the origin), not from the QP box.

---

## Q1 — Where and how is the envelope formulated?

**In BOTH the NMPC and the QP — and they use different quantities.**

**NMPC** (`centroidal_nmpc.py`, CasADi `path_constraints`, on `u=[f1,τ1,f2,τ2]` + state):
- **Wheel-torque rate cap (the envelope), EXACT:** `Ḣ_s = r_C1×f1 + τ1 + r_C2×f2 + τ2` about the **origin**
  (= structure CoM in R_s), bilateral box `−τ_w_max ≤ Ḣ_s ≤ τ_w_max` (`:279-282`). **Linear** in `u`.
- **SOC** on each wrench: `‖f‖²≤f_max²`, `‖τ‖²≤τ_max²` (`:262-268`) — second-order cone.
- Linear-momentum `‖m·v_com‖²≤p_max²` (`:287-288`); RWA conservation box `hw_k=c_simple−L_com−r_com×m·v_com
  ∈ [−h_max,h_max]` (`:296-300`, **bilinear in state**, IPOPT-native).
- **Decisive comment (`:275-278`):** the exact `Ḣ_s` *"Replaces the prior |L̇_com,i| proxy, which used
  lever-from-robot-CoM and bounded only the spin-rate part of the robot-momentum-rate — wrong quantity at
  non-zero standoff (campaign §9 documents the divergence)."*

**QP** (`wholebody_qp.py`, on `λ`): it **re-enforces** an envelope box — it does **not** only track
`lambda_ref`:
- **Momentum-rate box (`:525-538`):** `|L̇_robot| = |M_λ·λ| ≤ τ_w_max`, gated `np.isfinite(cfg.tau_w_max)`
  (canonical `τ_w_max=5 N·m`). 3 upper + 3 lower **linear** rows on `idx['lambda']`, coefficient `±M_λ`.
- Also the M5 hw-safety box `h_min ≤ hw−dt·M_λ·λ ≤ h_max` (soft slack, `:477-508`) and the `|L_com+dt·M_λ·λ|
  ≤ L_max` box (`:510-523`) — all linear in `λ`.
- **But `M_λ` here is the proxy** (lever-from-robot-CoM, see Q2) — the very quantity the NMPC dropped. So the
  QP enforces `|L̇_com(about robot CoM)| ≤ τ_w_max`, an ∞-norm box, which differs from the true envelope
  `‖L̇_com + r_com×m·v̇_com‖∞ ≤ τ_w,max` (about origin) by the orbital term.

⇒ **Form:** NMPC = exact `Ḣ_s` (linear) + SOC; QP = ∞-norm **box** (linear rows) on `λ`, but of the *proxy*
quantity. **The QP re-enforces the envelope (proxy), in addition to tracking `lambda_ref`.**

## Q2 — Where does `L̇_com` live in the QP, and is it linear in λ?

**Yes — `L̇_com = M_λ · λ`, fully linear in `λ`, with `M_λ` a 3×12 pre-solve parameter.**
`M_λ = compute_momentum_map(r_com, contact_config)` (`wholebody_qp.py:493`; fn `contact_phase.py:101-137`):
```
for each active contact j:   L̇_j = (r_Cj − r_com) × f_j + τ_j  = [ S(r_Cj − r_com) ,  I₃ ] · [f_j ; τ_j]
M_λ = [ S(r_CA − r_com) | I₃ | S(r_CB − r_com) | I₃ ]            (3×12; inactive columns zero)
```
`M_λ` depends only on `r_com` and the contact positions — **both parameters at the QP tick** (current CoM,
current anchor sites). So `L̇_com = M_λ·λ` is a **linear expression in the decision variable `λ`**, no state
terms beyond the parameter `M_λ`. (Numeric: superposition `M(3a−2b)=3Ma−2Mb` holds to 9e-16.) **Caveat:** the
lever is `r_Cj − r_com` ⇒ this is `L̇_com` about the **robot CoM** — the proxy, **not** the origin quantity
(Q1 / FLAG 2).

## Q3 — Is the envelope MARGIN extractable BEFORE the QP solve? (DECISIVE)

**Yes.** The realized `L̇_com = M_λ·λ` is only known post-solve (λ is a decision variable) — **but the
PLANNED margin is a pre-solve parameter**, because the NMPC solves first and hands its plan to the QP:
- **NMPC solves before the QP** and returns `lr` (planned `lambda_ref`): `sim_loop.py:2471`
  (`rp, vp, _, lr, info_n = self.nmpc.solve(...)`), and `af = compute_feedforward_acceleration(lr) =
  (1/m)Σf_ref` — the **planned CoM acceleration** (`:2477`).
- **Both reach the QP as parameters:** `qp.solve(..., lambda_ref=lr, a_com_ff=af, ...)` (`sim_loop.py:2789`).
- Therefore at QP entry we can compute, from parameters only:
  - planned **exact** usage `‖Ḣ_s(lr)‖∞ = ‖r_CA×lr_f1 + lr_τ1 + r_CB×lr_f2 + lr_τ2‖∞` (about origin), and/or
  - planned **proxy** usage `‖M_λ·lr‖∞`, plus the orbital term `r_com×m·af` (so the exact = proxy + orbital).
- The NMPC's own path constraint already enforced `‖Ḣ_s‖∞ ≤ τ_w_max` on the plan ⇒ the **planned margin
  `m_plan = τ_w_max − ‖Ḣ_s(lr)‖∞ ≥ 0`** — exactly the `W_budget ≥ 0` Piste A needs.

⇒ **A pre-solve (planned) margin exists ⇒ `W_budget` is a parameter ⇒ Piste A is LINEAR ⇒ FEASIBLE.**
(The `h_max` margin `h_max − ‖hw_planned‖∞`, also bounded by the NMPC, is available the same way as a second
budget source if desired.)

## Q4 — The `τ_q ↔ λ ↔ L̇_com` chain in the QP

**`τ_q` enters the centroidal rate ONLY through `λ` — the link is linear, no product.**
- Decision vector `z = [q̈_t, q̈, λ, τ_q]` (`wholebody_qp.py:427`): `λ` and `τ_q` are **separate** variables.
- They meet in **one linear equality**, the EoM `H q̈ + C = B_u τ_q + Jᵀλ`
  (`A_dyn[:,λ]=−Jᵀ` `:451`; `A_dyn[6:,τ_q]=−I` `:454`; `b=−C`). **Both linear, same equality, no `τ_q·λ`
  term.**
- The **momentum-rate box is purely `M_λ·λ`** (`:528,532`) — `τ_q` does **not** appear in it. The
  **passivity row is purely `τ_q`** (`A_pass[τ_q]=dq`, `:553`; "Linear in τ_q only", `:546`).
- So "positive joint work `dqᵀτ_q`" (a `τ_q` row) and "envelope usage `M_λ·λ`" (a `λ` row) are tied **only**
  through the linear EoM. **No bilinear coupling anywhere** ⇒ adding a budget that mixes them stays convex.

This is what makes Piste A clean: because the two never multiply, you can either (a) put a **pre-solve scalar**
`W_budget` on the passivity RHS (linear — Piste A), or even (b) fuse them **live** as
`dqᵀτ_q + 2α T_kin + ‖M_λ·λ‖∞ ≤ τ_w_max` which is still **convex** (∞-norm of a linear map = max of linear
rows) — that is the SOC/linear-rows form and is the bridge to Piste C. Neither is nonconvex.

---

## Decisive output

**Piste A formulable as a linear (or SOC) QP constraint `dqᵀτ_q + 2α T_kin ≤ W_budget` with `W_budget` a
pre-solve parameter from the planned envelope margin? — YES.**

**`W_budget` source (named):** the NMPC plan's envelope margin,
```
W_budget = κ · max(0,  τ_w_max − ‖Ḣ_s(lambda_ref)‖∞ )          [optionally min'd with the h_max margin]
   where  Ḣ_s(lambda_ref) = r_CA×f1_ref + τ1_ref + r_CB×f2_ref + τ2_ref   (NMPC-exact, about origin)
```
- `lambda_ref` and `r_C{A,B}`, `r_com`, `a_com_ff` are **all parameters at the QP tick** (`sim_loop.py:2789`)
  ⇒ `W_budget` is a pre-solve scalar ⇒ the constraint is a single **linear** row in `τ_q`.
- `κ` is the **units gain [1/s]** of FLAG 1 (torque-margin → power-budget). Modeling choice; pre-solve scalar.
- Non-negativity is guaranteed by the NMPC's own path constraint (`‖Ḣ_s(plan)‖∞ ≤ τ_w_max`), matching Piste
  A's `W_budget ≥ 0`.

**Why not the QP's existing box for the margin (FLAG 2):** that box is the abandoned `|M_λ·λ|`
proxy (lever-from-robot-CoM), which at the canonical **−0.35 m standoff** differs from the true envelope by
the orbital term `r_com×Σf`. Numeric demo: with `r_com=(0.10,−0.05,0.35)`, sample `lambda_ref` →
`‖Ḣ_exact − L̇_proxy‖∞ = 0.33 N·m` (non-trivial vs `τ_w_max=5`). So derive `W_budget` from the NMPC-exact
`Ḣ_s(lambda_ref)`, not the QP proxy box.

---

## Mismatches / refinements vs the prior J2-audit facts

| prior fact | status on `ae0673e` |
|---|---|
| envelope SOC/wheel-torque caps in NMPC `~251-268` | **refine:** SOC at `:262-268`; the wheel-torque (envelope) cap `Ḣ_s` is at `:270-282` (just below 268). ✔ present |
| momentum map `compute_momentum_map` 3×12 | **CONFIRMED** (`contact_phase.py:101-137`); lever `r_Cj − r_com` (robot-CoM frame) |
| QP wrench task `A_wrench=I, b_wrench=lambda_ref` `1093-1135` | **CONFIRMED** (`:1094-1096`) — the QP *tracks* `lambda_ref` **and additionally re-enforces** the envelope box (`:525-538`) |
| passivity inequality `wholebody_qp.py:549` | **CONFIRMED** — linear in `τ_q` only (`:546-553`) |
| **NEW (not in the prior audit):** the QP has its **own** envelope box `|M_λ·λ| ≤ τ_w_max` (`:525-538`), using the **proxy** the NMPC dropped | this is FLAG 2; the envelope lives in **both** solvers, with **different** quantities (NMPC exact, QP proxy) |

**Bottom line for the formulation step:** build Piste A as the existing passivity row with RHS
`W_budget = κ·max(0, τ_w_max − ‖Ḣ_s(lambda_ref)‖∞)` — linear, `W_budget` a pre-solve parameter. Decide `κ`
(units gain) and whether to also gate on the `h_max` margin. If a *live* (post-solve-accurate) coupling is
ever wanted, the convex fused form `dqᵀτ_q + 2α T_kin + ‖M_λ·λ‖∞ ≤ τ_w_max` is available (SOC) — that is
Piste C, also QP-formulable, so the fallback is **not** forced by any nonconvexity.

## Reproducer

`scripts/audit_j2_envelope.py` — READ-ONLY. Source anchors (Q1-Q4) + a numeric demo on the real
`compute_momentum_map`: linearity of `L̇_com` in `λ`, the planned margin as a scalar, the transport identity
`Ḣ_s = L̇_com + r_com×Σf`, and the proxy-vs-exact divergence at a non-zero standoff.
```
MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/audit_j2_envelope.py
→ VERDICT: 17/17 checks confirmed.  PISTE A IS FORMULABLE (YES).  (exit 0)
```

**STOP — doc-first.** The formulation discussion (and the spec) follow the digest. No `crawlbot/` change, no
`main` write, no PR, no implementation.
