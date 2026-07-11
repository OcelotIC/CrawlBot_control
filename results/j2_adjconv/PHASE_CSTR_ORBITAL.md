# Phase CSTR-ORBITAL — exact quantity in the solver hard rate constraint ‖·‖ ≤ τ_w_max

READ-ONLY code trace for the §V rewrite. Question: does the hard rate cap bound (i) the **centroidal**
`L̇_com = Σ_j (r_Cj − r_com)×f_j + τ_j` (about the ROBOT CoM, **no** orbital), or (ii) the **about-origin**
`Ḣ_s = Σ_j r_Cj×f_j + τ_j` (about the STRUCTURE origin, orbital `r_com×Σf_j` **included**)? No `crawlbot/`
change, no run. `Ḣ_s = L̇_com + r_com×Σf_j` (the orbital term is the difference).

## HEADLINE — the two solvers DISAGREE
| solver | constrained quantity | moment arm | orbital `r_com×Σf` | case |
|---|---|---|---|---|
| **pre-planner** (`coarse_preplanner.py:342`) | `L̇_com` | `(r_C − r_com)` about **robot CoM** | **NOT included** | **(i)** |
| **NMPC** (`centroidal_nmpc.py:279`) | `Ḣ_s` | `r_Cj` about **structure origin** | **included** | **(ii)** |

The logged/plotted `Ḣ_s` (export) is case (ii) → it matches the **NMPC** constraint, **not** the pre-planner.

## Pre-planner — case (i): centroidal L̇_com, NO orbital
`crawlbot/planning/coarse_preplanner.py`. State `X = opti.variable(9, M+1)  # [r(3); v(3); L(3)]` (`:297`),
so `xk[0:3]` is the **robot CoM** `r_com`. `p_r_C` = "stance contact point in R_s" (`:303`), single contact.
Constraint formed (`:341-343`), quoted:
```python
rk = xk[0:3]
L_dot = ca.cross(p_r_C - rk, fk) + tauk
opti.subject_to(opti.bounded(-cfg.tau_w_max, L_dot, cfg.tau_w_max))
```
- Moment arm = `p_r_C - rk = (r_C − r_com)` → **about the robot CoM**. The bounded quantity is
  `(r_C − r_com)×f + τ = L̇_com` (single contact). **The orbital term `r_com×f` is absent.** ⇒ **case (i).**
- Same expression drives the state ODE (`:317`, `L_dot = ca.cross(p_r_C - r, f) + tau`), so the plan's `L`
  state is the centroidal momentum and the cap is on its rate. No other rate constraint exists in the file.

## NMPC — case (ii): about-origin Ḣ_s, orbital INCLUDED
`crawlbot/solvers/centroidal_nmpc.py`. State `x = [r_com(3), v_com(3), L_com(3)]` (`:13,159`); `r_com = x[0:3]`
is available (`:162`). Constraint formed (`:270-282`), quoted:
```python
# Wheel-torque rate cap: |Ḣ_s,i| ≤ τ_w_max
# Ḣ_s = Σ [r_Cⱼ × fⱼ + τⱼ] is the exact moment the AOCS wheels
# must counter to keep the structure stationary (Newton's 3rd
# law about structure CoM = origin in R_s). ...
# Replaces the prior |L̇_com,i| proxy, which used
# lever-from-robot-CoM and bounded only the spin-rate part
# of the robot-momentum-rate — wrong quantity at non-zero
# standoff (campaign §9 documents the divergence).
H_dot_s = (ca.cross(r_C1, f1) + tau1 +
           ca.cross(r_C2, f2) + tau2)
tw = cfg.tau_w_max
Hdot_s_ineq = ca.vertcat(H_dot_s - tw, -H_dot_s - tw)   # imposed as an SOC-block inequality (parts, :284)
```
- Moment arms = `r_C1, r_C2` (the raw contact points) → **about the structure origin** (`r_com` is NOT
  subtracted). Bounded quantity `Σ_j r_Cj×f_j + τ_j = Ḣ_s`. Since `r_Cj×f_j = (r_Cj−r_com)×f_j + r_com×f_j`,
  the **orbital term `r_com×Σf_j` is included**. ⇒ **case (ii).**
- The code comment states this explicitly and records that it **deliberately replaced** the centroidal
  `L̇_com` proxy (the pre-planner's case-(i) form) as the **"wrong quantity at non-zero standoff."**
- **Distinct from the state ODE:** the NMPC's `L_dot` at `:175-178`
  (`L̇_com = Σ (r_Cj − r_com)×f_j + τ_j`, about CoM) integrates the `L_com` **state** — it is NOT the
  constraint. Within the NMPC: state = centroidal `L_com`, hard cap = about-origin `Ḣ_s`.

## Constraint vs logging
- The realized/planned `Ḣ_s` we export & plot (`export_figure_data.py:190-191`,
  `cross(anchor_a, f_a)+τ_a + cross(anchor_b, f_b)+τ_b`, structure-frame anchors) is about the **origin**
  = case (ii). It is the **same expression** the **NMPC** hard-constrains — so for the NMPC, logged == constrained.
- For the **pre-planner**, logged (ii, about-origin) ≠ constrained (i, centroidal about-CoM). They differ by
  the orbital term `r_com×Σf_j`.

## What to flag for the paper (V-B.3 / §VIII)
1. **Paper V-B.3 writes the cap on `L̇_com` (centroidal, no orbital) = case (i).** That matches the
   **pre-planner** but **contradicts the NMPC**, whose hard cap is the about-origin `Ḣ_s` (ii). The online
   primary solver bounds the orbital-inclusive quantity; the paper currently states the orbital-free one.
2. **§VIII says the reaction is dominated by the orbital term `r_com×Σf_j`.** Consistent with the **NMPC**
   (orbital IN) and with the logged `Ḣ_s`. The **pre-planner** omits exactly this dominant term from its
   rate budget → it caps only the spin-rate part, so at non-zero standoff (`|r_com|` grows to ~0.9→2.4 m
   across the traversal, ADJ-CONV) the pre-planner's `τ_w_max` bound is **not** the true wheel torque.
3. **Solver-to-solver inconsistency (code-correctness):** pre-planner (i, no orbital) vs NMPC (ii, orbital
   in). The NMPC's own comment labels the pre-planner's form the "wrong quantity at non-zero standoff"
   (its `§9` divergence note); the pre-planner was not updated to match. This is a real code + paper↔code
   discrepancy to resolve before the §V/§VIII rewrite.

## Deliverable summary
- **Pre-planner:** `coarse_preplanner.py:342` — `L̇_com = (r_C − r_com)×f + τ`, about **robot CoM**,
  **orbital NOT in** → case (i).
- **NMPC:** `centroidal_nmpc.py:279-282` — `Ḣ_s = Σ r_Cj×f_j + τ_j`, about **structure origin**,
  **orbital IN** → case (ii).
- **constraint == logged-Ḣ_s?** NMPC: **yes** (ii == ii). Pre-planner: **no** (constrains i, logs ii).
- **They differ** — flagged above.
