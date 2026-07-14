# Phase PREPLAN-ROLE — how the NMPC consumes the pre-planner output (does the optimistic budget contaminate closed-loop Ḣ_s?)

READ-ONLY data-flow trace, `j2/ds-active-rework`. Follows CSTR-ORBITAL: the pre-planner caps the
centroidal `L̇_com` (orbital-free, optimistic at standoff, case i); the NMPC caps the about-origin
`Ḣ_s` (orbital-in, correct, case ii). Question: does the pre-planner's optimistic budget contaminate
the closed-loop `Ḣ_s` guarantee? That depends on how the NMPC consumes the pre-planner output. No
`crawlbot/` change, no run.

## Data flow (file:line)
1. **Pre-planner runs once per step** (`sim_loop.py:_run_preplanner`, ~`:1396-1470`) → `CoarsePlanResult`
   with **`T_step`** and a momentum-feasible **CoM trajectory** (`r_com_at(t)`, `v_com_at(t)`). Stored as
   `self._coarse_plan` (+ `self._coarse_plan_t0`).
2. **`T_step` is installed in the scheduler's SS phase** (`:1398`); the reference and the phase both run
   over `[t_ss_start, t_ss_start + T_step]`.
3. **Each NMPC solve** (at `dt_nmpc`), the coarse plan is **sampled at the current horizon time** and
   passed as the reference (`:2636-2657`):
   ```python
   if (self._coarse_plan is not None) and (not settle_mode):
       rp_coarse = self._coarse_plan.r_com_at(tau_rel)   # (or compressed tau_comp)
       vp_coarse = self._coarse_plan.v_com_at(tau_rel)
       cref_r = rp_coarse ; cref_v = vp_coarse
   ```
4. **NMPC call** (`:2723-2728`):
   ```python
   rp, vp, _, lr, info_n = self.nmpc.solve(
       r_com=rs.r_com, v_com=rs.v_com, L_com=rs.L_com,   # x0 = TRUE current state (rs = robot.update(live), :2666)
       r_com_ref=cref_r, v_com_ref=cref_v,               # reference = pre-planner CoM trajectory
       contact_config=cc_nmpc, warm_start=True,
       hw_current=hw_for_nmpc, L_com_ref=L_com_ref_nmpc)
   ```
   The reference enters the NMPC **cost** (`centroidal_nmpc.py:205`, `e_r = r_com - r_ref`). No `r_goal`
   / terminal-equality is passed to the NMPC (the hard terminal `X[0:3,M]==r_goal` lives only in the
   **pre-planner**, `coarse_preplanner.py:357`).

## Classification (a/b/c)
- **(b) TRACKING REFERENCE.** The pre-planner CoM trajectory is the NMPC's `r_com_ref`/`v_com_ref`,
  penalized in the cost. It biases the target.
- **NOT (a) warm-start-only:** `warm_start=True` seeds from the NMPC's **own** previous solution
  `_last_x_opt` (`centroidal_nmpc.py:136,450`; cleared at phase transitions via `reset_warm_start`,
  `sim_loop.py:2162/2210/2374`), not the coarse plan.
- **NOT (c) hard boundary:** no pre-planner terminal state / `r_goal` is imposed as an NMPC hard
  constraint; the terminal appears only as a soft `kappa_terminal` cost toward the (tracked) reference.

## Timing, horizon, cadence
- **Timing inherited: YES (soft).** `T_step` (sized by the pre-planner) sets the SS phase duration
  (`:1398`) and times the tracked reference; the scheduler ends SS at `T_step`. But `T_step` is **not**
  an NMPC hard constraint — it shapes the reference's timeline, not the NMPC dynamics.
- **Horizon << step.** `N=8`, `dt=0.1` → horizon `= 0.8 s` (`t_horizon = t + nmpc_N·nmpc_dt`, `:2620`)
  vs `T_step ≈ 2.8 s`. The NMPC re-plans a **short ~29 % receding window**, not the whole step.
- **Re-solve cadence: full receding-horizon.** `n_qp_per_nmpc = round(dt_nmpc/dt_qp) = 10` (`:75`): the
  NMPC re-solves **every `dt_nmpc = 0.1 s` from the TRUE current state** (`rs.r_com`, live), control
  piecewise-constant over each 0.1 s window (`:2769`), with 10 QP sub-steps between solves. It does
  **not** replay a fixed pre-planned trajectory open-loop; it re-solves from truth and re-samples the
  reference each solve.

## Does the pre-planner's optimistic budget contaminate the closed-loop Ḣ_s guarantee?
**NO — the `Ḣ_s` guarantee is not contaminated.**
- The pre-planner enters only as a **soft tracking reference** (b) and via **`T_step` timing** — never
  as a constraint on the NMPC's wrench or momentum-rate.
- The NMPC enforces its **own** hard cap on the **correct about-origin `Ḣ_s ≤ τ_w_max`**
  (`centroidal_nmpc.py:279-282`, case ii), independent of the pre-planner, and re-solves
  receding-horizon from the **true** state every 0.1 s.
- Hence the closed-loop `Ḣ_s` is bounded by the NMPC's correct constraint, not the pre-planner's
  optimistic centroidal budget. If the pre-planner's reference is optimistic (would require
  `Ḣ_s > τ_w_max` once the orbital term is included), the NMPC clips the wrench to the correct cap and
  **deviates from the reference** — the momentum bound holds; only tracking fidelity suffers.
  (Consistent with the C-run realized `Ḣ_s` sitting at exactly 5.00, ABL-HDOT-2.)

### CAVEAT (real, but not a guarantee breach)
Because the pre-planner IS a tracking reference (b) and `T_step` is inherited, the **target and timing**
the NMPC chases are shaped by the optimistic centroidal budget (i). An over-optimistic / inconsistent
pre-planner therefore produces a CoM reference + `T_step` that are **infeasible to track exactly** under
the correct `Ḣ_s` cap → NMPC lag → dock **position error** at the fixed `T_step` boundary (matches the
~4.8 mm-near-5 mm-gate observations), and a `T_step` sized against the wrong quantity. This is a
**tracking/timing** degradation, not an `Ḣ_s` violation.

## Verdict for the paper
- Honest framing: **the closed-loop `Ḣ_s` guarantee is owned by the NMPC's correct hard constraint**
  (case ii) with full receding-horizon re-solve from the true state — **the pre-planner's optimistic
  budget does NOT contaminate it.**
- **But** the pre-planner is a tracking **reference** (not mere warm-start) and supplies `T_step`, so the
  CSTR-ORBITAL inconsistency (pre-planner caps orbital-free `L̇_com`) still warrants **fixing the
  pre-planner** (Option a) for a consistently-sized reference/`T_step`, or a caveat that the reference is
  sized against the optimistic budget while the guarantee is enforced against the correct one.

## Deliverable summary
- **Data flow:** pre-planner `CoarsePlanResult` → `self._coarse_plan` → sampled `r_com_at`/`v_com_at`
  (`sim_loop.py:2648-2657`) → NMPC `r_com_ref`/`v_com_ref` (`:2725`); `T_step` → scheduler SS phase
  (`:1398`).
- **Classification:** **(b) tracking reference** (cost, `centroidal_nmpc.py:205`); not (a) warm-start,
  not (c) hard boundary.
- **T_step/timing inherited:** yes (phase duration + reference timeline; soft, not an NMPC hard
  constraint).
- **Re-solve/horizon:** full receding-horizon from true state every 0.1 s; horizon 0.8 s ≪ step 2.8 s.
- **Contaminates closed-loop `Ḣ_s` guarantee?** **NO** (NMPC owns the correct hard cap + receding
  horizon). Caveat: pre-planner shapes the tracked target/timing, so the CSTR-ORBITAL fix is still
  warranted for reference/T_step consistency.
