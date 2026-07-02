# Phase 1h — the TRUE cause of the storage-off / loose-box failure: FORCE-vs-time, not box/rate/conditioning

Factor isolation via five controlled step-0 pre-planner solves (fresh opti, default guess =
the sim's guess, verbose IPOPT). ZERO `crawlbot/` change (monkeypatch only). Step-0 boundary
conditions captured from a canonical h_max=5 run: CoM distance **0.195 m**, T5=**2.775 s**,
T_short=**0.500 s** (= sim heuristic floor for a loose box), M=15, m=71.06, f_max=**25 N**,
ε_v=0.005, ε_L=0.05. Data: `factor_isolation_1h.json`, `factor_isolation_1h_summary.log`,
`scripts/diag_1h_factor_isolation.py`.

## R1–R5 (verbatim IPOPT EXIT each)
| run | box | T_step | rate cap | success | cost | \|h_w\|pk | \|L\|pk | term r_err (m) | \|v_N\| | \|L_N\| | \|Ḣ_s\|pk | **\|f\|pk** | IPOPT EXIT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **R1** | 5 | 2.775 | 5 | ✅ | 11.205 | 4.310 | 0.134 | 0.00000 | 0.00815 | 0.00622 | 0.701 | 9.19 | **Optimal Solution Found** |
| **R2** | **100** | 2.775 | 5 | ✅ | 11.205 | 4.310 | 0.134 | 0.00000 | 0.00815 | 0.00622 | 0.701 | 9.19 | **Optimal Solution Found** |
| **R3** | 100 | **0.500** | 5 | ❌ | 122.617 | 0.774 | 1.705 | 0.02759 | 0.00866 | 0.06332 | 5.000 | **25.00** | Converged to a point of local infeasibility |
| **R4** | 100 | 0.500 | **1e6** | ❌ | 478.981 | 3.311 | 7.480 | 0.03058 | 0.00722 | 0.06401 | **32.485** | **25.00** | Converged to a point of local infeasibility |
| **R5** | **5** | 0.500 | 5 | ❌ | 122.492 | 0.774 | 1.721 | 0.02574 | 0.00866 | 0.07883 | 5.000 | **25.00** | Converged to a point of local infeasibility |

(term ε_v = 0.005, ε_L = 0.05 — the terminal soft bounds; the terminal r is a HARD equality
`X[0:3,M]==r_goal`, coarse_preplanner.py:357, so term r_err > 0 means that constraint is violated.)

## Interpretation matrix — resolved
- **R2 SUCCEEDS** (box 5→100 at fixed T5): cost **11.205, identical to R1**, `Optimal Solution
  Found`. **⇒ the box value is IRRELEVANT.** Kills "loose box fails" and "storage box regularizes
  the pre-planner." (Consistent with Phase 1d.)
- **R5 FAILS** (box=5, short T): the short step is infeasible **even with the tight canonical box**.
  **⇒ the storage box is NOT load-bearing / "mortal" — it does not make the steps hold.** The short
  T_step does.
- **R4 FAILS** (box=100, short T, rate cap OFF): removing the rate cap does **not** rescue it. With
  the cap off the plan freely uses **\|Ḣ_s\| = 32.5** (≫5, the plan the cap was forbidding), yet
  **\|f\| still pins at 25.0 = f_max** and the hard terminal r_goal is still unreachable (r_err
  0.031 m, \|L_N\| 0.064 > ε_L). **⇒ the failure is NOT rate-cap-vs-time either.**

## Root cause — FORCE-vs-time terminal infeasibility
In **every** short-T failure (R3, R4, R5) the **contact-force limit is saturated: \|f\|pk = 25.0 =
`preplanner_f_max`** (config.py:254), and the hard terminal position `r(M)=r_goal` is violated
(r_err ≈ 0.026–0.031 m). The rate cap and the box are red herrings; the **force limit** is the
binding actuator constraint.

Physical check: `v̇ = f/m`, so f≤25 N ⇒ a_max = 25/71 ≈ **0.35 m/s²**. In a 0.5 s window with an
accelerate-then-decelerate profile the CoM can move at most a_max·(T/2)² ≈ 0.35·0.0625 ≈ **0.022 m**
— but the step needs **0.195 m** (≈ 9× further). The 0.195 m transfer is **physically impossible in
0.5 s within f ≤ 25 N**, so the pre-planner is genuinely infeasible.

The 0.5 s comes from the sim's `T_step_guess ∝ 1/h_max` heuristic (sim_loop.py:1888): a loose box
drives T_step to its 0.5 s floor, which is far below the force-feasible traversal time for a 0.195 m
step. **So the storage-off / loose-box "failure" is a T_step-heuristic-induced force-vs-time terminal
infeasibility — not the storage box, not the rate cap, not conditioning, not an unbounded state.**
Clean, non-mortal, and localized to the heuristic + force limit.

## Addendum — does an EXTENDED execution window (T_max) fix the h_max=6 closed-loop dock miss?
h_max=6 with `t_hold_max` raised 3 → **200 s** (execution-side lever only; the plan is feasible).
`tmax_ee_gap.json`, `scripts/diag_1h_tmax.py`; raw dir gitignored.

| quantity | value |
|---|---|
| hold span | **205 s** (2.5 → 207.5 s), 2051 dock evals |
| EE-gap first / **global min** / end | 8.01 / **6.80** (at t=3.3 s) / 10.27 mm |
| last-20% of hold: mean / std | **10.27 mm / 0.000 mm** |
| docked? | **NO** (fired=0) |

The EE dips to **6.80 mm** early (t=3.3 s, swing→hold transition), then relaxes to a **perfectly flat
steady-state plateau at 10.27 mm** (std 0.000 over 185 s) and never approaches 5 mm. **⇒ the h_max=6
dock miss is a STEADY-STATE tracking/settling error, NOT an execution-window limit. T_max does NOT
help** — 25× more hold time changes nothing; the controller converges to a fixed 10.27 mm offset.

## Two failure types — what T_max addresses
- **Type 2 (planning): R3–R5 pre-planner infeasibility (force-vs-time).** T_max is execution-side and
  **cannot** touch a solve that returns infeasible at plan time.
- **Type 1 (execution): h_max=6 closed-loop dock miss.** Here it is a **steady-state settling offset**,
  so T_max **also does not help** (the EE plateaus above 5 mm regardless of time).

Neither failure in this study is an execution-window (time-starvation) limit; T_max fixes neither.

## Caveat
Canonical h_max=5 is FINAL and unaffected. No fix attempted (a real fix would floor T_step at the
force-feasible minimum / decouple it from h_max — planning; and/or address the ~10 mm steady-state EE
offset — controller). The publish framing and any fix are cross-check (you + Idriss). No paper text.
Task 3 gated. Raw dumps (`_1h_factor/` [not written — solve exits early], `figC_hmax6_tmax/`) gitignored.
