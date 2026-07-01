# Phase-1e — momentum-budget (h_max) vs locomotion-speed frontier

**The two envelope constraints are coupled through speed.** The sim sets step duration from
the storage budget: `T_step = distance / (min|h_max|/m/lever)` (sim_loop.py:1888). A larger
`h_max` ⇒ shorter `T_step` ⇒ faster locomotion ⇒ higher required transfer rate
`L̇ = (r_C−r)×f + τ`, which the pre-planner caps at ±`τ_w,max` = 5 (coarse_preplanner.py:342).
So the **storage budget buys speed, and speed loads the rate cap.** This maps the trade-off.

**Headline:**
- **Plan-feasible** for `h_max ∈ {5,6,7}` (all 6 steps); **infeasible for `h_max ≥ 8`** — the
  short steps cannot reach their terminal within the rate (L̇≤5) *and* force (f≤25) limits in the
  shortened `T_step`. The failure is **physical, not solver** (Task B).
- **Realized-dockable only at `h_max = 5`.** At `h_max = 6, 7` a feasible plan exists but the
  closed loop **misses the 5 mm dock gate** (step-0 dock times out at 10.2 / 14.2 mm) and the
  traversal aborts. Canonical `h_max = 5` sits right at the realized frontier.

Method: capture the 6 steps' boundary conditions from the canonical `h_max=5` traversal, then
for each `h_max` recompute the per-step `T_step` from the real heuristic and re-solve each
step's pre-planner (fresh opti). Realized rows are full closed-loop sims. ZERO `crawlbot/`
change (monkeypatch of `_make_m7_config` only). Data: `hmax_frontier.json`,
`hmax_frontier_summary.log`, `scripts/diag_hmax_frontier.py`, `scripts/diag_run_hmax.py`.

## PLANNED frontier (pre-planner, per h_max)
Per-step CoM distances [m]: [0.195, 0.591, 0.202, 0.561, 0.215, 0.429] (steps 0–5).

| h_max | steps converged | min T_step (s) | planned SS speed† (m/s) | planned L̇ peak (cap 5) | planned h_w peak (box) |
|---|---|---|---|---|---|
| **5** | **6/6** | 2.775 | 0.0704 | 5.000 | 5.000 |
| **6** | **6/6** | 2.313 | 0.0844 | 5.000 | 6.000 |
| **7** | **6/6** | 1.982 | 0.0985 | 5.000 | 7.000 |
| 8 | 3/6 (fail 0,2,4) | 1.734 | — | — | — |
| 10 | 3/6 (fail 0,2,4) | 1.387 | — | — | — |
| 12 | 3/6 (fail 0,2,4) | 1.156 | — | — | — |

† planned SS-transit speed = Σdistance / Σ`T_step` (excludes DS settle). Realized full-traversal
speed at h_max=5 is 0.0376 m/s incl. DS (2.193 m / 58.3 s).

### Two step-regimes (per-step planned L̇ vs 5, h_w vs box)
- **Short steps 0,2,4 (~0.20 m): RATE-limited.** Their `L̇` climbs to the ±5 cap as `T_step`
  shrinks. At `h_max=5`: s0 L̇=0.70, s2 L̇=2.67, **s4 L̇=5.00** (already at the cap). By `h_max=6`
  **all three** are at L̇=5.00. At `h_max≥8` their `T_step` (≤1.73 s) is too short to reach the
  terminal within L̇≤5 → infeasible.
- **Long steps 1,3,5 (~0.5 m): STORAGE-limited.** Ample time (`T_step` 4–8 s), so `L̇≈0.01`; the
  planned `h_w` rises to **exactly fill the box** (peak = h_max at every h_max). They converge even
  at h_max=12. The storage box is an **active** constraint in the plan for these steps (planned
  h_w peak = h_max), refining Phase-1d's realized-only "interior" note (realized peak 4.885 < 5).

**Co-binding at canonical h_max=5:** step 4 runs at **both** L̇=5.00 (rate cap) **and** h_w=5.00
(storage box) simultaneously — the two envelope constraints meet at the operating point.

## REALIZED (full closed-loop sims)
| h_max | outcome | dock (mm) | τ_w saturation | θ_s peak | notes |
|---|---|---|---|---|---|
| **5** | **6/6 DOCK** | [4.94,4.41,4.88,4.42,4.76,4.92] | 3.7% | 0.591° | full traversal (committed C), 0.0376 m/s |
| 6 | **ABORT @ step 0** | 10.23 (TIMEOUT) | 12.3% (step 0) | 0.165° (step 0) | NMPC fails 10/114; `stop_on_failed_step` |
| 7 | **ABORT @ step 0** | 14.22 (TIMEOUT) | 12.7% (step 0) | 0.160° (step 0) | NMPC fails 10/110; `stop_on_failed_step` |

At `h_max=6,7` the plan is feasible but the **shorter SS window + faster swing** leave the EE at
10–14 mm when the window closes — over the 5 mm gate — so the step TIMES OUT and the traversal
aborts after step 0. τ_w saturation jumps 3.7% → 12+% and the NMPC fails 10×. (Realized rows
6,7 are step-0-only; the traversal never reaches steps 1–5.)

## Task B — classify the first pre-planner failure (h_max=8, step 0)
`T_step = 1.734 s`, CoM distance 0.195 m.
- **Warm-start from the nearest working budget (h_max=7 x*): still FAILS** (cost 231.1). Not solver.
- **Failed iterate: both actuator limits saturated** — |L̇|peak = 5.0 (= rate cap 5.0) **and**
  |f|peak = 25.0 (= f_max 25.0) — yet the terminal cannot be met: r_err = 0 m but
  **v = 0.00707 > ε_v (0.005)** and **L = 0.0517 > ε_L (0.05)**. No plan reaches the terminal
  (zero residual v, L) within the rate+force limits in 1.73 s.
- Verbatim IPOPT (default-guess verbose, `hmax_frontier_summary.log`):
  ```
  Number of Iterations....: 98
  Objective...............:   2.3280578540713776e+02
  Constraint violation....:   3.5294193163717846e-02
  Variable bound violation:   0.0000000000000000e+00
  EXIT: Converged to a point of local infeasibility. Problem may be infeasible.
  ```
- **Classification: (b) genuinely INFEASIBLE at this speed** — the binding/violated constraints are
  the **rate cap and the force limit** (both saturated), with the terminal velocity/momentum
  residual left unmet. This is a physical frontier, not a numerical wall.

## The crossover / what the constraints buy
- The **rate cap (5)** first binds the short steps at **h_max ≈ 6** (all of 0,2,4 → L̇=5.00; at
  h_max=5 only step 4 is at 5, steps 0,2 still have rate margin).
- The **storage box** binds in the *plan* from h_max=5 (planned h_w peak = box at the
  storage-limited steps 2,4,5).
- **Plan-feasibility wall: h_max = 8** (short steps infeasible — rate+force saturate, terminal unmet).
- **Realized dock wall: h_max = 6** (closed loop can't dock within 5 mm at the shortened window).
- So the storage budget of 5, the rate cap of 5, and the 5 mm dock tolerance are **jointly** at
  their limits at the canonical operating point: raising the budget makes the plan nominally faster
  (feasible to h_max=7) but the closed loop cannot dock beyond h_max=5.

## Caveat / scope
Canonical results (h_max=5) are **unaffected** — this sweep is exploratory, characterizing what the
constraint buys. The "is this a genuine Pareto frontier or a tunable dock/gain wall" judgement is
reserved for cross-check (you + Idriss); this report is factual measurement only. No `crawlbot/`
change, no fix, no paper text (Phase 2). Task 3 (soft `w·‖h_w‖²` counterfactual) remains gated.

## Artifacts (`results/j2_ablation_envelope/`)
| file | content |
|---|---|
| `scripts/diag_hmax_frontier.py` | planned frontier grid (h_max × step) + Task-B classification |
| `scripts/diag_run_hmax.py` | full closed-loop run at a chosen h_max (realized rows) |
| `hmax_frontier.json` | machine-readable grid + Task B |
| `hmax_frontier_summary.log` | frontier table + verbatim IPOPT EXIT (Task B) |
Raw sim dumps (`_frontier_cap/`, `figC_hmax6/`, `figC_hmax7/`) gitignored (regenerable).
