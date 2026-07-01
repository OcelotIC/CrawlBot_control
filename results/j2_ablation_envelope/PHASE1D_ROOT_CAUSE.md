# Phase-1d — root cause of the h_max=10 pre-planner failure

**Result: the failure is NOT the box value. It is the `T_step_guess` heuristic, which
inversely couples the planned step duration to `h_max`.** Enlarging the storage box
5→10 silently HALVED the planned step time (2.78 s → 1.39 s); the pre-planner cannot move
the CoM 0.195 m in 1.39 s feasibly, so IPOPT converges to a point of local infeasibility.
With the step time held at the original 2.78 s, the box=10 problem converges to the
**identical optimum** as box=5.

**This corrects Phase-1c** (`PHASE1C_LOOSE_BOX_DIAGNOSIS.md`), whose "numerically fragile
to the box VALUE" conclusion was confounded: the two real-sim runs it compared differed in
`h_max` **and** (via the heuristic below) in step duration. See "Corrections" at the end.

## The coupling
`crawlbot/simulation/sim_loop.py:1886-1890`:
```
h_max        = cfg.h_max_tight
v_max        = min|h_max| / m / lever_arm      # lever_arm = 1.0, m ≈ 71 kg
T_step_guess = max(0.5, distance / v_max)
```
`v_max ∝ h_max`, so `T_step_guess ∝ 1/h_max`. h_max 5→10 ⇒ v_max 0.070→0.141 m/s ⇒
T_step 2.78→1.39 s. The box value never enters the *solution* (it doesn't bind — Task B);
it enters only through this step-time heuristic.

## Task A + landscape matrix — step-0 pre-planner, factors separated
Fresh opti each cell (matches the sim's step-0: first solve on its opti). Params set exactly
as `solve()` (coarse_preplanner.py:471-489); only box / T_step / initial-guess vary.
`x*_5` = the converged h_max=5 primal. `T5=2.775 s` (sim's h_max=5 guess), `T10=1.388 s`
(sim's h_max=10 guess), CoM distance 0.195 m.

| # | box | T_step | initial guess | success | cost | h_w peak | iters | IPOPT EXIT |
|---|---|---|---|---|---|---|---|---|
| 1 | 5 | T5 | default | ✅ | 11.205 | 4.310 | 12 | Optimal Solution Found |
| **2** | **10** | **T5** | **default** | **✅** | **11.205** | **4.310** | **12** | **Optimal Solution Found** |
| 3 | 10 | T10 | default | ❌ | 194.914 | 2.579 | 133 | Converged to a point of local infeasibility |
| **4** | **10** | **T5** | **warm x*_5** | **✅** | **11.205** | 4.310 | 11 | Optimal Solution Found |
| 5 | 10 | T10 | warm x*_5 | ❌ | 194.460 | 2.579 | 92 | Converged to a point of local infeasibility |

- **Task A (row 4):** the h_max=10 problem **warm-started from x*_5 converges to 11.205** —
  the same optimum. So the box=10 feasible set contains that optimum and it is reachable.
- **The clincher (rows 1 vs 2):** with the *default* guess and the step time held at T5,
  **box=10 converges to the identical optimum (11.205, h_w 4.310)** as box=5, `Optimal
  Solution Found`. The box value is irrelevant to convergence.
- **The cause (rows 2 vs 3):** the *only* difference is T_step (2.78 vs 1.39 s). Row 3 (the
  short step) fails. Warm-start does not rescue the short step (row 5 still fails) — T10 is a
  genuinely harder problem, not a basin/init artifact.

### Verbatim IPOPT EXIT (real solve options, `landscape_probe.log`)
```
# row 2  box=10, T5, default:
Number of Iterations....: 12
Objective...............:   1.1205447604044167e+01
Constraint violation....:   4.8677312181055754e-12
Variable bound violation:   0.0000000000000000e+00
EXIT: Optimal Solution Found.

# row 3  box=10, T10, default:
Number of Iterations....: 133
Objective...............:   1.9491405224373173e+02
Constraint violation....:   9.1339744365410480e-02
Variable bound violation:   0.0000000000000000e+00      <-- the ±10 box is never touched
EXIT: Converged to a point of local infeasibility. Problem may be infeasible.
```

## Task B — is the box near binding at any step? (committed C, zero new solve)
Per-step REALIZED h_w peak-∞ per axis, `results/j2_canonical_revalidation/runfix_traversal.csv`:

| step | \|hw_x\| | \|hw_y\| | \|hw_z\| | peak-∞ | margin to 5 |
|---|---|---|---|---|---|
| 0 | 0.580 | 1.586 | 2.359 | 2.359 | 2.641 |
| 1 | 0.148 | 1.957 | 2.943 | 2.943 | 2.057 |
| 2 | 0.658 | 2.116 | 4.004 | 4.004 | 0.996 |
| 3 | 0.175 | 2.196 | 3.008 | 3.008 | 1.992 |
| **4** | 0.711 | 2.293 | **4.885** | **4.885** | **0.115** |
| 5 | 0.222 | 2.356 | 3.027 | 3.027 | 1.973 |

Global realized peak **4.885** (step 4, z), margin **0.115** to the box. The box is
**interior at every step** — it never binds. (Step-0 *planned* peak from the probe = 4.310,
also interior.) So the storage box shapes no step's solution; it is globally slack. This is
consistent with the matrix: since the box doesn't bind, enlarging it changes nothing about
the solution (only the heuristic's step-time output changes).

## Task C — landscape multi-modality at the nominal box (box=5, T5)
Perturb the default guess by ±5 % (3 seeds), re-solve step 0:

| seed | success | cost | h_w peak | iters |
|---|---|---|---|---|
| 1 | ✅ | 11.205 | 4.310 | 12 |
| 2 | ✅ | 11.205 | 4.310 | 12 |
| 3 | ✅ | 11.205 | 4.310 | 12 |

All reach the same optimum (11.205) in 12 iters. At the correct step time the NLP is
**unimodal and robust** — not a fragile multi-basin landscape. (The fragility only appears
when the step time is squeezed to T10, an infeasibly-short duration.)

## Bottom line
The pre-planner is well-conditioned at the nominal operating point (box=5, T5): robust to
±5 % guess perturbations and to enlarging the box to 10. The h_max=10 failure is an artifact
of the `T_step_guess` heuristic silently halving the step time when the box is enlarged, not
a property of the box or the optimization landscape. **The storage box is globally slack
(never binds, margin ≥ 0.115 Nms); it is not physically necessary and does not shape the
plan.** No physical story; no fix applied.

## Corrections to Phase-1c (honesty standard)
1. **"Numerically fragile to the box VALUE" — WRONG (confounded).** The box value is
   irrelevant to convergence (rows 1↔2, identical optimum). The confound was the T_step
   coupling; Phase-1c's two real runs differed in step duration, not just h_max.
2. **"Sweep failed at h_max=5 because it lacked the sim warm-start" — WRONG.** The Phase-1c
   sweep captured `kw` from the h_max=10 run, so `kw['T_step'] = T10 = 1.39 s`; every sweep
   solve (including the h_max=5 positive control) used T10 and failed for the *same* T_step
   reason, not a missing warm-start. The default guess it used is the sim's own guess.

## Artifacts
| file | content |
|---|---|
| `scripts/diag_landscape_probe.py` | the (box × T_step × guess) matrix + Task C, on fresh opti |
| `landscape_probe.json` | machine-readable matrix + Task C + x*_5 |
| `landscape_probe.log` | full run incl. verbatim IPOPT EXIT for rows 2 & 3 |
| `taskB_perstep_hw.json` | per-step realized h_w peak/axis vs the 5 box |

No `crawlbot/` or MJCF change. No paper text (Phase 2). No fix applied (the T_step_guess
heuristic coupling is documented, not modified). Task 3 (soft `w·‖h_w‖²` counterfactual)
remains gated.
