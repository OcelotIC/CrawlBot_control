# Task 2 — storage-box removal: factual pre-planner-failure diagnosis

**Question.** With the storage constraint removed (`enforce_hw_conservation=False` + `h_max_tight=1e6`,
rate cap `tau_w_max=5` kept — this is DIAG B), the coarse pre-planner's IPOPT solve fails. Classify the
failure: **(a) unbounded** (objective / iterates diverging) vs **(b) local-solver / restoration failure**
(iterates bounded, solver stuck).

**Method (zero `crawlbot/` change).** `scripts/diag_preplanner_storage.py` monkeypatches
`CoarsePrePlanner.solve` (class attribute) to, on the first failing storage-off solve, dump the last-iterate
trajectory, then re-solve the *same* NLP with `ipopt_print_level=5, max_iter=1000` to read IPOPT's own EXIT
status. It reuses the full canonical config via `runpy` + `--envelope-constraint storage-off --n-steps 1`.
Full transcript: `results/j2_ablation_envelope/task2_storage_diag.log`.

## Result (fresh run, this commit)

```
FIRST solve (as run by the sim):
  success=False   cost=122.6133   h_max(in)=1e+06
  |h_w|inf per-knot max = 0.7739 Nms      (box was ±1e6 — effectively removed)
  |r_com|max=0.683  |v_com|max=0.089  |L_com|max=1.705
  h_w inf-norm per knot: [0.0, 0.19, 0.371, 0.535, 0.676, 0.703, 0.723, 0.774,
                          0.744, 0.66, 0.548, 0.489, 0.453, 0.408, 0.356, 0.297]

RE-SOLVE (verbose IPOPT):
  Number of Iterations....: 87   (restoration iters 76r–87r)
  Objective...............: 1.2261e+02
  Constraint violation....: 1.5383e-01
  Variable bound violation: 0.0000e+00        <-- h_w never touched a bound
  EXIT: Converged to a point of local infeasibility. Problem may be infeasible.
  RE-SOLVE last-iterate |h_w|inf max = 0.7739   |L_com|max=1.705
```

## Classification: **(b) local-solver / restoration failure** — NOT unbounded, NOT physically infeasible

Evidence, all from the run above:

1. **Iterates are bounded.** Objective = 122.6 (finite), `|h_w|∞` = 0.774 Nms, `|L_com|` = 1.705,
   `|r_com|` = 0.683, `|v_com|` = 0.089. Nothing diverges. An unbounded problem would show the objective
   and/or `h_w` marching to ∞; here the last iterate is small and physical. ⇒ **rule out (a)**.
2. **The removed box was never the binding issue.** With the box at ±1e6 the last-iterate `|h_w|∞` is 0.774
   — four orders of magnitude inside the (removed) box — and **Variable bound violation = 0.0**. The storage
   box being open did not "let `h_w` run away"; `h_w` stayed ~0.77.
3. **IPOPT states it explicitly.** EXIT = *"Converged to a point of local infeasibility."* This is IPOPT's
   restoration phase (the `…r` iterations) converging to a stationary point of the constraint-violation
   measure with a small residual (`Constraint violation = 0.154`), i.e. the solver could not walk from its
   initialization to a feasible point — a **numerical/formulation** outcome, not a statement that no feasible
   point exists.

## Consequence for the paper

Removing a constraint enlarges the feasible set and **cannot** make a feasible problem infeasible. The
storage-off IPOPT failure is therefore a solver/formulation artifact (restoration failure from the current
warm-start + scaling), **not** evidence that the storage constraint is physically necessary. Accordingly the
storage-box blocker is **not reported as an ablation result.** The counterfactual that *would* be a result —
does `h_w` actually grow past ±5 when it is genuinely unconstrained — requires a formulation that stays
solvable without the hard box (e.g. a soft `w·‖h_w‖²` penalty replacing it). That is **Task 3**, which is a
`crawlbot/` change and is **gated on explicit approval** (not implemented here).

The clean, single-variable envelope result that *does* stand is the **rate-cap** ablation (Task 1): the box
was left untouched, only `tau_w_max` was raised, the pre-planner solved all 6 steps, and the planned wheel-rate
demand exceeded the 5 Nm cap (peak 6.27 Nm, 12.2% of SS ticks). See `task1_key_numbers.md`.
