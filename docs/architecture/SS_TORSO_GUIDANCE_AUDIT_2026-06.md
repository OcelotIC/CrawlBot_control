# SS Torso-Guidance Chain Audit + Arrival-vs-Target (read-only)

Branch `feat/ss-centroidal-momentum-task` @ `027c25c`. Read-only audit; one new plot
(`results/phase2_1_report/arrival_vs_target.png`) from EXISTING Phase-2.1 data (no re-run).
Answers: **who decides where the torso must be at end-of-SS, and does holding the CoM
reference mechanically bring the torso there?**

## Part A — the guidance chain (code evidence)

**1. TorsoPlanner target.** The end-of-SS torso target is `p_t1, R_t1` = the **FK torso pose
at the IK dock configuration `q_end`** at the *next* anchor pair:
`q_end = self.torso_map.get((end_a,end_b))` / `dock_configuration(model, se3_a, se3_b)`
(`sim_loop.py:1299-1301`); `p_t1 = rs_e.oMf_torso.translation` (`:1305-1306`); installed via
`torso_planner.add_phase(t0, t0+T_step, p_t0,R_t0, p_t1,R_t1, delta_com_start=δ0,
delta_com_end=δ1, …)` (`:1545-1548`). So the torso target is driven by the **next-foothold
IK** (both EEs at their anchors), not a heuristic.
- **κ values (paper vs code):** the paper's κ_f=0.70 is **`preplanner_kappa=0.7`** — the *CoM
  pre-planner* terminal-margin multiplier (`config.py:190`, set `run_m7_single_step.py:60`),
  **not** a fractional scaling of the torso target (p_t1 is the *full* FK torso at q_end).
  `kappa_terminal=1.0` (`config.py:177`). κ_d≈0.20 ↔ the planner ramp fraction
  `ramp=0.20` (`torso_planner.py:575`). `reference_source='task_space'` default
  (`config.py:392`; `_make_m7_config` does not override) ⇒ torso ref is the quintic p_t0→p_t1.

**2. r_com* origin — DERIVED from the torso target, not independent.**
`r_com*(t) = p_torso(t) + R_torso(t)·δ_com(t)` (`torso_planner.py:450-454`,
`com_reference_at`), where δ_com interpolates δ0→δ1 and **δ1 = R_t1ᵀ(r_com1 − p_t1)** is the
CoM-torso offset *at the IK config q_end* (`sim_loop.py:1308`; δ0 at `:1221`). The seam:
`cref = self.torso_planner.com_reference_at(t_horizon)` (`sim_loop.py:2331`) →
`self.nmpc.solve(r_com_ref=cref_r, v_com_ref=cref_v, …)` (`:2433-2435`). **The torso target
enters r_com* directly** (r_com* end = p_t1 + R_t1·δ1).

**3. The stance-chain link — there is NONE explicit; the torso is a passive outcome.**
The only mechanical coupling is the **stance-EE weld** (the stance tool frame is held at its
anchor by the contact equality `J_c q̈ = −J̇_c q̇`) plus whatever tasks remain. There is **no
term expressing "the stance arm reconfigures to bring the torso toward the next foothold."**
In **Variant A** (`ss_alpha_tl_weak=0`) the torso-linear P2 task is *removed* — T-MOM replaces
it (`wholebody_qp.py:851-866`) — so the torso linear DOF is driven only by **CoM task + weld +
swing-EE task + posture**. Algebraically: realized torso = `r_com* − R·δ_actual`, so it lands
on p_t1 **iff** the realized arm config reproduces the planned offset (`δ_actual → δ1`). Nothing
in Variant A pins δ_actual to δ1 except indirectly (swing-EE task pins the swing arm; the weld
makes the stance arm follow). The baseline instead has the explicit torso-linear task (via the
CoMToTorsoMapping reference) that *does* pin the torso to p_torso_ref.

**4. r_com* ↔ torso-target offset (sampled, step 1).** offset = `r_com* − p_torso_ref` =
`R·δ_com` ≈ `[−0.04, 0.21, 0.00] m`, |offset| 0.21–0.28 m, smooth across the step (y-dominated
≈ the arm-mass CoM offset). Consistent **by construction** (r_com* = torso_ref + planned δ).
So the *plan* is self-consistent; the question is whether the *realized* torso follows.

## Part B — arrival vs target (`arrival_vs_target.png`, 3 axes, step-1 SS)

End-of-SS realized torso vs `p_t1` (TorsoPlanner final target ≈ where step 2 expects the
torso; proxy = baseline torso-ref end, per brief — step-2 IK start not cleanly recoverable
from single-step data):

| run | \|torso_end − p_t1\| | per-axis [mm] |
|---|---|---|
| **baseline (OFF)** | **1.69 mm (on-target)** | [−1.6, 0.0, 0.4] |
| A@500 (failed) | 195 mm | [176, 75, 43] |
| **A@5000** | **35.5 mm** | [21.8, 27.4, −5.5] |
| A@30000 | 69.4 mm | [59, 36, −3] |
| B@5000 | 36.8 mm | [24, 28, −5] |
| B@30000 | 70.0 mm | [60, 36, −2] |

**What the plot shows:** the baseline (grey dashed) tracks the commanded torso reference
(red dotted) and lands on p_t1 (green, **1.69 mm**). **A@5000 (blue) docks the swing EE
(2.79 mm) but its torso arrives 35.5 mm displaced** from p_t1 (forward +22 mm x, lateral
+27 mm y) — a *different redundant whole-body config* that puts both EEs on their anchors
with the torso off-target. The displacement **grows with α_mom** (A@30000 69 mm), and the
weak torso-linear regulariser does **not** rescue it at the working weight (B@5000 ≈ A@5000).

## Verdict — REAL GAP (not benign)

Holding the CoM reference does **not** mechanically bring the torso to its planned pose under
T-MOM. r_com* is correctly derived from the torso target, but tracking CoM only fixes the
mass-weighted average; the torso's pose is left to the unconstrained arm-config redundancy,
which Variant A no longer pins (it removed the torso-linear task that did so in the baseline).
Result: the torso arrives **35–70 mm displaced** (vs baseline **1.69 mm**), worsening with
α_mom. The single step still docks, but the torso is not steered where the next step's IK
assumes it starts.

**Implication for Phase 3 (do not decide here):** the torso excursion is a real guidance gap,
not just a tuning/Variant-B amplitude question. Before/at Phase 3 either (a) restore an
explicit torso-position task alongside T-MOM (a strong torso-linear co-task, i.e. Variant B
with `ss_alpha_tl_weak ≫ 50`, since 50 is too weak vs α_mom≈5000), or (b) verify over the
5-step traversal whether the per-step displacement accumulates/drifts the IK start — if it
does, the CoM-reference/task-hierarchy needs rework before the gate. The data here does not
by itself kill Variant A (step 1 docks) but flags this as the item to resolve.
