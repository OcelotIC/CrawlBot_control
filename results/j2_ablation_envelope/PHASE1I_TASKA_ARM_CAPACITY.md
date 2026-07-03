# Phase 1i Task A — the true binding contact-force bound (READ-ONLY trace)

`preplanner_f_max = 25 N` is the Phase-1h root cause of the short-T planning infeasibility.
This traces its true physical value: **min(HOTDOCK, stance-arm along-step capacity, non-slip)**.
No sim run — static kinematics on the committed URDF + the 6 committed docked configs.

## (1) Origin + consumers of preplanner_f_max
- **Bare hardcoded constant, no derivation:** `crawlbot/simulation/config.py:254`
  `preplanner_f_max: float = 25.0  # [N] per active contact (also used by F-SAT clamp)`.
- **Only functional consumer = the pre-planner wrench box.** `sim_loop.py:411`
  (`f_max=cfg.preplanner_f_max` → `CoarsePrePlannerConfig`) → `coarse_preplanner.py:336`
  `opti.subject_to(opti.bounded(-cfg.f_max, fk, cfg.f_max))`.
- **The "also used by F-SAT clamp" comment is STALE.** The current F-SAT rate clamp uses
  `cfg.fsat_jitter_margin` and the planned torso-ref velocity (`sim_loop.py:2914-2917`); the
  comment at `sim_loop.py:2909` explicitly records that the OLD `(f_max/m_b)·dt²·2` cap was
  **replaced**. So `preplanner_f_max` no longer feeds any clamp. Other sites merely re-assign
  the same 25.0 (`run_m7_single_step.py:61`, `diag_cooperative_arms.py:272`).
- ⇒ **Changing preplanner_f_max affects ONLY the pre-planner wrench box.** No unrelated clamp breaks.

## (2) HOTDOCK interface ceiling
3.0 kN / 300 Nm (fully mated, tested — given). Upper ceiling only; far above the arm term.

## (3) Stance-arm contact-force capacity — CONFIGURATION-DEPENDENT
Static relation: to hold contact force `f` at the stance tool, the stance arm carries
`τ = J_lin^T f`, so `|(J_lin^T f)_i| ≤ τ_max` per joint. `J_lin` = 3×7 linear part of the
stance-arm tool Jacobian (`pin.computeFrameJacobian`, LOCAL_WORLD_ALIGNED) at the docked config.
Arm joint-torque limit: **MJCF ctrlrange ±50 Nm** (`VISPA_crawling_rwa3.xml:304-318`, all 14 arm
joints); SimConfig software cap **20 Nm** (tighter operational limit).
`f_along(d) = τ_max / max_i|(J_lin^T d)_i|`; along-step direction d = per-step torso displacement.

| step | stance | along-step dir | **f_along @50 Nm** | f_along @20 Nm | f_worst @50 | σ_min | cond |
|---|---|---|---|---|---|---|---|
| 0 | a | [0.90, 0.44, 0.07] | **62.9** | 25.2 | 46.7 | 0.387 | 3.1 |
| 1 | b | [1.00, −0.02, 0] | 74.3 | 29.7 | 46.9 | 0.400 | 3.1 |
| 2 | a | [0.85, 0.52, 0.06] | 63.1 | 25.2 | 47.3 | 0.374 | 3.2 |
| 3 | b | [1.00, 0.02, 0.01] | 70.6 | 28.2 | 50.9 | 0.417 | 2.8 |
| 4 | a | [0.87, 0.49, 0.03] | 67.0 | 26.8 | 49.4 | 0.373 | 3.0 |
| 5 | b | [0.99, 0.10, 0.04] | 69.4 | 27.8 | 55.7 | 0.401 | 2.7 |

- **min f_along across the 6 configs = 62.9 N @ τ=50 (physical); 25.2 N @ τ=20 (software).**
  Spread small (62.9–74.3 N). f_worst (weakest direction) = 46.7–55.7 N.
- **Not near a singularity:** σ_min ≈ 0.37–0.42, condition number ≈ 2.7–3.2 (well-conditioned).
  No direction where the capacity collapses regardless of τ.
- **The hardcoded 25 N ≈ the arm along-step capacity at the SOFTWARE torque cap (25.2 N).** So 25
  was (implicitly) the software-limited arm reach; the PHYSICAL actuator capacity (±50 Nm) is
  **62.9 N — 2.5× higher.**

## (4) Non-slip limit at the gripper anchor
**ABSENT.** The grip is a MuJoCo `<weld>` equality constraint (`VISPA_crawling_rwa3.xml:326-341`,
`solref="0.003 1"`) — a hard bilateral constraint with **no force/slip limit** (no friction cone,
no `<contact>` with `condim`/`friction` at the grip; `<contact/>` is empty). ⇒ non-slip = ∞ (not modeled).

## (5) Binding bound
`min( HOTDOCK 3000, arm along-step 62.9, non-slip ∞ ) = 62.9 N` → **the ARM binds** (physical),
or **25.2 N** at the software torque cap. NOT 3000.

## Proposed replacement + physical source
**`preplanner_f_max = 62.9 N`** — the stance-arm along-step force capacity at the MJCF physical
actuator limit (±50 Nm ctrlrange), taken as the **min over the 6 docked configs** (step 0, the
conservative binding config). Finite, physically traced, NOT inflated. Alternatives for Idriss:
- **46.7 N** — worst-direction (per-component-safe, since the pre-planner box is per-Cartesian-axis).
- **25.2 N** — software-torque-limited (≈ the current 25; keeps the operational cap).

### Necessary-condition arithmetic (position reachability, bang-bang d = a·(T/2)²)
At f=62.9 N → a_max = 62.9/71 = **0.886 m/s²** (vs 0.35 at f=25). Required CoM transfer ≈ 0.195 m.
| h_max | T_step | reach @0.886 | ≥ 0.195 m? |
|---|---|---|---|
| 6 | 2.31 s | 1.18 m | yes |
| 8 | 1.73 s | 0.66 m | yes |
| 10 | 1.39 s | 0.43 m | yes |
| 100 | 0.50 s | 0.055 m | **no (still infeasible)** |
So 62.9 N is expected to make the h_max={6,7,8,10} plans position-feasible, while extreme short-T
(box≳30, T→0.5 s floor) stays infeasible — the PHYSICAL result. (Position is necessary, not
sufficient: the terminal v≈0, L≈0 with the rate cap ±5 must also close; Task B measures the actual
IPOPT outcome.) The h_max=6 closed-loop STEADY-STATE dock offset (Phase 1h, 10.27 mm plateau) is a
controller issue, independent of f_max, and is not expected to change.

## Proposed diff (SHOWN, uncommitted — awaiting Idriss GO on the value)
```diff
--- a/crawlbot/simulation/config.py
+++ b/crawlbot/simulation/config.py
@@ -254 +254 @@
-    preplanner_f_max: float = 25.0          # [N] per active contact (also used by F-SAT clamp)
+    preplanner_f_max: float = 62.9          # [N] per active contact — stance-arm along-step capacity
+                                            # (J^-T·τ at MJCF ctrlrange ±50 Nm, min over 6 docked configs; Phase 1i).
+                                            # (F-SAT no longer uses this — see sim_loop.py:2914.)
--- a/scripts/run_m7_single_step.py
+++ b/scripts/run_m7_single_step.py
@@ -61 +61 @@
-        preplanner_f_max=25.0,
+        preplanner_f_max=62.9,   # stance-arm along-step capacity (Phase 1i)
--- a/scripts/diag_cooperative_arms.py
+++ b/scripts/diag_cooperative_arms.py
@@ -272 +272 @@
-    cfg.preplanner_f_max = 25.0  # F-SAT clamp source
+    cfg.preplanner_f_max = 62.9  # stance-arm along-step capacity (Phase 1i); F-SAT no longer uses this
```
All three sites carry the same hardcoded 25.0; `diag_cooperative_arms.py:272` is the LAST write
before the pre-planner is built, so it must change for the Task-B run to take effect (the other two
change for consistency). Only the value + comment change — no logic change.

## Artifacts
`arm_capacity_1i.json` (per-config capacity), `scripts/diag_1i_arm_capacity.py`. Read-only; no run.
STOP-GATE 1 — awaiting Idriss GO on the value (62.9 / 46.7 / 25.2) before Task B.
