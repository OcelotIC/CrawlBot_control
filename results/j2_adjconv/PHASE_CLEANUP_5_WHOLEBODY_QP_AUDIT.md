# PHASE CLEANUP-5 — WholeBody QP audit (READ-ONLY)

Audit of the Stage-2 QP: `crawlbot/solvers/wholebody_qp.py` (1385 lines) +
`crawlbot/solvers/hierarchical_qp.py` (528 lines). **No code changed.**

Method, learned from the retracted F1: the canonical config was **measured** by instrumenting
`WholeBodyQP.__init__` during a real `dca` run, and live/dead was **measured** by line coverage
of the full canonical replay — not inferred from dataclass defaults.

| module | statements | never executed on canonical | coverage |
|---|---|---|---|
| `wholebody_qp.py` | 521 | **175** | 66 % |
| `hierarchical_qp.py` | 220 | **67** | 70 % |
| total | 741 | **242** | 67 % |

That is a far larger dead surface than the NMPC's 28 — a third of the Stage-2 QP is
unreachable at the frozen config.

## The canonical config (measured, NOT the dataclass defaults)

Flags whose canonical value **differs from the default** are the ones that decide everything:

| field | default | **canonical** |
|---|---|---|
| `use_m2_stack` | False | **True** |
| `ee_null_space` | False | **True** |
| `cooperative_arms_mode` | False | **True** |
| `ss_two_task_mode` | False | **True** |
| `ds_centroidal_mode` | False | **True** |
| `qp_envelope_exact` | False | **True** |
| `alpha_lambda_int` | 0.0 | **1.0** |
| `alpha_com` | 1e3 | **0.0** |
| `alpha_com_soft` | 5.0 | **0.0** |
| `alpha_reg` | 1e-2 | **1.0** |
| `alpha_torque` | 1e0 | **5.0** |
| `alpha_wrench` | 1e1 | **1.0** |
| `w_hw_slack` | 8e2 | 800.0 |
| `dt_qp` | 0.008 | **0.01** |
| `tau_max` | 50 | **20** |
| `L_max` / `tau_w_max` | inf / inf | **10.0 / 2.5** |

Off at the canonical: `r_tube=0.0`, `alpha_torso_lin=0.0`, `alpha_reaction=0.0`,
`ss_centroidal_momentum_task=False`, `stance_thrust_correction=False`,
`passivity_W_budget=0.0`, `method='weighted'`.

**The decisive line is `wholebody_qp.py:678`:**
```python
_two_task = cfg.ss_two_task_mode and not settle_mode
```
In SS this is True, and the entire legacy task stack below is gated off with `not _two_task`.
In DS `settle_mode=True`, so the legacy stack is skipped *again* — this time by `not
settle_mode` — and `ds_centroidal_mode=True` substitutes the centroidal-DS tasks. **Every
path written for "SS without the two-task stack" is therefore unreachable**, which is where
most of the 175 dead statements come from.

## `wholebody_qp.py` — section map (line ranges, live/dead measured)

| lines | section | status |
|---|---|---|
| 1–51 | module docstring, imports | live |
| **52–221** | `WholeBodyQPConfig` — 60+ fields | mixed; ~11 are dead knobs (below) |
| 223–298 | class, `__init__`, `set_nominal_posture` | live (`234` = a defensive branch, dead) |
| **300–441** | `solve()` signature (≈50 kwargs) + docstring | live |
| 442–652 | **Build QP**: variables, dynamics/contact equalities, bounds, momentum box, passivity | **live** (`568`, `640–643` dead sub-branches) |
| 653–668 | CoM-Jacobian (`A_com`, `b_com`) assembly | live |
| **669–723** | **Phase-2.1 two-task stack: T-MOM + 6-D torso-pose + swing-EE + posture** | **LIVE — this IS the canonical SS controller** |
| 724–728 | Task 1 legacy CoM tracking | **DEAD** (`alpha_com=0`, `use_m2_stack`, `_two_task`) |
| 729–774 | Task 1b legacy torso 6D | **DEAD** (18 lines; pre-empted in SS, replaced in DS) |
| 775–808 | Cooperative-arms angular/linear split | **DEAD** (14 lines; pre-empted by two-task) |
| **809–865** | **Option D torso linear soft tube** | **DEAD** (33 lines — the largest single block; `r_tube=0`) |
| 866–882 | null-space projector setup | live (`880–881` dead) |
| 883–955 | Task 2 legacy EE 6D + null-space projection | **DEAD** (28 lines) |
| 956–997 | Cooperative P2 torso-linear + **T-MOM v1** | **DEAD** (11 lines; superseded by two-task) |
| 998–1027 | Task 2b soft CoM residual | **DEAD** (12 lines; `alpha_com_soft=0`) |
| 1028–1089 | Task 3 posture regulation | live; `1065–1082` (cooperative posture null-space projections) **DEAD** (17 lines) |
| 1090–1106 | Task 3b DS joint-space settle | live (DS) |
| 1107–1140 | Centroidal-DS tasks (CoM + torso-ang + posture) | **live (DS)** |
| 1141–1190 | Task 3 contact-wrench tracking | live; `1159–1179` **stance-thrust correction DEAD** (14 lines) |
| 1191–1205 | Task 3d explicit Σf penalty | **DEAD** (5 lines; abandoned in favour of Tikhonov ε=3) |
| 1206–1246 | Task 3c DS internal-stress regularization | **live** (`alpha_lambda_int=1.0`) |
| 1247–1264 | Task 3b reaction null-space | **DEAD** (6 lines; `alpha_reaction=0`) |
| 1265–1271 | Task 4 joint-torque minimization | live (α=5) |
| 1272–1279 | Task 5 acceleration regularization | live (α=1.0) |
| 1280–1298 | M5 h_w slack penalty | live (w=800) |
| 1299–1345 | solution extraction, h_w-slack telemetry, debug capture | live (`1318–1319`, `1323–1325` dead) |
| 1350–1385 | `_compute_indices`, `n_vars`, `variable_indices`, `__repr__` | `_compute_indices` live; **`1374`, `1379`, `1382` dead** (the three accessors never called) |

## `hierarchical_qp.py` — section map

| lines | section | status |
|---|---|---|
| 46–68 | `Task`, `QPSolveInfo` dataclasses | live |
| 93–128 | `__init__` | live |
| 129–170 | `add_task` | live (4 dead lines = validation branches) |
| 171–213 | `add_equality_constraint`, `add_inequality_constraint`, `set_bounds`, `clear_tasks`, `clear_constraints` | live except `clear_constraints` body (`204–207`) and `200` |
| 214–251 | `solve()` dispatch | live (3 dead) |
| 252–296 | `_solve_weighted` | **live — the canonical path** |
| **297–378** | **`_solve_strict`** | **DEAD — 36 uncovered lines** (`method='weighted'` canonically) |
| 379–492 | `_solve_qp_raw` | live; **15 dead lines** = solver-failure / fallback branches |
| 493–520 | `_get_solver_options` | live (2 dead) |
| 521–527 | `n_tasks`, `__repr__` | **dead** (accessors never called) |

Note `_solve_weighted` is where CLEANUP-1's canonical `regularization=1e-6` is applied by the
gate's replay shim; `κ_SS(H) ≈ 7.5e3` per the freeze.

## Removal candidates, ranked by (size × safety)

Exposure counted as *files* referencing the flag outside the solvers.

| # | feature | dead lines | tests | scripts | note |
|---|---|---|---|---|---|
| 1 | **`stance_thrust_correction`** | 14 | **0** | **0** | zero exposure **and** documented as REGRESSING step 0 (12.1 mm TIMEOUT vs 4.55 mm) — safest real deletion |
| 2 | **`alpha_reaction`** reaction null-space | 6 | **0** | **0** | zero exposure |
| 3 | **Option D tube** (`r_tube`, `w_tube_lin`) | 33 | **0** | 8 / 1 | largest block; no test depends on it |
| 4 | cooperative posture null-space (`alpha_torso_lin`) | 17 | **0** | 14 | |
| 5 | Σf settle penalty (`settle_alpha_sigf`) | 5 | **0** | 1 | superseded by ε=3 |
| 6 | **`_solve_strict`** | 36 | 2 | 6 | biggest single block; has test users |
| 7 | legacy EE + null-space (`ee_null_space`) | 28 | 1 | 2 | |
| 8 | legacy torso 6D + cooperative split | 32 | 1 | 3 | pre-empted by two-task |
| 9 | soft CoM residual (`alpha_com_soft`) | 12 | 1 | 17 | heavy script exposure |
| 10 | T-MOM v1 (`ss_centroidal_momentum_task`) | 11 | 1 | 1 | superseded by two-task |
| 11 | three unused accessors (`n_vars`, `variable_indices`, `n_tasks`, `__repr__`×2) | ~5 | 0 | 0 | trivial |

Rows 1–5 (≈75 lines) have **no test exposure at all**. Rows 6–10 (≈119 lines) each carry at
least one test and are the "supersession debris" of the architecture's evolution — removing
them is a deliberate decision about whether the superseded paths stay reproducible.

## Structural findings (not dead code)

1. **`solve()` is ~1045 lines** (300–1345) with ≈50 keyword arguments — one method holding the
   whole task stack, twelve `if` gates deep in places. This, not the dead code, is why the file
   is hard to work with. Splitting per-task assembly into private helpers would be a large but
   mechanical refactor, verifiable by the gate.
2. **`WholeBodyQPConfig` has 60+ fields** and mixes live canonical values, dead knobs, and
   abandoned-experiment switches with no marking. The dead-knob list above should at minimum be
   commented as such.
3. **Defaults are not canonical for 16 of the fields measured** (table at the top), including
   flags that change the entire task topology (`use_m2_stack`, `ss_two_task_mode`,
   `ds_centroidal_mode`). Instantiating `WholeBodyQPConfig()` gives a controller with a
   *different architecture*, not merely different gains — same class of trap as NMPC F6, but
   worse.
4. `_solve_qp_raw`'s 15 uncovered lines are solver-failure/fallback branches. Like the NMPC's
   `get_shifted_fallback`, they are uncovered *because* the QP never fails (`qp_fail=0` on the
   canonical) — **keep**.

## Recommendation

Rows 1–2 first (20 lines, zero exposure, one already documented as harmful), then row 3.
Rows 6–10 need your ruling on whether superseded paths stay reproducible — that is the same
question Tier B raised for the NMPC, and after the F1 retraction I would not assume "superseded"
means "safe to delete" without your call.
