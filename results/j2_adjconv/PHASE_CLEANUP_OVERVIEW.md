# CLEANUP chantier — overview

State of the repository-hygiene campaign on branch `cleanup-nmpc`. **24 commits**, CLEANUP-0
through CLEANUP-19. Every number below is measured from the repo, not recalled.

Companion documents:
- `CLEANUP_CARRYOVER.md` — live ledger of everything found but **not** acted on (resume-cold)
- `PHASE_CLEANUP_<n>_*.md` — the 13 per-phase reports
- `gate/README.md` — the safety gate this all runs behind

---

## 1. Net effect

**`crawlbot/` lost 1986 lines** — 561 insertions against 2547 deletions, ≈ 12 % of the package.

| file | before | after | Δ |
|---|---|---|---|
| `solvers/wholebody_qp.py` | 1385 | 949 | **−436** |
| `planning/constrained_geodesic.py` | 470 | *deleted* | **−470** |
| `planning/swing_planner.py` | 728 | 337 | **−391** |
| `simulation/sim_loop.py` | 3761 | 3386 | **−375** |
| `planning/torso_planner.py` | 702 | 480 | −222 |
| `simulation/config.py` | 610 | 506 | −104 |
| `solvers/nmpc_solver.py` | 657 | 649 | −8 |
| `solvers/centroidal_nmpc.py` | 681 | 701 | **+20** — bug fixes add code |

Coverage of the canonical replay confirms the removals were real, not relocations:
`swing_planner.py` **47 % → 95 %**, `torso_planner.py` **55 % → 81 %**.

---

## 2. The invariant, held throughout

**The canonical run is the thing that must not move.** The gate ran green on every
code-changing commit, and the final state is bit-for-bit identical to the frozen 2.5 canonical:

```
at-weld docks  6/6   4.02 / 4.89 / 4.99 / 4.97 / 4.95 / 4.62 mm
                     every step delta +0.0000   worst margin 0.01 mm
theta_s 0.540 deg    h_w 4.102 / 4.243 Nms    e_com 0.154 m    qp_fail 0
artifact identity    2077 rows x 132928 fields
```

---

## 3. The five campaigns

### CLEANUP-0 — the gate, built before the first broom stroke

`gate/run_gate.py`: four checks, one verdict, exit 0/1.

1. **Canonical replay** into a scratch dir, then export the 66-column fulldiag CSV
2. **Artifact identity** — field-by-field vs the committed baseline; only the two wall-clock
   timing columns are excluded
3. **Two-model consistency** — MJCF plant vs URDF controller model, per-link composite mass /
   COM / principal inertia, joint order and limits, frame placements
4. **Environment pin** — advisory; bit-identity is meaningless on an unpinned stack

It was proved to bite by **injecting the F1 mistake and watching it FAIL at row 11 of 2077 on a
1e-10 difference** — a change that would still have docked 6/6 and looked correct on every plot.
That is the whole argument for the gate: plots cannot see this class of regression.

### CLEANUP-1→4 — NMPC

Tier-A dead-code removal in `nmpc_solver.py`, an audit of `centroidal_nmpc.py`, then six real
fixes (F2 warm-start now gated on `info.success`; F5, F7, F8, F9, F10).

**This campaign also produced the retraction that shaped everything after it.** CLEANUP-2
claimed the h_w / M3 conservation path was dead, inferred from
`SimConfig.enforce_hw_conservation = False`. Wrong: `dca.main` builds its config through
`run_m7_single_step._make_m7_config()`, which sets it `True`. Measured live: `enforce_hw=True`,
`ng_path=17`, `ng_term=6`. Retracted publicly in the report and the commit message.

The rule adopted: **a dataclass default is not the canonical value.** Trace the config the run
actually builds, or instrument it. Confirm live/dead by line coverage of the replay, never by
reading.

### CLEANUP-5→11 — WholeBody QP, 1385 → 949

Audit, then a two-stage excision of the legacy pre-two-task SS stack and the abandoned
experiment tasks, then a config-surface prune, a readability pass separating live code from
removed-architecture scars, an anatomy audit of `solve()`, and finally an extraction of four
helpers — `solve()`'s body went **543 → 346 lines**.

Explicitly *not* done: merging or reordering task blocks. The order encodes the cost-assembly
sequence, so that would be a behavioural change dressed as a refactor.

### CLEANUP-12→15 — `sim_loop.py`, −375

Audit plus a cross-study against the already-cleaned QP/NMPC, then three passes removing
flag-gated dead paths: the FK-reference path, trajectory-IK, and the mid-waypoint reshape.

### CLEANUP-16→19 — `crawlbot/planning/`, 3258 → 2175 (−33 %)

Audit, deletion of `constrained_geodesic.py` (470 lines, never imported) with the planner FK
mode that was its only referent, then the SwingPlanner phase-override surgery, then a final
read-only audit of `coarse_preplanner.py` and `contact_scheduler.py`.

---

## 4. What measurement stopped

Deletions are the visible output; the refusals are the load-bearing part.

| what | proposed | measured | outcome |
|---|---|---|---|
| h_w / M3 path | dead (from a dataclass default) | `enforce_hw=True` on the canonical | **F1 retracted** |
| `locomotion_planner.py` | delete, 205 lines | 3 consumers, all import-clean; one is the M0/Lutze paper baseline | **kept** |
| `use_m2_stack` | looks dead — its QP twin was removed | gates torso-ref routing **and DS passivity** | **kept** |
| `coarse_preplanner` fallbacks | 45 dead statements | 25 are the failure ladder + API defaults | **kept** |
| `contact_scheduler` | 16 dead statements | 15 are validation/fallback | **file untouched** |
| `sequence_loader.py` | 0 % covered | backs `sim.setup(sequence_path=…)` | **kept** |

The operative distinction, applied at every step:

- **(a) research sediment behind opt-in flags** → remove
- **(b) failure/fallback branches dead *because the system is healthy*** → **keep**
- **(c) live debug hooks** (`_diag_freeze_ref`, `_diag_lock_arm_joints`, `_diag_pure_pd`) → **keep**

Class (b) is the one that looks most deletable on a coverage report and is most dangerous to
touch. `solve()`'s dead set in `coarse_preplanner.py` is *entirely* class (b) — the IPOPT
failure handler, the value-extraction fallback, the stats fallback, two API defaults.

---

## 5. Near-misses caught before they landed

| # | what nearly happened | what caught it |
|---|---|---|
| 1 | 11 undefined names — initialisers removed, a telemetry dict still read them | **pyflakes** as a pre-gate check |
| 2 | A cut based on coverage that predated the previous pass | noticing a symbol appear only in supposedly-dead branches → **re-measure after every removal** |
| 3 | A span-cut mid-signature writing a `SyntaxError` (multi-line `def`, closing `) -> None:` at method indent) | **`ast.parse` guard before write**; spans then switched to AST `lineno`/`end_lineno` |
| 4 | A silent default flip `False → True` while rewriting a comment | self-caught; the canonical sets it explicitly, so **the gate would not have seen it** |
| 5 | `run_m7_single_step.py` passing pruned kwargs — would have broken the gate's own config source | import-check |

No.4 is worth remembering: the gate is not a substitute for care about parameters the canonical
sets explicitly.

---

## 6. Tooling added

| path | role |
|---|---|
| `gate/run_gate.py` | 4 checks, 1 verdict, `last_verdict.json` |
| `gate/replay_canonical.py` | isolated managed-scenario replay, exact `dca.main` kwargs |
| `gate/dock_check.py` | **the physical result** — at-weld docks (Rule 10) + θ_s / h_w / e_com vs frozen |
| `gate/EXCEPTIONS.md` | Tier-0 bit-identity vs Tier-1 metric-equivalence, and the sign-off ledger |
| `gate/environment.lock` | pinned stack |

`dock_check.py` exists because the gate's verdict is a *hash* statement. It entails identical
docks but never showed them, and "the CSV is byte-identical" is not the sentence a reviewer
wants to read.

---

## 7. Where it stands

**Open, in priority order** (detail in `CLEANUP_CARRYOVER.md`):

1. **`setup_env.sh` cmeel pins** (§C1) — highest practical value. Without them nothing in the
   repo runs on a fresh container; it has been fixed by hand every session and never committed.
2. CLEANUP-19 steps 1–3 — `from_heuristic` + its four false comments (~85 lines),
   `contact_sequence_over_horizon` (19), `T_step_default` (4). Ready to execute.
3. Thirteen silent canonical values (§C4), including two hard boxes on every step's terminal
   state.
4. `sim_loop._step` decomposition (§A) — needs its own coupling measurement first.
5. **Awaiting a ruling, not a measurement** (§E): `a_cruise_max` and `locomotion_planner.py` —
   both documented-but-unused. Product calls.

**Not started:** repository-structure audit — `results/` holds 1920 files across 165
directories and `scripts/` holds 185 Python files, most of them one-off diagnostics. Separating
the canonical run's dependencies from research residue is the next campaign.
