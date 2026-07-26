# REPO_STATE — results résumé + file map

**Purpose.** A single entry point that (a) résumés the current results and (b) names the designated /
source-of-truth files. This is a navigation doc; it does not authorize deletion.

**Snapshot:** 2026-07-26, commit `eecbf94`, branch `claude/review-closure-bloc-2-uwu1x7`.
Supersedes the 2026-07-18 snapshot, which was taken on a **pre-chantier** branch and described the
flat pre-package layout, a 1907-file `results/` tree, and a test suite with 2 failures — none of
which is still true.

**Where the code truth lives:** `docs/architecture/STACK_OVERVIEW.md` (refreshed the same day).
This file maps the *repository*; that one maps the *controller*. Read §0 of STACK_OVERVIEW before
quoting any parameter — `SimConfig` defaults are not the canonical values.

---

## 1. Authoritative documents (ground truth — keep, central)

| doc | role |
|---|---|
| `docs/architecture/brainstorming_reworked_architecture.md` | Math derivation, control architecture, frame conventions (the *what*). |
| `docs/architecture/CLAUDE_CODE_HANDOFF.md` | File-level plan, diagnostic spec, anti-patterns (the *how*). |
| `docs/architecture/STACK_OVERVIEW.md` | **Code-ground-truth** for current-state questions + the canonical config chain. |
| `docs/architecture/STATUS.md` | Milestone status. |
| `docs/architecture/IK_FORMULATION.md` | IK derivation (§7–§9 carry a RETIRED banner — see CLEANUP-30). |
| `CLAUDE.md` | Project rules, Key Parameters table, Canonical Results (read at session start). |
| `docs/architecture/PORT_AUDIT.md`, `PORT_SYNTHESIS.md` | Modularity audit / port checklist. **Semantic map only — every `file:line` in them predates the package restructuring.** |
| `docs/crawlbot/` (34 files) | Per-module reference; the measured half is generated and **enforced** by `gate/sync_docs.py --check`. |
| `gate/README.md`, `gate/EXCEPTIONS.md` | What the gates check, and the Tier-0 / Tier-1 acceptance policy. |

---

## 2. Results résumé

### 2.1 Canonical operating point — FROZEN 2.5

Freeze commit `32aefaf`; τ_w,max = 2.5 N·m enforced **three times**: controller `config.py:79`,
AOCS clip `config.py:83`, plant MJCF wheel `ctrlrange="-2.5 2.5"`
(`models/VISPA_crawling_rwa3.xml:324-326`).
**Source of truth:** `results/j2_adjconv/canonical2p5_result.json` + `c25_fulldiag.csv` /
`u25_fulldiag.csv` (66 columns each; C = 2077 ticks, U = 1905).

| metric | C (managed) | U (management off, plant cap still active) |
|---|---|---|
| at-weld docks | **6/6** — 4.02 / 4.89 / 4.99 / 4.97 / 4.95 / 4.62 mm (worst margin **0.01 mm**) | 6/6 |
| planned Ḣ_s (per-axis peak) | capped at **2.500** on all six steps | up to **10.88** (4.4× envelope) |
| realized Ḣ_s (per-axis peak) | **2.500** (WQP box active at peak) | 3.451 |
| θ_s peak | **0.540°** | 1.194° |
| h_w (axis / norm) | 4.10 / 4.24 | 4.55 / 5.08 |
| κ_SS / qp_fail | 7.48e3 / 0 | 7.71e3 / 0 |
| traversal end time (log `final`) | **84.64 s** | — |

Rule 10 applies: **at-weld only**. Min-over-swing is a fly-by artifact.

### 2.2 Verified reproducible at this snapshot

`gate/run_gate.py` → **PASS** at `eecbf94`: byte-identical on 2077 rows × 132 928 fields against the
committed `c25_fulldiag.csv` (baselined at `bfd5509`), the two wall-clock columns excluded per
`gate/EXCEPTIONS.md`; two-model consistency PASS (15 links, 14 joints, 71.056 kg); environment pin
PASS. `gate/run_suite.py --fast` → **PASS** (198 passed, 0 failed, 0 errors, 0 skipped, 1 xfail).
Evidence: `results/review_closure/c0/C0_PROVENANCE.md`.

---

## 3. Designated core files (the controller — keep & organize)

`crawlbot/` — 34 tracked files (26 modules + 8 package `__init__`s) — is the clean, importable
package. Everything else (`scripts/`, `gate/`, `results/`, `benchmarks/`, `scenarios/`, `Misc/`)
consumes it.

| area | file | responsibility |
|---|---|---|
| **core** | `core/robot_interface.py` | Pinocchio wrapper; DOF-generic arm-slice detection; all controller quantities. |
| | `core/ik.py` | Docking IK + `manipulability_config` (IK 2). The Option-B trajectory/feasibility path was retired (CLEANUP-30) — **no interior path-feasibility guard remains**. |
| | `core/state_conversions.py` | MuJoCo(world) ↔ Pinocchio(structure-frame) bridge; quaternion conventions live here. |
| | `core/com_to_torso_mapping.py` | CoM→torso reference mapping — **DS path only** (SS bypasses it). |
| **solvers** | `solvers/centroidal_nmpc.py` + `nmpc_solver.py` | Momentum-aware NMPC (nx=9, nu=12, N=8, dt=0.1). |
| | `solvers/wholebody_qp.py` | Whole-body QP — the two-task weighted SS stack, no null-space projection. |
| | `solvers/hierarchical_qp.py` | Weighted/hierarchical solve backend. |
| | `solvers/contact_phase.py` | Contact config + momentum map. |
| **planning** | `planning/coarse_preplanner.py` | Per-step centroidal NLP (T_step + CoM plan). |
| | `planning/contact_scheduler.py` | Gait plan, anchor grid, phase timing. |
| | `planning/swing_planner.py` / `torso_planner.py` | Swing-EE / torso quintic+SLERP references. |
| | `planning/locomotion_planner.py`, `sequence_loader.py` | Higher-level plan, `.seq` loader (loader verified by **neither** gate). |
| **simulation** | `simulation/sim_loop.py` | The closed loop (largest file; DS/SS state machine, weld activation, AOCS call site). |
| | `simulation/tick_logging.py` | Per-tick recorders, split out of `sim_loop` (CLEANUP-32). |
| | `simulation/config.py` | `SimConfig` — the single tuning-knob dataclass. |
| | `simulation/logging.py`, `plotting.py` | Run logs + plots. |
| **aocs / estimation / diagnostics** | `aocs/force_estimator.py`, `estimation/contact_estimator.py`, `diagnostics/{metrics,plots,runner,snapshots}.py` | Wheel/force estimation, GMO contact, diagnostic suite. |

**Deleted since the last snapshot:** `planning/constrained_geodesic.py` (CLEANUP-17).

**Designated model files (canonical pair):**
- PLANT: `models/VISPA_crawling_rwa3.xml` (MuJoCo, nq=31, structure 7110 kg + 3 RWA).
- CONTROLLER: `models/VISPA_crawling_fixed.urdf` (Pinocchio, nq=21).

Both named in `gate/run_gate.py:45-46` and `scripts/diag_cooperative_arms.py:51,488`.
`models/VISPA_crawling.xml` and everything under `URDF_models/` are loaded by **no** code path.

**Canonical runner:** `scripts/diag_cooperative_arms.py` (`dca.main`), driven for the frozen point by
`gate/replay_canonical.py`. See STACK_OVERVIEW §0 for the full four-layer config chain.

---

## 4. Repository inventory (post-chantier)

Tracked-file counts by top-level dir:

| dir | files | nature | posture |
|---|---:|---|---|
| `Misc/` | **2100** | research sediment moved wholesale by the chantier | delete once its ~120 inbound citations are swept |
| `results/` | 117 | run artifacts, in **3** subdirs: `j2_adjconv/` (the canonical + all phase reports), `hero_render/`, `j2_figdata/` | keep |
| `docs/` | 43 | architecture ground truth + `docs/crawlbot/` per-module reference | keep |
| `crawlbot/` | 34 | **the core** | keep all |
| `gate/` | 19 | the eight checkers + lock + exceptions policy | keep |
| `tests/` | 19 | pytest suite (gated by `gate/run_suite.py`) | keep |
| `URDF_models/` | 18 | second model dir + meshes — **loaded by nothing** | see risk 1 |
| `models/` | 13 | canonical pair + mass-ratio variants + `archive/` | keep canonical pair |
| `benchmarks/`, `scenarios/`, `scripts/` | 6 / 5 / 5 | comparison, `.seq` fixtures, the 5 surviving runners | keep |
| **total tracked** | **2387** | | |

`scripts/` is down to five files: `diag_cooperative_arms.py` (canonical runner),
`diag_full_diag_export.py` (the 66-column exporter the gate uses), `export_figure_data.py`,
`render_traversal.py`, `run_m7_single_step.py` (`_make_m7_config`).

**The gate suite** (`gate/`, 8 checkers, each proven to bite on an injected fault before being
trusted): `run_gate.py` (canonical byte-identity + two-model consistency + env pin),
`run_suite.py` (the component tests — the half `run_gate` structurally cannot do), `dock_check.py`,
`sync_docs.py`, `verify_docs.py`, `verify_params.py`, `verify_roots.py`, `link_audit.py`.
**Both gates, not either:** `run_gate` protects the past (a wrong controller reproduces just as
faithfully), `run_suite` protects a change.

**Risks / ambiguities still open:**
1. **Two model directories.** `models/` and `URDF_models/` both carry a `VISPA_crawling_fixed.urdf`;
   the controller loads the `models/` one, but the STL/DAE meshes live only under
   `URDF_models/meshes/`, so the canonical URDF's `<mesh filename="meshes/…"/>` paths do not resolve
   next to it. Meshes are visualization-only, so the controller is unaffected — but decide one home.
2. **Model variants.** `models/` also carries `_8pct`, `_8pct_hw50/hw100`, bare `VISPA_crawling.xml`
   and `models/archive/`. CLAUDE.md's rule is one canonical MJCF with variations applied
   programmatically (`_mutate_mjcf`); confirm which variants are still exercised before pruning.
3. **`link_audit.py` reports 138 unresolved citations**, of which 125 are `Misc/…` (dangling by
   design until `Misc/` is swept) and 13 are genuinely deleted targets (`constrained_geodesic.py`,
   `docs/api`, retired `results/` dirs). It audits **prose** citations only — it cannot see paths
   computed in Python, which is how CLEANUP-21 silently disabled 7 tests for six passes.
4. **This file's and CLAUDE.md's prose `file.py:NNN` anchors are checked by nothing.**
   `verify_params.py` covers only the parameter table. Three CLAUDE.md refs had already drifted by
   the time CLEANUP-30 measured them.

---

## 5. Known issues (tracked, not introduced here)

- **Suite: 198 passed / 0 failed / 0 errors / 0 skipped / 1 xfail** in fast mode, and **gated**.
  The single xfail, `test_coarse_preplanner::test_far_infeasible_under_tight_rate`, is
  `strict=True`: the envelope-semantics question at cap 2.5 is open, and if the far case ever goes
  infeasible again the suite **fails** rather than turning green.
- **`MUJOCO_GL=osmesa` aborts pytest collection** in this container (PyOpenGL, via
  `tests/test_diagnostics.py`). Environment, not code — `run_suite.py` forces `disabled`.
- **Worst-case dock margin 0.01 mm** (step 3 of 6) — accepted operating point; 5 mm is the
  docking-mechanism capture radius.
- **CoM-reference export** snaps to the measured CoM at SS→DS entry (logging convention,
  `sim_loop._log_ds_tick`); decision pending whether to apply the torso export's terminal-hold fix.
- **Fig-3 ‖L_total‖ conservation channel** is not in the 66-column fulldiag export (it exists in
  `export_figure_data.py`'s traversal CSVs and as snapshots in `c25_fulldiag_meta.json`).
- **The canonical run does not honour Rule 3** — it exports via `scripts/diag_full_diag_export.py`,
  never `run_diagnostics()` (`CLEANUP_CARRYOVER` §A6).
- **No tag exists.** `git tag -l` is empty; `paper-2p5-base` was never created. The paper's
  provenance currently rests on the bare hash `bfd5509`.
- **`gate/sync_docs.py --check` fails on a fresh clone** (rc=1, "25 document(s) out of date"),
  because its coverage input `gate/_run/cov/cov.json` is untracked and the regeneration script
  it names — `gate/_run/cov_replay.sh`, cited by `CLAUDE.md:132-134` and `gate/sync_docs.py:34` —
  **does not exist in the repository**. `gate/api_live.py`, `gate/cov_compare.py` and
  `gate/gen_module_docs.py` read the same missing file and crash outright. A gate that fails
  unconditionally is a gate that gets overridden; the drift it exists to catch then hides inside
  25 standing failures. `run_gate.py` and `run_suite.py` are unaffected and both PASS.
  Detail: `results/review_closure/c0/C0_PROVENANCE.md`.

---

*Start from §1 for ground truth, §3 for the code map, §4 for what is still messy.*
