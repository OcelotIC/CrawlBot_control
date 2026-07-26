# REPO_STATE — results résumé + file map

**Purpose.** A single entry point that (a) résumés the current results and (b) names the designated /
source-of-truth files, so the next big effort — **cleaning the repo to make it readable** — starts from a
map instead of guesswork. This is a navigation doc; it does not authorize deletion. Snapshot date
2026-07-18, branch `claude/lucid-gates-rsigzt`.

---

## 1. Authoritative documents (ground truth — keep, central)

| doc | role |
|---|---|
| `docs/architecture/brainstorming_reworked_architecture.md` | Math derivation, control architecture, frame conventions (the *what*). |
| `docs/architecture/CLAUDE_CODE_HANDOFF.md` | File-level plan, diagnostic spec, anti-patterns (the *how*). |
| `docs/architecture/STACK_OVERVIEW.md` | Code-ground-truth for current-state questions. |
| `docs/architecture/STATUS.md` | Milestone status. |
| `CLAUDE.md` | Project rules + Key Parameters table + Canonical Results (read at session start). |
| `docs/architecture/PORT_AUDIT.md` | (new) Modularity audit — what it takes to swap in a new robot. |

---

## 2. Results résumé

### 2.1 Canonical operating point — FROZEN 2.5 (on `main` via merged PR #29)

Freeze commit `32aefaf`; τ_w,max = 2.5 N·m enforced 3× (controller `config.py:80`, AOCS clip
`config.py:84`, plant MJCF wheel `ctrlrange ±2.5`). **Source of truth:**
`results/j2_adjconv/canonical2p5_result.json` + `c25_fulldiag.csv` / `u25_fulldiag.csv`.

| metric | C (managed) | U (management off, plant cap active) |
|---|---|---|
| at-weld docks | **6/6** — 4.02 / 4.89 / 4.99 / 4.97 / 4.95 / 4.62 mm (worst margin **0.01 mm**) | 6/6 |
| planned Ḣ_s (per-axis peak) | capped at **2.500** on all six steps | up to **10.88** (4.4× envelope) |
| realized Ḣ_s (per-axis peak) | **2.500** (WQP box active at peak) | 3.451 |
| θ_s peak | **0.540°** | 1.194° |
| h_w (axis / norm) | 4.10 / 4.24 | 4.55 / 5.08 |
| κ_SS / qp_fail | 7.48e3 / 0 | 7.71e3 / 0 |

Canonical replay state (used by figures): `results/figC25_addfive/sim_log.json` (44 snapshots).

### 2.2 Post-freeze follow-up (this branch — 11 commits, the PR contents)

| work | result (one line) | artifacts |
|---|---|---|
| **T1** read-only audits | §VI-D QP-stack + §VI-E FF-orbital audits; 2 paper divergences flagged | `results/j2_adjconv/PHASE_QP_STACK_AUDIT.md`, `PHASE_FF_ORBITAL_AUDIT.md` |
| **T2** fulldiag export | added ω_s (structure rate) + L_total channels, logging-only, control byte-identical; L_total conserved ~1.5e-3, ω_s,z ≈ −h_w,z/I_comp (I_comp ≈ 2152 vs 2180) | `scripts/diag_full_diag_export.py`, `PHASE_DRIFT_CLOSURE_T2.md` |
| **T4** 450 s settle | 3/4 pass; C3 misses (θ_s 0.165°, slow τ ≈ 305 s asymptote, not a floor); traversal bit-identical to canonical | `PHASE_DRIFT_CLOSURE_T4.md` |
| **T4b** 900 s settle | z-drift crosses 0.05° at **t=860.44 s** (matches T4 extrapolation); tail τ ≈ 299.5 s; one-field config diff proven | `PHASE_DRIFT_CLOSURE_T4b.md` + committed plotting-grade trace CSV |
| **Hero figures** | frozen-pose MuJoCo renders (stroboscopic Camera A on structured white; full-system + gripper-inset Camera B, dark); dynamics-neutral (belt md5-identical) | `results/hero_render/` (composite, frame_0..7, sysview_full, sysview_gripper_inset, README, render_meta.json) |
| **Port audit** | 5-bin modularity inventory + port checklist | `docs/architecture/PORT_AUDIT.md` |

---

## 3. Designated core files (the controller — keep & organize)

`crawlbot/` (32 modules) is the clean, importable package. Everything else (scripts/, results/,
Misc/runs/q1_q2/, benchmarks/, Misc/lutze_baseline/, scenarios/) consumes it.

| area | file | responsibility |
|---|---|---|
| **core** | `core/robot_interface.py` | Pinocchio wrapper; DOF-generic arm-slice detection; all controller quantities. |
| | `core/ik.py` | Docking IK, manipulability configs, path feasibility. |
| | `core/state_conversions.py` | MuJoCo(world)↔Pinocchio(structure-frame) bridge. |
| | `core/com_to_torso_mapping.py` | CoM→torso reference mapping (DS path). |
| **solvers** | `solvers/centroidal_nmpc.py` + `nmpc_solver.py` | Momentum-aware NMPC (nx=9, nu=12). |
| | `solvers/wholebody_qp.py` | Whole-body QP (the two-task SS stack). |
| | `solvers/hierarchical_qp.py` | Weighted/hierarchical solve backend. |
| | `solvers/contact_phase.py` | Contact config + momentum map. |
| **planning** | `planning/coarse_preplanner.py` | Per-step centroidal NLP (T_step + CoM plan). |
| | `planning/contact_scheduler.py` | Gait plan, anchor grid, phase timing. |
| | `planning/swing_planner.py` / `torso_planner.py` | Swing-EE / torso quintic+SLERP references. |
| | `planning/locomotion_planner.py`, `constrained_geodesic.py`, `sequence_loader.py` | Higher-level plan, geodesic refs, `.seq` loader. |
| **simulation** | `simulation/sim_loop.py` | The closed loop (largest file; DS/SS state machine, weld activation, AOCS). |
| | `simulation/config.py` | `SimConfig` — the single tuning-knob dataclass. |
| | `simulation/logging.py`, `plotting.py` | Run logs + plots. |
| **aocs / estimation / diagnostics** | `aocs/force_estimator.py`, `estimation/contact_estimator.py`, `diagnostics/{metrics,plots,runner,snapshots}.py` | Wheel/force estimation, GMO contact, diagnostic suite. |

**Designated model files (canonical pair):**
- PLANT: `models/VISPA_crawling_rwa3.xml` (MuJoCo, nq=31).
- CONTROLLER: `models/VISPA_crawling_fixed.urdf` (Pinocchio, nq=21).

Canonical runner: `scripts/diag_cooperative_arms.py` (`dca.main`) — the entry point every campaign run uses.

---

## 4. Cleanup inventory (targets, candidates, risks)

Tracked-file counts by top-level dir (the readability problem in one table):

| dir | files | nature | cleanup posture |
|---|---:|---|---|
| `results/` | **1907** (168 subdirs) | run artifacts | **biggest target** — keep `j2_adjconv/`, `figC25_addfive/`, `hero_render/`; triage the rest |
| `scripts/` | **187** | mostly `diag_*.py` one-off runners | **second target** — most are single-investigation; keep `diag_cooperative_arms.py` + active exporters |
| `docs/` | 60 | architecture reports + api | keep ground-truth (§1); ~30 phase/memo reports are historical record |
| `crawlbot/` | 34 | **the core** | keep all; this is the readable nucleus |
| `tests/` | 25 | pytest suite | keep; fix the 2 pre-existing failures |
| `URDF_models/` | 18 | second model dir + meshes | **flag — see risks** |
| `models/` | 13 | canonical pair + variants + `archive/` | keep canonical pair; variants/archive reviewable |
| `Misc/runs/q1_q2/` | 12 | top-level scratch (separate from scripts/ & results/) | candidate to fold or drop |
| `Misc/lutze_baseline/`, `benchmarks/`, `scenarios/` | 10 / 6 / 5 | comparison + fixtures | keep if still referenced |

**Biggest `results/` subdirs** (file counts): `diagnostic_q2b` 156, `diagnostic_q2` 91, `logs` 76,
`j2_adjconv` 70, `frames` 65, `M7_abort_diag` 53, `diag_cooperative_arms` 38, `M7_step2_isolation` 33,
`M7_settle_diag` 26, `diag_cooperative_arms_torsopos07` 23. The `M7_*` and `diagnostic_q2*` trees are
historical single-phase scratch — prime triage candidates.

**Risks / ambiguities to resolve BEFORE deleting (found while mapping):**
1. **Two model directories.** `models/` and `URDF_models/` both contain a `VISPA_crawling_fixed.urdf`.
   The controller loads `models/VISPA_crawling_fixed.urdf` (per `scripts/diag_cooperative_arms.py:497`),
   but the STL/DAE **meshes live only under `URDF_models/meshes/`** — the canonical URDF's
   `<mesh filename="meshes/…"/>` paths do not resolve next to it. Meshes are visualization-only (MuJoCo
   uses primitive geoms; Pinocchio needs only the kinematic/inertial model), so the controller is
   unaffected — but the duplication + dangling mesh path is a readability trap. Decide one home.
2. **Model variants.** `models/` also carries mass-ratio variants (`_8pct`, `_8pct_hw50/hw100`, bare
   `VISPA_crawling.xml`) and `models/archive/` (6 already-archived variants incl. `rwa4_pyramid`, 7dof,
   01pct). Confirm which are still exercised (CLAUDE.md rule: one canonical MJCF, variations applied
   programmatically via `_mutate_mjcf`) before pruning.
3. **`Misc/reports/api/*.md` is stale.** It predates the current package layout (e.g. references a `dynamics`
   module that is not in `crawlbot/`; `contact_scheduler.py:36` still comments "matches dynamics.py").
   Regenerate or remove — do not trust as current.
4. **Session branches.** ~30 `origin/claude/*` remote branches exist; most are superseded by the merged
   PR #29. A branch sweep is part of "readable".

---

## 5. Known issues (tracked, not introduced here)

- **2 pre-existing pytest failures** — `test_far_infeasible_under_tight_rate` (re-examine its "tight
  rate" semantics now that the canonical cap IS 2.5) and `test_E7_t15_step2_dock_under_fk_mode`.
  Identical set before/after the freeze; zero new.
- **Worst-case dock margin 0.01 mm** (step 2) — accepted operating point (5 mm = docking-mechanism
  capture radius).
- **CoM-reference export** snaps to measured CoM at SS→DS entry (logging convention; decision pending).
- **Fig-3 ‖L_total‖ conservation channel** not yet in the fulldiag export (dedicated export pending).

---

*This doc is the recommended starting point for the repo-cleanup effort: work top-down from §4, using
§1–§3 as the "must-keep" list.*
