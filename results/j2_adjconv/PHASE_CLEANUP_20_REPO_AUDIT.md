# PHASE CLEANUP-20 — repository structure audit (READ-ONLY)

Folder-by-folder audit of the whole repository, to support the target layout: **a clean
structure for the canonical run and future development, with diagnostics / reports / run data
moved into `Misc/`.**

**No files moved, no files deleted.** This establishes what is load-bearing before anything
moves.

Builds on `REPO_STATE.md` (2026-07-18), which mapped the repo but explicitly *"does not
authorize deletion"*. This pass adds the measurement it lacked — and corrects two of its claims
(§5).

---

## 1. Method

Three independent measurements, because directory names lie:

1. **Static import closure** of the canonical run (`gate/_run/import_closure.py`) — AST walk of
   `import` / `from … import` from the gate roots, following first-party modules only.
2. **Runtime invocation trace** — the closure misses `subprocess` calls, so
   `gate/run_gate.py:225` (`diag_full_diag_export.py`) and `dca:621` (`render_traversal.py`)
   were added as roots by hand.
3. **Read-vs-write discrimination** for `results/` — a directory being *named* in code usually
   means a script *writes* there. Only files that are **read** are dependencies.

---

## 2. The decisive measurement

The canonical run — the thing the whole repository exists to reproduce — depends on:

| kind | count | what |
|---|---|---|
| **scripts** | **5 of 187** (2 %) | `diag_cooperative_arms.py`, `run_m7_single_step.py`, `diag_full_diag_export.py`, `export_figure_data.py`, `render_traversal.py` |
| **crawlbot modules** | 21 of 33 (63 %) | the rest are `__init__` files, the 4 `diagnostics/` modules, and `locomotion_planner.py` |
| **model files** | **2 of 31** | `models/VISPA_crawling_rwa3.xml` (plant), `models/VISPA_crawling_fixed.urdf` (controller) |
| **results files** | **1** | `results/j2_adjconv/c25_fulldiag.csv` — the gate's byte-identity baseline |

**98 % of `scripts/` and 96 % of `results/` are not on the canonical path.** That is the whole
case for the restructure, and it is measured rather than estimated.

Of the 5 scripts, only `diag_cooperative_arms.py` (the runner) and `run_m7_single_step.py`
(the config source) are pulled in by the replay itself; `diag_full_diag_export.py` +
`export_figure_data.py` serve the gate's CSV export, and `render_traversal.py` is invoked only
when rendering is enabled (off on the canonical).

---

## 3. Folder-by-folder

| dir | files | size | verdict |
|---|---:|---:|---|
| `crawlbot/` | 33 | 0.6 MB | **CORE** — keep at root, untouched |
| `gate/` | 8 | small | **CORE** — the reproduction gate |
| `models/` | 13 | 0.2 MB | **CORE** for 2 files; 5 variants + `archive/` reviewable |
| `tests/` | 25 | 0.3 MB | keep at root |
| `scenarios/` | 5 | tiny | keep — `.seq` files back `sim.setup(sequence_path=…)` |
| `docs/` | 61 | 8.6 MB | split: ground-truth vs historical reports |
| `scripts/` | 206 | 1.6 MB | **5 canonical, 181 diagnostics** → split |
| `results/` | 1921 | **377 MB** | **15.5 MB load-bearing (4.1 %)** → the main target |
| `lutze_baseline/` | 10 | tiny | keep — M0 paper baseline (CLEANUP-18 §3) |
| `benchmarks/` | 6 | tiny | pytest benchmarks; keep or fold into `tests/` |
| `diagnostic/` | 12 | 0.9 MB | **top-level scratch** — Q1/Q2 outputs → `Misc/` |
| `URDF_models/` | 18 | 1.0 MB | **unused by the canonical** → see §5.1 |
| root | 8 | small | 4 docs + 4 config files |

### `results/` — the number that matters

```
results/ total : 377.1 MB   (1921 files, 165 directories)
load-bearing   :  15.5 MB   (4.1 %)
residue        : 361.6 MB
```

The load-bearing set is eight directories:

| dir | why |
|---|---|
| `j2_adjconv/` | **`c25_fulldiag.csv` is the gate baseline** + 84 phase reports (the campaign record) |
| `j2_figdata/` | paper figure data |
| `hero_render/` | paper hero figures |
| `M2_tests/`, `M3_tests/`, `phase2_0_tmom/`, `M4_baseline_1pct/` | pytest **output** dirs (the §C2 PNG-churn problem) |
| `M7_1pct_3step_v22_t15_fk/` | soft fixture — `test_fk_reference_consistency.py:347` reads `sim_log.json` behind an existence guard |

Everything else is run residue. The largest single items: `logs/` 45 MB (76 old `sim_log.json`),
`diagnostic_q2b/` 43 MB, `diagnostic_q2/` 14 MB, `diag_qcurrent_fix/` 11 MB,
`diag_cooperative_arms/` 11 MB, `M7_abort_diag/` 10 MB. **64 of the 165 directories are not
referenced from anywhere in the repository** — not code, not docs, not reports.

### `scripts/` — 187 Python files by prefix

```
diag_ 70    run_ 32    audit_ 19    test_ 12    hero_ 6
diagnostic_ 5   debug_ 4   report_ 3   plot_ 3   ee_ 3   other 30
```

The `diag_*` / `debug_* `/ `audit_*` families are single-investigation runners. Note `scripts/`
also holds **12 `test_*.py` files that pytest does not collect** (the suite lives in `tests/`) —
they are legacy manual harnesses, and four of them are already non-functional (`CARRYOVER` §C3).

---

## 4. Proposed target layout

```
CrawlBot_control/
├── crawlbot/          the library                        (unchanged)
├── gate/              reproduction gate                  (unchanged)
├── models/            canonical plant + controller pair
├── tests/             pytest suite
├── scenarios/         .seq fixtures
├── scripts/           ONLY the 5 canonical-path scripts
├── docs/              ground-truth architecture only
├── lutze_baseline/    M0 comparison baseline
├── results/           canonical artifacts + paper figure data only
└── Misc/
    ├── diagnostics/   the ~181 one-off scripts
    ├── reports/       historical phase/memo reports
    ├── runs/          run residue (the 361 MB)
    ├── models/        variants + archive + URDF_models
    └── legacy/        non-functional scripts (CARRYOVER §C3)
```

Root-level `README.md`, `CLAUDE.md`, `LICENSE`, `pytest.ini`, `requirements.txt` stay;
`REPO_STATE.md` and `VISPA_OPEN_ITEMS_2026-06.md` are navigation/status docs that belong under
`Misc/reports/` once this audit supersedes them.

---

## 5. Traps found while mapping — resolve before moving anything

### 5.1 Two `VISPA_crawling_fixed.urdf`, different content

`models/VISPA_crawling_fixed.urdf` and `URDF_models/VISPA_crawling_fixed.urdf` share a name and
have **different md5s**. The controller loads the `models/` one (7 references;
`gate/run_gate.py:46`). Nothing in the repo references `URDF_models/` except `REPO_STATE.md`
itself.

`REPO_STATE.md` flagged this and reasoned that the canonical URDF's `<mesh filename="meshes/…">`
paths dangle because the meshes live only under `URDF_models/meshes/`. **Verified and benign:**
`crawlbot/core/robot_interface.py:161` calls `pin.buildModelFromUrdf` — the *model* builder, not
`buildGeomFromUrdf` — so geometry is never loaded and the mesh paths are never resolved. The
canonical needs no meshes at all.

So `URDF_models/` (1 MB, 18 files) is entirely off the canonical path. It is visualization/legacy
asset storage → `Misc/models/`. Moving it changes nothing at runtime; **leaving two same-named
URDFs in the tree is itself the hazard.**

### 5.2 `REPO_STATE.md` points at a directory that has never existed

§2.1 states: *"Canonical replay state (used by figures): `results/figC25_addfive/sim_log.json`
(44 snapshots)."*

```
ls results/figC25_addfive                        -> No such file or directory
git log --all --diff-filter=A -- results/figC25_addfive/*   -> empty
```

It has **never existed in the repository's history**. Anyone starting the cleanup from
`REPO_STATE.md` — which is precisely what it recommends — would look for the canonical figure
state in a path that is not there. The real figure data is `results/j2_figdata/` (672 K) and
`results/hero_render/` (2.4 MB).

Same class as the CLEANUP-2 F1 error and the CLEANUP-19 `from_heuristic` comments: **a
confidently-worded document asserting a fact about the code that measurement contradicts.** It
is the third instance this chantier has found, which is enough to call it a pattern rather than
an accident.

### 5.3 `run_diagnostics()` is not called on the canonical path

CLAUDE.md Rule 3: *"Every simulation produces diagnostics. Call `run_diagnostics()` at the end
of every sim."* The canonical import closure pulls in `crawlbot/diagnostics/__init__.py` (which
re-exports `run_diagnostics`) but **not** `runner.py`, `metrics.py`, `plots.py` or
`snapshots.py`, and neither `dca` nor `sim_loop` calls it.

Reporting, not fixing — this is a rule-compliance question for Idriss, and it is orthogonal to
the restructure. But `crawlbot/diagnostics/` should not be moved to `Misc/` on the strength of
"it's not in the closure" until it is settled.

### 5.4 The pytest output dirs block a clean `results/`

`M2_tests/`, `M3_tests/`, `phase2_0_tmom/`, `M4_baseline_1pct/` exist only because tests write
PNGs into the tracked tree, which is also `CARRYOVER` §C2 (the suite dirties five tracked files
on every run, so the repo can never be verified clean after `pytest`). Pointing those tests at a
scratch dir **fixes §C2 and removes four directories from `results/` in one change.** Best
value-per-risk item in this audit.

---

## 6. Migration plan

Ordered by risk, lowest first. Each step gate-verified — a move that breaks a path is exactly
what the gate's canonical replay catches.

| # | step | risk | note |
|---|---|---|---|
| 1 | `diagnostic/` → `Misc/diagnostics/q1_q2/` | **none** | top-level scratch, zero code references |
| 2 | `URDF_models/` → `Misc/models/urdf_legacy/` | **none** | zero references (§5.1) |
| 3 | Point the four test output dirs at a scratch path | low | **also fixes §C2** |
| 4 | `results/` residue → `Misc/runs/` | low | 361 MB, 157 directories; keep the 8 load-bearing |
| 5 | 181 non-canonical `scripts/` → `Misc/diagnostics/` | medium | many are self-referential; `sys.path` assumptions must be checked per file |
| 6 | historical `docs/` + `results/j2_adjconv/PHASE_*.md` → `Misc/reports/` | medium | many cross-references between reports would need rewriting |
| 7 | `models/` variants + `archive/` → `Misc/models/` | low | confirm none is exercised by `_mutate_mjcf` first |

**Steps 1–4 are mechanical and cover the bulk of the volume** (362 of the 377 MB, plus the two
top-level scratch dirs). Steps 5–7 are the ones that need care, because scripts carry implicit
path assumptions and reports cross-reference each other by relative path.

A caution on git: moving 1900 files does not shrink the repository — the history still carries
every blob. If the 377 MB is a *clone-size* problem rather than a *readability* problem, moving
directories will not solve it and the honest answer is history rewriting, which is a different
decision with different risks. **This audit treats it as a readability problem.**

---

## 7. What this does not answer

- Whether the 361 MB of run residue should be **moved or deleted**. Moving preserves it at zero
  readability cost, which is the conservative default; deletion is Idriss's call.
- Whether `benchmarks/` should fold into `tests/`. Six files, no strong signal either way.
- Whether the 12 `scripts/test_*.py` manual harnesses are worth repairing or should go to
  `Misc/legacy/` (§C3 already lists four as non-functional).
