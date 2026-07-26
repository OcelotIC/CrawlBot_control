# `crawlbot/` documentation

Per-package and per-module documentation, written from the code at the current
commit and from **line coverage of the canonical replay**
(`gate/replay_canonical.py`).

| package | overview | modules |
|---|---|---:|
| **solvers** — the controller | [`solvers/solvers.md`](solvers/solvers.md) | 5 |
| **planning** — reference generation | [`planning/planning.md`](planning/planning.md) | 6 |
| **core** — model, IK, frames | [`core/core.md`](core/core.md) | 4 |
| **simulation** — closed loop, config | [`simulation/simulation.md`](simulation/simulation.md) | 4 |
| **diagnostics** — metrics, figures | [`diagnostics/diagnostics.md`](diagnostics/diagnostics.md) | 4 |
| **aocs** — reaction wheels | [`aocs/aocs.md`](aocs/aocs.md) | 1 |
| **estimation** — contact observer | [`estimation/estimation.md`](estimation/estimation.md) | 1 |

**33 documents**: one folder per package, containing the package overview and
**one file per module**. Each module document carries its line count, canonical
coverage, the mathematics it implements, how the code realises it, and its API
table annotated canonical / not exercised.

---

## Where to start

Reading the controller for the first time, in order:

1. [`solvers/solvers.md`](solvers/solvers.md) — the two-stage architecture and
   why it is split that way.
2. [`solvers/centroidal_nmpc.md`](solvers/centroidal_nmpc.md) section 1.3 — the
   conservation law coupling the robot to the reaction wheels. This is the
   central idea of the whole system.
3. [`solvers/wholebody_qp.md`](solvers/wholebody_qp.md) sections 1.3-1.4 — the
   task stack, and why there is no null-space projection.
4. [`planning/coarse_preplanner.md`](planning/coarse_preplanner.md) section 1 —
   why step duration is computed rather than chosen.
5. [`simulation/sim_loop.md`](simulation/sim_loop.md) section 2 — how a step is
   sequenced, and why that order is forced.

---

## Method

Three sources, none of them memory:

1. **AST** for signatures and the public-symbol inventory
   (`gate/_run/api_inventory.py`).
2. **Line coverage of the canonical replay** to separate what runs from what does
   not (`gate/_run/api_live.py`).
3. **Reading the code** for everything else.

Every "canonical / not exercised" claim is therefore measured. The header and API
table of each module document are emitted by `gate/gen_module_docs.py` and are
correct by construction; the prose is written on top and is never overwritten by
a re-run.

### Why the precaution

This repository already had per-package documentation — `docs/api/`, now under
`Misc/reports/api/`. It rotted unnoticed until it carried a **SUPERSEDED** banner
and described a `dynamics` module that does not exist.

The CLEANUP chantier found **five** confidently-worded documents contradicted by
measurement: a dataclass default taken for the canonical value (F1); four
comments asserting that tests use `from_heuristic` (none does, across the whole
history); a `REPO_STATE.md` pointing at a directory that never existed; a
`STATUS.md` citing `crawlbot/planners/`, a package that does not exist; and a
line reference in CLAUDE.md that was stale by about 280 lines.

Hence the rule applied here: **a dataclass default is not the canonical value**,
and a function's name does not tell you whether it runs.

---

## Three cross-cutting traps

### 1. Defaults are not canonical values

`CentroidalNMPCConfig` declares `robot_mass=90`, `N=20`, `dt=0.05`; the canonical
run uses about 71 kg, `N=8`, `dt=0.1`. Source of truth: the "Key Parameters"
table in CLAUDE.md.

Measured exception: eight `WholeBodyQPConfig` fields and five
`CoarsePrePlannerConfig` fields are never overridden — for those the default
**is** the canonical value (`CLEANUP_CARRYOVER` C4). Two of them are hard boxes
on the terminal state of every planned step.

### 2. "Not exercised" does not mean "removable"

Three distinct classes:

| class | example | verdict |
|---|---|---|
| research sediment behind an opt-in flag | alternative AOCS modes, planner FK path | removable |
| **fallback, dead because the system is healthy** | `get_shifted_fallback`, `contact_scheduler` guards | **keep** |
| live diagnostic hook | `_diag_pure_pd`, `_diag_freeze_ref` | **keep** |

The second class looks most deletable on a coverage report and is the most
dangerous to touch.

### 3. Three exported log channels carry no signal

`H_rO` and `H_dot_est` are **identically zero** over all 2077 ticks (the estimator
is constructed but never updated); `gmo_contact_state` is **constant** (the
contact state machine is never advanced). Details in
[`aocs/force_estimator.md`](aocs/force_estimator.md) and
[`estimation/contact_estimator.md`](estimation/contact_estimator.md).

---

## Verification

These documents are **checkable**, which is the substantive difference from the
`docs/api/` that rotted:

```bash
PYTHONPATH=. python3 gate/sync_docs.py --check   # docs match the code
PYTHONPATH=. python3 gate/verify_docs.py         # every file:line and every symbol
PYTHONPATH=. python3 gate/link_audit.py          # every path cited in the repository
```

`sync_docs --check` is the one that makes staying current a routine rather than
a good intention: it exits non-zero when a symbol has been added, removed or
moved without the document following. CLAUDE.md rule 15 makes running it
mandatory after any change to `crawlbot/`.

Every symbol in the **Public API** table and every entry in the **Code map**
carries a line-anchored link into the source, and those links are *generated*.
A refactor that shifts line numbers is repaired by re-running the tool — which
is precisely the failure mode that left a stale line reference in CLAUDE.md
before this existed.

`verify_docs` fails if a `file:line` reference exceeds the file's length, or if a
cited symbol is no longer defined in `crawlbot/`.

It has already earned its keep: the first draft carried a torso-reference line
number inherited from CLAUDE.md, stale since the chantier shortened `sim_loop` by
375 lines. The real site is `sim_loop.py:2581-2584`. CLAUDE.md,
`CLEANUP_CARRYOVER` and these documents were corrected together.

**Not yet covered**: numeric values (weights, gains, thresholds), still verified
by hand against CLAUDE.md. That is the remaining soft spot.

---

## Out of scope here

- `lutze_baseline/` — the M0/Lutze comparison implementation.
- `gate/` — see `gate/README.md`.
- `Misc/` — research sediment, slated for removal.
- The deep architectural *why*:
  `docs/architecture/brainstorming_reworked_architecture.md` (specification) and
  `docs/architecture/STACK_OVERVIEW.md` (current code state).
