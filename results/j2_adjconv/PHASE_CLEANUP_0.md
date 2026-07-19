# PHASE CLEANUP-0 — founding session of the repository-hygiene chantier

**Safety apparatus only. No cleanup happened.** The gate exists before the first broom stroke.
Founding docs read first: `docs/architecture/PORT_AUDIT.md` (inventory, bins A–E) +
`docs/architecture/PORT_SYNTHESIS.md` (roadmap; added to the repo this session).

## STEP 0 — provenance tag

Annotated tag **`paper-2p5-base` → `bfd5509`** (main HEAD), message *"IEEE Access paper 2.5
artifact base; canonical config frozen at 32aefaf (ancestor)."* `bfd5509` is the first mainline
commit carrying the paper's **complete** artifact set (66-col fulldiag CSVs, `t4b_trace_900s.csv`,
the T4/T4b scripts); `32aefaf` (config freeze) is a documented, untagged ancestor.

⚠ **The managed git remote does not sync tags** — no `refs/tags` namespace exists on it; tag push
fails deterministically and the GitHub tooling exposes no create-tag call. The tag lives as a
**local** annotated tag + this documented equivalence to `bfd5509` (permanent on `main`).
Pushable from a developer machine: `git push origin paper-2p5-base`.

## STEP 1 — long-lived branch

**`cleanup-main`** created from `paper-2p5-base` (= `bfd5509`). Sub-branches: `cleanup/<topic>`.

⚠ **Naming deviation, needs Idriss's ruling.** Git cannot hold a branch `cleanup` **and** branches
`cleanup/<topic>` at once (ref dir/file conflict), and a stale `cleanup/01-index-md` (abandoned
2026-04-29 branch) already occupies that namespace. `cleanup-main` is the **non-destructive** choice:
it keeps the "cleanup" token, preserves the `cleanup/<topic>` sub-branch convention, and deletes
nothing. If the bare name `cleanup` is required, delete `cleanup/01-index-md` and rename (sub-branches
would then have to be `cleanup-<topic>`).

## STEP 2 — the gate (`gate/`) — the only code written this session

`PYTHONPATH=. python3 gate/run_gate.py` → four checks, one verdict, `gate/last_verdict.json`.

| # | check | what it does |
|---|---|---|
| 1 | canonical replay | `gate/replay_canonical.py` re-runs the frozen **managed (C)** scenario into `results/gate_run_scratch/` (exact `dca.main` C kwargs + `regularization=1e-6`, as the artifact was generated), then exports the 66-col fulldiag. No committed artifact touched. |
| 2 | artifact identity | field-by-field vs committed `results/j2_adjconv/c25_fulldiag.csv`. Byte-identical on all reproducible columns; the two wall-clock timing columns (`qp_time_ms`, `nmpc_time_ms`) are excluded by definition (`gate/EXCEPTIONS.md`). |
| 3 | two-model consistency | loads MJCF plant + URDF controller; diffs BIN A-a2 (per-link **composite** mass/COM/principal-inertia — `tool_*` lumped into `Link_6` to match Pinocchio's absorbed fixed joint — total mass, joint order + limits, tool/torso placements) + BIN B-b1 naming. Names any disagreement. |
| 4 | environment pin | live versions vs `gate/environment.lock`; **WARN** on mismatch (later: FAIL). Creates the lock on first run. |

Files: `gate/run_gate.py`, `gate/replay_canonical.py`, `gate/environment.lock`, `gate/EXCEPTIONS.md`,
`gate/README.md`, `gate/last_verdict.json`. Scratch (`gate/_run/`, `results/gate_run_scratch/`) is git-ignored.

## STEP 3 — baseline proof (must PASS trivially)

**VERDICT: PASS** (172 s total; replay 166 s, export 0.9 s).

| check | result |
|---|---|
| 1 canonical replay | rc=0 — 6-step managed traversal re-ran clean |
| 2 artifact identity | **PASS — byte-identical, 2077 rows × 132 928 fields** (all 64 reproducible columns; excl `qp_time_ms`, `nmpc_time_ms`) |
| 3 two-model consistency | **PASS** — 15 links + 14 joints, total mass 71.056 kg, all quantities agree to ≤ 1e-15 (limits ≤ 2.7e-6 rad = π-truncation) |
| 4 environment pin | WARN — lock created from live env this run (next run compares → PASS) |

The bit-identity of the re-run canonical against the committed baseline is the reference every future
session is measured against.

## Environment lock (founding baseline)

```json
{ "python": "3.11.15", "numpy": "2.3.5", "mujoco": "3.10.0",
  "pinocchio": "3.9.0", "casadi": "3.7.2", "scipy": "1.17.1", "ipopt_available": true }
```

## Standing rules recorded

- **Acceptance (two-tier):** Tier-0 = gate bit-identity (default, no sign-off). Tier-1 =
  metric-equivalence exception only with justification + Idriss's sign-off, logged in
  `gate/EXCEPTIONS.md`. (The 2 excluded timing columns are a definitional exclusion, not an exception.)
- **Governance:** work in `cleanup/<topic>` → PR into the integration branch (`cleanup-main` pending
  the naming ruling), **never `main`**; Idriss is the sole merger. `main` frozen except review-driven
  changes, cherry-picked onto the integration branch immediately (one-way sync). Promotion = one final
  PR integration→`main` after paper acceptance, reviewer = the gate.
- **Work order (sessions 1+, one topic per PR, gate green before & after each):** (1) `robot.yaml`
  consolidation; (2) the five PORT_SYNTHESIS latent-bug tickets; (3) dead-code sweep from the
  QP-STACK/FF-ORBITAL inventories; (4) no structural generalization without a named external need.

## Two rulings needed before the next session pushes

1. **Branch name** — accept `cleanup-main` (non-destructive), or delete the stale `cleanup/01-index-md`
   so the integration branch can be the bare `cleanup` (then sub-branches become `cleanup-<topic>`).
2. **Tag** — leave `paper-2p5-base` as a local tag + `bfd5509` anchor (you push the tag), or I add a
   pushable branch marker.
