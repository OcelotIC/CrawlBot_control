# PHASE CLEANUP-4 — `centroidal_nmpc.py` dead-code sweep (measured, not inferred)

Answers "is there any dead code left in `centroidal_nmpc.py`?" with **line coverage of a full
canonical replay** rather than static reasoning. Also **retracts CLEANUP-2 finding F1**, which
static reasoning got wrong. No code removed in this phase.

## Method

`gate/replay_canonical.py` (the full 6-step managed traversal) run under `coverage.py 7.15.2`,
scoped to the two solver modules. Result:

| module | statements | never executed | coverage |
|---|---|---|---|
| `crawlbot/solvers/centroidal_nmpc.py` | 235 | **28** | 88 % |
| `crawlbot/solvers/nmpc_solver.py` | 258 | 13 | 95 % |

## ⛔ Retraction of F1 — the h_w / M3 path is LIVE, not dead

CLEANUP-2 F1 claimed the whole wheel-momentum path was dead on the canonical because
`SimConfig.enforce_hw_conservation` defaults to `False` and `diag_cooperative_arms.py` never
sets it. **Both facts are true and the conclusion was still wrong**: `dca.main` does not use a
bare `SimConfig`. It takes

```
cfg = r_single._make_m7_config()          # diag_cooperative_arms.py:268
```

and `scripts/run_m7_single_step.py:49` sets **`enforce_hw_conservation=True`**, with
`h_max_tight=np.full(3, 5.0)` and `kappa_terminal=1.0`.

Measured on a real run by instrumenting `CentroidalNMPC.build()`:

```
enforce_hw=True  ng_path=17  ng_term=6  N=8  dt=0.1  tau_w=2.5  mass=71.056
```

`ng_path=17` (= 4 SOC + 6 Ḣ_s + 1 linear-momentum + **6 h_w box**) and `ng_term=6` confirm
both M3 blocks are active, and coverage shows lines 313–316 and 333–344 executing.

**Consequences of the retraction:**
- `compute_c_simple()`, the `hw_current` argument, `h_max_tight`, `kappa_terminal`, and
  `sim_loop.py:2704-2707`'s `hw_for_nmpc` computation are **all live**. Nothing there is dead.
- **"Tier B" is withdrawn as a removal candidate.** It is live canonical code implementing the
  published B2 Option-B mechanism — deleting it would have changed the controller.
- The CLEANUP-2 A1 probe remains valid for what it actually tested (h_w is ignored *when the
  flag is off*); the mistake was inferring the flag's canonical value from the dataclass
  default instead of tracing `dca`'s config construction.

Lesson, matching CLAUDE.md rule 1: a dataclass default is not the canonical value. Trace the
config that the run actually builds, or instrument it.

## What is genuinely never executed (all 28 statements, classified)

| lines | code | verdict |
|---|---|---|
| 138 | `config = CentroidalNMPCConfig()` — the `config is None` branch | **keep** — defensive; `sim_loop` always passes a config |
| 373 | `opts.update(solver_opts)` | **keep** — exercised by `tests/test_liabilities.py:283,323` |
| 431 | `raise RuntimeError("Call build() before solve().")` | **keep** — defensive guard |
| 459 | `self._last_success = False` | **keep** — the failure branch; 0 failures in 709 solves |
| **502–512** | **entire `get_shifted_fallback()` body** (11 stmts) | **keep** — the M5 infeasibility fallback. Dormant only because the NMPC never fails; it is the designed recovery path and removing it would delete the safety net |
| **587–599** | **entire `get_full_trajectory()` body** | **candidate** — zero production callers (`crawlbot/` = 0); used by 5 tests + 1 script |
| 692–697 | `__repr__` body | **keep** — debug/cosmetic |

Plus one static finding coverage cannot show:

- **`ContactPhase` is a dead import** — `centroidal_nmpc.py:69` imports
  `ContactPhase, ContactConfig`; `ContactPhase` occurs exactly once in the file (that import
  line). `ContactConfig` is used 6×. **Genuinely dead, zero risk** — the only Tier-A-class
  item left in this file.
- `get_last_trajectory()`'s third return value (`_last_success`) is discarded at its only
  production call site (`sim_loop.py:2770` unpacks into `_`). Not dead (2 scripts read the
  tuple), but the production contract only needs two of the three.

## Answer to the question

**Almost nothing removable is left.** After Tier A, the only *genuinely* dead code in
`centroidal_nmpc.py` is the one-word `ContactPhase` import. `get_full_trajectory()` is
production-dead but test-live (Tier C, needs a ruling). Everything else the coverage run
flags is either a defensive guard or dormant-by-design safety machinery — in particular
`get_shifted_fallback()` must stay: it is uncovered precisely *because* the controller is
healthy, which is the opposite of a reason to delete it.

The large deletion that looked available at the start of this chantier (the ~85-line M3
block) **does not exist as dead code** — that was the retracted F1.

## Also in this phase — F3 deferred per Idriss's ruling

Option 2 taken: the `nmpc_ok` encoding is left untouched so the frozen paper artifact stays
byte-identical, and the trap is documented instead —
- `scripts/diag_full_diag_export.py` module docstring: a CAVEAT block stating that
  `nmpc_ok = 0` in `DS_interstep` means *not called*, that a whole-column read gives a false
  34.1 %, that the true rate is 100 % (709/709), and that consumers must filter on `phase`;
- `results/j2_figdata/INTERNAL_figdata.md`: the same caveat with the per-phase table, plus why
  the encoding is deliberately unchanged (Tier-1 exception + baseline regeneration otherwise).

Both are documentation-only; the exported CSV is unaffected (gate re-run confirms).

## Still open

- **F1 replacement question**: none — the h_w path is live, nothing to remove.
- **Tier C** `get_full_trajectory()`: remove and drop/adapt 5 tests, or keep as tested API?
- Carried over: `setup_env.sh` cmeel ABI pins, and the 5 test-written PNGs that dirty the tree
  on every `pytest`.
