# PHASE CLEANUP-22 — migration steps 5 and 6

Completes the CLEANUP-20 restructure. **Step 7 (`models/` variants) is dropped by
instruction** — no model file is touched by this chantier.

Two deviations from the CLEANUP-20 sketch, both deliberate, both in §3.

---

## 1. Result — the final structure

| dir | tracked files | what |
|---|---:|---|
| `crawlbot/` | 33 | the library |
| `tests/` | 25 | pytest suite |
| `results/` | 120 | canonical baseline + paper figure data |
| `models/` | 13 | plant + controller (untouched) |
| `docs/` | **10** | **ground truth only** |
| `lutze_baseline/` | 10 | M0 paper baseline |
| `gate/` | 9 | reproduction gate |
| `benchmarks/` | 6 | pytest benchmarks |
| `scenarios/` | 5 | `.seq` fixtures |
| **`scripts/`** | **5** | **exactly the canonical import closure** |
| `URDF_models/` | 18 | untouched (model files) |
| `Misc/` | 2068 | `scripts/` 201 · `reports/` 51 · `runs/` 1816 |

`scripts/` now contains only the five scripts the canonical run actually needs:
`diag_cooperative_arms.py`, `run_m7_single_step.py`, `diag_full_diag_export.py`,
`export_figure_data.py`, `render_traversal.py`. **200 one-off runners moved to
`Misc/scripts/`.**

`docs/` keeps the eight ground-truth files named by CLAUDE.md and `REPO_STATE` §1 —
`brainstorming_reworked_architecture.md`, `CLAUDE_CODE_HANDOFF.md`, `STACK_OVERVIEW.md`,
`STATUS.md`, `PORT_AUDIT.md`, `PORT_SYNTHESIS.md`, `IK_FORMULATION.md`, `setup_env.sh` — plus
the two reference PDFs. **43 historical memos and campaign reports moved to `Misc/reports/`**,
including `docs/api/` (stale per `REPO_STATE` §4.3).

---

## 2. The real hazard was `_root`, not the move

104 scripts compute

```python
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # "one level up"
```

and then use `_root` for **both** `sys.path` **and** path construction — `URDF`, `MJCF`,
`OUT_DIR`. Moving a script one level deeper does not raise; it silently redirects every one of
those paths. A wrong `sys.path` fails loudly, but a wrong `OUT_DIR` just writes somewhere else.

So every rewritten expression is **evaluated after the move and asserted equal to the repo
root** (`gate/_run/verify_roots.py`): the `_root = …` / `_ROOT = …` assignment and the inline
`sys.path.insert(0, <expr>)` form are extracted, `eval`'d with `__file__` bound to the script's
real new path, and compared.

**It caught one.** `scripts/diagnostics/t15_post4_singularity.py` sat one level deeper than its
siblings, so it needed **two** extra levels, not one:

```
*** 1 WRONG ROOT ***
  Misc/scripts/diagnostics/t15_post4_singularity.py
     -> /home/user/CrawlBot_control/Misc
```

That script builds `URDF` and `RUN` from `_root`. Undetected, it would have loaded a model from
`Misc/models/…` — a path that does not exist — or written its output under `Misc/Misc/`.

Final: **111 root expressions checked, all resolve to the repo root.**

The fix was then re-expressed on one line, because the multi-line form defeated the verifier's
own regex and it reported the file as *"not statically evaluable"*. An unverifiable fix to a
verification finding is not a fix.

---

## 3. Two deliberate deviations from the CLEANUP-20 plan

### 3.1 `results/j2_adjconv/PHASE_*.md` stays put

The plan proposed moving the phase reports to `Misc/reports/`. **Revised on inspection:**
`j2_adjconv` is 43 reports interleaved with the 43 data files they analyse (32 JSON, 9 CSV,
2 PNG). Moving the reports would put every one a directory away from the JSON it cites, and
CLAUDE.md points at `results/j2_adjconv/PHASE_*.md` directly. The directory is coherent as a
unit; 86 files was never the problem, 1921 was.

### 3.2 `Misc/diagnostics/` → `Misc/scripts/`, and `q1_q2` moved to `runs/`

The sketch had `Misc/diagnostics/` for the one-off scripts, but CLEANUP-21 had already put the
former top-level `diagnostic/` **data** there. Scripts and their outputs under one name is the
confusion the restructure exists to remove. Now three buckets, each with one meaning:

```
Misc/scripts/   one-off runners
Misc/reports/   historical memos and campaign reports
Misc/runs/      every run artifact (including q1_q2)
```

---

## 4. Verification

### Gate — PASS

```
[1] canonical replay + export : replay rc=0 (281.9s), export rc=0
[2] artifact identity         : PASS  (2077 rows × 132928 fields)
[3] two-model consistency     : PASS   [4] environment pin : PASS
VERDICT: PASS  (285.1s)

at-weld docks 6/6   4.02 / 4.89 / 4.99 / 4.97 / 4.95 / 4.62 mm
                    every step delta +0.0000   worst margin 0.01 mm
theta_s 0.540 deg   h_w 4.102 / 4.243 Nms   e_com 0.154 m   qp_fail 0
```

### Link audit — 0 broken, after the tool was made honest

`gate/link_audit.py` was generalised from `results/`-only to every repo-relative path, and it
immediately earned its keep by catching a bug the previous narrow check missed:

**`Misc/runs/Misc/runs/q1_q2/…`** — a double prefix, 9 occurrences across 5 files. CLEANUP-21's
citation table mapped `diagnostic/` → `Misc/runs/q1_q2/`, but there is *also* a run directory
named `diagnostic` (`results/diagnostic/` → `Misc/runs/diagnostic/`), and the rule rewrote the
substring inside it. My earlier `grep "Misc/Misc"` check did not match `Misc/runs/Misc/runs`.
**A corruption check narrower than the corruption is worthless.**

The tool then had to be tightened twice, because its first two versions cried wolf:

| version | BROKEN BY MOVE | why the count was wrong |
|---|---:|---|
| basename exists elsewhere | 35 | `sim_log.json` lives in dozens of run dirs, most gitignored |
| + unique basename AND parent dir gone | 11 | still flagged gitignored scratch dirs created at run time |
| + skip gitignored paths and prose `...` | **5** | genuine |

The final 5 were **pre-existing rot, not migration damage**: `docs/architecture/STATUS.md` — a
*kept ground-truth document* — cited `crawlbot/planners/…` (6×) and another report cited
`crawlbot/control/…` (2×). Neither package directory exists; they are `planning/` and
`solvers/`. Corrected.

That is the **fourth** confidently-worded document this chantier has found contradicted by
measurement, after the F1 dataclass default, the `from_heuristic` comments, and
`REPO_STATE`'s `figC25_addfive`. Final state: **0 broken links**.

### Other checks

- **CLAUDE.md**: every path it cites still resolves (checked explicitly — it is the file read at
  session start, so a broken pointer there is the most expensive kind).
- **pyflakes** over all 199 moved scripts: 0 undefined names, 0 syntax errors.
- **`crawlbot/` diff**: comment-only. The library is untouched by the entire restructure.
- **7 cross-imports** of moved scripts rewritten to `Misc.scripts.<module>`; the three most-imported
  scripts (`run_m7_single_step` 45×, `diag_cooperative_arms` 24×, `export_figure_data` 17×) all
  stayed in `scripts/`, so those imports were unaffected.

---

## 5. What remains

1. **`out_dir_override` still resolves under `results/`** (carried from CLEANUP-21 §5.1). A
   future `dca --out-dir myrun` writes `results/myrun`. Fixing it means moving the base to
   `Misc/runs/` and updating three gate files plus `.gitignore` together — it touches the safety
   net, so it deserves its own commit.
2. **144 unresolved citations remain, none broken by any migration**: 7 refer to files deleted
   earlier in history (4 of them to `constrained_geodesic.py`, removed in CLEANUP-17), and 137
   have never existed. Worth a dedicated prose pass over the reports; entirely separate from the
   restructure.
3. `REPO_STATE.md` and `VISPA_OPEN_ITEMS_2026-06.md` remain at root. `REPO_STATE` is now largely
   superseded by CLEANUP-20 and contains the `figC25_addfive` phantom; deciding its fate is a
   documentation call, not a structural one.
4. `benchmarks/` (6 files) could fold into `tests/`. No strong signal either way.
