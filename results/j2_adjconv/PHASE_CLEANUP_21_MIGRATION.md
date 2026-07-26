# PHASE CLEANUP-21 — migration steps 1, 3, 4

Executes the CLEANUP-20 migration plan. **Step 2 (`URDF_models/`) is dropped by instruction —
no model file (`.xml`, `.urdf`, mesh) is touched by this pass.**

Everything is a **move**, not a delete. See §1.

---

## 1. Why move rather than delete

The question was whether deleting the 361 MB of run residue could be harmful. Three facts
decide it:

1. **Deleting does not shrink the repository.** Git history retains every blob. A fresh clone
   pulls the same bytes whether the files exist at HEAD or not. Deletion buys exactly the same
   readability as moving and zero size reduction.
2. **The residue is cited by path.** 432 citations across 191 tracked files point into those
   directories — the phase reports' provenance trail. Deleting turns every one into a pointer to
   nothing, which is *precisely* the `figC25_addfive` failure documented in CLEANUP-20 §5.2.
3. **Nothing is permanently lost either way** (`git checkout <commit> -- <path>` recovers a
   deleted file), so deletion's only distinctive effect is the dangling references.

Move + rewrite the citations: same clean tree, no dangling pointers, provenance intact.

If the 377 MB is a *clone-size* problem rather than a readability problem, neither moving nor
deleting helps — that would need history rewriting, which is a separate decision.

---

## 2. Result

| dir | tracked files before | after |
|---|---:|---:|
| `results/` | **1921** | **119** |
| `diagnostic/` | 12 | — (moved) |
| `Misc/` | — | **1815** |

`results/` is now **12.2 MB / 119 files**, and holds only what is load-bearing:

| kept | why |
|---|---|
| `j2_adjconv/` | `c25_fulldiag.csv` is the gate's byte-identity baseline, + the campaign reports |
| `j2_figdata/` | paper figure data |
| `hero_render/` | paper hero figures |
| `M7_1pct_3step_v22_t15_fk/` | soft fixture — `test_fk_reference_consistency.py:347` |
| `gate_run_scratch/` | gitignored gate scratch |

```
Misc/
├── diagnostics/q1_q2/    the former top-level diagnostic/ (12 files)
└── runs/                 160 run directories + 4 loose files (371 MB)
```

### Step 3 — pytest no longer dirties the tracked tree

Three `OUTPUT_DIR` constants now point at `results/test_scratch/`, which is gitignored:

| file | constant |
|---|---|
| `tests/test_reworked_qp.py:37` | `OUTPUT_DIR` |
| `tests/test_reworked_qp.py:560` | `TMOM_OUTPUT_DIR` |
| `tests/test_nmpc_conservation.py:36` | `OUTPUT_DIR` |

This closes **`CLEANUP_CARRYOVER` §C2**: the suite rewrote five tracked PNGs on every run with
byte-different matplotlib output, so the repository could never be verified clean after
`pytest` — which undercut the whole bit-identity discipline. `test_aocs_orbital.py` needed no
change; its `results/M4_baseline_1pct/` reference is a docstring mention, not a write.

---

## 3. The citation rewrite, in two passes — and why one was not enough

**Pass 1** rewrote string-literal `results/X` → `Misc/runs/X` for moved names only, with a
trailing `(?![A-Za-z0-9_])` boundary so `results/figA_canon` could not corrupt
`results/figA_canon_7step`. 432 citations, 191 files.

**Pass 1 was incomplete, and the gap was dangerous.** The runner scripts do not write string
literals — they build paths with `os.path.join(_root, 'results', 'X')`, which the regex cannot
see. Two consequences:

- **`scripts/render_traversal.py:40` still *read* a directory that had just moved.** A silent
  breakage: the docstring said `Misc/runs/…`, the code opened `results/…`.
- Docstrings and code disagreed across 75 files — the exact defect class this chantier keeps
  finding in *other people's* documents.

**Pass 2** rewrote the `os.path.join` form for moved names only: 109 paths across 75 files.

**Pass 3** fixed three `dca` branches that build their directory name with an f-string, so
neither regex could see the name. They had been left on `results/` beside three moved siblings —
the same `if/elif` chain writing to two different roots, which is worse than either choice.

### What was deliberately *not* rewritten

`scripts/diag_cooperative_arms.py:485`:

```python
else os.path.join(_root, 'results', out_dir_override))
```

`out_dir_override` resolves a relative **name** under `results/`, and the gate passes
`'gate_run_scratch'`. That path is referenced by `gate/run_gate.py`, `gate/replay_canonical.py`
and `gate/dock_check.py`, and is gitignored. Changing it would mean changing the safety net in
the same commit as the thing it guards. Left alone; noted in §5.

Two test docstrings also had to be corrected *back*: pass 1 rewrote them as moved-artifact
citations, but they describe where the test **writes**, which step 3 had just changed to
`results/test_scratch/`.

---

## 4. Verification

### Gate — PASS

```
[4] environment pin           : PASS
[3] two-model consistency     : PASS  (15 links, 14 joints, total 71.056 kg)
[1] canonical replay + export : replay rc=0 (268.9s), export rc=0
[2] artifact identity         : PASS  (2077 rows × 132928 fields)
VERDICT: PASS   (env PASS, 272.3s)

at-weld docks  6/6   4.02 / 4.89 / 4.99 / 4.97 / 4.95 / 4.62 mm
                     every step delta +0.0000   worst margin 0.01 mm
theta_s 0.540 deg    h_w 4.102 / 4.243 Nms    e_com 0.154 m    qp_fail 0
```

### `crawlbot/` is byte-identical

`git diff HEAD -- crawlbot/` is **empty**. The library was not touched by the restructure at
all — every change is in `scripts/`, `tests/`, `.gitignore` and file locations.

### Link audit — 0 links broken by the migration

`gate/link_audit.py` resolves every `results/…` and `Misc/runs/…` token cited in tracked
`.md`/`.py` and classifies each failure:

```
159 distinct paths cited; 62 do not resolve
  broken by migration      : 0
  f-string prefixes        : 5   (tokenizer artefacts, e.g. `results/M7_1pct_`
                                  from f"results/M7_1pct_{n_steps}step")
  pre-existing dangles     : 57  (never in history — predate this pass)
```

Zero breakage is the point: it is what distinguishes *move + rewrite* from *delete*.

### Two edits landed after the gate started

Both provably unreachable from the replay, so the verdict stands:

- two **docstring** corrections in `tests/` (no executable change);
- `run_m7_single_step.py:315`, inside `if __name__ == "__main__":`. `dca:48` does
  `import scripts.run_m7_single_step as r_single` — a module import, so the `__main__` block
  never executes on the canonical path.

---

## 5. Follow-ups this creates

1. **`out_dir_override` still resolves under `results/`.** A future `dca --out-dir myrun`
   writes `results/myrun`, re-polluting the directory this pass just cleaned. Changing the base
   to `Misc/runs/` means updating the three gate files and `.gitignore` together — a small
   change, but it touches the gate and deserves its own commit.
2. **57 cited paths do not resolve, all pre-existing** — including the CLEANUP-20 §5.2 phantom — not in `results/`, not in `Misc/runs/`,
   not in git history: `figC25_addfive` (12 citations — the CLEANUP-20 §5.2 phantom),
   `figures`, `figC_sw_s5_x1`, `figC_sw_s`, `figC_userw2`,
   `M7_1pct_3step_v22_t15_trajIK_ondemand`. Some are probably f-string templates quoted
   literally into prose. They were dangling *before* this migration and remain so; worth a
   dedicated pass over the reports.
3. ~~`Misc/runs/` has no index.~~ **Done** — `Misc/runs/README.md` indexes all 160 directories with file count, size and citing report.
4. Steps 5–7 of the CLEANUP-20 plan remain: the 181 non-canonical `scripts/`, the historical
   `docs/`, and the `models/` variants (the last now explicitly out of scope).
