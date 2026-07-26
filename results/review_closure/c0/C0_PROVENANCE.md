# C0 — Provenance (+ Step 0 ground truth)

**Review-Closure Bloc 2, Phase C0.** Read-only except for the two navigation-document
refreshes mandated by Step 0.3.

---

## Header (mandatory)

| item | value |
|---|---|
| commit worked on | **`eecbf94`** (`sim_loop readability`), branch `claude/review-closure-bloc-2-uwu1x7` |
| relation to `main` | `origin/main` = **`bfd5509`**; this branch = `bfd5509` + **51** commits, **0 behind** |
| date | 2026-07-26 |
| python | 3.11.15 (x86_64) |
| mujoco | 3.10.0 |
| pinocchio (`pin`) | 3.9.0 |
| casadi / ipopt | 3.7.2 / IPOPT present (`nlpsol('ipopt')` solves) |
| numpy / scipy | 2.3.5 / 1.17.1 |
| qpsolvers / osqp | 4.13.0 / 1.1.3 |
| env vs `gate/environment.lock` | **exact match on all 7 pinned entries** |

**Artifact paths cited by this report**

- `gate/last_verdict.json` (this run's machine-readable verdict)
- `gate/_run/suite_verdict.json`
- `results/gate_run_scratch/sim_log.json` (replay scratch — 2077 ticks)
- `results/j2_adjconv/{c25_fulldiag.csv, u25_fulldiag.csv, c25_fulldiag_meta.json, u25_fulldiag_meta.json, canonical2p5_result.json, t4b_trace_900s.csv, t4b_ltot_900s.csv, t4_settle450_{result,analysis}.json, t4b_settle900_{result,analysis}.json}`
- Refreshed this phase: `REPO_STATE.md`, `docs/architecture/STACK_OVERVIEW.md`

---

## Step 0 — ground truth

### 0.1 Tag — **CONFIRMED ABSENT (flagged)**

`git tag -l` returns nothing. `paper-2p5-base` does not exist. Placing it is Idriss's
action; this report only flags it, as instructed.

**The paper's provenance currently rests on a bare, untagged hash.** Two hashes matter and
they are not the same one:

- **`bfd5509`** — the artifact baseline. Present, ancestor of `main`, and `main` HEAD itself
  (see 0.2). This is the commit the committed `c25_fulldiag.csv` was baselined at, per
  `gate/README.md` and `gate/EXCEPTIONS.md`.
- **`32aefaf`** — `freeze(canonical-2p5): tau_w_max 2.5 (controller + plant) + Add-5 weights`.
  **Confirmed an ancestor of `bfd5509`** (Step 0.4). ✔

### 0.2 Branch base — **DEVIATION FROM THE BRIEF, resolved in the brief's favour**

The brief states "Branch `review-closure` from `main` HEAD (`eecbf94`)" and that the hygiene
chantier was "gate-passed and **promoted to `main`**". Measured:

```
origin/main                                = bfd5509   (pre-chantier)
claude/review-closure-bloc-2-uwu1x7 (HEAD) = eecbf94   = main + 51 commits, 0 behind
```

**The chantier is not on `main`.** `eecbf94` is the tip of the chantier work, and it is what
this session's designated branch already points at. So the *state* the brief intends to work
from is exactly the state this branch has; only the claim about where it lives is wrong.

Consequences, recorded so they are not rediscovered later:

1. Work proceeds on `eecbf94` — the brief's intended content — under the session's mandated
   branch name `claude/review-closure-bloc-2-uwu1x7`. No `review-closure` branch is created;
   creating one would fragment the work across two branch names for no gain.
2. The brief's separation rule ("do not import cleanup-branch changes into `review-closure`")
   **cannot be honoured as literally written**, because the base *is* the cleanup branch. What
   is honoured instead, and what the rule is actually for: **no PORT_AUDIT ticket and no
   chantier fix will be worked here.** If a phase turns out to need one, the phase stops and
   reports rather than reaching across streams.
3. The Tier-0 gate argument still holds, and was re-verified rather than assumed (C0.1): the
   replay at `eecbf94` is byte-identical to the artifact baselined at `bfd5509`.

### 0.3 Layout — verified; **both** navigation documents were stale; both refreshed

Package layout confirmed as the brief describes, with corrections:

- `crawlbot/` = 34 tracked files (26 modules + 8 `__init__`). `solvers/`, `planning/`,
  `aocs/`, `estimation/`, `simulation/`, `core/`, `diagnostics/` all present as listed.
- **Additions the brief's list omits:** `simulation/tick_logging.py` (per-tick recorders split
  out in CLEANUP-32), `core/state_conversions.py`, `core/com_to_torso_mapping.py`,
  `planning/coarse_preplanner.py`, `simulation/plotting.py`.
- **Deletion:** `planning/constrained_geodesic.py` is gone (CLEANUP-17).

**Which model the canonical run loads — answered:**

| role | file | md5 | cited at |
|---|---|---|---|
| PLANT (MuJoCo) | `models/VISPA_crawling_rwa3.xml` | `df2c210f5cefb79b9a945f95c322ce9a` | `gate/run_gate.py:45`, `scripts/diag_cooperative_arms.py:51` |
| CONTROLLER (Pinocchio) | `models/VISPA_crawling_fixed.urdf` | `4912179b0dc026a6f5165b0298b552f5` | `gate/run_gate.py:46`, `scripts/diag_cooperative_arms.py:488` |

**Not** `models/VISPA_crawling.xml` and **not** `URDF_models/VISPA_crawling.urdf`, both of which
the brief names. Neither is loaded by any code path in `crawlbot/`, `scripts/` or `gate/`
(grep over `VISPA_crawling` across all three trees returns only the two files above).
`URDF_models/` survives only because the STL/DAE meshes live there.

**Navigation documents.** The brief says refresh "whichever is wrong". Both were wrong, in
different ways, so both were refreshed in one commit:

- **`REPO_STATE.md`** — snapshot dated 2026-07-18 on a *pre-chantier* branch. Wrong on:
  `results/` 1907 files (now 117, 3 subdirs), `scripts/` 187 (now 5), the suite's "2 pre-existing
  failures" (now 0 failed / 1 xfail), `constrained_geodesic.py` listed as live, `ik.py` credited
  with "path feasibility" (retired, CLEANUP-30), and no mention of `gate/` at all.
- **`docs/architecture/STACK_OVERVIEW.md`** — the more serious one. Dated 2026-05-27 and declared
  "code-ground-truth", it described **a controller that no longer exists**: the cooperative split
  (strict-P1 torso-angular, co-equal P2 through `N_torso_ang`), `α_ee = 3000`, `w_hw_slack = 1e4`,
  soft-CoM (`alpha_com_soft` — a field since **removed**, not merely zeroed), the
  `legacy_corrected` AOCS, and "hard `L_max`/`τ_w` boxes are ∞ (dead)" when both are live at 10 Nms
  and 2.5 Nm. Rewritten against source read at `eecbf94`, with a new §0 documenting the
  **four-layer config chain** (dataclass → `_make_m7_config()` → `dca.main` overrides →
  canonical kwargs) — because three of the traps found this session were "a `SimConfig` default
  read in isolation is not the canonical value".

### 0.4 `32aefaf` ancestry — **CONFIRMED**

`git merge-base --is-ancestor 32aefaf bfd5509` → yes. `32aefaf` is also an ancestor of `eecbf94`.

### 0.5 Inherited audit anchors — **CONFIRMED DEAD; re-location protocol adopted**

`docs/architecture/PORT_AUDIT.md` (308 lines) and `PORT_SYNTHESIS.md` (48 lines) are committed
in-repo and were written against the flat pre-package layout. Every `file:line` in them is
positionally void. They are used here as **semantic maps only**; each anchor is re-located
against the current tree before it is quoted, and the report says so at the point of use.

Two anchors were already found to have drifted **inside** current documentation, i.e. this is
not confined to the inherited audits:

- `CLAUDE.md` cites `sim_loop.py:2573-2576` for the SS torso-reference routing. It is now at
  `sim_loop.py:2378-2380` (the two-task exclusion) and `2468-2471` (the raw-quintic branch).
- `CLAUDE.md` cites `config.py:71-72` / `:80` / `:84`; the live lines are `:70-71` (hw box),
  `:79` (`tau_w_max`), `:83` (`aocs_tau_w_max`). Values are correct; positions are off by one.

Both are consistent with CLAUDE.md's own "Remaining Work" note that its prose anchors are checked
by nothing. Not fixed here — that is a hygiene-stream ticket.

---

## C0.1 — Gate verdict: **PASS**

```
PYTHONPATH=. MUJOCO_GL=disabled python3 gate/run_gate.py
```

| check | verdict | detail |
|---|---|---|
| [1] canonical replay + export | PASS | replay rc=0 (272.7 s), export rc=0 |
| [2] artifact identity | **PASS** | **2077 rows × 132 928 fields byte-identical** |
| [3] two-model consistency | PASS | 15 links, 14 joints, total mass 71.056 kg, 0 fails |
| [4] environment pin | PASS (advisory) | 0 mismatches vs `gate/environment.lock` |
| **overall** | **PASS** | 277.1 s → `gate/last_verdict.json` |

**Exception invoked: exactly one, and it is the definitional exclusion, not a Tier-1 exception.**
Columns `qp_time_ms` and `nmpc_time_ms` are excluded from the byte-comparison as
nondeterministic wall-clock instrumentation (`gate/EXCEPTIONS.md`, "Definitional exclusion").
66 total columns − 2 excluded = 64 compared × 2077 rows = 132 928 fields. The Tier-1
metric-equivalence table in `EXCEPTIONS.md` remains **empty** — no signed-off numerical
deviation exists.

**Physical read-out** (`gate/dock_check.py results/gate_run_scratch/sim_log.json`, exit 0) —
because "the CSV is byte-identical" is a hash statement, not a physics one:

| step | at-weld d [mm] | vs frozen | margin to 5 mm |
|---:|---:|---|---:|
| 1 | 4.02 | delta +0.0000 | 0.98 |
| 2 | 4.89 | +0.0000 | 0.11 |
| 3 | **4.99** | +0.0000 | **0.01** |
| 4 | 4.97 | +0.0000 | 0.03 |
| 5 | 4.95 | +0.0000 | 0.05 |
| 6 | 4.62 | +0.0000 | 0.38 |

θ_s peak 0.540° · h_w peak axis 4.102 / norm 4.243 Nms · e_com peak 0.154 m · qp_fail 0 —
all matching frozen 2.5. Structure drift over the run: 22.49 mm position, max ‖ω_s‖ 1.97 mrad/s.

**Component gate** (the half `run_gate` structurally cannot do), run for completeness:
`gate/run_suite.py --fast` → **PASS**, 199 tests, **198 passed, 0 failed, 0 errors, 0 skipped,
1 xfail** (44 s). The xfail is
`test_coarse_preplanner::TestPositionDependentEnvelope::test_far_infeasible_under_tight_rate`,
`strict=True`, behaving as documented.

→ **The brief's Step-0.2 precondition is satisfied. Proceeding is safe.**

## C0.2 — Artifact set at `results/j2_adjconv/` — **CONFIRMED complete**

| artifact | shape | md5 |
|---|---|---|
| `c25_fulldiag.csv` | **66 cols × 2077 rows** ✔ | `c1cbc5e74fe4034523a23623dea44428` |
| `u25_fulldiag.csv` | **66 cols × 1905 rows** ✔ | `14fb3ace9f562a17145a4a029085a630` |
| `t4b_trace_900s.csv` | 13 cols × 4823 rows | `7fd69891a07195d9286d7817e14e5e04` |
| `t4b_ltot_900s.csv` | 5 cols × 1929 rows | `ea93d18fd1446644ad45d9739efcefb3` |
| T4 analysis JSONs | `t4_settle450_result.json`, `t4_settle450_analysis.json` | present |
| T4b analysis JSONs | `t4b_settle900_result.json`, `t4b_settle900_analysis.json` | present |
| canonical summary | `canonical2p5_result.json` | present |
| per-run metadata | `c25_fulldiag_meta.json`, `u25_fulldiag_meta.json` | present |

The 66-column count the brief cites is confirmed for both fulldiag CSVs. The directory carries
95 tracked files in total (55 `PHASE_*.md` reports + the JSON/CSV artifact set); nothing in the
expected set is missing.

Two data points lifted from `c25_fulldiag_meta.json` while confirming it, because later phases
depend on them:

- the traversal's last logged snapshot is `final` at **t = 84.64 s** (feeds C1.6);
- per-dock `twist_weld` is **already recorded** in the meta JSON for all six docks
  (0.020058 / 0.007735 / 0.005781 / 0.007170 / 0.004990 / 0.007020) — C2.1 must check what this
  scalar is before adding a 6-D twist channel, exactly as C2.2 was told to check before adding
  timing channels.

## C0.3 — Run config vs the `32aefaf` frozen values — **NO DRIFT**

**Two things were compared, because either alone would be misleading.**

**(a) `SimConfig` field-by-field, `32aefaf` → `eecbf94`** (`crawlbot/simulation/config.py`,
same path at both revisions). All 40 controller / envelope / weight / gain / gate / timing fields
checked are **identical**:

`tau_w_max 2.5` · `aocs_tau_w_max 2.5` · `aocs_K_hw 2.0` · `aocs_K_omega 50.0` ·
`aocs_K_d 25.0` · `aocs_K_theta 1.0` · `hw_min/max ∓5.0` · `L_max 10.0` ·
`ss_alpha_ee 1e3` · `ss_alpha_posture 2e1` · `ss_alpha_wrench 1.0` · `ss_alpha_mom 4e2` ·
**`alpha_torso_pose 2e3`** · `ss_Kp/Kd_torso 6.0/5.0` · `ss_Kp/Kd_ee 10.0/12.0` ·
`ss_Kp/Kd_ee_ang 6.0/4.5` · `ss_Kp/Kd_com 3.0/3.0` · `nmpc_N 8` · `nmpc_dt 0.1` ·
`dt_qp 0.01` · `dt_nmpc 0.1` · `preplanner_kappa 0.7` · `preplanner_M 15` ·
`preplanner_a_cruise_max 0.0` · `weld_radius 0.005` · `dock_ori_threshold_deg 5.0` ·
`dock_twist_max 0.05` · `dock_use_6d_twist True` · `h_max_tight 5.0` · `w_L_nmpc 1.0` ·
`com_z_standoff −0.35` · `gait_anchor_dx 0.8` · `tau_max 20.0` · `rwa_I_w 0.01`.

The brief's stated reference point is confirmed verbatim: `alpha_torso_pose: float = 2e3` at
**`crawlbot/simulation/config.py:303`**, annotated
`# 6-D torso-pose weight — CANONICAL-2p5 / Add-5 freeze (was 5e3; …)`.

**(b) The effective canonical kwargs**, which is what actually decides the run. The canonical
overrides several dataclass defaults, so (a) alone does not establish the operating point.
`gate/replay_canonical.py:37-47` is verbatim identical to the artifact-generating script
`Misc/scripts/diag_canonical2p5_run.py:126-140` (`_run('C', 2.5, 'figC25_addfive')`):

```
mass_ratio=0.01   n_steps=6   anchor_dx=0.8   legacy=False   alpha_torso_lin=0.0
aocs_mode='legacy_pid_numerical'   K_theta=1.0  K_omega=50.0  tau_w_max=2.5  settle_seconds=20.0
ss_two_task=True  ss_alpha_mom=400.0  alpha_torso_pose=2000.0  ss_alpha_ee=1000.0
ss_alpha_posture=20.0  ss_alpha_wrench=1.0  ss_kp_torso=3.0  ss_kd_torso=2.5
qp_envelope_exact=True  interstep_settle_alpha_wrench=3.0  interstep_settle_epsilon_v=5e-3
```
plus `HierarchicalQP.regularization = 1e-6` pinned explicitly (`gate/replay_canonical.py:20-29`).

**Three places where the dataclass default is NOT the canonical value** — recorded here because
each is a live mis-quotation risk for the paper:

| quantity | `config.py` default | canonical (effective) |
|---|---|---|
| torso task gains | `ss_Kp_torso = 6.0`, `ss_Kd_torso = 5.0` (`:351-352`) | **3.0 / 2.5** |
| AOCS mode | `'legacy'` (`:98`), forced to `'legacy_corrected'` by `_make_m7_config():49` | **`'legacy_pid_numerical'`** |
| QP envelope box | `qp_envelope_exact = False` (proxy `M_λ`) (`:338`) | **`True`** (exact, origin-referenced Ḣ_s) |

The first of these is the one the paper depends on, and it lands in the paper's favour: the r17
text's `K_p = 3, K_d = 2.5` **matches the canonical**, not the dataclass default. Formal
CONFIRMED/DEFECT adjudication is C1.2's job; this is recorded as provenance.

**Plant-side enforcement of the freeze** (the third of the three enforcement points) verified in
the MJCF the canonical actually loads:
`models/VISPA_crawling_rwa3.xml:324-326` — `<motor name="act_rw_{x,y,z}" … ctrlrange="-2.5 2.5"/>`.
The `_mutate_mjcf` structure-mass scaling is a **no-op at `mass_ratio = 0.01`** (guarded by
`abs(mass_ratio - 0.01) > 1e-9`, `scripts/diag_cooperative_arms.py:88`), so the canonical runs the
committed MJCF inertials unscaled, and the file is restored under an md5 assert after every run.

---

## Incidental finding — a gate that cannot pass on a fresh clone (hygiene stream, NOT fixed here)

Running the full documentation-gate routine from `CLAUDE.md` on the clean checkout:

| checker | rc | verdict |
|---|---:|---|
| `gate/verify_docs.py` | 0 | OK — 34 documents, every `file:line` + symbol resolves |
| `gate/verify_params.py` | 0 | OK — 15 CLAUDE.md parameter rows, every cited line declares its parameter and every value matches |
| `gate/verify_roots.py` | 0 | OK — 111 root expressions |
| `gate/link_audit.py` | 0 | OK (advisory: 138 unresolved citations, 125 of them `Misc/…`) |
| **`gate/sync_docs.py --check`** | **1** | **FAIL — "25 document(s) out of date with the code"** |

The `sync_docs` failure is **not** caused by this phase's edits: this phase touched no file under
`crawlbot/` and no file under `docs/crawlbot/` (`git status`: `REPO_STATE.md`,
`docs/architecture/STACK_OVERVIEW.md`, `results/review_closure/` only). It reproduces on the
untouched checkout at `eecbf94`. Root cause, from the tool's own first line of output:

```
note: gate/_run/cov/cov.json absent — coverage columns left as-is.
      regenerate with gate/_run/cov_replay.sh
```

- `gate/_run/` is **entirely untracked** (`git ls-files gate/_run` → empty; it holds only this
  session's replay scratch).
- **`gate/_run/cov_replay.sh` does not exist** — not tracked, and not present anywhere on the
  filesystem. It is the script `CLAUDE.md:132-134` and `gate/sync_docs.py:34` both instruct you
  to run.

So the `canonical? = yes / not exercised` coverage column can be neither reproduced nor
regenerated from a clean clone, the regenerated documents differ from the committed ones in that
column, and the check fails for all 25 module documents at once. Three further gate scripts read
the same missing file and crash outright rather than degrading:
`gate/api_live.py:14` (verified: `FileNotFoundError`, rc=1), `gate/cov_compare.py:17`,
`gate/gen_module_docs.py:14`.

**Why this matters and not just as tidiness.** `CLAUDE.md` Rule 15 states the doc gate is
"enforced, not requested" and lists `sync_docs.py --check` in the mandatory pre-commit routine.
A gate that fails unconditionally on a fresh clone is a gate that gets routinely overridden, and
the failure mode it was built to catch — a symbol added, removed or moved without its document
following — is then invisible inside the noise of 25 standing failures. That is the same class of
defect as CLEANUP-21 (a checker that could not see what it claimed to check).

**Not fixed here, by the brief's rule.** The gate is hygiene-stream property, and the fix is
either committing `cov_replay.sh` + `cov.json` or decoupling the coverage column from the
identity check — a chantier decision, not a review-closure one. Reported for Idriss to route.
No review-closure phase depends on `sync_docs`, and the two gates the brief does depend on
(`run_gate.py`, `run_suite.py`) both PASS.

---

## Deviations from the brief, recorded

| # | brief says | measured | action |
|---|---|---|---|
| 1 | chantier "promoted to `main`"; `main` HEAD = `eecbf94` | `main` = `bfd5509`; `eecbf94` is the chantier tip, = this session's branch | work on `eecbf94` (the intended *content*); no `review-closure` branch created — the session's mandated branch name is used |
| 2 | canonical MJCF `models/VISPA_crawling.xml`, URDF `URDF_models/VISPA_crawling.urdf` | canonical pair is `models/VISPA_crawling_rwa3.xml` + `models/VISPA_crawling_fixed.urdf`; neither file the brief names is loaded anywhere | reported; the correct pair is used throughout |
| 3 | "refresh whichever [nav doc] is wrong" | **both** wrong; `STACK_OVERVIEW.md` described a superseded controller while being declared code-ground-truth | both refreshed, one commit |
| 4 | C1.5 dock-gate thresholds live in `sequence_loader.py` / `contact_estimator.py` | they are `SimConfig` fields (`config.py:35-58`) | noted; C1.5 will report the real locations |

---

## STOP

C0 is complete and the precondition to continue is met: **gate PASS, byte-identical, one
definitional exclusion, no Tier-1 exception invoked, environment exactly on the lock.**

Awaiting explicit GO for **Phase C1 — Paper↔code exactness**. No C1 adjudication is recorded
above; the C1-adjacent facts noted here (torso gains, dock-gate location, structure inertia,
traversal end time) are provenance observations and will be re-derived with full `file:line`
evidence in `C1_EXACTNESS.md`.
