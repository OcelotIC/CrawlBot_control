# COM_GAIN_AUDIT_REPORT — CoM Gain Semantics Audit (Phases 0–2)

**Brief:** CoM Gain Semantics Audit and Controlled Fix (Idriss Chelikh)
**Branch:** `claude/com-gain-semantics-audit-j0u6yr` **Code state:** `eecbf94`
**Status: Phases 0, 1, 2 COMPLETE. Phase 3 BLOCKED at the human stop gate.**
**No behavioural code was changed. No committed artifact was modified. Nothing merged.**

Companion documents:
`COM_GAIN_STATIC_TRACE.md` (Phase 0) ·
`COM_GAIN_EXECUTABLE_PROOF.md` (Phase 1) ·
`COM_GAIN_ACTIVATION_EVIDENCE.md` (Phase 2)

---

## 8.1 Executive conclusion

> **The canonical run executed rank-one sum-and-broadcast feedback.**

Exactly one of the brief's four options, selected on executable evidence: the
canonical paper run at `32aefaf` and current `main` both apply
`a_com_des = a_com_ff + k_p 𝟙𝟙ᵀ e_r + k_d 𝟙𝟙ᵀ e_v` with `k_p = k_d = 3.0`, on every
tick, as a weighted QP task on 83.8 % of them.

**Second-order finding, equally material to the decision:** the affected feedback is
**active but not load-bearing**. It is bounded twice over — the error it differences
is architecturally capped at ~1.1 mm by NMPC re-anchoring, and the CoM task is only
~41 % served in SS and ~0.1 % served in DS. The defect's magnitude is **8.5× smaller
than the residual the weighted stack already leaves on this task**. This is brief
§7.5 **Outcome 4**.

---

## 8.2 Evidence table

| # | Claim | Commit | File:line | Runtime evidence | Confidence |
|---|---|---|---|---|---|
| 1 | `SimConfig` holds CoM gains as **scalars** `3.0` | `eecbf94` | `crawlbot/simulation/config.py:349-350` | `SimConfig().ss_Kp_com = 3.0`, type `float` | **Certain** |
| 2 | `sim_loop` promotes them to a **(3,3) matrix** | `eecbf94` | `crawlbot/simulation/sim_loop.py:957` | `np.diag([3.0]*3)` → shape `(3,3)` | **Certain** |
| 3 | `WholeBodyQPConfig` **declares a (3,) vector** — contradictory contract, no validation | `eecbf94` | `crawlbot/solvers/wholebody_qp.py:162-163` | default `100.0*np.ones(3)`, shape `(3,)`; matrix stored verbatim | **Certain** |
| 4 | `_com_task_rows` applies `np.diag` **again**, extracting the diagonal | `eecbf94` | `crawlbot/solvers/wholebody_qp.py:902-903` | `np.diag(cfg.Kp_com)` → shape `(3,)`, value `[3,3,3]` | **Certain** |
| 5 | the product collapses to a **scalar** | `eecbf94` | `crawlbot/solvers/wholebody_qp.py:905-906` | `type np.float64`, shape `()`, value `3.0` for `e_x` | **Certain** |
| 6 | the scalar **broadcasts** to all 3 axes | `eecbf94` | `crawlbot/solvers/wholebody_qp.py:904` | `a_com_des` shape `(3,)`, all components equal | **Certain** |
| 7 | law is **rank-one**, not diagonal | `eecbf94` | as above | all 5 sentinels match the rank-one table, none the diagonal table; 3 independent differential directions → `a_com_des = 0` exactly ⇒ rank ≤ 1 | **Certain** |
| 8 | common mode gets **3k_p**, not k_p | `eecbf94` | as above | `e_common=[1,1,1]` → `[9,9,9]` with `k_p=3` | **Certain** |
| 9 | the **K_d channel has the same defect** | `eecbf94` | `wholebody_qp.py:903, 906` | `v_com_ref=[1,−1,0]` → `a_com_des = 0` | **Certain** |
| 10 | **canonical freeze is identical** to `main` on this path | `32aefaf` | `sim_loop.py:1151`, `wholebody_qp.py:184,659-663` @ `32aefaf` | `git show` — same construction, same double-`np.diag`; only line numbers moved | **Certain** |
| 11 | `WholeBodyQPConfig()` **bare defaults are correct** (diagonal) | `eecbf94` | `wholebody_qp.py:162` | `k_p=100`, `e_x` → `[100,0,0]` | **Certain** |
| 12 | CoM gains are **not overridden** anywhere; canonical runs the 3.0 defaults | `eecbf94` | `gate/replay_canonical.py:36-47`; `dca.main` | `ss_Kp_com`/`ss_Kd_com` absent from `C_KWARGS` and from `dca`; repo-wide grep → 9 sites only | **Certain** |
| 13 | task **assembled in SS** at `α = 400` | `eecbf94` | `wholebody_qp.py:422, 425` | **5080** additions observed, `settle_mode=False` on all | **Certain** (measured) |
| 14 | task **assembled in DS** at `α = 100` | `eecbf94` | `wholebody_qp.py:514, 517` | **2010** additions observed | **Certain** (measured) |
| 15 | rank-one path live on **every** tick | `eecbf94` | — | `cfg.Kp_com` shape `(3,3)` on **8458/8458** solves | **Certain** (measured) |
| 16 | the error it acts on is **≤ 1.147 mm**, by NMPC re-anchoring | `eecbf94` | `sim_loop.py:2241-2242, 2289-2292, 2342-2343` | `|e_r|` med 0.165 / max 1.147 mm; **508/5080 = 10.0 %** of SS ticks have `e_r ≡ 0` (one per NMPC cycle) | **Certain** (measured) |
| 17 | **78.7 %** of the CoM error is in the invisible differential subspace | `eecbf94` | — | mean `|e_⊥|/|e_r|`: SS 75.4 %, DS 87.0 %, max 100.0 % | **Certain** (measured) |
| 18 | applied and intended feedback differ by **>100 %** typically | `eecbf94` | — | median `|Δa_fb|/|a_fb,diag|` = **1.206**; individual-axis **sign inversions** observed | **Certain** (measured) |
| 19 | the task is **largely not served** — the defect is not load-bearing | `eecbf94` | — | delivered fraction: SS med **0.407** (66.4 % of ticks < 50 %, 11.8 % wrong sign); DS med **0.0009** (100 % < 50 %, 48.8 % wrong sign); residual **8.5×** the median defect | **Certain** (measured) |
| 20 | **no test could have caught it** — suite exercises the *other* contract | `eecbf94` | `tests/test_reworked_qp.py:111-125`, `:566-570` | `grep -rn "Kp_com=" tests/` → **no match**; suite runs the `(3,)` defaults; the comparator re-uses the production expression, so it is a mirror | **Certain** |
| 21 | **every sibling gain is correct** — CoM is the lone outlier | `eecbf94` | `sim_loop.py:964-968` vs `wholebody_qp.py:428, 452-453, 530-531` | torso `(6,)`, EE `(3,)+(3,)`, posture scalar — all consumed correctly; `Kp_com` is the only one wrapped in `np.diag` at the call site | **Certain** |

Nothing in this table is inferred from source reading alone; rows 7–9, 11, 13–19 are
measured at runtime through the production helper.

---

## 8.3 Shape trace

Complete trace, canonical `k_p = k_d = 3.0`. Runtime types/shapes are **measured**.

| Stage | File:line | Expression | Runtime type | Runtime shape / value |
|---|---|---|---|---|
| CLI / config input | `gate/replay_canonical.py:36-47` | `ss_Kp_com` **not passed** → default | — | — |
| Config scalar | `config.py:349` | `ss_Kp_com` | `float` | `3.0` |
| Positional pass | `sim_loop.py:472` | `cfg.ss_Kp_com` → `kpc` | `float` | `3.0` |
| QP construction | `sim_loop.py:957` | `np.diag([kpc]*3)` | `ndarray float64` | **`(3,3)`** `[[3,0,0],[0,3,0],[0,0,3]]` |
| Dataclass storage | `wholebody_qp.py:162` | `cfg.Kp_com` (declared `(3,)`, no validation) | `ndarray float64` | **`(3,3)`** |
| QP helper | `wholebody_qp.py:902` | `np.diag(cfg.Kp_com)` | `ndarray float64` | **`(3,)`** `[3.0,3.0,3.0]` ← **extracted** |
| Error product | `wholebody_qp.py:905` | `Kp_com_mat @ e_r` | **`np.float64`** | **`()`** scalar `= 3·Σe_r` |
| Final addition | `wholebody_qp.py:904` | `a_com_ff + <scalar> + <scalar>` | `ndarray float64` | `(3,)` — **broadcast** |
| Task RHS | `wholebody_qp.py:910` | `b_com = a_com_des − Jdot_dq_com` | `ndarray float64` | `(3,)` |

Identical for `K_d`: `config.py:350` → `sim_loop.py:957` → `wholebody_qp.py:163` →
`:903` → `:906` → `:904`.

Sentinel confirmation (`k_p = 3`, `a_com_ff = 0`, `J̇q̇ = 0`, `q̇ = 0`):

| `e` | measured `a_com_des` | rank-one predicted | diagonal predicted |
|---|---|---|---|
| `[1,0,0]` | **`[3,3,3]`** | `[3,3,3]` ✔ | `[3,0,0]` ✘ |
| `[0,1,0]` | **`[3,3,3]`** | `[3,3,3]` ✔ | `[0,3,0]` ✘ |
| `[0,0,1]` | **`[3,3,3]`** | `[3,3,3]` ✔ | `[0,0,3]` ✘ |
| `[1,−1,0]` | **`[0,0,0]`** | `[0,0,0]` ✔ | `[3,−3,0]` ✘ |
| `[1,1,1]` | **`[9,9,9]`** | `[9,9,9]` ✔ | `[3,3,3]` ✘ |

Environment: NumPy **2.3.5**, Python 3.11, `pin==3.9.0`, `mujoco 3.11.0`,
`MUJOCO_GL=disabled`.

---

## 8.4 Activation trace

**The task WAS added to the canonical QP.**

| quantity | measured |
|---|---|
| `solve()` calls | 8458 |
| CoM task added | **7090 (83.8 %)** |
| SS additions — `add_task(A_com, b_com, cfg.ss_alpha_mom, priority=2)` (`wholebody_qp.py:425`) | **5080** at `α = 400.0` |
| DS additions — `add_task(A_com, b_com, cfg.ds_alpha_com, priority=1)` (`wholebody_qp.py:517`) | **2010** at `α = 100.0` |
| `ss_two_task_mode` / `ds_centroidal_mode` | `True` / `True` |
| `cfg.Kp_com` shape | `(3,3)` on **8458/8458** |
| `‖A_com‖` (SS) | 1.77 – 1.79 (non-degenerate) |
| `qp_ok` | `True` on all 8458 |

`_com_task_rows` itself is called **unconditionally** (`wholebody_qp.py:409`), so
`a_com_des` is computed on every tick regardless of mode; both consumers share the
same `b_com` object — one defect, two load paths.

Instrumentation proven inert: the replay reproduced the canonical docks
**4.02 / 4.89 / 4.99 / 4.97 / 4.95 / 4.62 mm** and `h_w` peak **4.24 Nms** exactly.

---

## 8.5 A/B table

**BLOCKED — Phase 3 not started.** Brief §3: "No behavioural code change before the
Phase-2 human checkpoint." Brief §6: "Do not implement a fix before explicit
approval."

The Phase-2 measurements support a **prediction**, recorded here so it can be
falsified rather than assumed: because the defect is 8.5× below the residual the
stack already leaves on the CoM task, and because the error it acts on is capped at
~1.1 mm, a corrected diagonal law should perturb the headline metrics only slightly.
**This is a hypothesis, not a result.** Two reasons it could fail:

1. the **worst-case tick perturbs the command by 100 %** of `|a_com_des|`
   (0.0591 m/s²), so the perturbation is not uniformly small;
2. the worst dock is **4.99 mm against a 5 mm capture gate — 0.01 mm of margin**.
   Any perturbation at all can flip a pass to a fail there.

Also to be measured, not assumed: `12/21` of the required metrics (§7.4) concern
momentum/attitude/envelope channels that the CoM task feeds only indirectly.

---

## 8.6 File-level diff

**Zero changes to `crawlbot/`, `gate/`, `scripts/`, `tests/`, `models/`.**

The audit added three documents under `results/j2_adjconv/` and nothing else:

```
results/j2_adjconv/COM_GAIN_STATIC_TRACE.md         (new, Phase 0)
results/j2_adjconv/COM_GAIN_EXECUTABLE_PROOF.md     (new, Phase 1)
results/j2_adjconv/COM_GAIN_ACTIVATION_EVIDENCE.md  (new, Phase 2)
results/j2_adjconv/COM_GAIN_AUDIT_REPORT.md         (new, this file)
results/j2_adjconv/com_gain_audit_ticks.csv         (new, Phase-2 per-tick evidence)
```

`git diff --stat HEAD` over tracked files: **empty**. That is the machine-checkable
statement that no behavioural change was made.

Probe scripts were written to the session scratchpad and deliberately **not
committed** — throwaway instrumentation; the durable form of their assertions is the
`tests/` additions specified in brief §7.3, which belong to Phase 3.

Because no file under `crawlbot/` changed, CLAUDE.md Rule 15 (doc-sync in the same
commit) does not apply to this audit; `gate/sync_docs.py --check` is unaffected.

---

## 8.7 Test results

| check | command | result |
|---|---|---|
| Fast suite gate | `PYTHONPATH=. python3 gate/run_suite.py --fast` | **PASS** — 199 tests, 198 passed, 0 failed, 0 errors, 0 skipped, 1 xfail (32 s) |
| Focused gain semantics probe | Phase-1 script against `WholeBodyQP._com_task_rows` | **ALL ASSERTIONS PASSED** (rank-one for the canonical construction, diagonal for the vector contract and for the bare defaults, rank ≤ 1 proven on 3 spanning directions, `K_d` channel confirmed) |
| Canonical reproduction under instrumentation | Phase-2 replay | **PASS** — docks and `h_w` peak byte-equal to the frozen canonical |
| Parameter-table check | `PYTHONPATH=. python3 gate/verify_params.py` | **PASS** — 15 rows, every cited line declares its parameter and every value matches |
| Path-citation audit | `PYTHONPATH=. python3 gate/link_audit.py` | **PASS** (exit 0) |
| Doc-sync check | `PYTHONPATH=. python3 gate/sync_docs.py --check` | **FAIL (exit 1), pre-existing on `eecbf94` and unrelated to this audit** — 25 false positives from the absent `gate/_run/cov/cov.json`. See §10. Not caused here: zero tracked files were modified. |

Not run, deliberately:

- **`gate/run_gate.py`** (canonical byte-identity) — no `crawlbot/` file changed, so
  there is nothing for it to protect against in this audit. It is **required** for
  Phase 3, where it would be the first thing to run.
- **Full `gate/run_suite.py`** (~90 s, pre-merge gate) — nothing is being merged.
  Required before any Phase-3 merge.

The focused gain tests specified in brief §7.3 (vector input, matrix input, scalar
rejection, wrong-length vector, non-square matrix, no cross-axis coupling for
diagonal gains) are **Phase-3 deliverables** and have not been written, since
committing them would mean committing the fix they validate.

---

## 8.8 Artifact declaration

| item | value |
|---|---|
| **Commit** | see the audit commit on `claude/com-gain-semantics-audit-j0u6yr` (code state audited: `eecbf94`; canonical freeze cross-checked: `32aefaf`) |
| **Artifact paths** | `results/j2_adjconv/COM_GAIN_{STATIC_TRACE,EXECUTABLE_PROOF,ACTIVATION_EVIDENCE,AUDIT_REPORT}.md` |
| **Run artifact** | `results/com_gain_audit_scratch/sim_log.json` — scratch, deleted after export (same convention as the untracked `results/gate_run_scratch/`); reproducible from the command below. **No committed artifact modified**: `results/j2_adjconv/canonical2p5_result.json` and `c25_fulldiag.csv` untouched |
| **Committed evidence** | `results/j2_adjconv/com_gain_audit_ticks.csv` — 7090 rows × 31 cols, per-tick `e_r` / `e_v` / mode split / applied-vs-diagonal feedback / `η_fb` / residual / delivered fraction |
| **Commands** | `bash docs/architecture/setup_env.sh` · `PYTHONPATH=. python3 gate/run_suite.py --fast` · `MUJOCO_GL=disabled PYTHONPATH=. python3 <scratchpad>/com_gain_proof.py` · `MUJOCO_GL=disabled PYTHONPATH=. python3 <scratchpad>/com_gain_activation.py` · `… com_gain_analyse.py` · `… com_gain_picks.py` |
| **Key numbers** | rank-one confirmed on 5/5 sentinels · `Kp_com` shape `(3,3)` on 8458/8458 ticks · 7090 task additions (5080 SS @ 400, 2010 DS @ 100) · docks 4.02/4.89/**4.99**/4.97/4.95/4.62 mm (canonical, exact) · `h_w` peak 4.24 Nms · `|e_r|` med 0.165 mm / max 1.147 mm · 508/5080 SS ticks with `e_r ≡ 0` (10.0 %) · `|e_⊥|/|e_r|` mean 78.7 % · `η_fb` med 0.0796 / P95 0.300 / max 0.906 · defect med 0.00309 / P95 0.0159 / max 0.0591 m/s² · median relative defect vs intended feedback 1.206 · delivered fraction SS 0.407 / DS 0.0009 · residual/defect ratio 8.5× |

---

## 9. Recommendation for the checkpoint

The semantic defect is proven and is not defensible as a design choice: `Kp_com` is
the only one of six gains handed to `WholeBodyQPConfig` as a matrix, its own field
declares a vector, and its five siblings all use the vector contract correctly. It
should be fixed regardless of whether the numbers move.

Per brief §7.5 **Outcome 4**, the recommended Phase 3 is therefore the *conservative*
one:

1. adopt the single vector contract `Kp_com: (3,)` with `np.full(3, kpc)` at
   `sim_loop.py:957`, keeping `np.diag` at `wholebody_qp.py:902-903`;
2. add the `as_gain_matrix()` validator from brief §7.3 so a matrix can never again
   be silently degraded — and apply it to **all** the gains, not just CoM, since the
   next slip will be in whichever one is left unguarded;
3. **no retuning** of `k_p`, `k_d`, weights, bounds, or references (brief §9);
4. run `gate/run_gate.py` **first** and expect byte-identity to **break** — that is
   the correct outcome for an intentional behavioural change, and the diff it reports
   is the A/B measurement;
5. full `gate/run_suite.py` plus the six focused gain tests;
6. re-measure the §7.4 metric set, with the 4.99 mm / 0.01 mm dock margin as the
   gate to watch first.

Two items worth Idriss's judgement independently of the gain fix, both surfaced by
this audit and neither part of it:

- **The DS centroidal CoM task is effectively inert** — median delivered fraction
  **0.0009**, wrong sign on 48.8 % of ticks. At `ds_alpha_com = 100` against
  torso-angular and posture, it is a task the QP does not serve. That is a weighting
  question, not a gain question, and under CLAUDE.md Rule 12 (one variable at a time)
  it must not be touched in the same experiment.
- **The test suite validates a code path production never takes.** `tests/` builds
  its own `WholeBodyQPConfig` and never passes `Kp_com`, so it exercised the correct
  diagonal law for the entire life of the defect. Whatever else Phase 3 does, at
  least one test should build the QP the way `sim_loop._build_qp` builds it.

---

## 10. Incidental finding — `gate/sync_docs.py` is destructive on a fresh clone

Found while running the CLAUDE.md pre-commit routine. **Not caused by this audit**
(zero tracked files were modified) and **not fixed here** — reported for a pass of
its own.

On this container `gate/sync_docs.py --check` reports **25 of 33 documents "out of
date with the code"** and exits 1. The cause is in its own first line of output:

```
note: gate/_run/cov/cov.json absent — coverage columns left as-is.
      regenerate with gate/_run/cov_replay.sh
```

`gate/_run/` is gitignored, so the coverage record does not exist on a freshly
cloned session. The 25 "stale" documents are **entirely** the coverage column: the
regeneration diff is **201 insertions / 201 deletions**, a pure 1:1 line swap of
`canonical coverage 97 %` → `not measured` and `**yes**` / `not exercised` → `—`.
There is no real code/doc drift.

The problem is the interaction: the note says "left as-is", but running
`gate/sync_docs.py` (Rule 15 step 1, the mandated first action after any
`crawlbot/` change) **erases** the committed coverage annotations rather than
preserving them. So on a fresh clone the routine as written silently destroys
data — and `--check` reports 25 false positives that mask any genuine staleness.
Verified here by running it, inspecting the diff, and reverting with
`git checkout -- docs/`.

Suggested fix (separate pass): have `sync_docs.py` **preserve** the existing
coverage column when `cov.json` is absent instead of overwriting it with `—`, and
have `--check` exclude the coverage column from the comparison in that case. Either
that, or commit `cov.json`.

---

**Awaiting explicit approval before any behavioural change.**
