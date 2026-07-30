# COM-GAIN-AUDIT Phase 3 — the fix, and the tracking measurement that reframes it

**Branch** `claude/com-gain-semantics-audit-j0u6yr`
**Parent** `c9e5854` (Phases 0–2, audit only)
**Scope** behavioural change to `crawlbot/solvers/wholebody_qp.py`; the canonical
run is **no longer byte-identical** and that is the intended outcome.

---

## 0. What changed relative to the Phase-2 recommendation

Phase 2 recommended forcing `sim_loop._build_qp` to hand over a **vector**.
**That was the wrong direction.** Idriss's call — *"the issue lies in the QP as
well, chosen approach is a diag (Kp)"* — is the correct one, and the repository
agrees unanimously:

| call site | handover |
|---|---|
| `crawlbot/simulation/sim_loop.py:957` | `np.diag([kpc]*3)` |
| `Misc/lutze_baseline/sim_lutze.py:79` | `np.diag([3., 3., 5.])` |
| `Misc/scripts/test_torso_task.py:108` | `np.diag([3., 3., 5.])` |
| `Misc/scripts/test_multi_step.py:90` | `np.diag([3., 3., 5.])` |
| `Misc/scripts/sim_torso6d.py:78` | `np.diag([3., 3., 5.])` |
| `Misc/scripts/test_integration.py:278` | `np.diag([50., 50., 100.])` |

**6/6 callers pass a diagonal matrix**, two of them anisotropic. The odd ones out
were the field default (`wholebody_qp.py:162-163`, a `(3,)`) and its consumer
`np.diag(cfg.Kp_com)`. The matrix is the chosen contract; the QP was the defect
site. Fix applied there.

**The defect is confined to the CoM channel.** `Kp_torso`/`Kd_torso` receive
`(6,)` and were always correct; so are the NMPC weights (`Wr`, `Wv`, `Qf_r`,
`Qf_v`). Audited exhaustively — `np.diag(cfg.…)` / `np.diag(self.config.…)`
appears at 8 sites, and only the 2 CoM ones were wrong.

---

## 1. The change

`as_gain_matrix(g, n, name)` — new module-level helper, `wholebody_qp.py:70-126`.
Accepts scalar → `g·I`, `(n,)` → `diag(g)`, diagonal `(n,n)` → unchanged.
**Rejects** a full matrix and any other shape with `ValueError` rather than
silently truncating or broadcasting.

Applied at four sites, replacing a bare `np.diag`:

| site | gain | numerically |
|---|---|---|
| `_com_task_rows` | `Kp_com`, `Kd_com` | **changed** — this is the fix |
| two-task stack | `Kp_torso`, `Kd_torso` | inert (input already `(6,)`) |
| DS angular block | `Kp_torso[3:,3:]`, `Kd_torso[3:,3:]` | inert |

The torso sites are converted only so the contract is uniform and the failure
mode is unrepresentable — they are proven inert by
`TestGainSemantics::test_torso_gain_handover_is_unchanged`.

---

## 2. Why nothing raised for the whole project history

`np.diag` is shape-polymorphic in the worst possible way:

| input | `np.diag` does | result |
|---|---|---|
| `(3,)` | **builds** | `diag(k)` — correct |
| `(3,3)` | **extracts** | `k` as a vector — wrong |

On the extracted vector, `k @ e` is a **scalar**, which NumPy broadcast back over
three axes. Applied vs intended, on the canonical isotropic gain `diag([3,3,3])`:

| error | intended [mm/s²] | applied [mm/s²] |
|---|---|---|
| `e = x only` | `[3, 0, 0]` | `[3, 3, 3]` |
| `e ⟂ [1,1,1]` | `[3, −3, 0]` | **`[0, 0, 0]`** |
| `e ∥ [1,1,1]` | `[3, 3, 3]` | `[9, 9, 9]` |

Both forms are `(3,)` and finite. Live on **8458/8458** canonical solves. The
applied term differed from the intended by **134 %** of the intended term's own
magnitude in SS (104 % in DS) — a *different* vector, not a perturbed one.

`tests/test_reworked_qp.py:568` reproduced the same bug in its own expected
value, so the probe would have agreed with the defect — it never surfaced only
because that module's configs leave the `(3,)` default in place. Fixed.

---

## 3. Tracking performance — the measurement Phase 2 omitted

Phase 2 measured the task-internal residual and concluded "not load-bearing".
That was the wrong quantity. Closed-loop, against the exported planner reference
(`scripts/audit_com_tracking_perf.py` → `com_tracking_perf.json`):

| quantity | SS median | SS p95 | SS max |
|---|---|---|---|
| `\|e_com\|` closed loop | **61.5 mm** | 128.8 mm | **153.7 mm** |
| component ⟂ `[1,1,1]` (invisible to the defect) | **81.1 %** | — | 99.99 % |

Per-axis signed error, SS: x median **+23.9 mm** (max **+145.1**), y −5.7, z
+3.1. The error is strongly anisotropic and x-dominated — exactly the structure a
sum-and-broadcast law cannot correct, and exactly what the anisotropic `Misc`
gains existed for.

`DS_interstep` exports `e_com ≡ 0` on all 1368 rows — that is the known
`_log_ds_tick` convention (`ref := measured`), not tracking. `DS_terminal`: median
3.5 mm, max 14.6 mm.

### 3.1 Why the gain fix cannot improve this

Two different references, traced to lines:

| reference | consumer | error seen |
|---|---|---|
| `rp_interp` — NMPC's own plan, sub-step interpolated | **the QP** (`sim_loop.py:2549`) | ≤ **1.147 mm** |
| `cref_r` — planner reference | **the NMPC** (`sim_loop.py:2243`); logged at `:2969` | **61–154 mm** |

The NMPC re-anchors its plan to the *measured* CoM every 100 ms, so `e_r` is
identically zero on **508/5080** SS ticks — exactly one per NMPC cycle — and
never exceeds ~1 mm. The QP's CoM task is a **plan-follower with ~1 mm of
authority**. The PD term is **9.8 %** of `|a_ff|` in SS (4.46 vs 54.2 mm/s²), and
the stack delivers a median **41 %** of the CoM acceleration it commands (0.09 %
in DS).

**So the gain was the smaller of two independent problems in the CoM channel**,
and the larger one is structural, not a gain value. Rerouting the task to the
planner reference is a control-architecture decision, and the DS weighting
question (`ds_alpha_com=100`, task near-inert) is a separate lever that rule 12
forbids moving in the same experiment. **Neither is done here.**

---

## 4. A/B against the frozen canonical

`gate/run_gate.py`: **byte-identity FAILS at row 1, `hw_x_Nms`** — as designed.
That break *is* the measurement. `gate/dock_check.py` on the post-fix replay:

| metric | frozen (rank-one) | post-fix (diagonal) | delta |
|---|---|---|---|
| dock 1 | 4.02 mm | 4.04 mm | +0.02 |
| dock 2 | 4.89 mm | 4.89 mm | 0.00 |
| **dock 3** | **4.99 mm** | **4.98 mm** | **−0.01** |
| dock 4 | 4.97 mm | 4.97 mm | 0.00 |
| dock 5 | 4.95 mm | 4.94 mm | −0.01 |
| dock 6 | 4.62 mm | 4.63 mm | +0.01 |
| **docks under 5 mm** | 6/6 | **6/6** | — |
| **worst margin** | 0.01 mm | **0.02 mm** | **+0.01 (better)** |
| θ_s peak | 0.540° | 0.539° | −0.001 |
| h_w peak axis / norm | 4.100 / 4.240 | 4.104 / 4.244 | +0.004 |
| **e_com peak** | **0.154 m** | **0.154 m** | **0.000** |
| qp_fail | 0 | 0 | 0 |

Two things to read here. The fix is **safe** — all six docks hold and the
tightest (the 0.01 mm-margin step 3) *improved*. And `e_com` peak is
**unchanged to three decimals**, which is the §3.1 prediction confirmed: the
task never saw the tracking error, so correcting its gain could not move it.

---

## 5. Verification

| gate | result |
|---|---|
| `gate/run_suite.py` (full) | **PASS** — 206 tests, 205 passed, 0 failed, 0 errors, 1 xfail |
| `gate/run_gate.py` | **FAIL** on artifact identity — intended; see §4 and §6 |
| `gate/sync_docs.py --check` | PASS |
| `gate/verify_docs.py` | PASS — 34 documents |
| `gate/link_audit.py` | PASS |
| `gate/verify_params.py` | PASS (after fixing 2 refs this change drifted) |
| `scripts/audit_com_gain_bite_check.py` | **PASS** — 5/5 predicates catch the pre-fix path, 0/5 fail post-fix |

Test-count delta is **exactly +6** (`git show HEAD:tests/test_reworked_qp.py`
declares 6 `def test_`, the tree declares 12) — no test was retired or disabled.

### 5.1 The new tests are proven to bite

`TestGainSemantics`, six tests. `scripts/audit_com_gain_bite_check.py` re-runs
each predicate against `old(g) = np.diag(g)` — the pre-fix expression verbatim —
and all five behavioural predicates raise. A test that only passes after the fix
is not evidence; these fail before it.

---

## 6. Open decisions — for Idriss, not taken here

1. **Re-freeze or not.** The canonical baseline has moved. `gate/run_gate.py`
   will FAIL on this branch until the frozen artifacts are regenerated. Whether
   to re-freeze at the corrected law is a call about the paper's numbers, not a
   code question. `gate/last_verdict.json` is committed recording the FAIL
   honestly rather than left showing a stale PASS.
2. **Paper impact.** θ_s moves 0.540° → 0.539°, docks by ≤0.02 mm, `e_com` peak
   not at all. If the paper's numbers are already propagated at 0.540°/0.54°,
   this is a sub-rounding change everywhere except the dock table.
3. **The structural finding in §3.1.** The CoM task tracking the NMPC plan rather
   than the planner reference is either the intended cascade design or a defect,
   and that is an architecture question. Deliberately not touched.
4. **`ds_alpha_com=100` leaves the DS CoM task near-inert** (delivered fraction
   median 0.0009). Separate lever, rule 12.

---

## 7. Incidental repo findings (not caused by this change, not fixed)

1. **`gate/_run/cov_replay.sh` does not exist.** CLAUDE.md's Rule-15 routine
   prescribes it for regenerating coverage annotations, but `gate/_run/` is
   gitignored in full (`.gitignore:109`), so it was never committed. Without
   `cov.json`, `sync_docs.py --check` reports **all 25** modules "out of date"
   (pure false positive) and a blanket `sync_docs.py` **erases** the committed
   coverage annotations. Regenerated here with
   `python3 -m coverage run --source=crawlbot gate/replay_canonical.py` +
   `coverage json`, after which `sync_docs.py` correctly touched **exactly one**
   document. `coverage` is not in the environment and had to be installed —
   worth adding to `setup_env.sh`, and worth committing the replay script.
2. **CLAUDE.md's suite count is stale**: it claims "210 passed"; the pre-change
   tree has **200** tests (206 after this change).
3. *(Checked and dismissed.)* `verify_params.py` **does** gate properly —
   `sys.exit(1)` at `gate/verify_params.py:96`. An apparent `rc=0` alongside its
   mismatch output was the exit status of `tail` through a pipe, not the
   checker's.

---

## 8. Artifacts

| path | what |
|---|---|
| `crawlbot/solvers/wholebody_qp.py` | `as_gain_matrix` + 4 call sites |
| `tests/test_reworked_qp.py` | `TestGainSemantics` (6) + probe fix |
| `docs/crawlbot/solvers/wholebody_qp.md` | §1.5 contract, §4 structural finding (Rule 15) |
| `scripts/audit_com_tracking_perf.py` | §3 measurement |
| `scripts/audit_com_gain_bite_check.py` | §5.1 bite proof |
| `results/j2_adjconv/com_tracking_perf.json` | §3 numbers |
| `CLAUDE.md` | 2 drifted parameter refs |

No committed canonical artifact was modified: the replay writes to
`results/gate_run_scratch/`, and `c25_fulldiag.csv` /
`canonical2p5_result.json` are untouched.
