# PHASE CLEANUP-25 — complete code pass (everything except `Misc/`)

Full audit of the in-scope code, with fixes applied where the finding was
unambiguous.

| dir | files | lines |
|---|---:|---:|
| `crawlbot/` | 33 | 14 017 |
| `tests/` | 23 | 6 359 |
| `gate/` | 30 | 2 601 |
| `scripts/` | 5 | 2 081 |
| `benchmarks/` | 6 | 529 |

`Misc/` excluded by instruction (slated for removal).

---

## 1. The dominant defect class: one trap, five victims

Everything serious this pass found traces to a single mechanism, documented in
`docs/crawlbot/core/robot_interface.md` §2 and now shown to be actively harmful:

> `FRAME_TORSO`, `FRAME_TOOL_A/B`, `JOINT_6A/B_ID`, `N_JOINTS`, `NQ`, `NV` are
> module-level constants holding **stale 6-DOF values**, rebound via `global` in
> `RobotInterface.__init__` to the real 7-DOF values. A `from … import NQ`
> **snapshots the stale value forever**.

```
at import          : NQ=19  NV=18  N_JOINTS=12  FRAME_TOOL_A=18  FRAME_TOOL_B=32
after construction : NQ=21  NV=20  N_JOINTS=14  FRAME_TOOL_A=20  FRAME_TOOL_B=36
```

Frame 18 is `Link_6_a`; frame 32 is **`Link_5_b`** — a mid-arm link, not a
gripper.

| victim | symptom | status |
|---|---|---|
| `Misc/…/tests/test_phase0.py` | `expected 20, got 18` | CLEANUP-23 |
| `Misc/…/tests/test_phase1.py` | identical | CLEANUP-23 |
| **`tests/test_reachability.py`** | **false reachability alarm** — §2 | **fixed** |
| **`tests/test_liabilities.py`** (3 tests) | IK onto a mid-arm link | **fixed** |
| `crawlbot/core/ik.py:17` | imported unused — a loaded gun | **removed** |

### Recommendation: make it structurally impossible

After this pass **nothing reads those module globals** — every remaining
occurrence is a docstring or an explanatory comment. They can therefore be
**deleted outright**, which eliminates the class rather than documenting it.
Not done here: it is a change to a live module's public surface and deserves its
own commit and gate run.

---

## 2. `tests/test_reachability.py` was generating a false alarm

It reported:

```
RESULTS: 1/3 steps reachable        pos_err = 325.00 mm
```

while the canonical run docks **6/6 under 5 mm**. The cause was the trap above:
line 82 used the snapshotted `FRAME_TOOL_A/B`, so the script was solving IK to
put **`Link_5_b`** on the anchor. The 325 mm was the offset between a mid-arm
link and the actual tool frame.

After switching to `robot.frame_tool_a` / `frame_tool_b`:

```
RESULTS: 3/3 steps reachable        pos_err = 0.00 mm
```

Anyone who had run this script would have concluded the robot has a reachability
limit it does not have. A wrong instrument does not merely fail to find problems
— it manufactures them.

The file remains a **non-collecting** `test_*` file (it defines `check_reachability`,
not `test_*`), like the two lutze scripts. It is now correct, but pytest still
collects nothing from it.

---

## 3. ⚠ 8 of 14 CLAUDE.md parameter references were stale

CLAUDE.md's "Key Parameters" table is the single source of truth for canonical
values, and every row cites a `file.py:LINE`. **Nothing checked those
citations.** When the chantier shrank `config.py` (610 → 507) and
`wholebody_qp.py` (1385 → 950), the references drifted 20–50 lines:

| parameter | cited | actual |
|---|---|---|
| `weight_ratio` | `wholebody_qp.py:75` | `:94` |
| α torso-pose | `config.py:351` | `:303` |
| α swing-EE | `config.py:319` | `:282` |
| α momentum | `config.py:336` | `:290` |
| w hw-slack | `wholebody_qp.py:181` | `:159` |
| α posture | `config.py:320` | `:283` |
| α wrench-track | `config.py:321` | `:284` |
| α torque-min / accel-reg | `sim_loop.py:1145` | `:1126` |
| ε (Tikhonov) | `hierarchical_qp.py:97` | `:98` |

**Every value was correct** — only the pointers rotted. But a pointer that lands
on `ss_Kp_torso: float = 6.0` while claiming to show `alpha_torso_pose = 2000` is
worse than no pointer: it invites a reader to "correct" the wrong number.

All corrected, and a new checker closes the gap:

```bash
PYTHONPATH=. python3 gate/verify_params.py
```

It checks, per row, that the file exists and is long enough, that the cited line
**declares the parameter the row names**, and that the value matches
**numerically** — so `2e3` in the source satisfies `2000` in the table, which a
string comparison would have flagged as a false positive.

This was the soft spot flagged when the documentation landed: *"verification
covers paths, lines and symbols — not numeric values."* It is now covered.

---

## 4. Test-suite integrity

| finding | status |
|---|---|
| `Misc/tests/test_fk_reference_consistency.py` — collection **ERROR**: imports `constrained_geodesic`, deleted in CLEANUP-17 | recorded (`CARRYOVER` C6), **not fixed** — retiring a test file is a coverage decision |
| three `test_*` files collect nothing (the lutze pattern) | two moved in CLEANUP-24; `test_reachability.py` remains |
| 228 tests collect from `tests/` + `benchmarks/` | — |

The collection error is worse than a failing test: it aborts the whole run. That
it went unnoticed since CLEANUP-17 is a consequence of the suite not being gated.

### ⚠ CLAUDE.md understates the suite's state by an order of magnitude

CLAUDE.md's Known Issues says *"2 pre-existing pytest failures"*. Measured (with
the collection-error file excluded so the run completes at all):

```
9 failed, 196 passed, 4 skipped, 3 errors
```

**Twelve problems, not two.** The full list:

| test | |
|---|---|
| `test_coarse_preplanner::test_far_infeasible_under_tight_rate` | the one CLAUDE.md names |
| `test_reworked_qp::TestPhase20TMOM` ×4 | `test_mass_scalar_sanity`, `test_pure_tracking_per_axis`, `test_static_hold`, `test_variant_b_weak_reference_coexistence` |
| `test_reworked_qp::TestT7TrackingSS::test_torso_and_ee_tracking` | |
| `test_reworked_qp::TestT8SoftCoMEffect::test_soft_com_reduces_rms` | `alpha_com_soft` is 0.0 canonically — the feature under test is off |
| `test_reworked_qp::TestT9DynamicsResidual::test_residual_small` | |
| `test_reworked_qp::TestT10DSPassivity::test_energy_decay` | |
| `test_mid_waypoint_reshape` ×3 | `FileNotFoundError` — fixtures for the reshape path removed in CLEANUP-15 |
| plus `test_fk_reference_consistency` | collection error, excluded above |

The "2" was true at the freeze and has not been revisited since; the chantier's
own removals account for at least four of these (the `mid_waypoint_reshape`
fixtures and `constrained_geodesic`). Since the suite is not gated, nothing
reported the drift.

> **Corrected in CLEANUP-27 §4:** "at least four" is wrong — it is **eleven of the
> twelve**. The 8 `test_reworked_qp` failures are also this chantier's, from the
> CLEANUP-6/9 removal of 9 `WholeBodyQPConfig` fields their shared helper still
> passes; measured `8 passed` at `4e2e8da^`. Only
> `test_far_infeasible_under_tight_rate` predates the chantier. The
> "pre-existing" label below was inherited, not measured.

**Not fixed here.** Each needs its own diagnosis — several are testing features
deliberately removed or disabled, which is a decision about what the suite should
cover, not a bug to patch.

### Zero regressions from this pass — measured, not asserted

The same suite was run at HEAD with these changes stashed, and after:

```
HEAD          : 9 failed, 196 passed, 4 skipped, 3 errors   (622.11s)
after CLEANUP-25: 9 failed, 196 passed, 4 skipped, 3 errors   (618.09s)
```

Identical. Necessary because this pass touched `crawlbot/`, and the gate proves
byte-identity of the *canonical run* but says nothing about the test suite.

---

## 5. Hygiene applied

- **8 unused imports removed** from `crawlbot/`.
- **`__all__` declared** on `core`, `aocs`, `planning`, `simulation` `__init__.py`.
  Re-exports are the package API, not leftovers; declaring the intent removes the
  false "imported but unused" without deleting the export.
- Two deliberately left: `centroidal_nmpc.ContactPhase` (Idriss said keep it) and
  `locomotion_planner.py`'s three (on the removal list, CLEANUP-23).

### A mistake made and caught in this pass

Removing the `FRAME_TOOL_A/B` import from `ik.py` **broke
`tests/test_liabilities.py`**, which imported them *transitively through
`ik.py`*. pyflakes correctly said they were unused *in `ik.py`* — and I read that
as "unused", which is a different claim.

Fourth instance in this chantier of an instrument narrower than the question.
The check for "can I delete this import" is not "is it used in this file" but
"does anything import it from here".

Repaired by pointing those three tests at the instance attributes, which also
fixed the pre-existing wrong-frame bug. All three pass.

---

## 6. Clean results

| check | result |
|---|---|
| `crawlbot/` module imports | **33/33** |
| canonical scripts import-check | **5/5** |
| pyflakes undefined names / syntax errors | **0** across all in-scope code |
| `TODO` / `FIXME` / `XXX` / `HACK` | **0** |
| hardcoded absolute paths | **0** |
| hardcoded 6-DOF arities in live code | **0** — every `np.zeros(12)` is the contact-wrench dimension (2 × 6-D), correctly not DOF-derived |

---

## 7. Left alone deliberately

- **15 unused locals, 3 placeholder-free f-strings.** Cosmetic, and each carries
  nonzero risk against a byte-identity gate — `coarse_preplanner.py:538`'s `m`,
  for instance, is part of a chained assignment `result._mass = m = float(...)`,
  so "removing the unused local" means rewriting a live statement.
- **`ContactObserverConfig.nv = 18`** — another stale 6-DOF default, but
  `sim_loop.py:461` overrides it with `robot.model.nv`. Latent, not active: a bare
  `ContactObserverConfig()` would carry 18.
- `Misc/` — out of scope.

---

## 8. Verification

```
gate VERDICT: PASS   2077 rows x 132928 fields byte-identical (150.1s)
docks 6/6   4.02 / 4.89 / 4.99 / 4.97 / 4.95 / 4.62 mm   delta +0.0000
theta_s 0.540 deg   h_w 4.102 / 4.243 Nms   e_com 0.154 m   qp_fail 0

gate/verify_params.py   14/14 rows OK  (was 9 mismatches)
gate/sync_docs.py --check   in sync
gate/verify_docs.py         33 documents OK
gate/link_audit.py          0 BROKEN BY MOVE
```

`crawlbot/` was modified in this pass, so the gate result is the load-bearing
one: the canonical run is byte-identical after the import cleanup.
