# CLEANUP chantier — carry-over ledger

Live list of everything the chantier has found but **not** acted on, with enough context to
resume cold. Kept current as phases land. `sim_loop.py` is the next target, so its items are
first and most detailed.

Method note that applies to everything below, learned the hard way (CLEANUP-2 F1 retraction):
**a dataclass default is not the canonical value.** Trace the config the run actually builds
(`dca.main` → `run_m7_single_step._make_m7_config()`) or instrument it. Confirm live/dead by
line coverage of `gate/replay_canonical.py`, not by reading.

---

## A. `sim_loop.py` — the next target

**Size:** 3756 lines. It is the largest remaining file and the last one carrying architecture
sediment.

### A1. The 40-parameter `WholeBodyQP.solve()` call — deferred deliberately

CLEANUP-10 recommended *against* restructuring the signature in the same pass as the
`Build QP` extraction, because it is an **API change that touches `sim_loop` call sites** and
would have obscured the diff proving the extraction was inert. That reasoning still holds, and
this is now a `sim_loop`-side decision.

Measured facts to start from (CLEANUP-10):
- 40 parameters; **30 are read in exactly one block** of `solve()` — 17 by the QP-assembly
  helpers, 11 by the SS two-task stack, 2 by contact-wrench tracking.
- Only 10 span multiple blocks: `settle_mode` (5), `dq` (4), `ds_centroidal_active` (3), and
  `q`, `J_torso`, `Jdot_dq_torso`, `R_torso`, `R_torso_ref`, `v_torso_ref`, `contact_config` (2 each).
- Two call sites, both keyword-only: `sim_loop.py` DS loop (~772) and SS loop (~3059).

If this is done, the natural grouping mirrors the helper boundaries (dynamics/contact block,
torso-reference block, EE block, mode flags). Any such change must be gate-verified
byte-identical.

### A2. `use_m2_stack` — a live trap, do not "clean" it

`SimConfig.use_m2_stack` **looks** dead (its `WholeBodyQPConfig` twin was removed in
CLEANUP-8) but is load-bearing on two paths that have nothing to do with the QP task stack:

| site | what it gates |
|---|---|
| `sim_loop.py:~2581` | torso-reference routing — CoM→torso δ-mapping vs raw TorsoPlanner quintic |
| `sim_loop.py:~2728` | `passivity_active` — **the DS passivity constraint** |

Deleting it would silently disable DS passivity. Its declaration in `config.py` now carries a
NOTE saying so. Same-named fields, opposite fates.

### A3. `sim_loop.py` lint debt (pre-existing, never introduced by the chantier)

From pyflakes, unchanged by any cleanup commit:

```
:40   'crawlbot.core.ik.solve_ik'            imported but unused
:40   'crawlbot.core.ik.solve_ik_waypoints'  imported but unused
:40   'crawlbot.core.ik.precompute_torso_map' imported but unused
:462  'contact_estimator.ContactState'       imported but unused
:486  f-string is missing placeholders
:1523 local variable 'traj_suffix'           assigned but never used
:2054 local variable '_ds_log_swing_to'      assigned but never used
:2130 local variable 'R0'                    assigned but never used
:2482 local variable 't_ds_start'            assigned but never used
:2635 local variable 'tref'                  assigned but never used
```

Cheap and safe, but verify each "unused local" is not a deliberate side-effect capture before
deleting.

### A4. Known `sim_loop` behavioural quirks (from CLAUDE.md, not yet addressed)

- **CoM-reference export snaps to the measured CoM at SS→DS entry** — logging convention in
  `_log_ds_tick` (`e_com=0`, `ref := measured`, ~`sim_loop.py:1038-1041`). Same class as the
  torso-export terminal-hold fix that was applied; decision pending on whether to apply it here.
  Reviewer-reported magnitude ~76 mm, **not repo-verified** (the fulldiag CSV has no CoM-ref
  channel).
- `dca.main` sets `cfg.ds_centroidal_mode=True` for the whole run including the trailing
  settle, so flags keyed on it cannot discriminate the DWELL.

---

## B. Blocked on a decision

### B1. F3 — `nmpc_ok` conflates "not called" with "failed" *(deferred by Idriss, option 2)*

The NMPC runs only in SS and the terminal DS settle. `DS_interstep` ticks are exported as
`nmpc_ok = 0`, which means *not called*. On the canonical that is 1368 of 2077 ticks, so a
whole-column read gives a **false 34.1 %** success rate; the true rate is **100 % (709/709)**.

Encoding left untouched on purpose: any fix (different sentinel, or an `nmpc_called` column)
changes the fulldiag CSV and therefore requires regenerating the frozen paper baseline under a
**Tier-1 exception** (`gate/EXCEPTIONS.md`). Documented instead, in
`scripts/diag_full_diag_export.py` and `results/j2_figdata/INTERNAL_figdata.md`.
**Revisit after submission.**

### B2. `_solve_strict` in `hierarchical_qp.py`

36 uncovered lines — the largest single dead block left. Dead because `method='weighted'`
canonically. **Has 2 test users + 6 script users**, so removal is a call about whether the
strict-hierarchy path stays reproducible. See `PHASE_CLEANUP_5`.

### B3. `get_full_trajectory()` in `centroidal_nmpc.py`

Production-dead (zero `crawlbot/` callers) but used by 5 tests + 1 script. Tier C.

---

## C. Environment & repo hygiene

### C1. `setup_env.sh` produces a broken pinocchio on a fresh container — **RESOLVED (CLEANUP-26)**

Pinned in `setup_env.sh` alongside `pin==3.9.0`, and added to the error hint the script
prints when the import check fails. Confirmed by `ldd` on pin's compiled extension: it links
`liburdfdom_*.so.4.0` and `libtinyxml2.so.10`, exactly the majors `~=4.0` / `~=10.0` force.
Original text kept for the record:


It pins `pin==3.9.0` but not its cmeel ABI dependencies, so pip resolves
`cmeel-urdfdom 6.0.0` / `cmeel-tinyxml2 11.0.0` while pin's binary needs
`liburdfdom_*.so.4.0` / `libtinyxml2.so.10`. **Nothing in the repo runs** until you add:

```bash
pip install --break-system-packages 'cmeel-urdfdom~=4.0' 'cmeel-tinyxml2~=10.0'
```

Fixed locally in every session so far, never committed. Should be pinned in `setup_env.sh`.

### C2. The test suite dirties five tracked files on every run — RESOLVED (CLEANUP-21)

Fixed by pointing the three `OUTPUT_DIR` constants at `results/test_scratch/` (gitignored).
Original text kept for the record:


`Misc/runs/M2_tests/{t10_passivity,t7_tracking}.png`, `Misc/runs/M3_tests/t4_hw_bounds.png`,
`Misc/runs/phase2_0_tmom/{t_mom_sine_x,t_mom_step_x}.png` are rewritten by
`test_nmpc_conservation.py` / `test_reworked_qp.py`, and matplotlib's encoding differs
byte-wise run to run (±1 kB on identical plots). The repo can therefore never be verified clean
after `pytest`, which undercuts the gate's bit-identity discipline. Fix: gitignore them or point
the tests at a scratch dir.

### C6. CLEANUP-17 broke a test module — **RESOLVED (CLEANUP-26)**

The module was **retired** to `Misc/tests/`, with its data fixture
(`Misc/runs/M7_1pct_3step_v22_t15_fk/`) following, since it was the only consumer. The suite
now collects **228 tests with zero errors** — end to end for the first time since CLEANUP-17.
Retiring rather than repairing was the right call: 8 of the 9 tests reach the deleted module
through the `smoothed` fixture, and the 9th validates an FK-mode run, a path removed from
`sim_loop` in CLEANUP-15 whose generator script is already under `Misc/scripts/`. Repairing
would have meant restoring the feature. Original analysis kept:


`Misc/tests/test_fk_reference_consistency.py:28` imports six symbols from
`crawlbot/planning/constrained_geodesic.py`, deleted in CLEANUP-17. The result is a
**collection error**, which interrupts the whole pytest run rather than failing one test:

```
ImportError while importing test module 'Misc/tests/test_fk_reference_consistency.py'
E   ModuleNotFoundError: No module named 'crawlbot.planning.constrained_geodesic'
```

**Root cause is a too-narrow search, again.** The CLEANUP-16 audit justified the deletion with
*"never imported"* — but what it measured was *never imported during the canonical run*, and it
enumerated importers only under `crawlbot/`. It did not look in `tests/`. Coverage of the
canonical replay cannot see a test-only consumer.

Third instance of the same methodological failure in this chantier, after the `link_audit`
basename heuristic and the import-resolution check in §C5. In each case the instrument was
weaker than the claim.

**Recommendation, not applied** (deleting a test file is a coverage decision):
the module's 9 tests all exercise the FK-reference / geodesic path that CLEANUP-15 and -17
deliberately removed, so the tests are testing a feature that no longer exists. Retiring the
file to `Misc/` is consistent — but note `test_E7_t15_step2_dock_under_fk_mode` is one of the two
known pre-existing failures in CLAUDE.md, and it is the only consumer of the
`Misc/runs/M7_1pct_3step_v22_t15_fk/` fixture. Moving the file orphans that fixture too.

### C3. Legacy research scripts now referencing removed symbols

Non-functional since CLEANUP-6/8; they existed only to exercise retired features, so deletion
is probably cleaner than repair:

| script | references |
|---|---|
| `test_qp_tracking{,_v19,_v20,_v21}.py` | `H_base_swing` / `swing_v_slice` kwargs |
| `bisect_qp_cascade.py` | same |
| `diag_option_d_tube.py` | the `_tube_*` counters |
| `test_integration.py` (CLEANUP-18) | `SwingPlanner.swing_trajectory` — its only caller |

**Correction (CLEANUP-18).** The CLEANUP-16 audit stated that `Misc/scripts/test_integration.py` and
`Misc/scripts/sim_torso6d.py` were "already in the non-functional list §C3". They were **not** on
this list, and measurement contradicts it: all `crawlbot` imports in *both* scripts — and in
`Misc/lutze_baseline/sim_lutze.py` — resolve at HEAD. Do not assume a script is already broken;
import-check it. This is what caused step 5 to be reverted (§C5).

### C5. `locomotion_planner.py` — KEPT in CLEANUP-18, **retracted in CLEANUP-23**

⚠ **This entry is superseded.** The reversal below rested on `Misc/lutze_baseline/sim_lutze.py`
being a working consumer. CLEANUP-23 measured that it **cannot run** — it dies on the first
state conversion with `shape (12,) into shape (14,)`, having never been migrated to the 7-DOF
model. Import resolution was the wrong instrument for the question "is this alive".

`locomotion_planner.py` therefore has no working consumer and goes back on the removal list.
Original reasoning kept below for the record.



CLEANUP-16 ranked "delete `locomotion_planner.py` (205 lines)" as step 5, risk "low, but breaks
2 legacy scripts". Measured: it has **three** consumers, all import-clean at HEAD, and one of
them is **`Misc/lutze_baseline/sim_lutze.py`** — a *package*, not a script, carrying the M0 / Lutze
comparison baseline that backs the paper's §II differentiation table. `LocomotionPlanner` is
live there (`sim_lutze.py:175-266`: construct, `calibrate_from_config`, two `reference_at`
calls). That is a paper artifact, not research sediment — the same KEEP class as
`sequence_loader.py` ("unused on the canonical ≠ retired").

Step 5 was executed then reverted within CLEANUP-18. To revisit, the question is not "is it dead
on the canonical" (it is) but "is the Lutze baseline still to be re-run" — an Idriss call.

### C4. Thirteen silent canonical values (Rule 5)

Dataclass fields `sim_loop` never overrides, so the default **is** the canonical value.

`WholeBodyQPConfig` (8, CLEANUP-8): `method`, `solver`, `weight_ratio`, `w_hw_slack`,
`alpha_settle`, `Kd_settle`, `qdd_max`, `tau_contact_max`. Only `w_hw_slack` is cited in
CLAUDE.md. Same class as the six NMPC weights already hoisted in CLEANUP-3.

`CoarsePrePlannerConfig` (5, CLEANUP-19): `eps_v_terminal` (5e-3 m/s) and `eps_L_terminal`
(5e-2 Nms) — **hard boxes on the terminal state of every step's plan**, i.e. the constraint
deciding where a step may end — plus `w_v_terminal` / `w_L_terminal` (1e2 each, soft penalties
on the same residual) and `ipopt_tol` (1e-6). None is in CLAUDE.md. The first four are physics,
not numerics, and are the higher-priority half of this item.

Also `CoarsePrePlannerConfig.T_step_default = 6.0` is never used at all — `sim_loop` always
passes `T_step` explicitly (its own comment at `sim_loop:403` calls the field "only a
bootstrap"), so the `if T_step is None` branch is dead. Removable rather than documentable.

---

## D. Optional next step in `wholebody_qp.py`

CLEANUP-11 did step 1 (extract `Build QP`). **Step 2**, not done: extract the SS two-task block
(~55 lines) as `_add_ss_two_task_stack`, taking `solve()`'s body from ~346 to ~290 and making
the canonical controller a named unit. Explicitly **not** recommended: merging or reordering
task blocks — the order encodes the cost-assembly sequence, so that would be a behavioural
change dressed as a refactor.

---

## E. Awaiting an Idriss ruling (not measurement questions)

### E1. `a_cruise_max` — delete a documented-but-off capability?

The pre-planner's cruise-phase acceleration constraint (`coarse_preplanner.py:349-356`, M7 v21)
is gated on `a_cruise_max > 0.0`. `SimConfig.preplanner_a_cruise_max` defaults to `0.0`, is not
exposed by `dca.main` or `run_m7_single_step`, and is reachable only by hand-editing
`SimConfig` — research sediment by the chantier's usual test (7 dead statements + 2 config
fields, ~12 lines).

But CLAUDE.md documents it as a parameter ("CoM shaping — a_cruise_max=0.0 (off)"), so removing
it deletes a documented capability rather than dead plumbing. Not a call the chantier should
make on its own.

### E2. `locomotion_planner.py` — see §C5

Same shape: dead on the canonical, live in the M0/Lutze paper baseline. The question is whether
that baseline is still to be re-run.
