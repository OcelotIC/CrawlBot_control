# PHASE CLEANUP-28 — `test_reworked_qp.py`, audited against the code

The brief: audit this file against the updated documentation and the code it
claims to test, on the working hypothesis that it is deprecated relative to the
canonical two-task design and should be deleted — "the old kwargs are not meant
to exist anymore".

The kwargs half of that is exactly right, and the code says so itself
(`wholebody_qp.py:126-131`):

> CLEANUP-6/7 removed the stacks this superseded (legacy CoM / torso-6D P1, the
> cooperative split, the Option D tube, the projected EE task, T-MOM v1, the
> soft-CoM residual) and their config fields.

That list names, one for one, the mechanisms this file's harness switched on.
So the file *cannot* be repaired by adjusting numbers.

**But the harness is not the assertions.** Measured test by test: **2 of 8 are
dead, 6 are about live code and all 6 pass right now against the canonical
stack.** Retiring the file wholesale would have deleted working component
coverage of the module the paper rests on — which currently has none.

---

## 1. Method: express each assertion on the current stack, or fail to

For every test, one question: can its assertion be *set up* with the config
surface that exists today (`ss_two_task_mode` + `alpha_torso_pose` +
`ss_alpha_mom`)? If yes it is portable and the only question is the number. If
expressing it at all would require restoring a removed feature, it is dead.

Seven probes were written and run against the current code **before any edit** to
the test file — every number in §2 and §3 comes from those runs, not from reading
the code and reasoning about it. The probes are not committed: the ported tests
*are* them, permanently, which is the point. Re-deriving the verdict means running
`pytest tests/test_reworked_qp.py`; the one comparison the tests do not carry (the
torso-ON vs torso-OFF harness choice) is tabulated in §3.

---

## 2. Per-test verdict — measured

| test | built on | in the code today | verdict | measured on the two-task stack |
|---|---|---|---|---|
| T7 `test_torso_and_ee_tracking` | `ee_null_space` (projected EE task) + `alpha_torso` | torso-pose 6-D **2000** + swing-EE 6-D **1000**, both direct | **PORT** | torso **1.800 mm** (< 5), EE pos **0.481 mm** (< 10), EE ori **0.318°** (< 5) |
| T8 `test_soft_com_reduces_rms` | `alpha_com_soft` | **gone** — soft-CoM residual task deleted; the QP has no direct CoM feedback | **DELETE** | not expressible: nothing to compare α=0 against |
| T9 `test_residual_small` | nothing — the hard equality constraint | `_add_equality_constraints`, live, architecture-independent | **PORT** | max residual **8.882e-16** (gate 1e-6) |
| T10 `test_energy_decay` | `alpha_passivity`, `passivity_active` | both live (`wholebody_qp.py:826`) | **PORT** | 0 violations in **both** DS modes; decay 137450× / **272×** |
| T-MOM/1 `test_static_hold` | `cooperative_arms_mode` + `ss_centroidal_momentum_task` | the task row survives verbatim in `_com_task_rows`; only the gate flag changed | **PORT** | ‖q̈‖ **5.3e-10**, drift **0.0000 mm**, task resid **3.1e-10** |
| T-MOM/2 `test_pure_tracking_per_axis` | ditto | ditto; the J̇ half needs no QP at all | **PORT** | J̇ FD rel **1.536e-07** (gate 1e-3); worst CoM track **3.22 mm** (guard 50) |
| T-MOM/3 `test_mass_scalar_sanity` | ditto | ditto | **PORT** | ratio **0.0689 → 0.4774 → 0.8447**, monotonic, top in [0.60, 1.40] |
| T-MOM/4 `test_variant_b_weak_reference_coexistence` | `ss_alpha_tl_weak` | **gone** — the weak torso-linear regulariser was half of the cooperative split | **DELETE** | not expressible: the two-task stack has one 6-D torso task, so no Variant A/B |

### Why the two deletions are deletions and not stale thresholds

Both fail the same way the CLEANUP-26 retirement of
`test_fk_reference_consistency` failed: **repairing them would mean restoring
the feature.**

- **T8** compares CoM-tracking RMS at `alpha_com_soft = 0` against `= 5`. The
  task that weight scaled no longer exists, and CLAUDE.md records the design
  decision as final: *"Soft-CoM residual disabled (QP has no direct CoM
  feedback)"*. There is no α > 0 arm of the comparison to run.
- **T-MOM/4** asserts Variant B (weak torso-linear reference ON) does not fight
  CoM tracking relative to Variant A (OFF). Both variants were the cooperative
  split's torso-linear channel. The two-task stack replaced that channel with a
  single 6-D torso-pose task, so the A/B contrast has no referent.

---

## 3. The port needed **zero** new constants

This is the finding that settled the delete-vs-port question. Every threshold in
the ported tests is the original one, unchanged:

```
5 mm / 10 mm / 5 deg     torso + EE tracking      -> 1.800 / 0.481 / 0.318
1e-6                     dynamics residual        -> 8.882e-16
0 violations, 1.5x       DS passivity             -> 0, 272x (and 137450x)
1e-2 / 2e-4 / 1e-6       static-hold q̈/drift/res  -> 5.3e-10 / 0.0 / 3.1e-10
1e-3                     J̇_com·q̇ FD              -> 1.536e-07
(0.05, 2.00) / (0.60, 1.40)  mass-scalar bands    -> 0.0689 / 0.8447
50 mm                    divergence guard         -> 3.22 mm worst
```

A test whose numbers all still hold against a re-architected controller is not
testing the architecture — it is testing physics and formulation. That is the
class worth keeping, and it is why the recommendation here differs from the
opening hypothesis.

One harness choice was needed, and it is a translation rather than a new number.
The retired momentum tests isolated T-MOM by holding torso **angular** and
leaving torso **linear** free — a split the cooperative stack allowed. The
two-task torso task is 6-D: on or off. Isolating the task under test therefore
means the torso task **off** (`p_torso_ref=None`, which `wholebody_qp.py:427`
already supports). Measured both ways before choosing:

| mass-scalar sweep | ratios | monotonic | in original bands |
|---|---|---|---|
| torso task ON | 0.0418 → 0.3528 → 0.7657 | yes | **no** — 0.0418 < 0.05 floor |
| torso task OFF (chosen) | 0.0689 → 0.4774 → 0.8447 | yes | **yes** |

With the torso task on, the floor would have had to be retargeted — an invented
number. With the faithful translation, the original bands hold. The 1/71 ≈ 0.0141
bug signature stays discriminable either way.

The one deliberate deviation from canonical weights is inherited, not new:
`alpha_wrench = 1e-2` instead of 1. With no NMPC in this loop `lambda_ref = 0`,
so the canonical weight penalises the contact force that is the *only* means of
accelerating the CoM through the stance weld (net external force = contact force
= m·a_com). The retired file needed the same correction for the same reason.

---

## 4. A defect the port surfaced

The two DS-passivity cases reported different initial energies from the same
seed — `T0 = 0.2947` vs `0.2856`:

```python
rng = np.random.default_rng(1)          # same seed both cases
v_raw[6:] = rng.normal(...) * 0.5       # same v_raw
Jc_full, _ = robot.get_contact_jacobians(True, True)
v = N_contact @ v_raw                   # <- differs
```

`get_contact_jacobians` reads the interface's **internal** state
(`robot_interface.py:445`, `s = self.state`), and the `robot` fixture is
module-scoped. So the null-space projector — and therefore the initial condition
— was built at whatever configuration the *previously run test* left behind.
Order-dependent initial conditions, inherited from the original T10.

Fixed with an explicit `robot.update(q0, v0)` before the Jacobian read. Both
cases now report `T0 = 0.2943`, and a single case run in isolation reports
`T0 = 0.2943, decay 272.2x` — identical, so order-independence is measured, not
asserted.

---

## 5. What changed on disk

`tests/test_reworked_qp.py` rewritten: 8 tests → 6 (7 items — passivity is
parametrised over both DS modes). Classes renamed off the retired milestone
labels (`TestT7TrackingSS` → `TestTrackingSS`, `TestPhase20TMOM` →
`TestMomentumTask`); `_make_m2_qp` → `_build_qp`; the removed `q_t` argument of
`solve()` dropped; the `_integrate` docstring corrected from the 6-DoF era
(`nv=18`, 12 joints) to `nv=20`, 14 joints.

**The filename is deliberately unchanged.** `test_reworked_qp` is an M2-era name
and reads wrong now, but 21 citations across 9 files point at it — most of them
historical report references describing what the file was. Renaming would either
dangle them or require rewriting history to say something it did not say. The
module docstring carries the current description instead.

```
before:  8 tests, 8 failures, 0 test bodies executed
after :  7 items, 7 passed in 76 s
```

---

## 6. Verification

```
pytest tests/test_reworked_qp.py         7 passed in 75.68s
pytest tests/  (MUJOCO_GL=disabled)      see §7
gate/sync_docs.py --check                in sync
gate/verify_params.py                    14/14 rows OK
gate/verify_docs.py                      33 documents OK
gate/link_audit.py                       0 BROKEN BY MOVE
```

`crawlbot/` was **not** touched — this pass is `tests/` and reports only, so the
canonical run cannot have moved.

⚠ `MUJOCO_GL=osmesa` now breaks *collection* of `tests/test_diagnostics.py` in
this container (`PyOpenGL: 'NoneType' object has no attribute 'glGetError'`) —
an environment regression, not a code one, and it aborts the whole run. Use
`MUJOCO_GL=disabled`, which is what `gate/run_gate.py` already does. Worth a
line in the suite-gating decision: a run mode that aborts on collection is
indistinguishable from a broken suite to anyone reading CI output.

---

## 7. Suite state after this pass

Full run, `MUJOCO_GL=disabled`, 649 s:

```
before:  9 failed, 196 passed, 4 skipped, 3 errors
after :  1 failed, 203 passed, 4 skipped, 3 errors
```

The arithmetic closes exactly: −8 failures (the whole `test_reworked_qp` set),
+7 passes (the ported items). Nothing else moved.

What remains, and neither item is this file's:

| remaining | status |
|---|---|
| `test_coarse_preplanner::test_far_infeasible_under_tight_rate` | **the only failing test in the repository**, and the only one that predates the chantier (CLEANUP-27 §4, measured at `4e2e8da^`). Envelope semantics at cap 2.5 — already on CLAUDE.md's Remaining Work |
| `test_mid_waypoint_reshape` ×3 errors | `FileNotFoundError`, fixtures for the reshape path removed in CLEANUP-15. CLEANUP-27 class D, retirement recommended |

Retiring the reshape trio would leave the suite at **one** failure with a written
diagnosis behind it. At that point gating the suite is a policy decision, not a
project — which is the state CLEANUP-27 §5 argued for.

---

## 8. Carried forward

`robot_interface.get_contact_jacobians` documents its return as
`(6*nc, 18)` (`robot_interface.py:440`) — a 6-DoF-era dimension; the model has
`nv = 20`. Same staleness class as the `_integrate` docstring fixed here, but in
`crawlbot/`, so it carries the Rule-15 doc-and-gate routine with it. Deliberately
not mixed into a tests-only commit; recorded in `CLEANUP_CARRYOVER`.
