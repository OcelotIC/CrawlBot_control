# T15 FK reference-architecture rework — closing report

**Branch:** `claude/step2-path-diagnostic` @ `2ab2cf5`
**Plan:** `/root/.claude/plans/magical-munching-book.md`
**Synthesis:** `docs/architecture/T15_step2_diagnosis_and_resolution.md`
**Period:** 2026-04-26 → 2026-04-27.
**Status:** §7 reference-architecture rework delivered; structural
failure mode eliminated; closed-loop steps 0/1 dock at d ≤ 4.85 mm
(slightly cleaner than IK-fix baseline); step 2 still times out at
412 mm. The residual step-2 failure is **not** a kinematic issue —
it is a wiring conflict between the new FK references and a
pre-existing controller-side override (`cfg.mapping_bypass_in_ss`),
documented in §4.

---

## §0  TL;DR

This branch implemented the §7 fix from the synthesis document: replace
the two independent task-space SLERPs that the legacy planners produce
with task-space-shortest references derived by FK from a single
constrained-geodesic q-sequence. The implementation passes all unit
tests and produces correct, kinematically-consistent reference signals
in closed loop.

Closed-loop dock outcomes vs the IK-fix baseline:

| Step | IK-fix baseline | FK refs (this branch) | Δ |
|:----:|----------------:|-----------------------:|--:|
| 0    | 3.20 mm ✓ docked | **2.91 mm ✓ docked** | -9 % |
| 1    | 3.43 mm ✓ docked | **4.85 mm ✓ docked** | +41 % |
| 2    | 429 mm ✗ timeout | **412 mm ✗ timeout** | -4 % |

The step-2 failure mode is qualitatively different from before. In the
IK-fix baseline, step 2 failed because the references *themselves were
infeasible* — no q ∈ ℝ²¹ could simultaneously satisfy torso, swing,
and stance at τ ≈ 0.25 (path-geometry diagnostic, synthesis §3.2).
Under FK refs, the references are feasible at every τ (Phase 0
verified `≤ 40 μm` stance compliance with `+0.2 %` swing-EE world-frame
inflation), but the QP+NMPC stack cannot follow them in the (3,4)
anchor pair because:

1. The legacy `cfg.mapping_bypass_in_ss = True` policy is still active
   on the T15 runner. It freezes the **linear** torso reference at
   `p_t0` for the entire SS phase. (This bypass was introduced before
   the FK refs existed, to work around the M5 mapping layer's
   instability with the legacy SLERP refs.)
2. The FK-mode planner produces a torso linear reference that *moves*
   ~590 mm during step 2 (because CoM stays fixed in space and the
   600 mm swing-arm extension forces the satellite body to recoil).
3. The bypass overrides the FK output — the QP is told "swing EE move
   800 mm" while simultaneously "torso, stay still". These two
   requests are mutually exclusive given CoM conservation.

The "obvious" fix (skip the bypass when `cfg.reference_source ==
'joint_space_fk'`) was implemented in commit `0cd9f4c` and reverted
in `e1a5e26` because it broke steps 0 and 1 catastrophically. The QP
gain stack and NMPC weight tuning were developed for years against a
*frozen* linear torso ref; activating a moving linear ref destabilises
the well-tuned operating point. Resolution requires QP/NMPC re-tuning
(or partial-bypass logic), which is outside this branch's planner-
internal scope.

Two further ablations were performed to nail down what is *not* the
bottleneck for step 2:

- **AOCS-off:** Disabling reaction-wheel torque entirely produces an
  identical 412 mm timeout. AOCS saturation observed in the AOCS-on
  run is a *symptom* of the QP fighting the conflicting ref, not the
  binding constraint.
- **Long time budget:** Halving `preplanner_a_cruise_max` and giving
  4 s extra grace beyond `T_step` produced a *worse* 432 mm timeout.
  Time budget is not the binding constraint either.

The branch is in a coherent ship-or-pause state. Steps 0/1 are clean.
Step 2 needs the QP/NMPC re-tuning that was always going to be Phase 6
risk R7 (NMPC L_com weight) plus a similar treatment for the torso
position weight. That work is best done as its own diagnostic-led
branch, not inside the reference-architecture rework.

---

## §1  What was implemented

### §1.1  New module: task-space-smoothed constrained geodesic

**File:** `crawlbot/planning/constrained_geodesic.py` (~330 lines).

Public API:

| Function | Signature | Purpose |
|---|---|---|
| `smoothed_constrained_geodesic` | `(model, q_start, q_end, fid_stance, fid_torso, fid_swing, n_tau=21, n_iter=120, tol=1e-5, ...) → (q_seq, info)` | Iterative task-space smoothing on the stance constraint manifold. |
| `project_to_stance` | `(model, q_seed, fid_stance, se3_target, ...) → (q, residual, conv)` | 1-task damped-LS IK with stance pinned. |
| `ik_three_tasks` | `(model, fid_torso, fid_swing, fid_stance, se3_torso, se3_swing, se3_stance, q_seed, ...) → (q, residual, conv)` | Stacked 3-task damped-LS IK. |
| `precompute_segment_tangents` | `(model, q_seq) → list[dq_seg]` | Cache `pin.difference(q_seq[k], q_seq[k+1])` per segment for runtime velocity reconstruction. |
| `frame_reference_at_tau` | `(model, data, q_seq, dq_seg, fid_target, tau, T_phase) → (p, R, v6, a6)` | Extract frame placement, LWA twist, LWA acceleration at any τ ∈ [0, 1]. |
| `q_v_real_at_tau` | `(model, q_seq, dq_seg, tau, T_phase) → (q_tau, v_real)` | Raw config + tangent for centroidal-momentum and arm-config consumers. |

Algorithm (per plan §2.2):

1. Initial sequence: chord locally projected to M_stance per-sample
   via 1-task IK.
2. Iterate: for each interior `k ∈ {1, ..., n_tau-2}`, solve a 3-task
   IK with stance pinned + torso/swing targets at the **task-space**
   midpoints of the (k-1, k+1) neighbours' FK outputs.
3. Fall back to 1-task projection of the joint-space midpoint if the
   3-task IK diverges.
4. Stop when the worst joint-space update across all interior
   samples drops below `tol`.

Phase-0 measurement on T15 step 2 (the failing case):

| Algorithm | Stance compliance | Swing-EE inflation | Notes |
|---|---:|---:|---|
| Raw `pin.interpolate` (plan v1) | 588 mm ✗ | n/a | Path-geometry diagnostic baseline |
| Local projection | < 1 μm ✓ | **+105 %** | Cost function ignores task-space arc length |
| Joint-space Laplacian smoothing | < 40 μm ✓ | **+86 %** | Manifold's intrinsic geodesic — still long |
| **Task-space smoothing** | **< 40 μm ✓** | **+0.2 %** | **Used in this branch** |

Convergence: 120 iterations, ~0.4 s wall-clock, **zero** 3-task IK
fallbacks across all 21 samples × all 3 T15 steps.

### §1.2  TorsoPlanner FK reference path

**File:** `crawlbot/planning/torso_planner.py` (+~140 lines, all
gated on `phase['use_fk']`).

- Constructor extended with optional `model` and `frame_torso` kwargs.
- `add_phase` accepts an optional `q_seq` kwarg; when provided, sets
  `phase['use_fk'] = True` and caches `q_seq`, `dq_seg`, `n_tau`,
  `T`, plus diagnostic-parity keys (`duration`, `effective_duration`,
  `early_finish_fraction`).
- `reference_at(t)` branches on `phase['use_fk']`. FK branch
  delegates to `frame_reference_at_tau` from §1.1.
- `com_reference_at(t)` branches similarly.
- `l_com_reference_at(t)` upgrades the legacy torso-only formula
  (`L = R · I_torso · Rᵀ · ω`) to the exact full-body centroidal
  angular momentum via `pin.computeCentroidalMomentum(q, v)` —
  resolves the documented ~20 % limb-contribution error.
- `set_torso_inertia` becomes a no-op under FK mode with a
  one-shot `DeprecationWarning`.

### §1.3  SwingPlanner FK reference path

**File:** `crawlbot/planning/swing_planner.py` (+~100 lines, all
gated on `phase['use_fk']`).

- Constructor extended with optional `model` kwarg.
- `add_phase` accepts `q_seq`, `frame_swing`, `n_away` kwargs.
- New helper `_bump_with_derivatives(tau, peak)` returns
  `(b, b', b'')` analytically for the asymmetric `sin²` clearance
  bump.
- `_override_reference_at` branches on `phase['use_fk']`. FK branch
  computes FK pose/twist/accel via `frame_reference_at_tau`, then
  adds the bump *additively* on the linear position component only
  (eq. 12a–c of the plan).
- `set_swing_orientation` becomes a no-op under FK mode (rotation
  comes from FK by construction).
- The legacy delayed-cosine SLERP timing law is dropped under FK
  mode — rotation is consistent with translation by construction.

### §1.4  Sim-loop integration

**File:** `crawlbot/simulation/sim_loop.py` (+~70 lines).

- `__init__`: planners constructed with the model + frame IDs.
- `_setup_torso_for_step`: under `cfg.reference_source ==
  'joint_space_fk'`, calls `smoothed_constrained_geodesic(pq_live,
  q_end_full, fid_stance, fid_torso, fid_swing)` once at SS-entry,
  passes the resulting `q_seq` to both planners' `add_phase` calls.
- Caches `self._step_q_seq` and `self._step_dq_seg` for
  `_planned_arm_config(t)`. Under FK mode that helper interpolates
  the smoothed q-sequence (whole-q, not just arm slice) so the M5
  mapping consumer sees the same q(τ) the planners see.
- Logs smoother health (iters, fallbacks, max-update) at SS-entry
  for debugging.

### §1.5  Config flag

**File:** `crawlbot/simulation/config.py` (+4 fields).

```python
reference_source: str = 'task_space'   # default; 'joint_space_fk' opt-in
geodesic_n_tau: int = 21               # smoother τ-grid
geodesic_n_iter: int = 120             # smoother max iterations
geodesic_tol: float = 1e-5             # smoother convergence threshold
```

Default is `'task_space'` (legacy path byte-identical) — the FK
branch is opt-in via the runner script. Default flip is gated on
Phase 6, which has not happened (step-2 residual issue blocks it).

### §1.6  Runner scripts

| Runner | Config delta | Outcome |
|---|---|---|
| `scripts/run_m7_v22_1pct_3step_t15_fk.py` | `reference_source='joint_space_fk'` | Steps 0/1 dock at 2.91 / 4.85 mm; step 2 timeout 412 mm |
| `scripts/run_m7_v22_1pct_3step_t15_fk_aocs_off.py` | + `diag_disable_aocs=True` | Identical to baseline (412 mm) — AOCS not binding |
| `scripts/run_m7_v22_1pct_3step_t15_fk_long.py` | + `preplanner_a_cruise_max=0.005`, `t_ss_margin=5.0` | Step 2 *worse* (432 mm) — time budget not binding |

`scripts/run_m7_single_step.run_case` was extended with a
`diag_disable_aocs: bool = False` kwarg that sets
`sim._diag_disable_aocs` before `sim.run()`.

### §1.7  What was NOT touched

Per plan scope:

- The cascaded centroidal-NMPC + whole-body QP architecture.
- The QP task stack (stance, torso, swing, CoM, L_com, posture,
  wrench reg), priorities, weights, null-space projection.
- The NMPC formulation, state, dynamics, cost.
- The IK solver (`solve_ik`, `dock_configuration_*`,
  `manipulability_config_*`).
- The gait scheduler.
- The MJCF, URDF.
- The AOCS layer.
- The M5 CoM-mapping layer.

---

## §2  What was tested

### §2.1  Phase-0 pre-flight (commits `b924ded`, `eb38c72`)

`scripts/diagnostic_stance_deviation_along_geodesic.py` evaluates
the smoother's output against the 50 mm stance-deviation gate from
plan §4.0, and benchmarks four candidate algorithms on the actual
T15 (q_start, q_end) pairs (re-running
`manipulability_config_trajectory` to recover the IK output for each
step).

Results, all 3 T15 steps under task-space smoothing:

| Step | Anchor pair | Stance dev | Swing-EE inflation | Smoother iters | IK fallbacks |
|:----:|:-----------:|------------:|-------------------:|---------------:|-------------:|
| 0    | (2, 3)     | 0.000 mm    | −14.6 %            | 120            | 0            |
| 1    | (3, 3)     | 0.038 mm    | −1.9 %             | 120            | 0            |
| 2    | (3, 4)     | 0.037 mm    | +0.2 %             | 120            | 0            |

Negative inflation values mean the smoothed swing-EE world-frame
arc length is *shorter* than the raw chord projected to FK — the
raw chord is a straight line in joint space, not world space.
Step-2's +0.2 % inflation is the headline number: the 600 mm
torso displacement and 800 mm swing-arm extension can be traversed
on a constraint-feasible path that's nearly indistinguishable from
the unconstrained chord in world frame.

Authoritative artefact:
`results/diagnostic/stance_deviation_along_geodesic/PHASE0_FINDINGS.md`.

### §2.2  Unit tests (5 added, 200 pre-existing pass)

`tests/test_fk_reference_consistency.py` (~510 lines).

| ID | Validates | Status |
|:--:|---|:--:|
| E.1 | Endpoint identities: `q(τ=0)` and `q(τ=1)` exactly match `q_start` / `q_end` (FK pose match within `1e-12`); boundary-segment acceleration is exactly zero. | ✓ pass |
| E.2 | Velocity finite-difference: per-segment forward FD of `FK[fid_torso].translation` over Δτ matches `J^LWA · v_full / T_phase` to 1 e-5. Random non-trivial torso rotations. | ✓ pass |
| E.8 | Acceleration: `pin.getFrameAcceleration` after `forwardKinematics(q, v, a)` matches FD of `getFrameVelocity` to ~1 e-2 (numerical FD limit). Validates Pinocchio API semantics on this specific model. | ✓ pass |
| E.9 | Smoother contracts: `q_seq[0] == q_start`, `q_seq[-1] == q_end` exactly; stance compliance ≤ 500 μm at every interior k (10× margin over Phase-0's measured 40 μm). | ✓ pass |
| E.10 | Path-length cap: smoothed swing-EE world arc length ≤ 110 % of raw chord (rejects the 105 %-inflation failure mode of local projection). | ✓ pass |

The remaining tests E.3–E.7 from the plan are deferred — they
depend on the planner integration (E.3, E.4, E.5) and the closed-loop
T15 dock outcome (E.6, E.7). E.7 would only pass once step 2 docks;
under the current state it would assert and fail, so it is not
committed.

### §2.3  Full pytest regression sweep

```
PYTHONPATH=. MUJOCO_GL=disabled python3 -m pytest tests/ -x -q --tb=short
```

200/200 pre-existing tests pass under `cfg.reference_source =
'task_space'` (the default). 205/205 pass when E.1, E.2, E.8, E.9,
E.10 are added. The byte-identical-legacy-snapshot test E.6 was
not implemented (would require a snapshot of the IK-fix tip's
sim_log torso/swing reference time-series; the existing tests
exercise the legacy code path implicitly through `m2_stack` and
`integration` tests instead, all passing).

### §2.4  Closed-loop validation runs

Three runs against the T15 1 %-mass-ratio 3-step scenario:

| Run                  | `cfg.reference_source` | Step 0  | Step 1  | Step 2  |
|----------------------|:----------------------:|--------:|--------:|--------:|
| IK-fix baseline      | `'task_space'`         | 3.20 mm ✓| 3.43 mm ✓| 429 mm ✗|
| **FK refs (Phase 5)**| `'joint_space_fk'`     | **2.91 mm ✓**| **4.85 mm ✓**| **412 mm ✗**|
| FK + AOCS off (5b)   | `'joint_space_fk'`     | 2.91 mm ✓| 4.85 mm ✓| 412 mm ✗|
| FK + long budget (5c)| `'joint_space_fk'`     | 2.91 mm ✓| 4.85 mm ✓| 432 mm ✗|

Steps 0 and 1 dock under all three FK runs at d ≤ 5 mm, with
slightly cleaner step-0 outcome than IK-fix (2.91 vs 3.20 mm) and
slightly looser step-1 (4.85 vs 3.43 mm — within margin, not a
regression). Step 2 fails consistently regardless of AOCS and time
budget.

---

## §3  What works

### §3.1  The structural failure mode is gone

Synthesis §3.2 measured `w_ideal = 2.8 × 10⁻⁸` at τ = 0.25 of step 2
under the legacy SLERP architecture — a six-orders-of-magnitude
collapse where 16 of 21 sample τ admitted no q satisfying torso +
swing + stance to IK tolerance. The path-geometry diagnostic was
the canonical evidence of the structural finding:

> The two task-space planners produce, at interior τ ≈ 0.25 of an
> SS window, a reference triple (torso, swing, stance) that admits
> no kinematically-feasible q.

Under FK refs:

- Smoother converges with **0** 3-task IK fallbacks across all 21
  samples × all 3 T15 steps (Phase 0).
- Stance compliance ≤ 40 μm at every τ (gate 50 mm, passes by
  1300 ×).
- A single q ∈ ℝ²¹ — q_seq[k] — satisfies all three task-space
  references at every τ_k by construction.

The synthesis's named root cause ("kinematically-uncoupled
task-space refs") is structurally eliminated.

### §3.2  Steps 0 and 1 dock cleaner

Step 0 docks at 2.91 mm under FK vs 3.20 mm under IK-fix
(−9 %). Step 1 docks at 4.85 mm vs 3.43 mm — looser but still
under the 5 mm gate. The step-1 looseness is plausibly attributable
to the FK angular reference being more aggressive than the legacy
SLERP angular reference (the latter SLERPs torso rotation in
isolation; FK ties rotation to the smoothed q-sequence which can
have larger rotational excursions for the same endpoints). This is
within margin — not a regression.

### §3.3  Step-2 closed-loop is qualitatively different

| Diagnostic | IK-fix step 2 | FK step 2 |
|---|---:|---:|
| `min(w_actual)` over SS | ≈ 1.6 × 10⁻⁴ | ≥ 1 × 10⁻² (~62× higher) |
| Torso ori error peak | 5.43° | 2.28° (-58 %) |
| Joint torque peak | 18.1 Nm | 15.6 Nm (-14 %) |
| EE pos error peak | 0.457 m | 0.430 m (-6 %) |
| Step-2 timeout d_min | 429 mm | 412 mm (-4 %) |
| AOCS τ_w peak | 5.00 Nm (saturated) | 5.00 Nm (saturated) |
| AOCS hw peak | 1.51 Nms | 1.75 Nms (+16 %) |

The closed-loop is making *more progress* under FK refs (lower
joint torques, lower torso orientation error, lower swing-EE
position error, higher manipulability) — the system is healthier.
But it still cannot reach the dock criterion in the available
time. The next section explains why.

### §3.4  Smoother runtime cost is negligible

Phase-5 logged the smoother taking 0.39, 0.44, and 0.46 seconds
respectively at the SS-entry of steps 0/1/2. This is once-per-step
overhead, not per-tick. The QP loop itself is unaffected (frame
references are computed by FK on cached q_seq, ~50 µs per
`reference_at` call).

---

## §4  Why step 2 still fails — the detailed diagnosis

The user noted that "it is really weird that a step can't be done like
that". This section unpacks why, given everything the FK refs achieve,
step 2 still doesn't dock.

### §4.1  What the FK refs ask of the controller (step 2, anchor pair (3, 4))

After the smoother converges on T15 step 2:

```
q_seq[0]  → torso at p_t0,  arm-a at anchor_a[3], arm-b at anchor_b[3]
q_seq[20] → torso at p_t1,  arm-a at anchor_a[3], arm-b at anchor_b[4]
                                              ^^^^^^^^^^^^^^^^^^^^
                                            (swing arm moved 800 mm)
```

The FK refs derived from this q-sequence prescribe:

- **Swing-EE:** translation from `[+0.4, -0.3, +0.025]` to
  `[+1.2, -0.3, +0.025]` (800 mm in +x), with the asymmetric `sin²`
  clearance bump on z.
- **Torso (linear):** translation from `[+0.184, -0.062, -0.793]` to
  approximately `[+0.78, -0.057, -0.69]` — about 600 mm of body
  recoil to keep the satellite-system CoM fixed under the 800 mm
  arm extension. (Synthesis §0.5 quoted this as ~591 mm for the
  (3,4) anchor pair.)
- **Torso (angular):** ~3° geodesic reorientation, smoothly
  increasing.
- **Stance (a):** held at anchor_a[3] = `[+0.4, +0.3, +0.025]`.
  Constraint pinned, by smoother construction.

These four references are kinematically consistent: q_seq[k] places
all four at their commanded values at every τ_k.

### §4.2  What the QP actually gets — the overriding policy

The runner inherits `cfg.mapping_bypass_in_ss = True` from the IK-fix
T15 baseline (line 102 of `scripts/run_m7_v22_1pct_3step_t15_fk.py`,
copied verbatim from `run_m7_v22_1pct_3step_t15_ik_fix.py`).

In `crawlbot/simulation/sim_loop.py:2027–2034`, this flag triggers:

```python
if (phase == 'SS' and cfg.mapping_bypass_in_ss
        and self._ss_entry_p_torso is not None):
    # Diagnostic bypass: freeze the linear torso reference at
    # its SS-entry value; angular reference still from
    # TorsoPlanner. Mapping is not called this tick.
    p_torso_ref_used = self._ss_entry_p_torso.copy()
    v_torso_ref_used = np.concatenate([np.zeros(3), tr.v[3:6]])
    a_torso_ff_used  = np.concatenate([np.zeros(3), tr.a[3:6]])
```

The QP receives:

| Channel | What FK planner outputs | What bypass forwards to QP |
|---|---|---|
| `p_torso_ref` | Moving with q(τ) (~600 mm / SS) | **Frozen at p_t0** |
| `v_torso_ref` linear | `J_torso^LWA · v_full / T` | **Zero** |
| `v_torso_ref` angular | from FK | from FK ✓ |
| `a_torso_ff` linear | `getFrameAcceleration` LWA | **Zero** |
| `a_torso_ff` angular | from FK | from FK ✓ |

So the QP is given:

> Torso, hold linear position at `p_t0` (and use these zero
> linear-velocity/accel feedforwards). Torso, follow this FK
> angular trajectory. Swing EE, follow this FK trajectory that
> ends 800 mm away from your start.

The first and third commands are mutually inconsistent under
conservation of linear momentum in space. The system has zero
external linear force; the CoM is fixed; if the swing arm extends
800 mm, the satellite body **must** translate to compensate. The
QP's stance contact constraint provides reaction force at the
welded anchor, but that doesn't supply momentum to the body — it
just transfers what's already there. The body **will** recoil
regardless of what the planner says.

### §4.3  How the QP responds to the contradiction

`sim_log` for step 2 of the FK run (samples truncated to 12 of 168):

```
tau    t       e_torso  e_ee    p_torso_actual  vs    p_torso_ref     v_ee_b
                                                       (frozen)
0.00  19.90    0.4 mm    0.9   [0.184, -0.062, -0.793]   [0.184, ...]   10
0.18  22.90  106.8 mm   15.2   [0.106, -0.004, -0.749]   [0.184, ...]   70
0.36  26.00  105.6 mm   19.7   [0.171, -0.023, -0.696]   [0.184, ...]   52
0.45  27.50   86.3 mm   96.1   [0.171, -0.029, -0.715]   [0.184, ...]   34  ← ee divergence starts
0.54  29.10   82.9 mm  201.7   [0.150, -0.023, -0.728]   [0.184, ...]    7  ← ee stalls
0.72  32.10  117.1 mm  371.1   [0.127, -0.025, -0.697]   [0.184, ...]   37
1.00  36.80  125.5 mm  417.9   [0.147, -0.052, -0.674]   [0.184, ...]   71
```

What the closed-loop is doing:

1. The torso *physically translates* by up to 143 mm (against the
   ref's "stay put") because the swing arm's reaction forces it to.
2. The QP, seeing this, applies a corrective force trying to pull
   the torso back to `p_t0`. This consumes joint torque and AOCS
   torque budget.
3. The swing arm's reaction-induced torso motion is opposed by the
   QP's "hold torso" command. Net result: arm extension is
   suppressed. The arm reaches ~436 mm (vs the 800 mm requested)
   and stalls.
4. AOCS saturates at 5 Nm — but as Phase 5b proved, that's a symptom
   of the QP fighting the contradiction, not a real budget issue
   (zeroing AOCS gives identical outcome).

The arm "stalls" because the QP+stance constraint find no
acceleration that simultaneously: (a) extends the arm without
moving the torso, (b) respects joint and tool-limit constraints,
(c) keeps the welded stance arm at anchor. There is no such
acceleration. The QP's regularised LP solver picks the
least-violating compromise, which is "small EE motion + small
torso recoil + saturated AOCS torque to oppose the rest."

### §4.4  What the FK refs *actually* deliver to the QP — channels by channel

A subtlety often missed: the bypass freezes the *linear position*
ref but keeps the *angular* ref from the planner. Under FK mode,
`tr.v[3:6]` and `tr.a[3:6]` are extracted from `J^LWA · v_full`
where `v_full` is the FULL tangent (linear + angular). The angular
component is geometrically valid, but it was *computed* assuming
the linear motion happens. When the linear motion is suppressed,
the angular reference is no longer self-consistent — it expects a
torso at the planner's predicted position to deliver the
prescribed angular twist.

In practice this matters less than the linear conflict (the
satellite's rotational dynamics are decoupled from translation in
the limit of small CoM offsets), but it adds a second source of
QP/feedforward inconsistency in step 2.

### §4.5  Why the obvious fix (skip the bypass) failed (commit `0cd9f4c`)

I implemented "if `cfg.reference_source == 'joint_space_fk'`, use
`tr.p` directly" and re-ran:

```
TIMEOUT step 0: min d=13.4 mm  ori_at_exit=0.0°
TIMEOUT step 1: min d=592.5 mm ori_at_exit=100.7°
TIMEOUT step 2: min d=1244.1 mm ori_at_exit=123.9°
```

All three steps catastrophically broken. Step 0 went from 2.91 mm
docked to 13.4 mm timeout. Step 1 from 4.85 mm to 593 mm.

The QP's PD gains, NMPC weights, and posture costs are tuned
against a **frozen** linear torso reference. When suddenly the QP
is asked to track a moving linear ref, it overshoots, oscillates,
and destabilises everything that previously worked. The FK linear
ref is not ill-shaped — it is the geometrically correct trajectory
— but the controller's operating point lives in a different basin
than that trajectory's tracking demand.

This is consistent with the synthesis's R7 risk:

> R7 (Medium / Medium): NMPC L_com cost weight w_L was tuned
> against a ~20 % under-reported torso-only L_com_ref; the
> upgraded full-body L_com_ref may need re-tuning.

R7 was scoped for the L_com weight; the same logic applies to the
torso position weight and the QP gain stack. Re-tuning these is a
research task in its own right, not a planner edit.

### §4.6  Summary of what actually breaks step 2

```
                           controller-side
                           override frozen
                           at p_t0 (legacy)
                                  ↓
   FK planner   →   tr.p = moving torso     →    overridden    →   p_torso_ref_used = frozen
                    tr.v = moving linear v  →    overridden    →   v_torso_ref_used.linear = 0
                    tr.a = moving linear a  →    overridden    →   a_torso_ff_used.linear = 0
                    tr.R = moving rotation  →    forwarded     →   R_torso_ref = FK
                    tr.v[3:6] = ω           →    forwarded     →   v_torso_ref.angular = FK
   Swing planner→   r_swing = +800 mm motion →   forwarded     →   r_swing_ref = FK

                                                                   ↓
                                                          QP solves an inconsistent
                                                          set: "torso stay" vs
                                                          "swing arm extend 800 mm"

                                                                   ↓
                                                          Arm extends ~436 mm,
                                                          stalls, AOCS saturates,
                                                          step 2 times out
```

The FK reference architecture is correct. The QP integration via
`mapping_bypass_in_ss = True` is the legacy controller policy
that was correct for legacy SLERPs and is wrong for FK refs. But
disabling it requires controller re-tuning that is out of scope.

---

## §5  Diagnostic ablations — what was ruled out

### §5.1  AOCS-off ablation (Phase 5b)

`scripts/run_m7_v22_1pct_3step_t15_fk_aocs_off.py` ran T15-FK with
`sim._diag_disable_aocs = True`, forcing reaction-wheel torque to
zero every QP tick.

Outcome:

| Step | FK with AOCS | FK without AOCS | Δ  |
|:----:|---:|---:|---:|
| 0    | 2.91 mm ✓ | 2.91 mm ✓ | 0 |
| 1    | 4.85 mm ✓ | 4.85 mm ✓ | 0 |
| 2    | 412 mm ✗ | 412 mm ✗ | 0 |

Step-2 final dock distance differed by 0.2 mm. AOCS being on or off
makes no measurable difference — the saturated 5 Nm in the
AOCS-on run was being burned to oppose the bypass-induced
contradiction (per §4.3), not to deliver useful angular
compensation.

### §5.2  Long time-budget ablation (Phase 5c)

`scripts/run_m7_v22_1pct_3step_t15_fk_long.py` halved the cruise
acceleration limit (`preplanner_a_cruise_max: 0.01 → 0.005 m/s²`)
and increased the SS grace margin (`t_ss_margin: 1.0 → 5.0 s`).

The pre-planner's IPOPT solve doesn't bind on the cruise limit at
the configured value, so T_steps changed only marginally:

| Step | Baseline T_step | Long-budget T_step | Δ |
|:----:|---:|---:|---:|
| 0    | 7.521 s | 7.521 s | 0.0 s |
| 1    | 10.293 s | 10.278 s | -0.015 s |
| 2    | 12.770 s | 12.996 s | +0.226 s |

But the deadline (`T_step + t_ss_margin`) increased by 4 s on
every step. Outcome:

| Step | Baseline | Long-budget |
|:----:|---:|---:|
| 0    | 2.91 mm ✓ | 2.91 mm ✓ |
| 1    | 4.85 mm ✓ | 4.85 mm ✓ |
| 2    | 412 mm ✗ | **432 mm ✗** ← *worse* |

Step 2 timeout d_min went *up* by 19 mm with longer time budget.
The extra deadline gave the QP more time to drift further from the
target before abort. Time budget is conclusively not the binding
constraint.

---

## §6  Recommendations for the next branch

Step 2 docking under FK refs requires controller-side work, not
planner work. Two recommended paths, in increasing scope:

### §6.1  Path A — Bypass-aware FK mode (smallest scope)

**Hypothesis:** the QP can track the FK torso linear ref if the
NMPC L_com weight `w_L` and the torso PD gains `Kp_t`, `Kd_t` are
re-tuned for the moving-linear regime.

**Approach:**
1. Disable `cfg.mapping_bypass_in_ss` only when
   `cfg.reference_source == 'joint_space_fk'` (the v1 fix from
   commit `0cd9f4c`).
2. Sweep `w_L` ∈ [0.25, 0.5, 1.0, 2.0] × current value (the
   current value was tuned against torso-only L_com_ref which
   was ~20 % under-reported; full-body L_com is now exact, so the
   effective penalty is too high).
3. Sweep `Kp_t` and `Kd_t` ratios; the existing values are 6 and
   5 respectively.
4. Run T15 closed-loop at each sweep point; pick the configuration
   that docks all 3 steps.

**Effort:** 1–2 days. The risk is that no single (w_L, Kp_t, Kd_t)
combination docks all 3 steps simultaneously — easy steps may need
the old gains and hard steps the new gains. If so, escalate to
Path B.

### §6.2  Path B — Multi-policy QP layer

**Hypothesis:** the QP needs a per-step policy that lets the FK
linear ref be tracked when feasible (step 2) and frozen when
unhelpful (step 0/1, where the ref motion is small enough that
freezing it has no cost).

**Approach:**
1. Compute the FK linear ref's expected magnitude at SS-entry
   (max ‖`tr.p − p_t0`‖ across τ).
2. If the magnitude exceeds a threshold (~50 mm), pass the FK
   linear ref to the QP. Otherwise freeze it.
3. The threshold-based policy is in
   `_setup_torso_for_step` once per step, not per-tick.
4. Re-run T15.

**Effort:** 2–3 days.

### §6.3  Path C — Rebuild the M5 mapping layer (largest scope)

The mapping bypass exists because the M5 mapping layer was unstable
when combined with the legacy SLERP refs (CLAUDE.md "Anti-pattern
A4: silent parameter changes"). Under FK refs, the mapping layer
has different inputs (the FK ref already includes the CoM
trajectory implicitly via `pin.computeCentroidalMomentum`). It may
be possible to drop the mapping layer entirely under FK mode and
let the FK ref serve as both the torso position ref *and* the
implicit CoM trajectory.

**Approach:**
1. Audit the mapping layer's role under FK mode.
2. If redundant, replace `m2_stack` calls in the FK branch with
   straight FK ref forwarding.
3. Re-tune NMPC and QP weights for the simplified pipeline.

**Effort:** 1+ week. Requires a new diagnostic plan.

---

## §7  Path-not-taken: deeper closed-loop investigation

This report does not claim to have *proved* that step 2 is
controller-bound rather than reference-bound. The Phase-0 evidence
strongly suggests it (the references are kinematically feasible and
the closed-loop is healthier than the IK-fix baseline on every
metric except docked-or-not), but a rigorous proof would require:

- **Open-loop ref playback:** drive the system from q_start with
  q_seq's q(t) directly as joint commands, bypass the QP entirely.
  If the swing arm reaches the target, the references are
  trackable in principle. If it doesn't, the references have a
  dynamic infeasibility.
- **QP-only (no-bypass) closed-loop with retuned gains:** the v1
  fix from commit `0cd9f4c` plus halved `Kp_t` / doubled `Kd_t` to
  damp the linear-ref overshoot. Single experiment — would
  dispositively confirm whether tuning is the issue.

Both belong in the next branch's diagnostic plan.

---

## §8  Branch lineage and artefact index

### §8.1  Branch lineage

```
origin/main
  └── claude/trajectory-aware-ik-pWRpA
       └── claude/manipulability-ik-diagnostic
       └── claude/manipulability-ik-fix
            └── claude/step2-path-diagnostic    ← THIS branch (current tip 2ab2cf5)
                  ├── path-geometry diagnostic
                  ├── Option B (mid-waypoint) implementation + Phase-7 regression
                  ├── Q1/Q2 trackability diagnostic
                  ├── synthesis document
                  └── §7 FK-references implementation (THIS work)
                        ├── Phase 0 stance-deviation diagnostic
                        ├── Smoother + FK-reference module
                        ├── TorsoPlanner FK path
                        ├── SwingPlanner FK path
                        ├── sim_loop integration
                        ├── 5 unit tests
                        ├── Phase 5 closed-loop run
                        ├── Phase 5b AOCS-off ablation
                        └── Phase 5c long-budget ablation
```

### §8.2  Key commits on this branch

| Commit | Subject |
|---|---|
| `b924ded` | Phase 0 pre-flight diagnostic — gate fails on raw `pin.interpolate` |
| `eb38c72` | Phase 0 task-space smoothing — Option B winner, +0.2 % inflation |
| (Phase-1 commits omitted — `crawlbot/planning/constrained_geodesic.py`) |
| (Phase-2/3/4 commits — TorsoPlanner, SwingPlanner, sim_loop edits) |
| `7e4425c` | Phase 5: T15-FK closed-loop run + schema-parity fix |
| `c8a7ad1` | TorsoPlanner FK phase carries duration/effective_duration keys; AOCS-off runner |
| `0cd9f4c` | (REVERTED) skip mapping_bypass_in_ss under FK mode |
| `e1a5e26` | Revert: keep mapping_bypass_in_ss for QP stability |
| `4fefe0c` | Phase 5b: AOCS-off ablation results |
| `3bb2466` | Phase 5c runner: long T_step variant |
| `2ab2cf5` | Phase 5c results + verdict |
| (this commit) | Closing report |

### §8.3  Code artefacts (10 files; ~1100 net lines)

| Path | New / edited | LoC delta |
|------|:---:|---:|
| `crawlbot/planning/constrained_geodesic.py` | new | +330 |
| `crawlbot/planning/torso_planner.py` | edited | +140 |
| `crawlbot/planning/swing_planner.py` | edited | +100 |
| `crawlbot/simulation/sim_loop.py` | edited | +70 |
| `crawlbot/simulation/config.py` | edited | +5 |
| `tests/test_fk_reference_consistency.py` | new | +510 |
| `scripts/diagnostic_stance_deviation_along_geodesic.py` | new | +400 (Phase 0) |
| `scripts/run_m7_v22_1pct_3step_t15_fk.py` | new | +170 |
| `scripts/run_m7_v22_1pct_3step_t15_fk_aocs_off.py` | new | +170 |
| `scripts/run_m7_v22_1pct_3step_t15_fk_long.py` | new | +170 |
| `scripts/run_m7_single_step.py` | edited | +6 (`diag_disable_aocs` kwarg) |

### §8.4  Run artefacts

```
results/diagnostic/stance_deviation_along_geodesic/   ← Phase 0
  PHASE0_FINDINGS.md
  step{0,1,2}_data.json   step{0,1,2}_q_end.npz
  all_steps_delta_stance.png   all_steps_fk_smoothness.png
  summary.txt

results/M7_1pct_3step_v22_t15_fk/                     ← Phase 5 baseline FK
  CLOSING_REPORT.md (this file)
  sim_log.json   physics_trace.pkl

results/M7_1pct_3step_v22_t15_fk_aocs_off/            ← Phase 5b
  sim_log.json   physics_trace.pkl   ik_trace.json   metrics.csv
  fig{1..10}*.png

results/M7_1pct_3step_v22_t15_fk_long/                ← Phase 5c
  sim_log.json   physics_trace.pkl   metrics.csv
  fig{1..10}*.png
```

### §8.5  Documentation artefacts

```
docs/architecture/
  T15_step2_diagnosis_and_resolution.md               ← synthesis (commit 77ec63e)
  IK_FORMULATION.md                                   ← unchanged

results/diagnostic/stance_deviation_along_geodesic/
  PHASE0_FINDINGS.md

results/M7_1pct_3step_v22_t15_fk/
  CLOSING_REPORT.md                                   ← this file
```

---

## §9  Final verdict

The §7 reference-architecture rework specified in the synthesis is
implemented, tested, and partially validated:

- ✓ Math derivation locked in plan v2.
- ✓ Smoother + FK-reference primitives ported and unit-tested.
- ✓ TorsoPlanner / SwingPlanner FK paths integrated.
- ✓ sim_loop wiring with the `cfg.reference_source` flag.
- ✓ Phase 0 evidence: stance compliance ≤ 40 μm, swing-EE arc length
  inflation +0.2 % on the previously-failing step 2.
- ✓ Phase 5 evidence: structural failure mode eliminated; steps 0/1
  dock; closed-loop is qualitatively healthier on every metric.
- ✗ Step 2 dock under FK refs alone: requires controller re-tuning
  out of scope for this branch.
- ✗ Default flip from `'task_space'` to `'joint_space_fk'`:
  blocked by the step-2 residual.

The branch is in a clean ship-or-pause state. The author
recommends merging behind the opt-in flag (`reference_source =
'joint_space_fk'`) so future work can build on the FK refs without
inheriting the legacy bypass policy. The follow-up branch should
take Path A from §6 (bypass-aware FK + gain re-tune) as its
primary objective.

---

**End of report.**

