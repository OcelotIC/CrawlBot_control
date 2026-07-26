# T15-post-6 — Origin and rationale of `mapping_bypass_in_ss`

Read-only investigation. Facts only.

---

## §1 Introducing commit

### 1.1 Identity

| Field | Value |
|---|---|
| Commit SHA | `61dcfcaf7f09efad972ea36d9bd6f76fd7a9d41e` |
| Short SHA | `61dcfca` |
| Date | 2026-04-20 00:52:09 +0000 |
| Author | Claude |
| Subject | `M7: add mapping_bypass_in_ss flag + v21 mapping-off run` |
| Files touched | 16 (1 config + 1 sim_loop + 1 new runner script + 1 JSON + 10 PNG + 1 CSV + 1 PKL) |

### 1.2 Full commit message

```
M7: add mapping_bypass_in_ss flag + v21 mapping-off run

New SimConfig flag mapping_bypass_in_ss (default False). When True,
SimulationLoop._step bypasses the CoM->torso mapping during SS only:
QP receives p_torso_ref frozen at the SS-entry torso position
(captured in _setup_torso_for_step as _ss_entry_p_torso) with
v_torso_ref_lin = 0 and a_torso_ff_lin = 0. Angular reference still
comes from TorsoPlanner. DS phase is unchanged.

Misc/scripts/run_m7_v21_mapping_off.py mirrors the v21 baseline runner
with the flag turned on. Output:
Misc/runs/M7_1pct_1step_v21_mapping_off/.

192/192 tests still pass with the flag at its default False.

https://claude.ai/code/session_01DDnmj7RygRCQMS9vXdfiJe
```

### 1.3 `crawlbot/simulation/config.py` diff (entire addition)

Appended to the end of `SimConfig` (after the
`diag_disable_passivity_on_abort` block):

```diff
+
+    mapping_bypass_in_ss: bool = False
+    # M7 EE bisection follow-up. When True, sim_loop bypasses the
+    # CoM->torso mapping during SS only: the QP receives
+    #   r_torso_ref = r_torso(t = t_ss_start)   (frozen at SS entry)
+    #   v_torso_ref_lin = 0
+    #   a_torso_ff_lin  = 0
+    # for the linear components of the torso reference. Angular
+    # reference still comes from TorsoPlanner (orientation tracking
+    # unchanged). DS phase is unchanged (mapping still active there).
```

`default = False` at introduction.

### 1.4 `crawlbot/simulation/sim_loop.py` diff (three hunks)

**Hunk 1 — new `__init__` field (at line 97, today ~101):**

```diff
@@ -97,6 +97,10 @@ class SimulationLoop:
         self._step_q_end: Optional[np.ndarray] = None
         self._step_t_ss_start: float = 0.0
         self._step_T_step: float = 0.0
+        # M7 EE-bisection follow-up: torso linear position at SS entry
+        # (set in _setup_torso_for_step). Read by _step() when
+        # cfg.mapping_bypass_in_ss is True; otherwise unused.
+        self._ss_entry_p_torso: Optional[np.ndarray] = None
         # Simulation time at which the active coarse plan was anchored
         # (so r_com_at(t - t0) gives the right reference at current time).
         self._coarse_plan_t0: float = 0.0
```

**Hunk 2 — snapshot at SS setup (inside `_setup_torso_for_step`):**

```diff
@@ -926,6 +930,12 @@ class SimulationLoop:
         self._step_q_end = q_end.copy()
         self._step_t_ss_start = float(t_ss_start)
         self._step_T_step = float(T_step)
+        # Snapshot the torso linear position at SS entry — read by the
+        # mapping_bypass_in_ss diagnostic in _step() to freeze the SS
+        # linear torso reference. p_t0 is the torso pose computed from
+        # the live state above (line ~810), so it equals the actual
+        # torso position at the moment SS begins.
+        self._ss_entry_p_torso = p_t0.copy()

         return (q_end, T_step, True)
```

**Hunk 3 — new branch in the per-tick reference assembly (today's Branch A at `sim_loop.py:1734–1742`):**

```diff
@@ -1679,7 +1689,15 @@ class SimulationLoop:
             else:
                 tq_planner = tq
             tr = self.torso_planner.reference_at(tq_planner)
-            if phase in ('SS', 'DS') and self.mapping is not None and cfg.use_m2_stack:
+            if (phase == 'SS' and cfg.mapping_bypass_in_ss
+                    and self._ss_entry_p_torso is not None):
+                # Diagnostic bypass: freeze the linear torso reference at
+                # its SS-entry value; angular reference still from
+                # TorsoPlanner. Mapping is not called this tick.
+                p_torso_ref_used = self._ss_entry_p_torso.copy()
+                v_torso_ref_used = np.concatenate([np.zeros(3), tr.v[3:6]])
+                a_torso_ff_used = np.concatenate([np.zeros(3), tr.a[3:6]])
+            elif phase in ('SS', 'DS') and self.mapping is not None and cfg.use_m2_stack:
                 af_for_mapping = np.zeros(3) if self._diag_pure_pd else af
                 if phase == 'SS':
                     q_map, dq_map = self._planned_arm_config(tq, rs)
```

The new `if` branch has **higher priority** than the pre-existing
mapping branch — when the flag is True, the mapping path is
skipped entirely in SS. The `elif` ensures DS behaviour is
unchanged (mapping remains active for DS even when the flag is
set).

### 1.5 Labels in the commit and docstring

- Introduced as `M7 EE-bisection follow-up`.
- Called a `diagnostic bypass` in the sim_loop.py hunk-3 inline
  comment.
- Default `False` at introduction, i.e. off-by-default
  instrumentation (not a committed closure).
- A companion runner script `Misc/scripts/run_m7_v21_mapping_off.py`
  was added in the same commit to exercise it.

*(§2 Rationale follows.)*

---

## §2 Rationale — commit messages, docstrings, memos

### 2.1 Introducing-commit message (`61dcfca`, 2026-04-20)

> "SimulationLoop._step bypasses the CoM->torso mapping during SS
> only: QP receives p_torso_ref frozen at the SS-entry torso
> position (captured in _setup_torso_for_step as
> _ss_entry_p_torso) with v_torso_ref_lin = 0 and a_torso_ff_lin
> = 0. Angular reference still comes from TorsoPlanner. DS phase
> is unchanged."

The commit itself describes the behaviour but not the motivation.
For the motivation, the commit labels the flag `M7 EE-bisection
follow-up` (in both the config.py docstring comment and the
`__init__` field comment) — pointing at `M7_EE_POSITION_BISECTION.md`
and the technical-log entries below.

### 2.2 Technical log — `Misc/reports/architecture/M7_TECHNICAL_LOG.md` §15 "Mapping Bypass in SS — Position Tracking Resolved (2026-04-20)"

Directly addresses the motivation. Relevant excerpts:

> "The EE position bisection (§M7_EE_POSITION_BISECTION.md)
> identified the CoM→Torso mapping as producing the bulk of the
> position inflation when comparing standalone QP (24 mm EE peak)
> to full closed-loop (162 mm EE peak). The chain:"

| case | description | EE pos peak SS [mm] | Δ from previous |
|---|---|---|---|
| A_swing | standalone, SwingPlanner EE | 3.82 | — |
| B_minus | + NMPC, torso constant | 4.59 | +0.77 |
| B_v21 | + mapping (planned-δ) | 164.79 | **+160.20** |
| D | full sim_loop SS | 162.38 | −2.41 |

> "Adding NMPC alone contributes under 1 mm of additional
> inflation; adding the mapping (planned-δ, v21 configuration)
> adds 160 mm. The mapping was providing a moving torso reference
> that consumed QP torque budget and prevented the EE task from
> tracking its reference."

Fix description:

> "`SimConfig.mapping_bypass_in_ss = True` causes the SS-phase
> torso reference to hold at its SS-entry pose:
> ```python
> if cfg.mapping_bypass_in_ss:
>     r_b_ref = self._ss_entry_p_torso      # frozen at SS start
>     v_b_ref = np.zeros(3)
>     a_b_ff = np.zeros(3)
> else:
>     r_b_ref, v_b_ref, a_b_ff = self.mapping.compute(...)
> ```
> Angular reference still comes from TorsoPlanner (unchanged)."

Verification:

> "Closed-loop v21 with mapping bypass: EE position peak 32 mm,
> closest approach at abort 8 mm. A 5× reduction vs mapping
> active (162 mm / 41 mm)."

### 2.3 T11 closure summary — `M7_TECHNICAL_LOG.md` §17 "T11 Closed (2026-04-20)"

Lists `mapping_bypass_in_ss = True` as one of four independent
architectural changes that closed T11:

> "1. Pinocchio armature — closed 16° of EE orientation drift (§13)
>  2. MJCF damping → 0, armature alone retained — theorem-consistent
>     physics on both sides (§14)
>  3. Mapping bypass in SS — closed 130 mm of EE position inflation (§15)
>  4. Swing early-finish — ensured dock activates after reference completes (§16)
>  The controller logic was not changed during this session. All
>  four findings are model-consistency or reference-generation
>  fixes at the simulation-controller boundary."

### 2.4 Bisection memo — `Misc/reports/architecture/M7_EE_POSITION_BISECTION.md`

The bisection that motivated the flag. Scope (L4):

> "Decompose the 6.7× EE position inflation in SS across three
> orthogonal contributions."

Protocol (L35–65) defines three cases: `A_swing` (standalone
SwingPlanner), `B_minus` (+ NMPC, **torso ref constant**),
`B_v21` (+ mapping with planned-δ). The "torso ref constant" in
`B_minus` is described at L58 as:

> "**Torso reference remains constant** (bypass mapping).
>  Isolates NMPC's direct-to-QP contribution."

Expected outcome (L122):

> "**P3:** `B_v21` is the dominant step, > 80 mm added."

i.e. the bisection was designed to test whether the mapping is
the dominant source. The `mapping_bypass_in_ss` flag was
introduced to make B_minus reproducible in the production
sim_loop and, having confirmed the bisection prediction, was
then kept on as the T11 closure.

### 2.5 T12 memo — `Misc/reports/architecture/M7_T12_MEMO.md` §3.4 "Role of `mapping_bypass_in_ss`"

Retrospective assessment from the T12 investigation:

> "The flag was installed in T11 to close an EE-position inflation
> from the SS-phase mapping loop. It works for its intended
> purpose (SS metrics at 14% are clean). The side effect,
> unobserved at 1% because the downstream consequences were mild,
> is that the SS-frozen position reference diverges from the
> mapping output over the full SS duration, producing the large
> position-reference step at SS→DS. At 14% this amplifies into
> the DS1 divergence above.
>
> The flag was a closure on a T11 problem that produced a new
> problem at T12 through a different mechanism. Removing the flag
> (Option B in §5) re-opens the T11 problem; keeping it requires
> addressing the SS→DS discontinuity differently."

And the explicit code behaviour under the flag (M7_T12_MEMO
§Q3, L99–109):

> "Per `sim_loop.py:1697–1725`, under `mapping_bypass_in_ss = True`:
>   - SS: `p_ref = self._ss_entry_p_torso` (frozen at SS-entry pose).
>   - DS: `p_ref = self.mapping.compute(q_current)` (live mapping at post-impact q).
>  `R_ref = torso_planner.reference_at(t)` in both phases. The
>  position reference discontinuously jumps at weld; the
>  orientation reference does not. The position step magnitude
>  equals the full planned torso excursion accrued in SS
>  (~591 mm on the 14% case)…"

The T12 memo identifies this SS→DS position-reference jump as
the trigger of the T12 DS1 divergence, and proposes three options
(§5 of that memo) — one of which (Option B) is to remove the
flag. Option B was **rejected** ("Re-opens the T11 EE-position
inflation issue that the flag was introduced to close. Rejected
unless T11 mechanism can be closed differently first.").

### 2.6 `config.py` docstring (current)

The docstring installed by the introducing commit, unchanged to
date (lines appended to `SimConfig`):

```
mapping_bypass_in_ss: bool = False
# M7 EE bisection follow-up. When True, sim_loop bypasses the
# CoM->torso mapping during SS only: the QP receives
#   r_torso_ref = r_torso(t = t_ss_start)   (frozen at SS entry)
#   v_torso_ref_lin = 0
#   a_torso_ff_lin  = 0
# for the linear components of the torso reference. Angular
# reference still comes from TorsoPlanner (orientation tracking
# unchanged). DS phase is unchanged (mapping still active there).
```

### 2.7 `sim_loop.py` inline comment (hunk 3 in §1.4)

```
# Diagnostic bypass: freeze the linear torso reference at
# its SS-entry value; angular reference still from
# TorsoPlanner. Mapping is not called this tick.
```

Uses the word "**diagnostic**" to describe the branch.

### 2.8 Compact rationale

| Source | Rationale |
|---|---|
| Introducing commit (`61dcfca`) | Adds a flag (default False) that bypasses CoMToTorsoMapping in SS; angular reference unchanged; DS unchanged. Presented as "M7 EE-bisection follow-up". |
| `M7_TECHNICAL_LOG.md` §15 | The CoM→Torso mapping was responsible for **~160 mm of the 162 mm SS EE-position inflation** (v21 closed-loop 162 mm vs standalone 24 mm). Freezing the torso linear reference at SS-entry reduced EE peak from 162 mm to 32 mm (5× reduction). |
| `M7_TECHNICAL_LOG.md` §17 | One of four independent changes that closed T11's first successful single-step dock. |
| `M7_EE_POSITION_BISECTION.md` | Experimental protocol that isolated the mapping as the dominant SS EE inflation source; the flag was introduced to make the `B_minus` configuration reproducible in production sim_loop. |
| `M7_T12_MEMO.md` §3.4 | Acknowledges the flag is a **closure that produced a new problem**: the frozen-in-SS / live-in-DS asymmetry creates a discontinuous SS→DS position-reference step equal to the full planned torso excursion (~591 mm at 14%). At 14% this triggered the DS1 `|h_w|` saturation and attitude divergence. Removing the flag would re-open T11's EE inflation; the memo **rejects** removal. |

*(§3 T11 vs T15 usage follows.)*

---

## §3 T11 vs T15 usage

### 3.1 Reproducer scripts

| Validation | Script | `mapping_bypass_in_ss` set in script? |
|---|---|---|
| **Introducing commit companion** | `Misc/scripts/run_m7_v21_mapping_off.py` (`sim_loop.py` at 61dcfca) | `cfg.mapping_bypass_in_ss = True` |
| **T11** (1 %, 1-step) | `Misc/scripts/run_m7_v22_with_swing_hold.py` | `cfg.mapping_bypass_in_ss = True` (line 55) |
| **T12** (14 %, 1-step) | `Misc/scripts/run_m7_v22_14pct_with_swing_hold.py` | `cfg.mapping_bypass_in_ss = True` (line 76) |
| **T15 original** (1 %, 3-step) | `Misc/scripts/run_m7_v22_1pct_3step_t15.py` | `cfg.mapping_bypass_in_ss = True` (line 79) |
| **T15 bug1fix_vel** (Phase 3 rerun) | same script (OUT repointed) | `cfg.mapping_bypass_in_ss = True` |

Every M7 v22 validation run in the T11→T15 series has used
`mapping_bypass_in_ss = True`. The flag default in `SimConfig`
is still `False`.

### 3.2 Full config-echo comparison

The T11, T12, and T15 reproducer scripts echo identical SS-side
configuration. T15 adds one T12-closure flag; nothing else
differs.

| Parameter | T11 script | T12 script | T15 script |
|---|---|---|---|
| `preplanner_a_cruise_max` | `0.01` | `0.01` | `0.01` |
| `preplanner_cruise_ramp_frac` | `0.2` | `0.2` | `0.2` |
| `mapping_bypass_in_ss` | `True` | `True` | **`True`** |
| `swing_early_finish_fraction` | `0.80` | `0.80` | `0.80` |
| `aocs_off_in_ds` | not set (default `False`) | not set (default `False`) | **`True`** (explicit) |
| `ds_ramp_duration_s` | not set (default `2.0`) | not set (default `2.0`) | not set (default `2.0`; documented in script header) |
| MJCF `damping` mutation | `0.0` | `0.0` | `0.0` |
| MJCF `armature` mutation | `0.05` | `0.05` | `0.05` |
| mass_ratio | `0.01` | `0.14` (via MJCF structure-block mutation) | `0.01` |
| `n_steps` | `1` | `1` | `3` |

Relative to T11, T15 introduces:

1. `aocs_off_in_ds = True` — the T12 closure (documented in
   `M7_T12_MEMO.md`). Orthogonal to `mapping_bypass_in_ss`; it
   gates the AOCS Mode B during DS, not the torso-reference path.
2. `n_steps = 3` — the multi-step exercise. Orthogonal to the
   flag.
3. (Applied at branch HEAD in T15-fix-1 Phase 2) — `_t_plan_offset`
   reset at SS entry (Option Z, commit `7c8f01a`). Also orthogonal
   to `mapping_bypass_in_ss` — Option Z only affects the swing
   planner's time axis alignment, not the torso reference.

### 3.3 Configuration consistency across the series

T11, T12, T15: same `mapping_bypass_in_ss = True`.
No parametric change to the flag across the three validations.
T12's DS1 divergence analysis (documented in `M7_T12_MEMO.md`)
identified the flag as the trigger of the 14 % DS failure but
explicitly did not toggle it — the T12 closure went through the
separate `aocs_off_in_ds` route instead (see §5 of this report).

### 3.4 Tests that keep the flag at its default `False`

The flag's default is `False`, and `61dcfca`'s commit message
notes "192/192 tests still pass with the flag at its default
False". Any test that instantiates `SimConfig()` without
overriding `mapping_bypass_in_ss` — i.e. the unit-test suite —
exercises the `elif` branch of hunk 3, i.e. the original mapping
path. Production validation scripts (T11 / T12 / T15) all set
`= True`.

*(§4 Impact of toggling follows.)*

---

## §4 Impact of toggling — code-inspection only

Hypothetical: `mapping_bypass_in_ss = False` set, all other T15
configuration kept (`aocs_off_in_ds = True`,
`swing_early_finish_fraction = 0.80`, `use_m2_stack = True`,
`mapping_bypass_in_ss = False`). No simulation run; branch
analysis only.

### 4.1 Which branches activate vs deactivate

At `sim_loop.py:1734–1777` (three branches; §2 of
`T15_post5_pipeline_audit.md`):

```python
if (phase == 'SS' and cfg.mapping_bypass_in_ss             # Branch A
        and self._ss_entry_p_torso is not None):
    p_torso_ref_used = self._ss_entry_p_torso.copy()
    v_torso_ref_used = np.concatenate([np.zeros(3), tr.v[3:6]])
    a_torso_ff_used  = np.concatenate([np.zeros(3), tr.a[3:6]])
elif phase in ('SS', 'DS') and self.mapping is not None    # Branch B
        and cfg.use_m2_stack:
    af_for_mapping = np.zeros(3) if self._diag_pure_pd else af
    if phase == 'SS':
        q_map, dq_map = self._planned_arm_config(tq, rs)
    else:
        q_map, dq_map = rs.q, rs.v
    r_b_ref_m, v_b_ref_m, a_b_ff_m, _ = self.mapping.compute(
        r_com_ref=rp_interp, v_com_ref=vp_interp,
        a_com_ff=af_for_mapping, q_current=q_map, dq_current=dq_map)
    # (post-dock DS blend logic follows here — DS only)
    p_torso_ref_used = r_b_ref_m
    v_torso_ref_used = np.concatenate([v_b_ref_m, tr.v[3:6]])
    a_torso_ff_used  = np.concatenate([a_b_ff_m, tr.a[3:6]])
else:                                                       # Branch C
    p_torso_ref_used = tr.p
    v_torso_ref_used = tr.v
    a_torso_ff_used  = tr.a
```

With `mapping_bypass_in_ss = False` and other T15 config:

| Phase | Branch that activates (T15 default) | Branch that activates (flag toggled to False) |
|---|---|---|
| SS | **A** (frozen) | **B** (live mapping, planned-δ) |
| DS | B (live mapping, live-q + post-dock blend) | B (**unchanged** — the flag only affects SS) |

The SS-only behaviour change is: **replace the frozen SS-entry
linear torso reference with a `mapping.compute(...)` call every
QP sub-step (100 Hz)**, fed by the same `rp_interp, vp_interp`
NMPC interpolants that DS already consumes.

### 4.2 What `mapping.compute` produces in SS with the toggle flipped

Per `crawlbot/core/com_to_torso_mapping.py:133–170`, the closed-form
map is:

```
r_b_ref = (m_total / m_b) · r_com_ref − δ(q_current) / m_b
v_b_ref = (m_total / m_b) · v_com_ref − δ̇(q_current, dq_current) / m_b
a_b_ff  = (m_total / m_b) · a_com_ff     # δ̈ dropped, see 4.3
```

with (`com_to_torso_mapping.py:97–132`):

```
δ(q)      = Σ_{i ≠ torso} m_i · r_i(q)                 # (3,) world-frame sum
δ̇(q, q̇) = Σ_{i ≠ torso} m_i · J_i_translational(q) · q̇  # via LOCAL_WORLD_ALIGNED frame Jacobians
```

Inputs in SS with the flag off:

- `r_com_ref = rp_interp` — NMPC `x_opt[0:3, 1]` interpolated
  between knot 0 and knot 1 over the 10 sub-steps of the NMPC
  tick (`sim_loop.py:1708–1709`).
- `v_com_ref = vp_interp` — same for velocity.
- `a_com_ff = af` — `nmpc.compute_feedforward_acceleration(lr)`
  (`sim_loop.py:1621`).
- `q_current = q_map` — `_planned_arm_config(tq, rs)`
  (`sim_loop.py:759`): live floating-base `(q, v)` from
  `rs.q, rs.v`, but arm-joint portion **quintic-interpolated**
  between `_step_q_start` (live at SS entry) and `_step_q_end`
  (IK end-pose) over `[t_ss_start, t_ss_start + T_step]`.
- `dq_current = dq_map` — matching `ṡ(τ) · Δq_arm` quintic
  derivative.

### 4.3 Known code-level caveats (from module docstrings and comments)

`crawlbot/core/com_to_torso_mapping.py:1–25` (module header):

> "`a_b_ff = (m_total/m_b) * a_com_ff` [drop delta_ddot] … δ̈
> is dropped at v1 (PD handles it); see
> docs/architecture/STATUS.md §7 (cascade bisection 2026-04-16)."

Same caveat in-line at `com_to_torso_mapping.py:168`:

> "delta_ddot dropped at v1; PD handles the residual accel."

So the mapping's feedforward acceleration is incomplete by design:
the `(1/m_b) · δ̈(q, q̇, q̈)` term is omitted, and the QP's
torso-PD (`cfg.Kp_torso · e_pos + cfg.Kd_torso · e_vel`,
`wholebody_qp.py:540–557`) is expected to absorb the residual. No
other warning is present in the module's docstrings.

### 4.4 Historical evidence on planned-δ vs live-δ behaviour

From `Misc/reports/architecture/M7_TECHNICAL_LOG.md` §2 (EE Position
Inflation bisection):

> "**Problem with live δ(q_current)**: The arm's tracking error
> creates fluctuations in δ(q) that get amplified (×1.78) into
> torso reference noise. The EE tracks imperfectly → δ changes
> → torso reference jitters → QP fights jitter → EE tracking
> degrades. A feedback loop."

> "**Current best (v19): Mapping with planned δ(q_planned).**
> Uses swing planner's planned arm configuration instead of
> actual. Produces smooth feedforward torso reference that
> anticipates arm motion. Torso tracks to 42 mm (vs 22 mm with
> live δ, 121 mm with no mapping)."

At v19 the SS mapping used planned-δ (current `_planned_arm_config`
quintic — the same code path the toggle-off branch would use at
T15 HEAD). v19 reported 42 mm SS torso-tracking error at 1 %. The
standalone QP floor at that time was 22 mm (live-δ) / 24 mm
(v21 standalone test, per §15 bisection).

From `M7_TECHNICAL_LOG.md` §8:

> "The standalone test with v19 config proved this: the
> mapping-based moving torso reference saturates joints at 20 Nm,
> leaving zero torque for the EE task. The quintic demands ~13 Nm
> on the peak stance joint just for the torso (40 kg × 0.67 m/s²
> peak accel × 0.5 m moment arm), leaving 7 Nm for everything
> else. The EE needs 5–7 Nm. Budget: maxed."

**Correction of an earlier phrasing in this report.** An earlier
draft of this paragraph stated the observation was "from before
the v22 trajectory-shaping fixes". That is partially wrong.
`M7_EE_POSITION_BISECTION.md` L30 states the bisection cases
"All use v21 config (`preplanner_a_cruise_max=0.01`, planned-δ,
…)". The v21 preplanner cruise-box settings are the same ones
T11, T12, and T15 carry forward to v22. So the 164.79 mm mapping
contribution in §2.2 and the 20 Nm torque-budget observation are
measurements **with** the cruise-box shaping already active, not
from before it. What they are "before" is the later flag
introductions (`swing_early_finish_fraction = 0.80` from
2026-04-21, `aocs_off_in_ds = True` from 2026-04-22, and the
Option Z `_t_plan_offset` reset from 2026-04-22). See §7 for
the full source-freshness audit.

Nothing in the code as shipped prevents `mapping.compute` from
producing a valid reference — the limitation that was documented
is a **torque-budget** limitation at the QP level, not a
mapping-side divergence.

### 4.5 What CoMToTorsoMapping has no documented safeguard against

- **δ(q) singularities.** `δ(q)` is a sum of per-body positions
  reached via forward kinematics (`com_to_torso_mapping.py:97–108`).
  FK does not have singularities; `δ(q)` is always well-defined.
  No FK-related caveat.
- **δ̇(q, q̇) rank deficiency.** `δ̇ = Σ m_i · J_i · q̇`
  (`com_to_torso_mapping.py:110–131`). The per-body Jacobians
  `J_i` depend on q. No docstring flags singular configurations
  of the summed Jacobian; the map passes `q̇` through
  geometrically without pseudo-inverse.
- **Stability.** The map is algebraic and open-loop — no
  recursion, no integrator, no time-lagged state. It cannot
  "diverge" on its own; any problematic behaviour has to come
  from its inputs (NMPC `r_com_ref`, planned-δ `q_map`) or its
  downstream consumer (QP PD).

### 4.6 Interaction with current T15 configuration

With `mapping_bypass_in_ss = False` under T15 config (at branch
HEAD = `b364e0b`, after Option Z fix):

- **T11 EE-inflation mechanism re-exposed.** The bisection
  numbers (§2.2) were measured at pre-T11 config; any re-run
  would face the same mapping-induced torso-reference evolution
  that added ~160 mm to SS EE peak at 1 %.
- **T12 SS→DS discontinuity closes.** Per `M7_T12_MEMO.md` §3.4,
  the SS→DS position step under flag=True was the full planned
  torso excursion. With flag=False, SS and DS both use
  `mapping.compute`; the SS-exit and DS-entry references differ
  only by the small change due to contact-set change — no
  discontinuity.
- **AOCS route-around is orthogonal.** `aocs_off_in_ds = True`
  disables AOCS Mode B in DS. Flipping the mapping flag does
  not touch AOCS behaviour.
- **Option Z (`_t_plan_offset` reset at SS entry) is orthogonal.**
  Option Z affects the swing-planner time axis (§3 of
  `T15_fix1_diff.md`); the mapping flag affects the torso
  reference only. Both flags co-exist independently.

### 4.7 Documented technical obstacles to setting the flag False

From the memos and docstrings surveyed:

| Source | Claimed obstacle |
|---|---|
| `M7_TECHNICAL_LOG.md` §2 | "feedback loop" with live-δ — mitigated by using planned-δ (already the SS-configuration `_planned_arm_config` uses when the flag-off branch runs). |
| `M7_TECHNICAL_LOG.md` §8 | Torque-budget saturation at 20 Nm with planned-δ in v19 regime (before v22 trajectory shaping). Status: observation under older config, not a statement about the current v22 config with `preplanner_a_cruise_max = 0.01` + `preplanner_cruise_ramp_frac = 0.2`. |
| `M7_TECHNICAL_LOG.md` §15 | Bisection found mapping adds ~160 mm of SS EE inflation. Status: direct motivation for introducing the flag. |
| `M7_T12_MEMO.md` §3.4 / §5 Option B | "Re-opens the T11 EE-position inflation issue that the flag was introduced to close. Rejected unless T11 mechanism can be closed differently first." — rejection is explicit but **conditional** on not having a different T11-closure mechanism. |
| `com_to_torso_mapping.py` docstring | "δ̈ dropped at v1 (PD handles it)" — acknowledges the FF incompleteness; not an obstacle to the mapping itself, but a reminder that the torso-PD is carrying the residual. |

**No code-level assertion or docstring flags the map as unsafe
or unstable to call in SS.** The documented obstacles are all
empirical (measurements from pre-T11/T12 experiments), not
structural. The flag was introduced as a **diagnostic bypass**
(its own inline comment, §1.4 hunk 3), later retained as a T11
closure, and its removal is conditional on finding an alternative
T11-closure mechanism per the T12 memo's explicit text.

*(§5 Related fixes follows.)*

---

## §5 Related fixes in the same timeframe

### 5.1 Timeline of `SimConfig` changes near the flag's introduction

From `git log --oneline -- crawlbot/simulation/config.py`:

| Commit | Date | Subject | Fields added / changed |
|---|---|---|---|
| `6128db9` | 2026-04-19 | M7 post-abort DS diagnostic decomposition — four orthogonal runs | diag_* flags |
| **`61dcfca`** | **2026-04-20** | **M7: add mapping_bypass_in_ss flag + v21 mapping-off run** | **`mapping_bypass_in_ss: bool = False`** |
| `f5dea6d` | 2026-04-21 | feat(sim): T11 closed — clean dock via swing_early_finish_fraction=0.80 | `swing_early_finish_fraction` |
| `5b17051` | 2026-04-22 | T12-fix Option A: smooth SS->DS torso position-reference release | `ds_ramp_duration_s` |
| `bcd3d7c` | 2026-04-22 | T12: close DS1 divergence at 14% mass ratio | `aocs_off_in_ds`, `ds_ramp_duration_s` (field reused by Option A bundle), NMPC logging fields |

Five consecutive-day commits modifying `config.py` — four of them
directly related to the torso-reference / SS→DS path (all except
`6128db9` which is diagnostic-only). `61dcfca` sits at the start
of the cluster; the next three build on the T11 closure it
enabled.

### 5.2 `swing_early_finish_fraction` (commit `f5dea6d`, 2026-04-21)

Closes T11 in combination with `mapping_bypass_in_ss`. Commit
message (truncated):

> "First closed T11 dock in the project. Dock at t=6.01s: d=2.70mm,
> ori=0.06°, ||v_rel_lin||=13.6mm/s receding, ||v_rel_ang||=4.6mrad/s.
> Post-dock residual absorbed by DS passivity QP per the theorem's
> settle guarantee.
>
> Combined effect of three independent fixes from this session:
> - Pinocchio armature install (commit 63a072f) — wrist q̈ modeling
>   correct, EE orientation drift closed (16.7° → 0.88°).
> - MJCF damping=0 (Part 2 sweep: damping not load-bearing, armature
>   alone stabilizes discrete-time settle).
> - mapping_bypass_in_ss=True — mapping's moving torso reference was
>   burning torque budget the QP needed for EE tracking.
> - swing_early_finish_fraction=0.80 — quintic completes at 0.80·T_step;
>   dock gate requires swing complete before firing. Prevents
>   activation during terminal deceleration ramp."

**Relation to the mapping flag.** Listed together as one of the
"three independent fixes" (the commit message counts four bullets
but the armature + damping fixes are bundled as one at the fix
level) that closed T11. The commit message does not claim they
interact in any way, and §17 of `M7_TECHNICAL_LOG.md` affirms
they are "independent architectural changes".

### 5.3 `ds_ramp_duration_s` / Option A (commit `5b17051`, 2026-04-22)

Per commit message:

> "Per Misc/reports/architecture/M7_T12_MEMO.md §5. Blends the DS torso
> linear position reference from the frozen SS-exit pose
> (_ss_entry_p_torso) to the live mapping.compute output over
> cfg.ds_ramp_duration_s (default 2.0 s), using the quintic
> s(τ) = 10 τ³ − 15 τ⁴ + 6 τ⁵. SS behavior, orientation reference,
> and velocity/acceleration feedforward are unchanged. Ramp state
> is captured at weld activation and cleared at SS entry."

**Relation to the mapping flag.** Option A is a *direct
descendant* of the `mapping_bypass_in_ss=True` closure. It exists
specifically because of the SS→DS position-reference step that
the flag creates (identified as the T12 DS1 divergence trigger,
per `M7_T12_MEMO.md` §3.4, also quoted in §2.5 of this report).
Option A does **not** touch SS; it only smooths the transition
from the (frozen) SS-exit pose to the (live-mapping) DS reference
over 2.0 s. The code changes in `sim_loop.py` are localized to
the DS-branch of the reference-assembly block
(`sim_loop.py:1755–1770` per §2.2 of
`T15_post5_pipeline_audit.md`).

### 5.4 `aocs_off_in_ds` (commit `bcd3d7c`, 2026-04-22)

Per commit message:

> "Mechanism: AOCS Mode B (legacy_corrected) produced spurious
> wheel torque during DS phases due to missing frame-rotation
> transport term in the feedforward. Wheel reaction rotated the
> structure body via Newton's third law; gain scaled as
> 1/I_struct, producing mass-ratio-dependent failure.
>
> Changes:
> - config: aocs_off_in_ds flag (default False); ds_ramp_duration_s
>   field for post-dock reference ramp (Option A).
> - sim_loop: phase-aware AOCS dispatcher gate (AOCS off during DS
>   when flag set); post-dock reference-ramp state machine and
>   quintic blend; NMPC status-string and iteration-count logging.
> …
> Validation: T12 single-step at 14% with aocs_off_in_ds=True
> passes DS1 closure criteria (torso_ori end-DS1 = 2.25 deg vs
> 9.92 deg baseline; |h_w| end-DS1 = 0.22 Nm·s vs 6.28; zero NMPC
> failures). SS metrics bit-identical to prior runs. Structure
> rotation matches ballistic prediction within 1%."

**Relation to the mapping flag.** Bundled into the same T12
closure commit as `ds_ramp_duration_s` but **semantically
orthogonal** to the mapping flag. `aocs_off_in_ds` addresses a
separate Mode-B Jacobian-frame-rotation bug in the AOCS path
(documented in `M7_T12_MEMO.md` and `AOCS_CONCERN_MEMO.md`). It
does not touch the torso reference. Both flags contribute to T12
closure but through different mechanisms: Option A (smooth
reference release) + AOCS-off addresses the two identified
triggers of the DS1 divergence.

### 5.5 Sibling flags introduced across the T11 / T12 cluster

Cumulative config summary across the T11–T12 fix cluster (all
pre-existing to T15):

| Flag / field | Default | T11 script | T12 script | T15 script | Rationale (per commits) |
|---|---|---|---|---|---|
| `mapping_bypass_in_ss` | `False` | `True` | `True` | `True` | T11 — SS EE-position inflation closure |
| `swing_early_finish_fraction` | `1.0` | `0.80` | `0.80` | `0.80` | T11 — prevent weld activation during swing deceleration |
| `ds_ramp_duration_s` | `2.0` | default used | default used | default used | T12 Option A — smooth SS→DS torso-ref release |
| `aocs_off_in_ds` | `False` | default used (`False`) | default used (`False`) | `True` | T12 — close DS1 structure-attitude divergence at 14 % |

T15 inherits the T11 fixes and the T12 closures without
re-examining any of them.

### 5.6 None of the related fixes supersede `mapping_bypass_in_ss`

Each related fix in the cluster addresses a separate failure
mechanism:

- `swing_early_finish_fraction`: fixes a weld-activation timing
  issue (`post-swing terminal deceleration → weld impulse`);
  unrelated to torso reference.
- `ds_ramp_duration_s` (Option A): fixes the **symptom** that
  `mapping_bypass_in_ss=True` creates in DS1 (the discontinuous
  reference step), not the underlying SS torso-reference
  problem. Explicitly labels itself as downstream of the mapping
  flag.
- `aocs_off_in_ds`: fixes an AOCS-side frame-rotation bug
  (`force_estimator.py:286`, Mode B); unrelated to torso
  reference.

None of these supersede or conflict with `mapping_bypass_in_ss`.
The T15 script's config echo (§3.2) inherits all three in their
T11/T12 configurations together with `mapping_bypass_in_ss=True`.

*(§6 Factual summary follows.)*

---

## §6 Factual summary

### 6.1 Motivation (facts from sources)

1. **Empirical trigger.** A pre-T11 bisection
   (`M7_EE_POSITION_BISECTION.md`, quantitative results recorded
   in `M7_TECHNICAL_LOG.md` §15) measured SS EE-peak error of
   **164.79 mm** with the CoMToTorsoMapping active vs **4.59 mm**
   with the torso linear reference held constant in SS — a 160 mm
   gap directly attributable to the mapping path.

2. **Code intent.** Commit `61dcfca` (2026-04-20, "M7: add
   mapping_bypass_in_ss flag + v21 mapping-off run") added the
   flag with `default = False`. Its own inline comment in
   `sim_loop.py` calls it a "**diagnostic bypass**". The commit
   message describes the mechanical behaviour (freeze linear
   torso reference at SS entry) without claiming a closure.

3. **Closure adoption.** The day after, commit `f5dea6d`
   (2026-04-21, T11 closure) listed `mapping_bypass_in_ss = True`
   as one of three independent fixes that together produced the
   first successful single-step dock (d = 2.70 mm, ori = 0.06°).
   `M7_TECHNICAL_LOG.md` §15 credits it with "closed 130 mm of
   EE position inflation".

4. **Retention under T15.** T11, T12, and T15 reproducer scripts
   all explicitly set `cfg.mapping_bypass_in_ss = True` (§3.1).
   `SimConfig` default remains `False`; all 192 unit tests run
   against the default.

### 6.2 Current behavioural effect (under T15 config, at branch HEAD)

With the flag on, during SS:

- **Linear torso reference** = `self._ss_entry_p_torso` (constant
  over the whole SS window). Set in `_setup_torso_for_step` from
  the live torso pose at SS-entry.
- **Linear torso velocity / FF acceleration** = `0`.
- **Angular torso reference, velocity, FF acceleration** —
  unchanged: still from TorsoPlanner's quintic interpolation
  between IK endpoints.
- **DS phase** — unaffected (still uses `CoMToTorsoMapping.compute`
  with the Option A quintic blend from the frozen SS-exit pose
  for the first `ds_ramp_duration_s = 2.0 s`).

### 6.3 Is there a documented technical obstacle to setting the flag False in SS?

Inventory of statements from the surveyed sources:

| Source | Statement | Type |
|---|---|---|
| `com_to_torso_mapping.py` docstring | "δ̈ is dropped at v1 (PD handles it)" | Design note on feedforward completeness; not an obstacle to calling the map |
| `com_to_torso_mapping.py` inline | No stability, singularity, or instability warnings | — |
| `sim_loop.py:1734` inline | "Diagnostic bypass" label | Labels the flag as diagnostic; does not forbid removal |
| `M7_TECHNICAL_LOG.md` §2 | "live δ(q_current) … feedback loop". Mitigated by planned-δ (already what `_planned_arm_config` provides to `mapping.compute` in the flag-off SS branch). | Resolved (by planned-δ) |
| `M7_TECHNICAL_LOG.md` §15 | Empirical: mapping adds ~160 mm SS EE inflation at v21 config | Empirical observation at v21 |
| `M7_TECHNICAL_LOG.md` §8 | v19 planned-δ mapping saturates 20 Nm torque budget → no room for EE tracking | Empirical observation at pre-v22 config (before `preplanner_a_cruise_max = 0.01` / `preplanner_cruise_ramp_frac = 0.2` were introduced) |
| `M7_T12_MEMO.md` §3.4 / §5 Option B | "Re-opens the T11 EE-position inflation issue that the flag was introduced to close. Rejected unless T11 mechanism can be closed differently first." | Explicit conditional rejection |
| `STATUS.md` §7 | Diagnostic protocol descriptions; no obstacle claim | — |

**Only one of the listed sources explicitly addresses removing
the flag**: `M7_T12_MEMO.md` §5 Option B. Its rejection is
**conditional**: the removal is not forbidden on structural
grounds; it is deferred pending an alternative closure of the
T11 EE-position inflation mechanism. No source in the inventory
claims the mapping cannot be run in SS, produces invalid output
in SS, or has a structural instability when called per-sub-step
in SS.

### 6.4 What would need to be true for the flag to be safely set False in SS (from the surveyed sources)

Per the T12 memo and technical log, two documented mechanisms
would be re-activated:

1. **SS EE-position inflation** via the mapping-induced moving
   torso reference consuming QP torque budget
   (`M7_TECHNICAL_LOG.md` §15 & §8). **Correction on the
   empirical baseline:** the 164.79 mm bisection figure in §2.2
   was measured with `preplanner_a_cruise_max = 0.01` and
   `preplanner_cruise_ramp_frac = 0.2` already active
   (`M7_EE_POSITION_BISECTION.md` L30). The v19 torque-saturation
   observation in §4.4 predates the cruise-box, but the 160 mm
   SS-EE-inflation measurement does not — it is a
   cruise-box-active figure.
   What was **not** present when the bisection ran (2026-04-20):
   `swing_early_finish_fraction = 0.80` (2026-04-21, `f5dea6d`),
   `aocs_off_in_ds = True` (2026-04-22, `bcd3d7c`), and the
   Option Z `_t_plan_offset` reset (2026-04-22, `7c8f01a`).
   Whether the 160 mm SS EE inflation still dominates under
   the full T15 config (Option Z + T11 + T12 closures combined)
   with flag=False has not been measured in the surveyed
   artefacts.

2. **Potentially-resolved side effects.** Removing the flag
   would also remove the SS→DS position-reference step that
   `M7_T12_MEMO.md` §3.4 identifies as the T12 DS1 trigger. If
   flag=False is set, Option A's DS-ramp blend (`ds_ramp_duration_s`)
   becomes unnecessary (SS and DS use the same mapping path —
   no discontinuity to smooth).

### 6.5 Scope note on Candidate 5 of `T15_post5_pipeline_audit.md` §8.4

`T15_post5_pipeline_audit.md` Candidate 5 posits
`CoMToTorsoMapping` as an architectural surface for manipulability-
awareness, noting that it is "currently bypassed in SS under T15
config". This report documents that:

- The bypass is intentional (T11 closure) with empirical
  motivation at v21 config.
- Removing the bypass is conditionally rejected by `M7_T12_MEMO.md`
  §5 Option B — conditional on an alternative T11-closure
  mechanism being demonstrated first.
- No source claims a **structural** (code-level or model-level)
  obstacle to setting the flag False in SS.
- The most recent v22 empirical measurement of the flag-off SS
  behaviour in the surveyed artefacts is from **pre-T11**
  bisection (§2.2 table). No flag-off run exists at T15 config
  (Option Z fix + `aocs_off_in_ds = True` + v22 trajectory
  shaping) in the surveyed artefacts.

Whether Candidate 5 is feasible in practice depends on whether
the T11 EE-inflation mechanism remains dominant under v22
trajectory-shaping parameters — a question the surveyed
documentation does not answer.

---

## §7 Source-freshness audit

Added in revision after the initial report drafting. The original
§§1–6 cited the archived memos and the technical log without
systematically flagging their ages relative to the flag's
introduction, the T11 closure, the T12 closure, and the T15 /
Option Z fix. This section catalogs each source, its age, what it
documents, and where in §§1–6 its age meaningfully affects
interpretation.

### 7.1 Source-age table

| Source | Created | Last touched | Pre-/post-flag (2026-04-20)? | Pre-/post-T11 (2026-04-21)? | Pre-/post-T12 (2026-04-22)? |
|---|---|---|---|---|---|
| `docs/architecture/STATUS.md` | 2026-04-11 | 2026-04-15 | **pre** | **pre** | **pre** |
| `Misc/reports/architecture/M7_EE_POSITION_BISECTION.md` | 2026-04-20 | 2026-04-20 | same-day | pre | pre |
| `Misc/reports/architecture/M7_TECHNICAL_LOG.md` | 2026-04-17 | 2026-04-21 | post | same-day | **pre** |
| `Misc/reports/architecture/M7_T12_MEMO.md` | 2026-04-22 | 2026-04-22 | post | post | post |
| `Misc/reports/architecture/AOCS_CONCERN_MEMO.md` | 2026-04-22 | 2026-04-22 | post | post | post |
| `crawlbot/core/com_to_torso_mapping.py` docstring | 2026-04-16 (mapping v1 commit `103bcc6`) | unchanged | pre | pre | pre |
| `sim_loop.py:1734` inline "diagnostic bypass" comment | 2026-04-20 (commit `61dcfca`) | unchanged | same-day | pre | pre |
| `crawlbot/simulation/config.py` docstring on the flag | 2026-04-20 (commit `61dcfca`) | unchanged | same-day | pre | pre |

### 7.2 Dates of the reference events

- **Flag introduced** (`mapping_bypass_in_ss`): 2026-04-20,
  commit `61dcfca`.
- **T11 closed** (flag used in closure): 2026-04-21, commit
  `f5dea6d`.
- **T12 closed** (flag retained; Option A + AOCS-off added):
  2026-04-22, commits `5b17051`, `bcd3d7c`.
- **T15 config adopted** (this report's target): 2026-04-22
  (original T15 script) + 2026-04-22 (Option Z fix commit
  `7c8f01a`).

### 7.3 Where source age changes interpretation

| Section | Source cited | Age concern |
|---|---|---|
| §2.4 | `M7_EE_POSITION_BISECTION.md` | Same-day-as-flag design memo. Protocol description only; not a retrospective. Correctly used as the documented motivation, but does not witness outcomes. |
| §2.2 | `M7_TECHNICAL_LOG.md` §15 | Written 2026-04-21 (T11 closure day); describes the bisection that was run the previous day. The 164.79 mm figure **is** at v21 cruise-box shaping — not "pre-v22 shaping". Post-T11 closure is *not yet reflected* (T11 closed that same day, but §15 doesn't evaluate whether T11's `swing_early_finish_fraction` or other post-v21 changes would change the 160 mm number). |
| §4.4 | `M7_TECHNICAL_LOG.md` §8 | Observations from v19 standalone tests (pre-v21 cruise-box). Distinct from the §15 bisection. An earlier draft of §4.4 conflated the two; the corrected paragraph (§4.4 ¶3 of this report) flags the discrepancy. |
| §2.5 | `M7_T12_MEMO.md` §3.4 / §5 | **Most recent retrospective on the flag.** Written post-T11, during T12 investigation. Its statements about the flag's side effect (SS→DS position-reference step) and its conditional rejection of removal are the freshest signal in the surveyed sources. |
| §4.3 | `com_to_torso_mapping.py` docstring | Module-level docstring from the v1 mapping commit (2026-04-16). The "δ̈ dropped at v1, PD handles it" note has not been revisited in later commits. Age: pre-flag-adoption, pre-T11. |
| §4.7 | `STATUS.md` §7 | Written 2026-04-15, pre-bisection, pre-flag. Used only for API-signature confirmation. Its "SimConfig state at end of session" is a snapshot of the codebase before the mapping flag existed. |
| §6.4 | `M7_TECHNICAL_LOG.md` §8 + §15 | The claim that the "documented obstacles are all empirical" uses both sources; §15 is cruise-box-active but pre-T11-post-closure fixes, §8 is pre-cruise-box. Neither source witnessed a flag-off run with all of {cruise-box, `swing_early_finish_fraction=0.80`, `aocs_off_in_ds=True`, Option Z} active. |

### 7.4 What is **not** in any surveyed source

- A run with **all T15-current flags** (`mapping_bypass_in_ss=False`,
  `swing_early_finish_fraction=0.80`, `aocs_off_in_ds=True`,
  Option Z `_t_plan_offset` reset) — i.e. flag-off under the
  current closure bundle.
- A re-measurement of the 160 mm SS EE inflation after
  `swing_early_finish_fraction=0.80` landed.
- Any post-T12 revisit of the `M7_TECHNICAL_LOG.md` §8 torque-
  budget observation.

Neither the T11 closure (`M7_TECHNICAL_LOG.md` §17) nor the T12
closure (`M7_T12_MEMO.md`) re-ran the bisection with their own
additions to see whether the 160 mm figure still holds.

### 7.5 Freshness summary for §6.3's obstacle inventory

Repeating §6.3's "is there a documented technical obstacle"
question with source-age annotations:

| Source | Age rel. T15 | Is it current enough to be load-bearing for a "flag is still needed" claim? |
|---|---|---|
| `com_to_torso_mapping.py` docstring | pre-flag | No (predates the architectural regime the flag addresses) |
| `sim_loop.py:1734` "diagnostic bypass" comment | same-day-as-flag | Does not evaluate outcomes; only labels |
| `M7_TECHNICAL_LOG.md` §2 (live-δ feedback loop) | pre-T11 | Resolved by planned-δ, which the flag-off branch already uses |
| `M7_TECHNICAL_LOG.md` §15 (160 mm inflation) | pre-T11-closure-side-fixes | Partial — bisection uses cruise-box but no `swing_early_finish_fraction` |
| `M7_TECHNICAL_LOG.md` §8 (20 Nm saturation) | pre-v21-cruise-box | Least load-bearing — observation is from before cruise-box + `swing_early_finish_fraction` |
| `M7_T12_MEMO.md` §3.4 / §5 Option B (conditional rejection) | post-T11, contemporaneous-with-T12 | **Most current.** Still does not include Option Z or measure v22 flag-off. |
| `STATUS.md` §7 | pre-flag | Used for API only |

### 7.6 Revised bottom line

The T12 memo's conditional rejection is the newest surveyed
signal, but even it does not test `mapping_bypass_in_ss=False`
with the full T15 closure bundle. The empirical obstacles cited
in §6.3 and §6.4 are from bisections and runs that **predate
at least one** of: `swing_early_finish_fraction=0.80`,
`aocs_off_in_ds=True`, Option Z `_t_plan_offset` reset. The
assertion "no structural obstacle" is unchanged by this
freshness audit; the assertion "empirical obstacles still hold
at v22 config" was not supported by fresh-enough data and should
not be read into this report.

---

*End of T15-post-6 report (revised with §7 source-freshness
audit).*
