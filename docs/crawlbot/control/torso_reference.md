# `crawlbot.control.torso_reference`

**File**: [`crawlbot/control/torso_reference.py`](../../../crawlbot/control/torso_reference.py) — **67 lines** — canonical coverage **77 %**

**The torso reference the whole-body QP tracks.** Extracted as is from
`WholeBodyController.track` (C2, branch `refactor/sim-loop-split`). It
*generates* a reference and commands nothing — in the ROS 2 layout it sits on the
reference side, next to the planners.

---

## 1. The reference

`TorsoReferenceShaper.shape(tr)` returns a `TorsoRef(p, R, v, a)` (structure
frame): the planner's reference (`tr.p`, `tr.R`, `tr.v`, `tr.a`, TorsoPlanner
quintic + SLERP) in SS and DS alike, then the diagnostic overrides — `freeze_ref`
holds the first sample of `p` and `R` with zero `v`, `a`; `pure_pd` zeroes `a`
(the controller zeroes λ_ref and a_com_ff).

## 2. Retired: the CoM→torso δ-mapping

The linear part used to be replaced by `r_b = (m/m_b)·r_com_ref − δ(q)/m_b`
(`CoMToTorsoMapping`, with F-RATE caching and a post-dock DS blend). The paper
uses none of it.

**R2a** retired its SS uses — the SS mapping bypass (`mapping_bypass_in_ss`), the
non-two-task SS path (`ss_two_task_mode=False`) and F-SAT, its SS-only rate
limiter (`fsat_jitter_margin`; its counters, all 0 on the canonical, stay
readable as constants for the driver's `sat_stats.txt`). Coverage-proved
unexecuted by the canonical and the Table 2 scenarios.

**R2b** retired the rest: the DS δ-mapping path, the post-dock DS blend
(`ds_ramp_duration_s`, Option A) with the `on_ss_entry` / `on_dock` events that
armed it, and the `crawlbot/core/com_to_torso_mapping.py` module (its test and
document with it). This path **did** run on the canonical — in DS — but was
inert: every DS tick the gait sequencer drives runs with `settle_mode=True`,
where the QP drops the linear torso task and keeps only the angular row.

Inertness probe (C2, 2026-09-29, worktree of `6c0bffc`, not committed): the DS
path's linear output (p, v[:3], a[:3]) replaced by NaN left the canonical replay
bit-identical to the host-local reference (453 193 floats, logs included) and all
26 gate scenarios identical. The removal itself confirms it: the canonical
bit-identical, and every control output of the 20 scenarios too. The only
telemetry it fed, `step_log.json` `delta_q` / `delta_q_current`, is written in
SS, where the mapping never ran: `None` on all 5080 canonical entries, and in the
`dwell` scenario a stale δ left over from the DWELL's last DS tick. The keys
stay, always `None` — the one difference R2b makes (see the R2 memo).

Revival path (an active DS with a tracked linear torso task, P3): the module is
at `4b988ec:crawlbot/core/com_to_torso_mapping.py`; see
`results/j2_adjconv/PHASE_R2_REFERENCE_PATHS_RETIRED.md`.

## 3. What C2 removed

Three stores that were written but never read: `_ds_ramp_p_end` (only ever set
to `None`), `_mapping_nmpc_tick`, and the local `R_b_ref_frozen` of the
`freeze_ref` branch.

Tests: `tests/test_torso_reference.py` (raw path, freeze_ref + pure_pd),
synthetic inputs.

## Public API

| symbol | signature | canonical? | code |
|---|---|---|---|
| **`TorsoRef`** *(dataclass)* |  |  | [L23](../../../crawlbot/control/torso_reference.py#L23) |
|   `p` | `` | _field_ | [L27](../../../crawlbot/control/torso_reference.py#L27) |
|   `R` | `` | _field_ | [L28](../../../crawlbot/control/torso_reference.py#L28) |
|   `v` | `` | _field_ | [L29](../../../crawlbot/control/torso_reference.py#L29) |
|   `a` | `` | _field_ | [L30](../../../crawlbot/control/torso_reference.py#L30) |
| **`TorsoReferenceShaper`** |  |  | [L33](../../../crawlbot/control/torso_reference.py#L33) |
| `.shape` | `(tr)` | **yes** | [L43](../../../crawlbot/control/torso_reference.py#L43) |

## Code map

| unit | source |
|---|---|
| `class TorsoRef` | [L23-30](../../../crawlbot/control/torso_reference.py#L23-L30) |
| `class TorsoReferenceShaper` | [L33-66](../../../crawlbot/control/torso_reference.py#L33-L66) |
| `TorsoReferenceShaper.shape` | [L43-66](../../../crawlbot/control/torso_reference.py#L43-L66) |

---

## See also

- the caller: [`controller.md`](controller.md)
- the planner reference: [`planning/torso_planner.md`](../planning/torso_planner.md)
