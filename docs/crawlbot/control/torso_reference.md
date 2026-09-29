# `crawlbot.control.torso_reference`

**File**: [`crawlbot/control/torso_reference.py`](../../../crawlbot/control/torso_reference.py) — **234 lines** — canonical coverage **78 %**

**The torso reference the whole-body QP tracks.** Extracted as is from
`WholeBodyController.track` (C2, branch `refactor/sim-loop-split`). It
*generates* a reference and commands nothing — in the ROS 2 layout it sits on the
reference side, next to the planners.

---

## 1. Paths

`TorsoReferenceShaper.shape(tr, rs, qs, tq, phase, rp_interp, vp_interp, af)`
returns a `TorsoRef(p, R, v, a)` (structure frame). In order of precedence:

| path | when | linear reference |
|---|---|---|
| mapping bypass | SS, `cfg.mapping_bypass_in_ss`, after an SS entry | frozen at the SS-entry torso position |
| δ-mapping | SS or DS, `use_m2_stack`, **not** two-task SS | `r_b = (m/m_b)·r_com_ref − δ(q)/m_b` |
| raw planner | two-task SS (the canonical SS) | `tr.p`, `tr.v`, `tr.a` |

The angular part is always the planner's (`tr.R`, `tr.v[3:]`, `tr.a[3:]`).
Inside the δ-mapping path:

- **F-RATE** — δ(q), δ̇(q, q̇) are recomputed once per NMPC tick (`qs == 0`) and
  cached across the QP sub-steps, which stops the q → δ → r_b → q loop at 100 Hz.
- **F-SAT** (SS only) — the per-tick increment of r_b is capped at
  `(|v_b_ref| + cfg.fsat_jitter_margin)·dt`; counters `_sat_*` are read by the
  canonical driver after the run.
- **Post-dock DS blend** (Option A, DS only) — quintic blend over
  `cfg.ds_ramp_duration_s` from the SS-entry torso position, armed by `on_dock`,
  cleared by `on_ss_entry`.

Then the diagnostic overrides: `freeze_ref` holds the first sample of `p` and `R`
with zero `v`, `a`; `pure_pd` zeroes `a` (the controller zeroes λ_ref and a_com_ff).

## 2. What the canonical actually uses — measured

On the canonical, SS is two-task (raw planner path) and every NMPC-tracked DS
tick (DWELL, trailing DS) runs with `settle_mode=True`, where the QP drops the
linear torso task and keeps only the angular row (`wholebody_qp.py`, two-task
switch and centroidal-DS block). The δ-mapping path therefore runs but should
have no effect.

**Inertness probe** (2026-09-29, worktree of `6c0bffc`, not committed): under
`settle_mode`, the δ-mapping path's LINEAR output (p, v[:3], a[:3]) was replaced
by NaN. Result:

- canonical replay: **bit-identical** to the host-local reference — every file,
  logs included (453 193 floats); no NaN reaches the QP or the telemetry;
- all 26 gate scenarios (`gate/scenarios.py`, vs the `36c44bf` baseline):
  **identical** — including `abortdiag`, which runs DS **without** centroidal
  mode: under `settle_mode` the QP ignores the linear torso reference either
  way.

So on the canonical this block is inert. Whether to retire it (CLEANUP-30 style)
or keep it for an active-DS future (NMPC in DS, P3) is a decision for Idriss; the
extraction kept it as is. Its DS path is inert in every run the gait sequencer
drives (all its DS ticks are `settle_mode=True`); what remains live is the SS
path with `ss_two_task_mode=False` (scenario `legacy_stack`, F-SAT included)
and external `_step(settle_mode=False)` callers.

## 3. What C2 removed

Three stores that were written but never read: `_ds_ramp_p_end` (only ever set
to `None`), `_mapping_nmpc_tick`, and the local `R_b_ref_frozen` of the
`freeze_ref` branch.

Tests: `tests/test_torso_reference.py` (raw path, bypass, SS-entry / dock events,
freeze_ref + pure_pd), synthetic inputs, no mapping.

## Public API

| symbol | signature | canonical? | code |
|---|---|---|---|
| **`TorsoRef`** *(dataclass)* |  |  | [L32](../../../crawlbot/control/torso_reference.py#L32) |
|   `p` | `` | _field_ | [L36](../../../crawlbot/control/torso_reference.py#L36) |
|   `R` | `` | _field_ | [L37](../../../crawlbot/control/torso_reference.py#L37) |
|   `v` | `` | _field_ | [L38](../../../crawlbot/control/torso_reference.py#L38) |
|   `a` | `` | _field_ | [L39](../../../crawlbot/control/torso_reference.py#L39) |
| **`TorsoReferenceShaper`** |  |  | [L42](../../../crawlbot/control/torso_reference.py#L42) |
| `.shape` | `(tr, rs, qs, tq, phase, rp_interp, vp_interp, af)` | **yes** | [L85](../../../crawlbot/control/torso_reference.py#L85) |
| `.on_ss_entry` | `(p_torso_entry)` | **yes** | [L218](../../../crawlbot/control/torso_reference.py#L218) |
| `.on_dock` | `(t)` | **yes** | [L225](../../../crawlbot/control/torso_reference.py#L225) |

## Code map

| unit | source |
|---|---|
| `class TorsoRef` | [L32-39](../../../crawlbot/control/torso_reference.py#L32-L39) |
| `class TorsoReferenceShaper` | [L42-233](../../../crawlbot/control/torso_reference.py#L42-L233) |
| `TorsoReferenceShaper.shape` | [L85-214](../../../crawlbot/control/torso_reference.py#L85-L214) |
| `TorsoReferenceShaper.on_ss_entry` | [L218-223](../../../crawlbot/control/torso_reference.py#L218-L223) |
| `TorsoReferenceShaper.on_dock` | [L225-233](../../../crawlbot/control/torso_reference.py#L225-L233) |

---

## See also

- the caller: [`controller.md`](controller.md)
- the mapping itself: [`core/com_to_torso_mapping.md`](../core/com_to_torso_mapping.md)
- the planner reference: [`planning/torso_planner.md`](../planning/torso_planner.md)
