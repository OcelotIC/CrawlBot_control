# `crawlbot.planning.torso_planner`

**File**: [`crawlbot/planning/torso_planner.py`](../../../crawlbot/planning/torso_planner.py) — **481 lines** — canonical coverage **77 %**

> Module docstring: *"TorsoPlanner — Generates 6D torso + CoM reference trajectories."*

Torso pose reference: quintic in position, SLERP in orientation, one phase per
step. Also supplies the CoM and `L_com` references consumed by stage 1.

---

## Public API

| symbol | signature | canonical? | code |
|---|---|---|---|
| **`TorsoReference`** *(dataclass)* |  |  | [L40](../../../crawlbot/planning/torso_planner.py#L40) |
|   `p` | `` | _field_ | [L42](../../../crawlbot/planning/torso_planner.py#L42) |
|   `R` | `` | _field_ | [L43](../../../crawlbot/planning/torso_planner.py#L43) |
|   `v` | `` | _field_ | [L44](../../../crawlbot/planning/torso_planner.py#L44) |
|   `a` | `` | _field_ | [L45](../../../crawlbot/planning/torso_planner.py#L45) |
| **`ComReference`** *(dataclass)* |  |  | [L49](../../../crawlbot/planning/torso_planner.py#L49) |
|   `r_com` | `` | _field_ | [L51](../../../crawlbot/planning/torso_planner.py#L51) |
|   `v_com` | `` | _field_ | [L52](../../../crawlbot/planning/torso_planner.py#L52) |
| **`TorsoPlanner`** |  |  | [L55](../../../crawlbot/planning/torso_planner.py#L55) |
| `.set_torso_inertia` | `(I_body)` | **yes** | [L88](../../../crawlbot/planning/torso_planner.py#L88) |
| `.set_hold` | `(p, R, r_com=None)` | **yes** | [L105](../../../crawlbot/planning/torso_planner.py#L105) |
| `.add_phase` | `(t_start, t_end, p_start, R_start, p_end, R_end, delta_c...)` | **yes** | [L126](../../../crawlbot/planning/torso_planner.py#L126) |
| `.clear_phases` | `()` | **yes** | [L269](../../../crawlbot/planning/torso_planner.py#L269) |
| `.reference_at` | `(t)` | **yes** | [L280](../../../crawlbot/planning/torso_planner.py#L280) |
| `.has_phase_at` | `(t)` | not exercised | [L289](../../../crawlbot/planning/torso_planner.py#L289) |
| `.reference_at_clamped` | `(t)` | **yes** | [L296](../../../crawlbot/planning/torso_planner.py#L296) |
| `.com_reference_at` | `(t)` | **yes** | [L317](../../../crawlbot/planning/torso_planner.py#L317) |
| `.l_com_reference_at` | `(t)` | **yes** | [L340](../../../crawlbot/planning/torso_planner.py#L340) |
| `._hold_reference` | `()` | **yes** | [L395](../../../crawlbot/planning/torso_planner.py#L395) |
| `._profile_params` | `(t, phase)` | **yes** | [L411](../../../crawlbot/planning/torso_planner.py#L411) |
| `._quintic_params` | `(t, phase)` | **yes** | [L423](../../../crawlbot/planning/torso_planner.py#L423) |
| `._interpolate_phase` | `(t, phase)` | **yes** | [L436](../../../crawlbot/planning/torso_planner.py#L436) |
| `._interpolate_com` | `(t, phase)` | **yes** | [L458](../../../crawlbot/planning/torso_planner.py#L458) |

---

---

## 1. Interpolation

Within a phase over `[t_start, t_end]`, with `tau` the normalised time:

**Position** — the same rest-to-rest quintic as the swing planner:

```
p(tau) = p0 + (p1 - p0) * s(tau)        s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5
```

zero velocity *and* zero acceleration at both ends.

**Orientation** — SLERP on the rotation, driven by the same `s(tau)`:

```
R(tau) = R0 * exp( s(tau) * log( R0^T R1 ) )
```

Interpolating in the Lie algebra rather than component-wise on the quaternion is
what makes the path a geodesic on SO(3) — constant angular velocity direction,
no wobble, and no need to re-normalise.

Implemented in `_interpolate_phase` / `_quintic_params`; the CoM counterpart is
`_interpolate_com`.

## 2. Three outputs, three consumers

| method | consumer |
|---|---|
| `reference_at(t)` | the QP torso-pose task (stage 2) |
| `com_reference_at(t)` | the CoM reference |
| `l_com_reference_at(t)` | **`L_ref` in the NMPC cost** (stage 1) |

`l_com_reference_at` is worth noting: it is what makes `L_ref` in
`centroidal_nmpc` a live planned momentum profile rather than a stub.

## 3. Per-step re-anchoring

`add_phase(t_start, t_end, p0, R0, p1, R1)` installs the current step's phase
(`sim_loop.py:1544`) after the pre-planner has produced `T_step`. The reference
is **re-anchored on the measured pose at the start of each step**, so tracking
error does not accumulate across a traversal.

## 4. ⚠ In SS this quintic is used raw

Explicit project rule:

> *SS uses the raw TorsoPlanner quintic (`sim_loop.py:2581-2584`); the mapping
> (delta(q_current) + F-SAT) remains a DS-only path.*

In single support the QP's torso-pose task receives `tr.p / v / a` directly. The
CoM->torso mapping only intervenes in double support.

## 5. `reference_at_clamped` — for the log, not the control

On DS settle ticks where no phase covers `t`, this returns the **frozen terminal
quintic pose**. It exists only for the export: without it the logged torso
reference jumped at the SS->DS transition.

The continuity fix is **logging-only** — control was proven byte-identical by a
full re-run. Project rule 11:

> *A logged reference that jumps at a phase transition is either a control bug or
> an export artefact — find out which before plotting.*

Here it was the second. Determining that required a full re-run diff, not an
argument.

## 6. Real tracking, and why the excursion is not an error

Boundary residual: **18-27 mm** steady-state (98.6 mm on the initial step). The
~150 mm mid-swing excursion is **genuine free-floating recoil** against the
momentum envelope, not a tracking failure (TORSO-REF-AUDIT).

The distinction matters: tightening the torso gains to chase that excursion would
fight physics the NMPC deliberately allows, and would spend wheel momentum to do
it.

Removed in CLEANUP-18: `set_from_waypoints` (orphaned by CLEANUP-14 when the
`ds_mobile_com_magnitude` block went) and `_trapezoidal_params` (zero callers,
including internal).

## Code map

| unit | source |
|---|---|
| `class TorsoReference` | [L40-45](../../../crawlbot/planning/torso_planner.py#L40-L45) |
| `class ComReference` | [L49-52](../../../crawlbot/planning/torso_planner.py#L49-L52) |
| `class TorsoPlanner` | [L55-480](../../../crawlbot/planning/torso_planner.py#L55-L480) |
| `TorsoPlanner.set_torso_inertia` | [L88-103](../../../crawlbot/planning/torso_planner.py#L88-L103) |
| `TorsoPlanner.set_hold` | [L105-118](../../../crawlbot/planning/torso_planner.py#L105-L118) |
| `TorsoPlanner.add_phase` | [L126-267](../../../crawlbot/planning/torso_planner.py#L126-L267) |
| `TorsoPlanner.clear_phases` | [L269-270](../../../crawlbot/planning/torso_planner.py#L269-L270) |
| `TorsoPlanner.reference_at` | [L280-287](../../../crawlbot/planning/torso_planner.py#L280-L287) |
| `TorsoPlanner.has_phase_at` | [L289-294](../../../crawlbot/planning/torso_planner.py#L289-L294) |
| `TorsoPlanner.reference_at_clamped` | [L296-315](../../../crawlbot/planning/torso_planner.py#L296-L315) |
| `TorsoPlanner.com_reference_at` | [L317-338](../../../crawlbot/planning/torso_planner.py#L317-L338) |
| `TorsoPlanner.l_com_reference_at` | [L340-391](../../../crawlbot/planning/torso_planner.py#L340-L391) |
| `TorsoPlanner._hold_reference` | [L395-408](../../../crawlbot/planning/torso_planner.py#L395-L408) |
| `TorsoPlanner._profile_params` | [L411-421](../../../crawlbot/planning/torso_planner.py#L411-L421) |
| `TorsoPlanner._quintic_params` | [L423-434](../../../crawlbot/planning/torso_planner.py#L423-L434) |
| `TorsoPlanner._interpolate_phase` | [L436-456](../../../crawlbot/planning/torso_planner.py#L436-L456) |
| `TorsoPlanner._interpolate_com` | [L458-480](../../../crawlbot/planning/torso_planner.py#L458-L480) |

---

## See also

- package overview: [`planning.md`](planning.md)
