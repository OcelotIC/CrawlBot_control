# `crawlbot.planning.torso_planner`

**File**: `crawlbot/planning/torso_planner.py` — **481 lines** — canonical coverage **81 %**

> Module docstring: *"TorsoPlanner — Generates 6D torso + CoM reference trajectories."*

Torso pose reference: quintic in position, SLERP in orientation, one phase per
step. Also supplies the CoM and `L_com` references consumed by stage 1.

---

## Public API

| symbol | signature | canonical? |
|---|---|---|
| **`TorsoReference`** *(dataclass)* |  |  |
|   `p` |  | _field_ |
|   `R` |  | _field_ |
|   `v` |  | _field_ |
|   `a` |  | _field_ |
| **`ComReference`** *(dataclass)* |  |  |
|   `r_com` |  | _field_ |
|   `v_com` |  | _field_ |
| **`TorsoPlanner`** |  |  |
| `.set_torso_inertia` | `(I_body)` | **yes** |
| `.set_hold` | `(p, R, r_com=None)` | **yes** |
| `.add_phase` | `(t_start, t_end, p_start, R_start, p_end, R_end, delta_c...)` | **yes** |
| `.clear_phases` | `()` | **yes** |
| `.reference_at` | `(t)` | **yes** |
| `.has_phase_at` | `(t)` | **yes** |
| `.reference_at_clamped` | `(t)` | **yes** |
| `.com_reference_at` | `(t)` | **yes** |
| `.l_com_reference_at` | `(t)` | **yes** |
| `._hold_reference` | `()` | **yes** |
| `._profile_params` | `(t, phase)` | **yes** |
| `._quintic_params` | `(t, phase)` | **yes** |
| `._interpolate_phase` | `(t, phase)` | **yes** |
| `._interpolate_com` | `(t, phase)` | **yes** |

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

## See also

- package overview: [`planning.md`](planning.md)
