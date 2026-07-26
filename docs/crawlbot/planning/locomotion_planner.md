# `crawlbot.planning.locomotion_planner`

**File**: [`crawlbot/planning/locomotion_planner.py`](../../../crawlbot/planning/locomotion_planner.py) — **206 lines** — canonical coverage **17 %**

> Module docstring: *"LocomotionPlanner — CoM reference trajectory generation for VISPA."*

Previous-generation CoM planner. **Dead on the canonical, kept for the M0/Lutze
paper baseline.**

---

## Public API

| symbol | signature | canonical? | code |
|---|---|---|---|
| **`LocomotionPlanner`** |  |  | [L32](../../../crawlbot/planning/locomotion_planner.py#L32) |
| `.calibrate_from_config` | `(r_com_init)` | not exercised | [L64](../../../crawlbot/planning/locomotion_planner.py#L64) |
| `._build_waypoints` | `()` | not exercised | [L75](../../../crawlbot/planning/locomotion_planner.py#L75) |
| `._equilibrium_com` | `(phase, r_a, r_b)` | not exercised | [L85](../../../crawlbot/planning/locomotion_planner.py#L85) |
| `.reference_at` | `(t)` | not exercised | [L116](../../../crawlbot/planning/locomotion_planner.py#L116) |
| `.full_trajectory` | `(dt)` | not exercised | [L179](../../../crawlbot/planning/locomotion_planner.py#L179) |

### Module constants

| name | value |
|---|---|
| `DEFAULT_COM_HEIGHT` | `-0.47` |

---

---

## 1. What it was

An equilibrium-based CoM reference generator: `_equilibrium_com` computes a
support-consistent CoM target, `_build_waypoints` chains them, `reference_at`
interpolates. It predates the pre-planner + torso-planner split.

`sim_loop.py:46` carries the comment *"LocomotionPlanner removed — CoM reference
comes from TorsoPlanner"*, and indeed `sim_loop` never constructs it. Coverage
**17 %**; `full_trajectory` has zero callers.

## 2. Why it is still here

CLEANUP-16 ranked it "delete, 205 lines, low risk". **Revised on measurement**
(CLEANUP-18 section 3): it has three consumers, all of whose imports resolve at
HEAD, and the decisive one is **`lutze_baseline/sim_lutze.py`** — a *package*,
not a research script, carrying the M0/Lutze comparison behind the paper's
section II differentiation table.

`LocomotionPlanner` is load-bearing there: constructed at `sim_lutze.py:175`,
calibrated at `:176`, evaluated at `:231` and `:266`.

Deleting it would have traded 205 lines for a broken paper baseline.

## 3. The method lesson

The audit had asserted its consumers were "already non-functional". Import-checking
all three showed that **false for every one**. Hence the rule adopted: do not
assume a script is already broken — test it.

Revisiting is a project question, not a code one: *is the Lutze baseline still to
be re-run?* (`CLEANUP_CARRYOVER` C5).

## Code map

| unit | source |
|---|---|
| `class LocomotionPlanner` | [L32-205](../../../crawlbot/planning/locomotion_planner.py#L32-L205) |
| `LocomotionPlanner.calibrate_from_config` | [L64-73](../../../crawlbot/planning/locomotion_planner.py#L64-L73) |
| `LocomotionPlanner._build_waypoints` | [L75-83](../../../crawlbot/planning/locomotion_planner.py#L75-L83) |
| `LocomotionPlanner._equilibrium_com` | [L85-114](../../../crawlbot/planning/locomotion_planner.py#L85-L114) |
| `LocomotionPlanner.reference_at` | [L116-177](../../../crawlbot/planning/locomotion_planner.py#L116-L177) |
| `LocomotionPlanner.full_trajectory` | [L179-205](../../../crawlbot/planning/locomotion_planner.py#L179-L205) |

---

## See also

- package overview: [`planning.md`](planning.md)
