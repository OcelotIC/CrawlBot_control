# `crawlbot.planning.locomotion_planner`

**File**: `crawlbot/planning/locomotion_planner.py` — **206 lines** — canonical coverage **17 %**

> Module docstring: *"LocomotionPlanner — CoM reference trajectory generation for VISPA."*

Previous-generation CoM planner. **Dead on the canonical, kept for the M0/Lutze
paper baseline.**

---

## Public API

| symbol | signature | canonical? |
|---|---|---|
| **`LocomotionPlanner`** |  |  |
| `.calibrate_from_config` | `(r_com_init)` | not exercised |
| `._build_waypoints` | `()` | not exercised |
| `._equilibrium_com` | `(phase, r_a, r_b)` | not exercised |
| `.reference_at` | `(t)` | not exercised |
| `.full_trajectory` | `(dt)` | not exercised |

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

## See also

- package overview: [`planning.md`](planning.md)
