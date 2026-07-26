# `crawlbot.solvers.contact_phase`

**File**: [`crawlbot/solvers/contact_phase.py`](../../../crawlbot/solvers/contact_phase.py) — **138 lines** — canonical coverage **83 %**

> Module docstring: *"Contact phase definitions for crawling multi-arm robot locomotion."*

The shared vocabulary of both controller stages: which grippers are holding,
where, and how that turns into a linear map from contact wrench to momentum
rate. Small file, but both stages depend on it agreeing with itself.

---

## Public API

| symbol | signature | canonical? | code |
|---|---|---|---|
| **`ContactPhase`** |  |  | [L19](../../../crawlbot/solvers/contact_phase.py#L19) |
| **`ContactConfig`** *(dataclass)* |  |  | [L27](../../../crawlbot/solvers/contact_phase.py#L27) |
|   `phase` | `` | _field_ | [L43](../../../crawlbot/solvers/contact_phase.py#L43) |
|   `nc` | `` | _field_ | [L44](../../../crawlbot/solvers/contact_phase.py#L44) |
|   `active_contacts` | `` | _field_ | [L45](../../../crawlbot/solvers/contact_phase.py#L45) |
|   `r_contact_A` | `` | _field_ | [L46](../../../crawlbot/solvers/contact_phase.py#L46) |
|   `r_contact_B` | `` | _field_ | [L47](../../../crawlbot/solvers/contact_phase.py#L47) |
| `.from_phase` | `(cls, phase, r_contact_A, r_contact_B)` | **yes** | [L50](../../../crawlbot/solvers/contact_phase.py#L50) |
| `.active_contact_positions` | `()` | not exercised | [L70](../../../crawlbot/solvers/contact_phase.py#L70) |
| `skew` | `(v)` | **yes** | [L80](../../../crawlbot/solvers/contact_phase.py#L80) |
| `compute_momentum_map` | `(r_com, contact_config)` | **yes** | [L101](../../../crawlbot/solvers/contact_phase.py#L101) |

---

---

## 1. Phases

`ContactPhase`: `DOUBLE` (both arms welded), `SINGLE_A` (A holds, B swings),
`SINGLE_B` (B holds, A swings).

⚠ The architecture is **two-phase, not three**. `SINGLE_A` and `SINGLE_B` are two
instances of the same SS regime — there is no third "EXT" state. Explicit
project rule (spec 7.1).

`ContactConfig.from_phase(phase, r_contact_A, r_contact_B)` derives `nc` (number
of active contacts) and the `active_contacts` mask. This is the object handed to
both the NMPC and the QP so they agree on how many wrenches are in play.

## 2. The momentum map

The one piece of real mathematics in the file. For each active contact `j`, the
contribution of its wrench to the centroidal angular-momentum rate is

```
L_dot_j = (r_Cj - r_com) x f_j + tau_j
```

which is **linear** in `[f_j ; tau_j]`, so it can be written as a matrix block:

```
L_dot_j = [ S(r_Cj - r_com)   I_3 ] @ [ f_j ; tau_j ]
```

where `S(v)` is the skew-symmetric matrix with `S(v) w = v x w`. Stacking both
contacts gives

```
L_dot_com = M_lambda @ lambda        M_lambda in R^{3x12}
```

with **zero columns for inactive contacts** — which is how the same fixed-size
map serves both DS and SS without reshaping.

Implementation: `compute_momentum_map` (`:101`), `skew` (`:80`).

### Why it matters to both stages

| stage | use |
|---|---|
| `centroidal_nmpc` | the same relation appears inside the ODE, to **predict** momentum |
| `wholebody_qp` | `h_w - dt * M_lambda * lambda` is the momentum box, to **bound** the commanded wrench |

If this map and the NMPC's ODE ever disagreed on the moment arm, stage 2 would
be enforcing a box that stage 1 never planned against. Keeping the map in one
place is what prevents that.

⚠ Note the arm here is `r_Cj - r_com` — from the **robot CoM**, because this is
the *centroidal* momentum. That is a different quantity from the wheel-torque cap
in `centroidal_nmpc`, which uses `r_Cj` from the **structure origin**. Both are
correct in their own context; confusing them is the exact error corrected in
CAMPAIGN_5STEP section 9.

## 3. Unexercised

`ContactConfig.active_contact_positions` — a convenience accessor with no
callers.

## Code map

| unit | source |
|---|---|
| `class ContactPhase` | [L19-23](../../../crawlbot/solvers/contact_phase.py#L19-L23) |
| `class ContactConfig` | [L27-77](../../../crawlbot/solvers/contact_phase.py#L27-L77) |
| `ContactConfig.from_phase` | [L50-67](../../../crawlbot/solvers/contact_phase.py#L50-L67) |
| `ContactConfig.active_contact_positions` | [L70-77](../../../crawlbot/solvers/contact_phase.py#L70-L77) |
| `skew()` | [L80-98](../../../crawlbot/solvers/contact_phase.py#L80-L98) |
| `compute_momentum_map()` | [L101-137](../../../crawlbot/solvers/contact_phase.py#L101-L137) |

---

## See also

- package overview: [`solvers.md`](solvers.md)
