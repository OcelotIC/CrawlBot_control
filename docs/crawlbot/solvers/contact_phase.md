# `crawlbot.solvers.contact_phase`

**File**: `crawlbot/solvers/contact_phase.py` — **138 lines** — canonical coverage **85 %**

> Module docstring: *"Contact phase definitions for crawling multi-arm robot locomotion."*

The shared vocabulary of both controller stages: which grippers are holding,
where, and how that turns into a linear map from contact wrench to momentum
rate. Small file, but both stages depend on it agreeing with itself.

---

## Public API

| symbol | signature | canonical? |
|---|---|---|
| **`ContactPhase`** |  |  |
| **`ContactConfig`** *(dataclass)* |  |  |
|   `phase` |  | _field_ |
|   `nc` |  | _field_ |
|   `active_contacts` |  | _field_ |
|   `r_contact_A` |  | _field_ |
|   `r_contact_B` |  | _field_ |
| `.from_phase` | `(cls, phase, r_contact_A, r_contact_B)` | **yes** |
| `.active_contact_positions` | `()` | not exercised |
| `skew` | `(v)` | **yes** |
| `compute_momentum_map` | `(r_com, contact_config)` | **yes** |

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

## See also

- package overview: [`solvers.md`](solvers.md)
