# `crawlbot.estimation.contact_estimator`

**File**: `crawlbot/estimation/contact_estimator.py` — **261 lines** — canonical coverage **69 %**

> Module docstring: *"Generalized Momentum Observer (GMO) for sensorless contact detection."*

Generalized Momentum Observer (De Luca, 2006): detects contact **without force
sensors and without measuring acceleration**.

That second point is the reason it exists — differentiating joint velocity twice
to get acceleration would drown a contact signal in noise.

---

## Public API

| symbol | signature | canonical? |
|---|---|---|
| **`ContactObserverConfig`** *(dataclass)* |  |  |
|   `K_O` | `80.0` | _field_ |
|   `dt` | `0.01` | _field_ |
|   `nv` | `18` | _field_ |
|   `F_threshold` | `5.0` | _field_ |
|   `d_proximity` | `0.02` | _field_ |
|   `d_contact` | `0.01` | _field_ |
|   `d_reset` | `0.03` | _field_ |
|   `debounce_count` | `5` | _field_ |
| **`ContactState`** |  |  |
| **`GeneralizedMomentumObserver`** |  |  |
| `.reset` | `(M, v)` | **yes** |
| `.update` | `(M, v, C_matrix, tau_applied)` | **yes** |
| `.residual` | `()` | **yes** |
| `.initialized` | `()` | not exercised |
| `.swing_residual_norm` | `(swing_v_slice)` | **yes** |
| **`ContactStateMachine`** |  |  |
| `.update` | `(r_swing_norm, d_FK, force_mode=False)` | not exercised |
| `.reset` | `()` | **yes** |
| `.state` | `()` | **yes** |
| `.is_docked` | `()` | not exercised |

---

---

## 1. Derivation

Equations of motion, zero gravity:

```
M(q) v_dot + C(q,v) v = S^T tau + J_c^T f_ext
```

Define the generalized momentum `p = M(q) v`. Using the standard skew-symmetry
property `M_dot = C + C^T`, its derivative is

```
p_dot = S^T tau + C^T v + J_c^T f_ext
```

which is remarkable: **no acceleration term**. That is what makes the observer
possible.

Now integrate a copy of that equation with output feedback:

```
beta_dot = S^T tau + C^T v + r
r        = K_O ( p - beta )
```

Subtracting gives the error dynamics

```
r_dot = -K_O r + K_O tau_ext
```

so `r` converges to `tau_ext = J_c^T f_ext` with time constant `1/K_O`. The
residual **is** the external generalized force, low-pass filtered — no
differentiation anywhere.

With `K_O = 80` the time constant is 12.5 ms, about one QP tick.

## 2. One step of the implementation

Forward Euler, `dt = 0.01`:

```python
CT_v  = C_matrix.T @ v
beta += dt * (tau_applied + CT_v + r)
p     = M @ v
r     = K_O * (p - beta)
```

`tau_applied` is `[0_6 ; tau_joints]` — zeros on the floating base, because the
base is unactuated. This is why `RobotInterface` computes the full `C_matrix`
and not just the `C @ v` vector: the observer needs `C^T v`, which is a
different quantity.

## 3. Where it runs

`update()` is called **in single support only** (`sim_loop.py:3123`); in double
support the log records `0.0` (`:1057`), because `gmo_swing_residual` needs a
swing-velocity slice that DS does not track.

Measured on the canonical:

```
gmo_residual_norm   max = 8.088   mean = 1.017   non-zero = 2067/2077
```

A real signal.

## 4. ⚠ `ContactStateMachine` is inert

Constructed (`sim_loop.py:468`), reset (`:1948`), its state read every tick for
the log (`:1061`) — but **`update()` is never called** (0/59 lines).

```
gmo_contact_state   distinct values = [0]      (NO_CONTACT, constant)
```

**This is architecture, not a bug.** Docking is decided geometrically, never by
the GMO. Project rule: *require both `d < 5 mm` AND `ori < 5 deg`*. The observer
provides a measurable, logged residual; the state machine that would turn it into
a contact decision is not wired to the canonical path.

The thresholds in `ContactObserverConfig` (`F_threshold`, `d_proximity`,
`d_contact`, `d_reset`, `debounce_count`) belong to that unwired machine and are
therefore **not canonical values** — nothing reads them on the canonical run.

Worth knowing before using `gmo_contact_state` in a figure.

## See also

- package overview: [`estimation.md`](estimation.md)
