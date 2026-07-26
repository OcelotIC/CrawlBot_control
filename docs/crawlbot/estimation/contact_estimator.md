# `crawlbot.estimation.contact_estimator`

**File**: [`crawlbot/estimation/contact_estimator.py`](../../../crawlbot/estimation/contact_estimator.py) — **260 lines** — canonical coverage **68 %**

> Module docstring: *"Generalized Momentum Observer (GMO) for sensorless contact detection."*

Generalized Momentum Observer (De Luca, 2006): detects contact **without force
sensors and without measuring acceleration**.

That second point is the reason it exists — differentiating joint velocity twice
to get acceleration would drown a contact signal in noise.

---

## Public API

| symbol | signature | canonical? | code |
|---|---|---|---|
| **`ContactObserverConfig`** *(dataclass)* |  |  | [L35](../../../crawlbot/estimation/contact_estimator.py#L35) |
|   `K_O` | `80.0` | _field_ | [L37](../../../crawlbot/estimation/contact_estimator.py#L37) |
|   `dt` | `0.01` | _field_ | [L38](../../../crawlbot/estimation/contact_estimator.py#L38) |
|   `nv` | `18` | _field_ | [L39](../../../crawlbot/estimation/contact_estimator.py#L39) |
|   `F_threshold` | `5.0` | _field_ | [L40](../../../crawlbot/estimation/contact_estimator.py#L40) |
|   `d_proximity` | `0.02` | _field_ | [L41](../../../crawlbot/estimation/contact_estimator.py#L41) |
|   `d_contact` | `0.01` | _field_ | [L42](../../../crawlbot/estimation/contact_estimator.py#L42) |
|   `d_reset` | `0.03` | _field_ | [L43](../../../crawlbot/estimation/contact_estimator.py#L43) |
|   `debounce_count` | `5` | _field_ | [L44](../../../crawlbot/estimation/contact_estimator.py#L44) |
| **`ContactState`** |  |  | [L47](../../../crawlbot/estimation/contact_estimator.py#L47) |
| **`GeneralizedMomentumObserver`** |  |  | [L55](../../../crawlbot/estimation/contact_estimator.py#L55) |
| `.reset` | `(M, v)` | **yes** | [L75](../../../crawlbot/estimation/contact_estimator.py#L75) |
| `.update` | `(M, v, C_matrix, tau_applied)` | **yes** | [L92](../../../crawlbot/estimation/contact_estimator.py#L92) |
| `.residual` | `()` | **yes** | [L136](../../../crawlbot/estimation/contact_estimator.py#L136) |
| `.initialized` | `()` | not exercised | [L141](../../../crawlbot/estimation/contact_estimator.py#L141) |
| `.swing_residual_norm` | `(swing_v_slice)` | not exercised | [L144](../../../crawlbot/estimation/contact_estimator.py#L144) |
| **`ContactStateMachine`** |  |  | [L160](../../../crawlbot/estimation/contact_estimator.py#L160) |
| `.update` | `(r_swing_norm, d_FK, force_mode=False)` | not exercised | [L186](../../../crawlbot/estimation/contact_estimator.py#L186) |
| `.reset` | `()` | **yes** | [L247](../../../crawlbot/estimation/contact_estimator.py#L247) |
| `.state` | `()` | **yes** | [L253](../../../crawlbot/estimation/contact_estimator.py#L253) |
| `.is_docked` | `()` | not exercised | [L257](../../../crawlbot/estimation/contact_estimator.py#L257) |

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

`update()` is called **in single support only** (`sim_loop.py:2913`); in double
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

## Code map

| unit | source |
|---|---|
| `class ContactObserverConfig` | [L35-44](../../../crawlbot/estimation/contact_estimator.py#L35-L44) |
| `class ContactState` | [L47-52](../../../crawlbot/estimation/contact_estimator.py#L47-L52) |
| `class GeneralizedMomentumObserver` | [L55-157](../../../crawlbot/estimation/contact_estimator.py#L55-L157) |
| `GeneralizedMomentumObserver.reset` | [L75-90](../../../crawlbot/estimation/contact_estimator.py#L75-L90) |
| `GeneralizedMomentumObserver.update` | [L92-133](../../../crawlbot/estimation/contact_estimator.py#L92-L133) |
| `GeneralizedMomentumObserver.residual` | [L136-138](../../../crawlbot/estimation/contact_estimator.py#L136-L138) |
| `GeneralizedMomentumObserver.initialized` | [L141-142](../../../crawlbot/estimation/contact_estimator.py#L141-L142) |
| `GeneralizedMomentumObserver.swing_residual_norm` | [L144-157](../../../crawlbot/estimation/contact_estimator.py#L144-L157) |
| `class ContactStateMachine` | [L160-259](../../../crawlbot/estimation/contact_estimator.py#L160-L259) |
| `ContactStateMachine.update` | [L186-245](../../../crawlbot/estimation/contact_estimator.py#L186-L245) |
| `ContactStateMachine.reset` | [L247-250](../../../crawlbot/estimation/contact_estimator.py#L247-L250) |
| `ContactStateMachine.state` | [L253-254](../../../crawlbot/estimation/contact_estimator.py#L253-L254) |
| `ContactStateMachine.is_docked` | [L257-259](../../../crawlbot/estimation/contact_estimator.py#L257-L259) |

---

## See also

- package overview: [`estimation.md`](estimation.md)
