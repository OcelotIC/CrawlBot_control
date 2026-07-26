# `crawlbot.estimation.contact_estimator`

**File**: [`crawlbot/estimation/contact_estimator.py`](../../../crawlbot/estimation/contact_estimator.py) — **261 lines** — canonical coverage **69 %**

> Module docstring: *"Generalized Momentum Observer (GMO) for sensorless contact detection."*

Generalized Momentum Observer (De Luca, 2006): detects contact **without force
sensors and without measuring acceleration**.

That second point is the reason it exists — differentiating joint velocity twice
to get acceleration would drown a contact signal in noise.

---

## Public API

| symbol | signature | canonical? | code |
|---|---|---|---|
| **`ContactObserverConfig`** *(dataclass)* |  |  | [L36](../../../crawlbot/estimation/contact_estimator.py#L36) |
|   `K_O` | `80.0` | _field_ | [L38](../../../crawlbot/estimation/contact_estimator.py#L38) |
|   `dt` | `0.01` | _field_ | [L39](../../../crawlbot/estimation/contact_estimator.py#L39) |
|   `nv` | `18` | _field_ | [L40](../../../crawlbot/estimation/contact_estimator.py#L40) |
|   `F_threshold` | `5.0` | _field_ | [L41](../../../crawlbot/estimation/contact_estimator.py#L41) |
|   `d_proximity` | `0.02` | _field_ | [L42](../../../crawlbot/estimation/contact_estimator.py#L42) |
|   `d_contact` | `0.01` | _field_ | [L43](../../../crawlbot/estimation/contact_estimator.py#L43) |
|   `d_reset` | `0.03` | _field_ | [L44](../../../crawlbot/estimation/contact_estimator.py#L44) |
|   `debounce_count` | `5` | _field_ | [L45](../../../crawlbot/estimation/contact_estimator.py#L45) |
| **`ContactState`** |  |  | [L48](../../../crawlbot/estimation/contact_estimator.py#L48) |
| **`GeneralizedMomentumObserver`** |  |  | [L56](../../../crawlbot/estimation/contact_estimator.py#L56) |
| `.reset` | `(M, v)` | **yes** | [L76](../../../crawlbot/estimation/contact_estimator.py#L76) |
| `.update` | `(M, v, C_matrix, tau_applied)` | **yes** | [L93](../../../crawlbot/estimation/contact_estimator.py#L93) |
| `.residual` | `()` | **yes** | [L137](../../../crawlbot/estimation/contact_estimator.py#L137) |
| `.initialized` | `()` | not exercised | [L142](../../../crawlbot/estimation/contact_estimator.py#L142) |
| `.swing_residual_norm` | `(swing_v_slice)` | **yes** | [L145](../../../crawlbot/estimation/contact_estimator.py#L145) |
| **`ContactStateMachine`** |  |  | [L161](../../../crawlbot/estimation/contact_estimator.py#L161) |
| `.update` | `(r_swing_norm, d_FK, force_mode=False)` | not exercised | [L187](../../../crawlbot/estimation/contact_estimator.py#L187) |
| `.reset` | `()` | **yes** | [L248](../../../crawlbot/estimation/contact_estimator.py#L248) |
| `.state` | `()` | **yes** | [L254](../../../crawlbot/estimation/contact_estimator.py#L254) |
| `.is_docked` | `()` | not exercised | [L258](../../../crawlbot/estimation/contact_estimator.py#L258) |

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

## Code map

| unit | source |
|---|---|
| `class ContactObserverConfig` | [L36-45](../../../crawlbot/estimation/contact_estimator.py#L36-L45) |
| `class ContactState` | [L48-53](../../../crawlbot/estimation/contact_estimator.py#L48-L53) |
| `class GeneralizedMomentumObserver` | [L56-158](../../../crawlbot/estimation/contact_estimator.py#L56-L158) |
| `GeneralizedMomentumObserver.reset` | [L76-91](../../../crawlbot/estimation/contact_estimator.py#L76-L91) |
| `GeneralizedMomentumObserver.update` | [L93-134](../../../crawlbot/estimation/contact_estimator.py#L93-L134) |
| `GeneralizedMomentumObserver.residual` | [L137-139](../../../crawlbot/estimation/contact_estimator.py#L137-L139) |
| `GeneralizedMomentumObserver.initialized` | [L142-143](../../../crawlbot/estimation/contact_estimator.py#L142-L143) |
| `GeneralizedMomentumObserver.swing_residual_norm` | [L145-158](../../../crawlbot/estimation/contact_estimator.py#L145-L158) |
| `class ContactStateMachine` | [L161-260](../../../crawlbot/estimation/contact_estimator.py#L161-L260) |
| `ContactStateMachine.update` | [L187-246](../../../crawlbot/estimation/contact_estimator.py#L187-L246) |
| `ContactStateMachine.reset` | [L248-251](../../../crawlbot/estimation/contact_estimator.py#L248-L251) |
| `ContactStateMachine.state` | [L254-255](../../../crawlbot/estimation/contact_estimator.py#L254-L255) |
| `ContactStateMachine.is_docked` | [L258-260](../../../crawlbot/estimation/contact_estimator.py#L258-L260) |

---

## See also

- package overview: [`estimation.md`](estimation.md)
