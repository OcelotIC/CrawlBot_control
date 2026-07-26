# `crawlbot.planning.swing_planner`

**File**: [`crawlbot/planning/swing_planner.py`](../../../crawlbot/planning/swing_planner.py) — **338 lines** — canonical coverage **78 %**

> Module docstring: *"Swing arm trajectory planner for crawling locomotion."*

Cartesian reference for the free end-effector during single support: where the
gripper is, how fast, and at what orientation, at every instant of a swing.

Everything is in the **structure body frame**, where the anchors are fixed
points — which is why no live-anchor transform machinery is needed.

---

## Public API

| symbol | signature | canonical? | code |
|---|---|---|---|
| **`SwingReference`** *(dataclass)* |  |  | [L48](../../../crawlbot/planning/swing_planner.py#L48) |
|   `p_ee` | `` | _field_ | [L50](../../../crawlbot/planning/swing_planner.py#L50) |
|   `v_ee` | `` | _field_ | [L51](../../../crawlbot/planning/swing_planner.py#L51) |
|   `a_ee` | `` | _field_ | [L52](../../../crawlbot/planning/swing_planner.py#L52) |
|   `R_ee` | `` | _field_ | [L53](../../../crawlbot/planning/swing_planner.py#L53) |
|   `omega_ee` | `` | _field_ | [L54](../../../crawlbot/planning/swing_planner.py#L54) |
|   `alpha_ee` | `` | _field_ | [L55](../../../crawlbot/planning/swing_planner.py#L55) |
|   `swing_arm` | `` | _field_ | [L56](../../../crawlbot/planning/swing_planner.py#L56) |
|   `is_swinging` | `` | _field_ | [L57](../../../crawlbot/planning/swing_planner.py#L57) |
|   `phase_progress` | `` | _field_ | [L58](../../../crawlbot/planning/swing_planner.py#L58) |
| **`SwingPlanner`** |  |  | [L61](../../../crawlbot/planning/swing_planner.py#L61) |
| `.set_swing_orientation` | `(R_start)` | **yes** | [L126](../../../crawlbot/planning/swing_planner.py#L126) |
| `.plan` | `()` | **yes** | [L137](../../../crawlbot/planning/swing_planner.py#L137) |
| `._quintic` | `(tau)` | **yes** | [L143](../../../crawlbot/planning/swing_planner.py#L143) |
| `._quintic_dot` | `(tau)` | **yes** | [L149](../../../crawlbot/planning/swing_planner.py#L149) |
| `._quintic_ddot` | `(tau)` | **yes** | [L154](../../../crawlbot/planning/swing_planner.py#L154) |
| `._bump` | `(tau)` | **yes** | [L164](../../../crawlbot/planning/swing_planner.py#L164) |
| `._bump_dot` | `(tau)` | **yes** | [L172](../../../crawlbot/planning/swing_planner.py#L172) |
| `._bump_ddot` | `(tau)` | **yes** | [L183](../../../crawlbot/planning/swing_planner.py#L183) |
| `._delayed_cosine` | `(tau, tau_d)` | **yes** | [L204](../../../crawlbot/planning/swing_planner.py#L204) |
| `._delayed_cosine_dot` | `(tau, tau_d)` | **yes** | [L213](../../../crawlbot/planning/swing_planner.py#L213) |
| `._delayed_cosine_ddot` | `(tau, tau_d)` | **yes** | [L221](../../../crawlbot/planning/swing_planner.py#L221) |
| `.reference_at` | `(t)` | **yes** | [L230](../../../crawlbot/planning/swing_planner.py#L230) |
| `._last_swing_position` | `(current_idx)` | not exercised | [L323](../../../crawlbot/planning/swing_planner.py#L323) |

### Module constants

| name | value |
|---|---|
| `DEFAULT_CLEARANCE` | `0.03` |
| `DEFAULT_AWAY_NORMAL` | `np.array([0.0, 0.0, -1.0])` |

---

---

## 1. Trajectory design

For a swing from `p_start` to `p_end` over duration `T`, with
`tau = (t - t_phase_start)/T` in [0,1]:

```
p(tau) = p_start + dp * s(tau) + clearance * n_hat * bump(tau)
```

Three ingredients, each with a job:

### s(tau) — the quintic, rest to rest

```
s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5
```

Chosen because `s(0)=0, s(1)=1` with **both first and second derivatives zero at
each end**. That gives `v(0)=v(1)=0` and `a(0)=a(1)=0`: no velocity step at
detach, no acceleration jump at attach. A cubic would give zero velocity but a
finite acceleration jump — a jerk impulse into a free-floating base, which
becomes momentum the wheels must absorb.

### bump(tau) — the clearance bell

Nominally `sin^2(pi tau)`: zero at both ends, maximum at mid-swing, C1
boundaries. The generalised form allows an asymmetric peak at `tau_p`:

```
rise    (tau <= tau_p) : sin^2( pi*tau       / (2*tau_p)     )
descent (tau >= tau_p) : sin^2( pi*(1-tau)   / (2*(1-tau_p)) )
```

Both branches reach 1 at `tau_p` and 0 at the ends, with zero first derivative
at 0, `tau_p` and 1 — so the two halves join smoothly. It reduces exactly to
`sin^2(pi tau)` when `tau_p = 0.5`.

`n_hat` is the unit normal away from the structure surface — canonically
`-z` in the structure frame, because the robot hangs *below* the structure.
Default clearance 0.03 m.

### sigma(tau) — delayed cosine, for orientation

```
sigma(tau) = 0                                        if tau <  tau_d
             0.5 * (1 - cos( pi (tau - tau_d)/(1 - tau_d) ))   otherwise
```

The orientation SLERP is driven by `sigma`, not by `s`. This **concentrates the
rotation in the second half of the swing**: the gripper first arcs over the
obstacle (clearance peaks at `tau = 0.5`) and only *then* rotates into the dock
orientation during the final approach.

Doing both at once would sweep the gripper through a wider volume and arrive
with residual angular velocity — bad when the gate demands `ori < 5 deg`.

All three profiles are implemented with their first and second derivatives
(`_quintic`, `_bump`, `_delayed_cosine` and `_dot` / `_ddot` variants), all
exercised on the canonical.

## 2. `reference_at(t)`

Queries the gait plan and branches:

- **DS** -> the last swing position, frozen, with zero velocity and acceleration;
- **SS** -> interpolate over `T_eff = T_step * early_finish_fraction`, with
  `tau` clipped to 1.

### Early finish

Once `tau` reaches 1 the reference is *at the target with zero velocity and
acceleration* — by construction, since all three profiles have `p_dot(1) = 0`.
Clipping `tau` therefore holds the gripper still rather than freezing a
mid-motion state.

The point is to let the end-effector arrive **before** the phase ends and settle,
so the docking gate (`d < 5 mm AND ori < 5 deg`) evaluates on a quiet pose
rather than on one still moving. The measured at-weld distances (4.02 to 4.99 mm)
sit inside a 5 mm capture radius; arriving hot would not be recoverable.

## 3. What CLEANUP-18 removed

The whole **phase-override mechanism**: `add_phase`, `_override_reference_at`,
`clear_phase_overrides`, the `_phase_overrides` list and its dispatch loop at the
head of `reference_at`, plus `adaptive_reference_at` and `swing_trajectory`.

`reference_at()` now always takes the scheduler-driven path — the one the
canonical has always used. Coverage went from **47 % to 95 %**, the file from 728
to 337 lines.

⚠ Do not confuse: **`torso_planner.add_phase` is live**. Only the *swing*
planner's `add_phase` was dead.

## Code map

| unit | source |
|---|---|
| `class SwingReference` | [L48-58](../../../crawlbot/planning/swing_planner.py#L48-L58) |
| `class SwingPlanner` | [L61-336](../../../crawlbot/planning/swing_planner.py#L61-L336) |
| `SwingPlanner.set_swing_orientation` | [L126-130](../../../crawlbot/planning/swing_planner.py#L126-L130) |
| `SwingPlanner.plan` | [L137-138](../../../crawlbot/planning/swing_planner.py#L137-L138) |
| `SwingPlanner._quintic` | [L143-146](../../../crawlbot/planning/swing_planner.py#L143-L146) |
| `SwingPlanner._quintic_dot` | [L149-151](../../../crawlbot/planning/swing_planner.py#L149-L151) |
| `SwingPlanner._quintic_ddot` | [L154-155](../../../crawlbot/planning/swing_planner.py#L154-L155) |
| `SwingPlanner._bump` | [L164-170](../../../crawlbot/planning/swing_planner.py#L164-L170) |
| `SwingPlanner._bump_dot` | [L172-181](../../../crawlbot/planning/swing_planner.py#L172-L181) |
| `SwingPlanner._bump_ddot` | [L183-192](../../../crawlbot/planning/swing_planner.py#L183-L192) |
| `SwingPlanner._delayed_cosine` | [L204-210](../../../crawlbot/planning/swing_planner.py#L204-L210) |
| `SwingPlanner._delayed_cosine_dot` | [L213-218](../../../crawlbot/planning/swing_planner.py#L213-L218) |
| `SwingPlanner._delayed_cosine_ddot` | [L221-226](../../../crawlbot/planning/swing_planner.py#L221-L226) |
| `SwingPlanner.reference_at` | [L230-316](../../../crawlbot/planning/swing_planner.py#L230-L316) |
| `SwingPlanner._last_swing_position` | [L323-336](../../../crawlbot/planning/swing_planner.py#L323-L336) |

---

## See also

- package overview: [`planning.md`](planning.md)
