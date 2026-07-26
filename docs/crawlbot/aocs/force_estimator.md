# `crawlbot.aocs.force_estimator`

**File**: `crawlbot/aocs/force_estimator.py` — **657 lines** — canonical coverage **39 %**

> Module docstring: *"MomentumDisturbanceEstimator — Estimate the disturbance torque applied by"*

Reaction-wheel control, and the estimator of the disturbance torque the moving
robot applies to the host structure.

This is the other half of the decentralised contract: the NMPC promises never to
demand more than `tau_w_max`; this module is what actually spends it.

---

## Public API

| symbol | signature | canonical? |
|---|---|---|
| **`EstimatorConfig`** *(dataclass)* |  |  |
|   `robot_mass` | `71.0` | _field_ |
|   `dt` | `0.01` | _field_ |
|   `filter_tau` | `0.016` | _field_ |
|   `include_transport` | `True` | _field_ |
| **`MomentumDisturbanceEstimator`** |  |  |
| `.reset` | `()` | not exercised |
| `.update` | `(r_com, v_com, L_com, omega_s)` | not exercised |
| `.update_analytical` | `(r_com, v_com, L_com, L_com_prev, a_com, omega_s)` | not exercised |
| `.H_rO` | `()` | **yes** |
| `.H_dot` | `()` | **yes** |
| `.initialized` | `()` | not exercised |
| `compute_aocs_command` | `(H_dot_est, omega_s, hw_current, hw_target=None, K_omega...)` | not exercised |
| `compute_aocs_command_legacy_corrected` | `(L_com, L_com_prev, r_com, v_com, v_com_prev, hw_current...)` | not exercised |
| `compute_aocs_command_legacy_pd_numerical` | `(L_com, L_com_prev, r_com, v_com, v_com_prev, omega_s, o...)` | not exercised |
| `compute_aocs_command_legacy_pd_model` | `(L_com, L_com_prev, r_com, v_com, v_com_prev, omega_s, t...)` | not exercised |
| `compute_aocs_command_legacy_pid_numerical` | `(L_com, L_com_prev, r_com, v_com, v_com_prev, omega_s, o...)` | **yes** |
| `compute_aocs_command_legacy_pid_model` | `(L_com, L_com_prev, r_com, v_com, v_com_prev, omega_s, t...)` | not exercised |

---

---

## 1. The physics

Total angular momentum of the robot about O (the structure CoM):

```
H_{r/O} = L_com + r_com x (m_r * v_com)
          -----   ----------------------
          spin           orbital
```

Two contributions, both real: joint motion (spin, order 5 Nms) and CoM
translation (orbital, order 20 Nms over three steps). The orbital term is the
larger one, which is why bounding `||m*v_com||` matters in stage 1.

The disturbance torque on the structure is the inertial derivative:

```
tau_dist = -dH/dt|_inertial = -( dH/dt|_struct + omega_s x H_{r/O} )
```

The `omega_s x H` transport term is what makes this an inertial rather than a
body-frame derivative — it is not optional once the structure itself rotates.

## 2. The canonical control law

Six laws are implemented; the canonical run selects
`aocs_mode='legacy_pid_numerical'`. The other five are alternative modes — not
dead code, but **not covered by the gate either**.

```
tau_w = ff_term
      + K_hw * ( clip(h_w) - h_w )         desaturation
      + K_theta * theta_s                  attitude
      + K_omega * omega_s                  rate damping
      + K_d * omega_s_dot                  numerical accel damping
tau_w <- clip( tau_w, +/- tau_w_max )
```

| parameter | canonical value | source |
|---|---|---|
| `K_theta` | **1.0** Nm/rad | passed explicitly |
| `K_omega` | **50.0** | passed explicitly |
| `tau_w_max` | **2.5** Nm | passed explicitly (frozen cap) |
| `K_d` | 25.0 | default |
| `K_hw` | 2.0 | default |

### Why K_theta is positive

Same derivation as `K_omega` and `K_d`: Newton-Euler about the structure CoM,
with `tau_w` on the wheels producing `-tau_w` reaction on the structure. For
`theta_s > 0` to decrease you need negative angular acceleration, hence
`tau_w > -H_s_dot`, hence a **positive** K_theta contribution. The sign is
counter-intuitive if you reason about the wheels instead of the structure.

### The attitude term is momentum-bound, not torque-bound

Rotating the structure back by `delta_theta` requires the wheels to transiently
carry `|h_w| = I_s * omega_max <= h_w_max`, so

```
omega_max = h_w_max / I_s
```

With `h_w_max = 5 Nms` and `I_s ~ 1500 kg*m^2` that is about **3.3 mrad/s**. A
typical per-traversal rotation (~2 deg = 35 mrad) therefore needs **~10 s
minimum**, whatever the torque budget. This is why K_theta is sized for a slow
(~60 s) rotate-back rather than a fast correction.

## 3. Two feedforwards — and why one is not enough

`ff_term` has two branches, and **both run** on the canonical:

### Single support: kinematic finite differences (`:585-588`)

```
ff = -L_com_dot - r_com x ( m_r * v_com_dot )       (both by FD)
```

Valid while the robot is kinematically free at the contact.

### Double support: contact-wrench feedforward (`:592`)

With both grippers welded, the closed loop carries internal stress. It exerts on
the structure a couple

```
( r_CA - r_CB ) x f
```

that is **invisible in `L_com`** — the two contact forces cancel in the
momentum balance while their moments do not. The kinematic feedforward is
therefore structurally incomplete in DS, not merely noisy.

`sim_loop.py:877-883` computes the correct term straight from the QP solution:

```
tau_struct_ff = - sum_i ( r_Ci x f_i + tau_i )
```

and passes it in, short-circuiting the FD branch.

This is the only difference in AOCS treatment between DS and SS.

## 4. ⚠ `MomentumDisturbanceEstimator` is not in the loop

The object is constructed (`sim_loop.py:445`) and its `H_rO` / `H_dot`
properties are read every tick for the log (`:1067-1068`), but **`update()` is
never called** — 0 of 54 lines covered.

Measured on the canonical log:

```
H_rO       shape=(2077, 3)   max|.| = 0   all-zero = True
H_dot_est  shape=(2077, 3)   max|.| = 0   all-zero = True
```

**Both exported channels are identically zero over the whole traversal.** The
feedforward actually used is the inline computation of section 3.

The theory in the module docstring — variant A (EMA-filtered finite differences)
and variant B (analytical, via `a_com`) — remains valid; neither is wired in.
The EMA-before-differentiation ordering described there is the right design if
it is ever reconnected: filtering after a finite difference amplifies the noise
you were trying to remove.

Worth knowing before plotting or analysing `H_dot_est`.

## See also

- package overview: [`aocs.md`](aocs.md)
