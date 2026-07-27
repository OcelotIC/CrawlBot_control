# `crawlbot.aocs.force_estimator`

**File**: [`crawlbot/aocs/force_estimator.py`](../../../crawlbot/aocs/force_estimator.py) — **685 lines** — canonical coverage **39 %**

> Module docstring: *"MomentumDisturbanceEstimator — Estimate the disturbance torque applied by"*

Reaction-wheel control, and the estimator of the disturbance torque the moving
robot applies to the host structure.

This is the other half of the decentralised contract: the NMPC promises never to
demand more than `tau_w_max`; this module is what actually spends it.

---

## Public API

| symbol | signature | canonical? | code |
|---|---|---|---|
| **`EstimatorConfig`** *(dataclass)* |  |  | [L43](../../../crawlbot/aocs/force_estimator.py#L43) |
|   `robot_mass` | `71.0` | _field_ | [L60](../../../crawlbot/aocs/force_estimator.py#L60) |
|   `dt` | `0.01` | _field_ | [L61](../../../crawlbot/aocs/force_estimator.py#L61) |
|   `filter_tau` | `0.016` | _field_ | [L62](../../../crawlbot/aocs/force_estimator.py#L62) |
|   `include_transport` | `True` | _field_ | [L63](../../../crawlbot/aocs/force_estimator.py#L63) |
| **`MomentumDisturbanceEstimator`** |  |  | [L66](../../../crawlbot/aocs/force_estimator.py#L66) |
| `.reset` | `()` | not exercised | [L110](../../../crawlbot/aocs/force_estimator.py#L110) |
| `.update` | `(r_com, v_com, L_com, omega_s)` | not exercised | [L118](../../../crawlbot/aocs/force_estimator.py#L118) |
| `.update_analytical` | `(r_com, v_com, L_com, L_com_prev, a_com, omega_s)` | not exercised | [L174](../../../crawlbot/aocs/force_estimator.py#L174) |
| `.H_rO` | `()` | **yes** | [L223](../../../crawlbot/aocs/force_estimator.py#L223) |
| `.H_dot` | `()` | **yes** | [L228](../../../crawlbot/aocs/force_estimator.py#L228) |
| `.initialized` | `()` | not exercised | [L233](../../../crawlbot/aocs/force_estimator.py#L233) |
| `compute_aocs_command` | `(H_dot_est, omega_s, hw_current, hw_target=None, K_omega...)` | not exercised | [L243](../../../crawlbot/aocs/force_estimator.py#L243) |
| `compute_aocs_command_legacy_corrected` | `(L_com, L_com_prev, r_com, v_com, v_com_prev, hw_current...)` | not exercised | [L291](../../../crawlbot/aocs/force_estimator.py#L291) |
| `compute_aocs_command_legacy_pd_numerical` | `(L_com, L_com_prev, r_com, v_com, v_com_prev, omega_s, o...)` | not exercised | [L379](../../../crawlbot/aocs/force_estimator.py#L379) |
| `compute_aocs_command_legacy_pd_model` | `(L_com, L_com_prev, r_com, v_com, v_com_prev, omega_s, t...)` | not exercised | [L444](../../../crawlbot/aocs/force_estimator.py#L444) |
| `compute_aocs_command_legacy_pid_numerical` | `(L_com, L_com_prev, r_com, v_com, v_com_prev, omega_s, o...)` | **yes** | [L514](../../../crawlbot/aocs/force_estimator.py#L514) |
| `compute_aocs_command_legacy_pid_model` | `(L_com, L_com_prev, r_com, v_com, v_com_prev, omega_s, t...)` | not exercised | [L626](../../../crawlbot/aocs/force_estimator.py#L626) |

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

### 2b. Reading the law term by term (C2.3)

The law is a sum of five contributions, and until C2.3 only the sum left the
function — so no artifact could say which term produced a given wheel torque,
and the saturation was invisible. `decomposition`, an optional out-dict, now
records them:

| key | term | note |
|---|---|---|
| `tau_ff` | feedforward | FD-on-`L_com` in SS, contact-wrench couple in DS — see §3 |
| `tau_att_p` | `K_θ · θ_s` | attitude proportional |
| `tau_rate_d` | `K_ω · ω_s` | rate damping |
| `tau_accel_d` | `K_d · ω̇_s` | second-order damping, ω̇ by one-step FD |
| `tau_antiwindup` | `K_hw · (sat(h_w) − h_w)` | **identically zero while `|h_w| ≤ h_max`** |
| `tau_w_preclip` | their sum | before the ±`tau_w_max` clip |
| `tau_w` | `clip(tau_w_preclip)` | what is commanded |

Two properties make this worth having rather than re-deriving downstream:

- **The identity holds by construction.** The logged objects *are* the summands
  the function adds, so `Σ terms == tau_w_preclip` exactly. A reader checks it;
  they do not reproduce the arithmetic and hope the gains they assumed match
  the ones that ran.
- **`tau_w_preclip` vs `tau_w` is the only way to measure saturation.** The
  clip is where commanded torque becomes applied torque, and on the unmanaged
  run the controller demands up to 26.9 N·m against a ±2.5 plant cap. Logging
  only the clipped value hides the entire demand.

⚠ **Signs.** All three feedback terms enter with **`+`**, not the `−` a reader
expects from a textbook PD. That is deliberate and derived in the docstring
(`:545-548`): τ_w on the wheels produces −τ_w on the structure, so driving
`θ_s → 0` needs a *positive* contribution. It is a genuine divergence from how
the paper writes the law — see review-closure `C1_EXACTNESS.md` §C1.4.

⚠ **Only `legacy_pid_numerical` fills the dict.** The `_model` variants take
the same terms but were left alone (they are not the canonical mode and are
unexercised); their callers pass nothing and the recorder writes its sentinel.

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

## Code map

| unit | source |
|---|---|
| `class EstimatorConfig` | [L43-63](../../../crawlbot/aocs/force_estimator.py#L43-L63) |
| `class MomentumDisturbanceEstimator` | [L66-240](../../../crawlbot/aocs/force_estimator.py#L66-L240) |
| `MomentumDisturbanceEstimator.reset` | [L110-116](../../../crawlbot/aocs/force_estimator.py#L110-L116) |
| `MomentumDisturbanceEstimator.update` | [L118-172](../../../crawlbot/aocs/force_estimator.py#L118-L172) |
| `MomentumDisturbanceEstimator.update_analytical` | [L174-218](../../../crawlbot/aocs/force_estimator.py#L174-L218) |
| `MomentumDisturbanceEstimator.H_rO` | [L223-225](../../../crawlbot/aocs/force_estimator.py#L223-L225) |
| `MomentumDisturbanceEstimator.H_dot` | [L228-230](../../../crawlbot/aocs/force_estimator.py#L228-L230) |
| `MomentumDisturbanceEstimator.initialized` | [L233-234](../../../crawlbot/aocs/force_estimator.py#L233-L234) |
| `compute_aocs_command()` | [L243-288](../../../crawlbot/aocs/force_estimator.py#L243-L288) |
| `compute_aocs_command_legacy_corrected()` | [L291-376](../../../crawlbot/aocs/force_estimator.py#L291-L376) |
| `compute_aocs_command_legacy_pd_numerical()` | [L379-441](../../../crawlbot/aocs/force_estimator.py#L379-L441) |
| `compute_aocs_command_legacy_pd_model()` | [L444-511](../../../crawlbot/aocs/force_estimator.py#L444-L511) |
| `compute_aocs_command_legacy_pid_numerical()` | [L514-623](../../../crawlbot/aocs/force_estimator.py#L514-L623) |
| `compute_aocs_command_legacy_pid_model()` | [L626-684](../../../crawlbot/aocs/force_estimator.py#L626-L684) |

---

## See also

- package overview: [`aocs.md`](aocs.md)
