# `crawlbot.aocs.force_estimator`

**File**: [`crawlbot/aocs/force_estimator.py`](../../../crawlbot/aocs/force_estimator.py) — **103 lines** — canonical coverage **88 %**

> Module docstring: *"MomentumDisturbanceEstimator — Estimate the disturbance torque applied by"*

The AOCS law — one law, the one the paper uses (`legacy_pid_numerical`).

This is the other half of the decentralised contract: the NMPC promises never to
demand more than `tau_w_max`; this law is what actually spends it. The block that
drives it (history, attitude error, wrench feedforward) is
[`control/attitude.md`](../control/attitude.md).

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

One law is implemented, `legacy_pid_numerical`. Its five variants (legacy,
legacy_corrected, legacy_pd_numerical/model, legacy_pid_model) and the H_{r/O}
estimator law (H_est) were never run by the canonical nor by the paper's Table 2
configurations and were retired (R1 — `results/j2_adjconv/PHASE_R1_AOCS_MODES_RETIRED.md`).

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

### Single support: kinematic finite differences

```
ff = -L_com_dot - r_com x ( m_r * v_com_dot )       (both by FD)
```

Valid while the robot is kinematically free at the contact.

### Double support: contact-wrench feedforward

With both grippers welded, the closed loop carries internal stress. It exerts on
the structure a couple

```
( r_CA - r_CB ) x f
```

that is **invisible in `L_com`** — the two contact forces cancel in the
momentum balance while their moments do not. The kinematic feedforward is
therefore structurally incomplete in DS, not merely noisy.

`control/attitude.py` computes the correct term straight from the QP solution:

```
tau_struct_ff = - sum_i ( r_Ci x f_i + tau_i )
```

and passes it in, short-circuiting the FD branch.

This is the only difference in AOCS treatment between DS and SS.

## 4. `MomentumDisturbanceEstimator` — retired (R1)

It was constructed and its `H_rO` / `H_dot` properties were read every tick for
the log, but `update()` was never called: both channels were identically zero
over all 2077 ticks. It was retired with the other modes; the log channels
`H_rO` / `H_dot_est` are kept (schema unchanged) and written as zeros directly —
bit-identical. Its theory (EMA-filtered finite differences on H_{r/O}, or the
analytical variant via `a_com`) is in git history if a disturbance estimator is
ever wanted; note the ordering it got right — filter before differentiating.

## Code map

| unit | source |
|---|---|
| `compute_aocs_command_legacy_pid_numerical()` | [L21-102](../../../crawlbot/aocs/force_estimator.py#L21-L102) |

---

## See also

- package overview: [`aocs.md`](aocs.md)
