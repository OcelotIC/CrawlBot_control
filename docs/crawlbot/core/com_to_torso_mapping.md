# `crawlbot.core.com_to_torso_mapping`

**File**: [`crawlbot/core/com_to_torso_mapping.py`](../../../crawlbot/core/com_to_torso_mapping.py) — **257 lines** — canonical coverage **52 %**

> Module docstring: *"CoM-to-torso reference mapping (M1, v1: with delta_dot)."*

Converts a *centroidal* reference (what the NMPC produces) into a *torso*
reference (what the QP tracks), using an exact mass-weighted identity.

It exists so the whole-body QP can run a rigid-body pose task on the torso
instead of a centroidal task on the CoM — the two are exactly equivalent at the
Jacobian level, but the first is far better conditioned.

---

## Public API

| symbol | signature | canonical? | code |
|---|---|---|---|
| **`CoMToTorsoMapping`** |  |  | [L44](../../../crawlbot/core/com_to_torso_mapping.py#L44) |
| `.compute_delta` | `(q)` | **yes** | [L97](../../../crawlbot/core/com_to_torso_mapping.py#L97) |
| `.compute_delta_dot` | `(q, dq)` | **yes** | [L110](../../../crawlbot/core/com_to_torso_mapping.py#L110) |
| `.compute_delta_local` | `(q)` | not exercised | [L143](../../../crawlbot/core/com_to_torso_mapping.py#L143) |
| `.compute_delta_local_dot` | `(q, dq)` | not exercised | [L160](../../../crawlbot/core/com_to_torso_mapping.py#L160) |
| `.compute` | `(r_com_ref, v_com_ref, a_com_ff, q_current, dq_current=None)` | not exercised | [L181](../../../crawlbot/core/com_to_torso_mapping.py#L181) |
| `.body_com_jacobian` | `(data, joint_idx)` | **yes** | [L224](../../../crawlbot/core/com_to_torso_mapping.py#L224) |
| `.torso_pos_jacobian_from_com` | `(q)` | not exercised | [L239](../../../crawlbot/core/com_to_torso_mapping.py#L239) |

### Module constants

| name | value |
|---|---|
| `TORSO_JOINT_IDX` | `1` |

---

---

## 1. The identity

Split the total mass between the torso and everything else:

```
m_total * r_com = m_b * r_b + delta(q)
```

where

```
delta(q)         = sum_{i != torso} m_i * r_i(q)          (world frame)
delta_dot(q, dq) = sum_{i != torso} m_i * J_i(q) @ dq
```

Solving for the torso gives the mapping:

```
r_b_ref = (m_total/m_b) * r_com_ref - (1/m_b) * delta(q)
v_b_ref = (m_total/m_b) * v_com_ref - (1/m_b) * delta_dot(q, dq)
a_b_ff  = (m_total/m_b) * a_com_ff                    [delta_ddot dropped]
```

`delta_ddot` is deliberately dropped at v1: the PD term absorbs it, and the
second derivative of a mass-weighted sum over a 14-DOF arm set is both expensive
and noisy.

### Why it is exact, not an approximation

Differentiating the identity gives the Jacobian relation

```
J_b_pos = (m_total/m_b) * J_com - (1/m_b) * sum_{i != torso} m_i * J_i
```

so a torso-position task and a CoM-position task span the same row space. This
is verified by test T3. Nothing is lost by tracking the torso; the conditioning
improves because `J_b_pos` is a rigid-body Jacobian rather than a mass-weighted
sum.

Implementation: `compute_delta` (`:97`) walks the non-torso bodies and
accumulates `m_i * r_i` from `data.oMi[i].act(inertias[i].lever)`;
`compute_delta_dot` (`:110`) does the same with per-body translational
Jacobians in `LOCAL_WORLD_ALIGNED`.

## 2. The feedback loop this creates, and the loop-free variant

`delta(q)` in the world frame contains the base position:

```
delta = m_arms * r_base + D_local
```

So the torso position being *controlled* re-enters its own *reference*. Fed with
live `q_current` at the 100 Hz QP rate, this closes a mapping -> q -> mapping
loop that was measured oscillating `r_b_ref` by up to **237 mm per tick** on
large swings.

Two mitigations exist in the code:

**F-RATE (canonical)** — recompute `delta` and `delta_dot` **once per NMPC tick**
(10 Hz) instead of per QP tick. The interpolated `(m_total/m_b) * r_com` term
still varies smoothly at 100 Hz. See the comment block at
`sim_loop.py:2584-2596`.

**Loop-free reformulation (implemented, not canonical)** —
`compute_delta_local` uses

```
D_local(q) = sum_{i != torso} m_i * (r_i - r_torso)
```

which is invariant to base *translation* (both CoMs translate together), giving
the exact and loop-free identity `r_b = r_com - D_local/m_total`. It carries no
base-position term at all, so live joint angles can be fed without jitter. It is
unexercised on the canonical.

## 3. ⚠ This is a DS-only path

Explicit project rule:

> *Do not route the SS torso reference through the delta-mapping in two-task
> mode — SS uses the raw TorsoPlanner quintic (`sim_loop.py:2581-2584`); the
> mapping remains a DS-only path.*

In single support the QP's torso-pose task is fed the planner's raw
quintic+SLERP directly. That is why the coverage looks the way it does:
`compute()` — the full mapping entry point — is **unexercised**, while the
building blocks `compute_delta` and `compute_delta_dot` are live, called from
the DS path.

`TORSO_JOINT_IDX = 1` because in a free-flyer URDF joint 1 is the root joint and
its body is the torso.

## Code map

| unit | source |
|---|---|
| `class CoMToTorsoMapping` | [L44-256](../../../crawlbot/core/com_to_torso_mapping.py#L44-L256) |
| `CoMToTorsoMapping.compute_delta` | [L97-108](../../../crawlbot/core/com_to_torso_mapping.py#L97-L108) |
| `CoMToTorsoMapping.compute_delta_dot` | [L110-127](../../../crawlbot/core/com_to_torso_mapping.py#L110-L127) |
| `CoMToTorsoMapping.compute_delta_local` | [L143-158](../../../crawlbot/core/com_to_torso_mapping.py#L143-L158) |
| `CoMToTorsoMapping.compute_delta_local_dot` | [L160-175](../../../crawlbot/core/com_to_torso_mapping.py#L160-L175) |
| `CoMToTorsoMapping.compute` | [L181-218](../../../crawlbot/core/com_to_torso_mapping.py#L181-L218) |
| `CoMToTorsoMapping.body_com_jacobian` | [L224-237](../../../crawlbot/core/com_to_torso_mapping.py#L224-L237) |
| `CoMToTorsoMapping.torso_pos_jacobian_from_com` | [L239-256](../../../crawlbot/core/com_to_torso_mapping.py#L239-L256) |

---

## See also

- package overview: [`core.md`](core.md)
