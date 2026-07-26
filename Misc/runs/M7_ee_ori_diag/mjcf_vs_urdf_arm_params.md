# MJCF vs URDF — arm B parameter audit

Side-by-side parameter dump for the swing arm (B) between
`models/VISPA_crawling_rwa3.xml` (MuJoCo) and
`models/VISPA_crawling_fixed.urdf` (Pinocchio). Read-only, no
interpretation. Indices in joints-only space (length 14): arm A occupies
0..6, arm B occupies 7..13. The wrist segment is `Joint_5_b` and
`Joint_6_b` at indices 12 and 13.

## 1. Arm B joint dynamic parameters

All seven `arm B` joints in the MJCF inherit `class="robot_joint"` and
have **no per-joint overrides**:

```xml
<default class="robot_joint">
  <joint damping="0.05" armature="0.05" limited="true"
         range="-3.14159 3.14159"/>
  ...
</default>
```

The URDF carries **no `<dynamics>` tag on any of the seven arm B joints**
(grep across the file returns zero matches for `<dynamics`,
`damping`, `friction`, `armature`, `stiffness`).

| joint idx (B-only) | joint idx (joints-only) | name           | MJCF damping | MJCF armature | MJCF frictionloss | MJCF stiffness | URDF damping | URDF friction |
|---|---|---|---|---|---|---|---|---|
| 0 |  7 | `Joint_1_b`     | 0.05 | 0.05 | 0 (default) | 0 (default) | 0 (default) | 0 (default) |
| 1 |  8 | `Joint_2_b`     | 0.05 | 0.05 | 0           | 0           | 0           | 0 |
| 2 |  9 | `Joint_swivel_b`| 0.05 | 0.05 | 0           | 0           | 0           | 0 |
| 3 | 10 | `Joint_3_b`     | 0.05 | 0.05 | 0           | 0           | 0           | 0 |
| 4 | 11 | `Joint_4_b`     | 0.05 | 0.05 | 0           | 0           | 0           | 0 |
| 5 | 12 | `Joint_5_b`     | 0.05 | 0.05 | 0           | 0           | 0           | 0 |
| 6 | 13 | `Joint_6_b`     | 0.05 | 0.05 | 0           | 0           | 0           | 0 |

Common joint definition (MJCF): `type="hinge"`, `range="-π, π"`, `limited="true"`.
Common joint definition (URDF): `type="revolute"`, `lower="-π"`, `upper="π"`,
`effort="50"`, `velocity="0.09395689"`.

## 2. Arm B link inertials

MJCF `fullinertia` ordering is `(ixx, iyy, izz, ixy, ixz, iyz)`. URDF
`<inertia>` uses named attributes `ixx, iyy, izz, ixy, ixz, iyz`. Tables
below compare each scalar entry; relative deltas are computed as
`|MJCF − URDF| / max(|MJCF|, |URDF|, 1e-12)` and rounded to four
decimals.

### Link_1_b

| field | MJCF | URDF | rel. Δ |
|---|---|---|---|
| mass     | 2.328 | 2.328 | 0.0000 |
| pos / origin x | 0.0     | 0.0     | 0.0000 |
| pos / origin y | -0.029  | -0.029  | 0.0000 |
| pos / origin z | -0.043  | -0.043  | 0.0000 |
| ixx      | 0.012  | 0.012  | 0.0000 |
| iyy      | 0.007  | 0.007  | 0.0000 |
| izz      | 0.008  | 0.008  | 0.0000 |
| ixy      | 0      | 0      | 0.0000 |
| ixz      | 0      | 0      | 0.0000 |
| iyz      | 0.003  | 0.003  | 0.0000 |

### Link_2_b

| field | MJCF | URDF | rel. Δ |
|---|---|---|---|
| mass     | 3.995 | 3.995 | 0.0000 |
| pos / origin | (0.25, 0, 0.142) | (0.25, 0, 0.142) | 0.0000 |
| ixx      | 0.010 | 0.010 | 0.0000 |
| iyy      | 0.578 | 0.578 | 0.0000 |
| izz      | 0.574 | 0.574 | 0.0000 |
| ixy/ixz/iyz | 0/0/0 | 0/0/0 | 0.0000 |

### Link_2_swivel_b

| field | MJCF | URDF | rel. Δ |
|---|---|---|---|
| mass     | 0.001 | 0.001 | 0.0000 |
| pos / origin | (0, 0, 0) | (0, 0, 0) | 0.0000 |
| ixx/iyy/izz | 1e-6/1e-6/1e-6 | 1e-6/1e-6/1e-6 | 0.0000 |
| ixy/ixz/iyz | 0/0/0 | 0/0/0 | 0.0000 |

### Link_3_b

| field | MJCF | URDF | rel. Δ |
|---|---|---|---|
| mass     | 2.328 | 2.328 | 0.0000 |
| pos / origin | (0, 0.043, 0.029) | (0, 0.043, 0.029) | 0.0000 |
| ixx      | 0.012 | 0.012 | 0.0000 |
| iyy      | 0.008 | 0.008 | 0.0000 |
| izz      | 0.007 | 0.007 | 0.0000 |
| ixy      | 0     | 0     | 0.0000 |
| ixz      | 0     | 0     | 0.0000 |
| iyz      | 0.003 | 0.003 | 0.0000 |

### Link_4_b

| field | MJCF | URDF | rel. Δ |
|---|---|---|---|
| mass     | 3.157 | 3.157 | 0.0000 |
| pos / origin | (0, -0.043, -0.210) | (0, -0.043, -0.210) | 0.0000 |
| ixx      | 0.199 | 0.199 | 0.0000 |
| iyy      | 0.193 | 0.193 | 0.0000 |
| izz      | 0.01  | 0.01  | 0.0000 |
| ixy      | 0     | 0     | 0.0000 |
| ixz      | 0     | 0     | 0.0000 |
| iyz      | 0.01  | 0.01  | 0.0000 |

### Link_5_b

| field | MJCF | URDF | rel. Δ |
|---|---|---|---|
| mass     | 2.695 | 2.695 | 0.0000 |
| pos / origin | (0, 0.125, -0.007) | (0, 0.125, -0.007) | 0.0000 |
| ixx      | 0.029  | 0.029  | 0.0000 |
| iyy      | 0.006  | 0.006  | 0.0000 |
| izz      | 0.027  | 0.027  | 0.0000 |
| ixy      | 0      | 0      | 0.0000 |
| ixz      | 0      | 0      | 0.0000 |
| iyz      | -0.001 | -0.001 | 0.0000 |

### Link_6_b

| field | MJCF | URDF | rel. Δ |
|---|---|---|---|
| mass     | 0.924 | 0.924 | 0.0000 |
| pos / origin | (0, 0, 0) | (0, 0, 0) | 0.0000 |
| ixx      | 0.002 | 0.002 | 0.0000 |
| iyy      | 0.002 | 0.002 | 0.0000 |
| izz      | 0.001 | 0.001 | 0.0000 |
| ixy/ixz/iyz | 0/0/0 | 0/0/0 | 0.0000 |

### tool_b

| field | MJCF | URDF | rel. Δ |
|---|---|---|---|
| mass     | 0.1 | 0.1 | 0.0000 |
| pos / origin | (0, 0, 0) | (0, 0, 0) | 0.0000 |
| ixx/iyy/izz | 0.0001/0.0001/0.0001 | 0.0001/0.0001/0.0001 | 0.0000 |
| ixy/ixz/iyz | 0/0/0 | 0/0/0 | 0.0000 |

## 3. Deltas (suspect list per the task spec)

Suspects per the spec criteria: damping/armature/frictionloss present in
the MJCF but absent or zero in the URDF; or any inertia value differing
by more than 1 %.

| suspect | applies to | MJCF value | URDF value |
|---|---|---|---|
| `damping`   | every arm B joint (`Joint_1_b` … `Joint_6_b`, `Joint_swivel_b`) | 0.05 | 0 (no `<dynamics>` tag) |
| `armature`  | every arm B joint                                              | 0.05 | not representable in URDF (no equivalent attribute), effective 0 in Pinocchio |
| `frictionloss` | every arm B joint                                          | 0 (default) | 0 | — *(not a suspect: identical)* |
| `stiffness` | every arm B joint                                              | 0 (default) | 0 | — *(not a suspect: identical)* |

No link-inertia suspects: every mass, origin, and inertia component
matches between the two files (rel. Δ = 0 within file precision).

For symmetry of record, the same MJCF-only suspects apply identically
to all seven `arm A` joints (they share the same `class="robot_joint"`
default).

## 4. Stance-arm A weld parameters (active during A_swing)

A_swing setup uses `start_a = 2`, so the stance weld engaged at run
time is `grip_a_to_2a` (sim_loop deactivates all welds at setup and
re-activates the start-pair). All `grip_a_to_*` welds in the MJCF share
identical solver settings:

```xml
<weld name="grip_a_to_1a" site1="gripper_a" site2="anchor_1a" solref="0.003 1" active="false"/>
<weld name="grip_a_to_2a" site1="gripper_a" site2="anchor_2a" solref="0.003 1" active="false"/>
<weld name="grip_a_to_3a" site1="gripper_a" site2="anchor_3a" solref="0.003 1" active="true"/>
<weld name="grip_a_to_4a" site1="gripper_a" site2="anchor_4a" solref="0.003 1" active="false"/>
<weld name="grip_a_to_5a" site1="gripper_a" site2="anchor_5a" solref="0.003 1" active="false"/>
<weld name="grip_a_to_6a" site1="gripper_a" site2="anchor_6a" solref="0.003 1" active="false"/>
```

| weld param | value | source |
|---|---|---|
| `solref` | `0.003 1`     | explicit on the `<weld>` element |
| `solimp` | not present   | inherits MuJoCo default `0.9 0.95 0.001 0.5 2` |
| `relpose` | not present  | inherits default `0 0 0 1 0 0 0` (identity 6D) |
| `active` (at MJCF compile time) | `true` only on `grip_a_to_3a`; sim_loop overrides this at setup | MJCF + `sim_loop._activate_weld` |

Global solver context (relevant to weld stabilisation behaviour):

```xml
<option gravity="0 0 0" timestep="0.001" integrator="RK4">
  <flag energy="enable" contact="enable"/>
</option>
```

`<size>` block: not present (defaults). `impratio`, `o_solref`,
`o_solimp`, `noslip_iterations`: not present (defaults).
