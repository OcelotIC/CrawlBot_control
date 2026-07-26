# `crawlbot.core`

**Foundation layer.**

| file | lines | canonical coverage | document |
|---|---:|---:|---|
| `ik.py` | 1468 | 40 % | [ik.md](ik.md) |
| `robot_interface.py` | 460 | 87 % | [robot_interface.md](robot_interface.md) |
| `com_to_torso_mapping.py` | 257 | 52 % | [com_to_torso_mapping.md](com_to_torso_mapping.md) |
| `state_conversions.py` | 232 | **100 %** | [state_conversions.md](state_conversions.md) |

## Role

Everything above this layer assumes a consistent robot state expressed in the
**structure frame R_s**. This package is what produces it:

- `state_conversions` performs the MuJoCo(world) <-> Pinocchio(R_s) change of
  frame, and owns the quaternion conventions;
- `robot_interface` computes, in one Pinocchio pass, every quantity the tick
  needs — dynamics, centroidal terms, Jacobians;
- `ik` picks docking configurations out of a redundant solution space;
- `com_to_torso_mapping` converts centroidal references into torso references.

## Two traps documented in detail

**Quaternion conventions differ** between the two libraries (Pinocchio `xyzw`,
MuJoCo `wxyz`). `state_conversions.py` is the reference and every conversion must
go through it.

**`robot_interface`'s module constants are rebound at construction.**
`from ... import NQ` freezes a stale 6-DOF value of 19; the real value is 21.
Use instance attributes.

## The largest verification blind spot

`ik.py` is 40 % covered: six of its nine public functions carry **no gate
coverage at all**, having been orphaned when CLEANUP-15 removed the FK-reference
path from `sim_loop`. Changes there will trip nothing.
