# INTERNAL — Structure-inertia mini-calc (READ-ONLY, `ae0673e`)

Inertias from the canonical model (`models/VISPA_crawling_rwa3.xml`), to convert the dock residual angular
momentum [N·m·s] → structure rotation rate [rad/s, °/s]. No sim run (model read + one eval at a committed
dock snapshot). Reproducer: `scripts/calc_struct_inertia.py`. **You do the final conversion** — numbers below.

## Table

| quantity | value | frame / point |
|---|---|---|
| **structure mass** `m_struct` | **7110.0 kg** | body id=1 "structure" |
| **structure inertia — principal** | **[1777, 1493, 597] kg·m²** | principal frame, about structure CoM (= body origin, `ipos=0`) |
| structure `body_iquat` | `[0, 0.7071, 0, 0.7071]` (wxyz) = **90° about Y** | principal→body |
| **structure inertia — BODY frame** | **diag([597, 1493, 1777]) kg·m²** (`Ixx,Iyy,Izz`) | structure body frame, about CoM (off-diag = 0) |
| **composite SYSTEM inertia** `I_sys` | diag **[631, 1712, 2009] kg·m²**; principal **[2011, 1716, 626]**; max off-diag 60 | **WORLD frame**, about **system CoM**, at the `final` dock snapshot |
| — structure-alone about system CoM | diag [597, 1495, 1779] kg·m² | system (arms+wheels) adds **+12.4%** to the trace |
| system CoM ↔ structure CoM offset | 0.018 m | tiny parallel-axis lever (robot = 71 kg vs 7110 kg) |
| **residual `subtree_angmom[0]`** | **[−0.00296, 0.00057, 0.00259] N·m·s**, ‖·‖=0.003977 | WORLD frame, about system CoM |
| **dominant residual axis** | **X** (\|L_x\|=0.00296, **75%** of norm; L_z=0.00259 comparable) | world |

**Frame/point notes.** `subtree_angmom[0]` is the **total-system** angular momentum about the **system CoM**,
in the **world frame** (validated as ground truth in the J1 work) — so the inertia that converts it to a rate
is `I_sys` (world, about system CoM), **not** the structure's principal inertia. At dock the attitude error
is <1°, so world ≈ structure body frame; the welded robot (71 kg + arms extended) adds ~12% to the structure's
inertia. The structure's own principal values `[1777, 1493, 597]` are in a frame rotated 90° about Y from the
body frame, so in the body/world frame the order is **`Ixx=597 (light) … Izz=1777 (heavy)`** — the residual's
dominant world-X axis is the **light** axis.

## Reference angular-momentum + envelope (already measured — not re-measured)

| | value |
|---|---|
| Fix-A residual (ae0673e, traversal-final) | ≈ **0.003977 N·m·s** |
| Fix-C residual @ ε_twist=0.005 | ≈ **0.001005 N·m·s** |
| `τ_w,max` (wheel-torque rate cap) | **5.0 N·m** per axis |
| `h_max` (wheel stored momentum) | **±5.0 N·m·s** per axis (robot `L_max` = 10) |

## For reference only (you do the conversion) — proper `ω = I_sys⁻¹·L`

The residual is **not** axis-aligned (X≈Z in magnitude) and `I_xx=631 ≪ I_zz=2009`, so use the full-tensor
solve, **not** `‖L‖/I_x`:

- **Fix-A** (vector known): `ω = [−4.84, 0.51, 1.39]×10⁻⁶ rad/s = [−2.77, 0.29, 0.80]×10⁻⁴ °/s`,
  **‖ω‖ ≈ 2.90×10⁻⁴ °/s**. X-dominated (small `I_xx`).
- **Fix-C** (only the norm 0.001005 retained, ≈ same direction → ×0.253): **‖ω‖ ≈ 7.3×10⁻⁵ °/s**.

Both are microscopic structure rates (sub-mdeg/s). The conversion to *settled drift over a window* and the
comparison against any AOCS/pointing budget is yours — these are the inertia + residual numbers to do it.

**Caveat:** the residual **direction** is from the `fixA_gate` run (‖L‖=0.003977, = the pose-gated /
ε_twist≥0.007 Fix-C residual); the ε_twist=0.005 run kept only the **norm** (0.001005), so its vector
direction is assumed equal — treat the 75%-X split as representative, not exact for the 0.001005 case.

**STOP** — read-only mini-calc; you do the momentum→rate conversion. No modification, no `main`, no PR.
