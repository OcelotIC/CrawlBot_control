# Two-Layer Control Architecture for the SpaceServicer on ASTROHUB

**Date:** 2026-04-09
**Authors:** Idriss
**Working notes assisted by:** Claude (Anthropic)

---

## 0. Conventions and Notation

### 0.1 System Terminology

| Term | Definition |
|---|---|
| **ASTROHUB** | The free-floating platform (spacecraft with AOCS/RWA) on which the robot crawls |
| **SpaceServicer** | The complete crawling robot: torso + two VISPA arms |
| **VISPA** | A single 7-DOF anthropomorphic arm (Airbus DS), two per SpaceServicer |
| **Torso** | The central body connecting both arms; the floating base in Pinocchio |
| **End-effector (EE)** | The terminal link of each arm, equipped with a docking interface (HOTDOCK SI) |
| **Arm 1 / Arm 2** | Fixed labels for the two VISPA arms (system description) |
| **Stance arm / Swing arm** | Role-based labels: stance = welded to platform; swing = free to move (control/planning) |
| **Weld** | A rigid bilateral holonomic constraint between an EE and a platform docking port |

### 0.2 Frames

| Symbol | Name | Origin | Description |
|---|---|---|---|
| $\mathcal{W}$ | World (inertial) | — | MuJoCo world frame, ECI-like |
| $\mathcal{P}$ | Platform body | $O_p$ | Rigidly attached to ASTROHUB, at platform CoM |
| $\mathcal{T}$ | Torso body | $O_t$ | Rigidly attached to the SpaceServicer torso |

### 0.3 Superscript/Subscript Convention

Following the Siciliano convention (Siciliano et al., 2009), the **left superscript** denotes the frame in which a vector is expressed:

$${}^{\mathcal{P}}\mathbf{v}_\text{com} \quad \text{means: CoM velocity, coordinates in platform frame } \mathcal{P}$$

**Subscripts** identify the quantity: $\text{com}$ (center of mass), $\text{ee}$ (end-effector), $p$ (platform), $t$ (torso), $w$ (reaction wheels), $1$/$2$ (arm index).

**Superscripts on the right** carry additional qualifiers:

| Qualifier | Meaning |
|---|---|
| $\text{rel}$ | Relative to the platform (velocities measured w.r.t. platform motion) |
| $\text{in}$ | Inertial (absolute velocities in the world frame, then projected into $\mathcal{P}$) |
| $\text{ref}$ | Reference / desired value |
| $\text{plan}$ | Planned by the NMPC |
| $\text{est}$ | Estimated (e.g., finite-difference derivative) |

**Default convention (applies unless marked otherwise):**

> All quantities are **relative** to the platform and **expressed in the platform frame** $\mathcal{P}$, with origin at the platform CoM $O_p$.

When the default applies, the left superscript ${}^{\mathcal{P}}$ and the right qualifier ${}^\text{rel}$ are **dropped** for readability. For example:

$$\mathbf{v}_\text{com} \equiv {}^{\mathcal{P}}\mathbf{v}_\text{com}^\text{rel}$$

The full heavy notation ${}^{\mathcal{P}}\mathbf{L}_{\text{robot}/O_p}^\text{in}$ (angular momentum of the robot about $O_p$, inertial, in platform-frame coordinates) is used only when the default does not apply — specifically when discussing the conservation law (§3.5) and the inertial momentum decomposition (§3.4), where both relative and inertial quantities coexist in the same equation.

**Angular momentum reference point:** $\mathbf{L}_{\text{robot}/O_p}$ means "angular momentum of the robot computed about $O_p$." When the reference point is the robot's own CoM, it is written $\mathbf{L}_\text{com}$.

### 0.4 Symbol Glossary

| Symbol | Definition | Dimension | Unit |
|---|---|---|---|
| $\mathbf{q}$ | Joint angles (both arms) | $n_a \times 2$ | rad |
| $\dot{\mathbf{q}}$ | Joint velocities | $n_a \times 2$ | rad/s |
| $\mathbf{q}_\text{full}$ | Generalized velocity $[\mathbf{v}_t;\,\dot{\mathbf{q}}]$ | $n_v$ | m/s, rad/s |
| $\mathbf{r}_\text{com}$ | Robot CoM position in $\mathcal{P}$ | 3 | m |
| $\mathbf{v}_\text{com}$ | Robot CoM velocity (relative, in $\mathcal{P}$) | 3 | m/s |
| $\mathbf{L}_\text{com}$ | Centroidal angular momentum (relative, in $\mathcal{P}$) | 3 | N·m·s |
| $\mathbf{h}_w$ | RWA angular momentum in $\mathcal{P}$ | 3 | N·m·s |
| $h_\text{max}$ | RWA saturation limit (per axis) | 1 | N·m·s |
| $\boldsymbol{\tau}_q$ | Joint torques | $n_a \times 2$ | N·m |
| $\boldsymbol{\tau}_w$ | Wheel torque command | 3 | N·m |
| $\mathbf{f}_j$ | Contact force at contact $j$ | 3 | N |
| $\boldsymbol{\tau}_j$ | Contact torque at contact $j$ | 3 | N·m |
| $\mathbf{H}$ | Joint-space inertia matrix | $n_v \times n_v$ | kg·m² |
| $\mathbf{C}$ | Coriolis/centrifugal matrix | $n_v \times n_v$ | kg·m²/s |
| $\mathbf{A}$ | Centroidal momentum matrix | $6 \times n_v$ | mixed |
| $T$ | Kinetic energy (relative) | 1 | J |
| $\alpha$ | Passivity decay rate | 1 | 1/s |
| $w_r, w_v, w_L, w_u$ | NMPC cost weights | 1 | (dimensionless) |
| $N$ | NMPC horizon length | 1 | — |
| $\Delta T$ | NMPC time step | 1 | s |
| $T_\text{step}$ | Duration of one locomotion step | 1 | s |

### 0.5 System Dimensions

| Configuration | $n_q$ | $n_v$ | $n_u$ (joints) | Free DOFs in SS | Free DOFs in DS |
|---|---|---|---|---|---|
| 6-DOF VISPA (current) | 19 | 18 | 12 | 12 | 6 |
| 7-DOF VISPA (target) | 21 | 20 | 14 | 14 | 8 |

$n_q$: generalized positions (7 base + $2 n_a$ joints). $n_v$: generalized velocities (6 base + $2 n_a$ joints). Free DOFs: $n_v - 6 n_\text{welds}$.

| Controller | State dim | Control dim | Frequency |
|---|---|---|---|
| NMPC (B2) | 9 | 12 | ~10 Hz |
| Whole-body QP | — (acceleration-level) | $n_v + 6 n_c$ | 100 Hz |
| AOCS | — | 3 | 100 Hz |

### 0.6 Conventions for Thresholds and Parameters

- Equations use **radians** for angles and angular velocities.
- Human-readable thresholds (dock tolerances, error budgets) are stated in **degrees** and explicitly marked (e.g., "5°").
- All units are **SI** (m, kg, s, rad, N, N·m, N·m·s, J).

### 0.7 Notation in Equations

Throughout the body of this document, the subscript $s$ in equations (e.g., $\mathbf{r}_\text{com}^s$, $\boldsymbol{\omega}_s^s$, $\mathbf{R}_s$) refers to the **platform** frame $\mathcal{P}$ and the platform body. This legacy notation is retained from the development phase for consistency with the codebase (where the platform is called `structure`). In the publication, all $s$ subscripts/superscripts will be migrated to the Siciliano left-superscript convention defined in §0.3.

**Reading guide for equations:** $\mathbf{r}_\text{com}^s \equiv {}^{\mathcal{P}}\mathbf{r}_\text{com}$, $\boldsymbol{\omega}_s^s \equiv {}^{\mathcal{P}}\boldsymbol{\omega}_p$, $\mathbf{L}_\text{com}^{s,\text{rel}} \equiv \mathbf{L}_\text{com}$ (under the default convention).

---

## 1. Problem Statement


Multi-step docking fails due to three interacting issues:

1. **Stale NMPC warm start** after dock events (topology change invalidates primal/dual solution)
2. **CoM task competing with EE approach** during close approach (both at high priority)
3. **46° orientation error at dock** caused by a DOF budget deficit

### DOF Budget Analysis (Single-Support, 6-DOF arms → 7-DOF arms)

With 7-DOF arms: 14 free DOFs in single-support (20 total − 6 stance weld).

| Task | DOFs |
|---|---|
| CoM position | 3 |
| Torso 6D (pos + ori) | 6 |
| EE 6D (pos + ori) | 6 |
| **Total demand** | **15** |
| **Available** | **14** |
| **Deficit** | **1** |

The QP sacrifices EE orientation (priority 2, effective weight α/1000) while CoM and torso are priority 1.

### Key Insight

CoM position and torso position are strongly coupled (torso is ~80% of total mass). The 3-DOF CoM task is nearly redundant with the translational part of the 6-DOF torso task. **Solution:** map the NMPC's CoM reference into a torso position reference, eliminating the separate CoM task.


---

## 2. Related Work and Positioning

### Terrestrial Humanoid MPC (Belvedere et al.)


The reworked architecture converges toward the same structure used in Belvedere et al.:

| Aspect | Belvedere et al. (JIS-MPC) | SpaceServicer (reworked) |
|---|---|---|
| Stage 1 | Centroidal MPC (CoM/DCM) | Centroidal NMPC (CoM + momentum) |
| Mapping | CoM plan → base pose reference | CoM plan → torso pose reference (mass-weighted) |
| Stage 2 | Whole-body QP (base + swing foot) | Whole-body QP (torso + swing end-effector) |
| Balance | ZMP ∈ support polygon | $\mathbf{h}_w \in$ RWA polytope |
| Momentum in Stage 1 | In ZMP/DCM constraint | In $\mathbf{h}_w$ box + rate constraint |
| Momentum in Stage 2 | Not explicitly tracked | Soft backup constraint on $\mathbf{h}_w$ |
| Angular momentum in feasibility | Neglected ($M_{lx} = M_{ly} = 0$), flagged as future work | Included via orbital correction  |

 Even in terrestrial humanoid MPC, ignoring angular momentum in the feasibility analysis is recognized as a limitation (Belvedere et al., Section VI). This strengthens the argument that omission of centroidal angular momentum in prior space robotics work (Rognant et al.) is a real gap.


### Mishra, De Stefano & Ott (2022) — Passivity for in-orbit crawling

Passivity-based unified control for the MIRROR MAR system (same VISPA arms, HOTDOCK interfaces). Establishes Lyapunov stability for bilateral-contact crawling ($\dot{W} \leq 0$). **Gap:** no momentum management, structure assumed inertially fixed. **Our extension:** centroidal NMPC + passivity-constrained QP with guaranteed settling rate.

### Kalaycioglu & De Ruiter (2023) — Passivity-based NMPC

PNMPC for free-flying dual-arm spacecraft. Embeds passivity constraints (eq. 14–15) and terminal storage function (eq. 17) into NMPC for closed-loop stability. Following Raff et al. (2007). **Gap:** no contacts, no locomotion, no RWA. **Our extension:** passivity-NMPC concept applied to constrained crawling with conservation-law momentum box.

### Rognant et al. (2019, 2025) — Position-dependent velocity bound

Position-dependent velocity bound for assembly planning (eq. 6). Same VISPA use case. **Gap:** centroidal term dropped, scalar bound, no closed-loop validation. **Our extension:** analytical feasibility envelope (elliptic cylinder), closed-loop NMPC realization, coarse pre-planner.

### Contribution Positioning Table

| Capability | Mishra (2022) | Kalaycioglu (2023) | Rognant (2019, 2025) | **This work** |
|---|---|---|---|---|
| Passivity proof | Lyapunov (impedance) | Storage fn. in NMPC | — | Constraint in QP |
| MPC framework | — | PNMPC | — | Centroidal NMPC |
| Momentum mgmt | — | — | Scalar bound | Conservation-law box |
| Contact model | Bilateral (walking) | None | Single-step | Multi-step + transitions |
| Position-dep. feasibility | — | — | Identified | Analytical + planner |
| Energy-based settling | Asymptotic | Terminal cost | — | Exponential rate |

---

## 3. Theoretical Foundations


### Frame Conventions and Sensor Model

#### Frames

| Symbol | Frame | Description |
|---|---|---|
| $\mathcal{W}$ | World (inertial) | MuJoCo world frame, ECI-like |
| $\mathcal{P}$ | Platform | Origin $O_p$ at ASTROHUB CoM, rigidly attached |
| $\mathcal{T}$ | Torso | Origin $O_t$, rigidly attached to SpaceServicer torso |

Poses: $g_{ws} = (\mathbf{R}_s, \mathbf{p}_s) \in SE(3)$ is the platform in $\mathcal{W}$. The robot base in $\mathcal{P}$ is $g_{sb} = (\mathbf{R}_b^s, \mathbf{p}_b^s)$ where $\mathbf{R}_b^s = \mathbf{R}_s^\top \mathbf{R}_t$ and $\mathbf{p}_b^s = \mathbf{R}_s^\top(\mathbf{p}_t - \mathbf{p}_s)$.

#### Sensor Conventions

Due to sensor fidelity (gyroscopes measure body-frame angular velocity; navigation provides inertial linear velocity):

| Quantity | Coordinates | Source |
|---|---|---|
| $\boldsymbol{\omega}_s^s$ | platform body $\mathcal{P}$ | gyro on platform |
| $\boldsymbol{\omega}_t^t$ | torso body $\mathcal{B}$ | gyro on torso |
| $\mathbf{v}_s^w = \dot{\mathbf{p}}_s$ | inertial $\mathcal{W}$ | navigation |
| $\mathbf{v}_t^w = \dot{\mathbf{p}}_t$ | inertial $\mathcal{W}$ | navigation |

The controller operates entirely in platform frame $\mathcal{P}$. The SE(3) adjoint maps are used to rigorously transport quantities between frames.

---

### Relative Twist and Pinocchio Interface

The `mujoco_to_pinocchio` function computes the *relative* twist of the torso w.r.t. the platform, expressed in platform-frame coordinates:

$$\boldsymbol{\omega}_\text{rel}^s = \mathbf{R}_b^s\,\boldsymbol{\omega}_t^t - \boldsymbol{\omega}_s^s$$

$$\mathbf{v}_\text{rel}^s = \mathbf{R}_s^\top\!\left(\mathbf{v}_t^w - \mathbf{v}_s^w\right) - \boldsymbol{\omega}_s^s \times \mathbf{p}_b^s$$

These are fed to Pinocchio as the floating-base velocity. All subsequent quantities from Pinocchio — CoM position/velocity, Jacobians, centroidal momentum — are therefore computed from **relative** velocities in platform frame.

---

### Centroidal Momentum (Relative)

The centroidal momentum matrix $\mathbf{A}^s(\mathbf{q}) \in \mathbb{R}^{6 \times n_v}$ (Orin & Goswami, 2008) maps the relative generalized velocity to the relative centroidal momentum (using [angular; linear] convention):

$$\mathbf{h}_\text{com}^{s,\text{rel}} = \mathbf{A}^s(\mathbf{q}) \begin{bmatrix} \boldsymbol{\omega}_\text{rel}^s \\ \mathbf{v}_\text{rel}^s \\ \dot{\mathbf{q}} \end{bmatrix} = \begin{bmatrix} \mathbf{L}_\text{com}^{s,\text{rel}} \\ m\,\mathbf{v}_\text{com}^{s,\text{rel}} \end{bmatrix}$$

where $\mathbf{L}_\text{com}^{s,\text{rel}}$ is the angular momentum of the robot about its own CoM, computed from velocities **relative to the platform**. This is what Pinocchio returns via `pin.computeCentroidalMomentum()`. It is **not** the inertial momentum.

The rate form connects to the QP's acceleration-level formulation:

$$\dot{\mathbf{h}}_\text{com}^{s,\text{rel}} = \mathbf{A}^s(\mathbf{q})\,\ddot{\mathbf{q}}_\text{full} + \dot{\mathbf{A}}^s(\mathbf{q}, \dot{\mathbf{q}})\,\dot{\mathbf{q}}_\text{full}$$

#### Robot Momentum at the Base (Alternative Representation)

The same relative momentum can be extracted from the top 6 rows of the joint-space inertia matrix:

$$\mathbf{h}_b^{s,\text{rel}} = \mathbf{H}_{[0:6,:]}\,\dot{\mathbf{q}}_\text{full} = \mathbf{H}_t\,\mathbf{v}_t^{s,\text{rel}} + \mathbf{H}_{tq}\,\dot{\mathbf{q}}$$

This gives the 6D momentum at the torso frame origin $O_b$. Both representations encode the same physical quantity; the angular parts are related by the co-adjoint transport:

$$\mathbf{L}_b^{s,\text{rel}} = \mathbf{L}_\text{com}^{s,\text{rel}} + (\mathbf{r}_\text{com}^s - \mathbf{p}_b^s) \times m\,\mathbf{v}_\text{com}^{s,\text{rel}}$$

---

### Inertial Robot Momentum in Platform Frame

For the conservation law, we need the **inertial** momentum of the robot. Each body $i$ of the robot at position $\mathbf{r}_i^s$ in platform frame has inertial velocity (expressed in $\mathcal{P}$ coordinates):

$$\mathbf{v}_i^{s,\text{in}} = \underbrace{\mathbf{R}_s^\top \mathbf{v}_s^w + \boldsymbol{\omega}_s^s \times \mathbf{r}_i^s}_{\text{platform drag}} + \mathbf{v}_i^{s,\text{rel}}$$

$$\boldsymbol{\omega}_i^{s,\text{in}} = \boldsymbol{\omega}_s^s + \boldsymbol{\omega}_i^{s,\text{rel}}$$

The total **inertial** angular momentum of the robot about $O_p$, in platform-frame coordinates, is:

$$\mathbf{L}_\text{robot}^{O_s,s,\text{in}} = \sum_i \left[\mathbf{I}_i^s\,\boldsymbol{\omega}_i^{s,\text{in}} + \mathbf{r}_i^s \times m_i\,\mathbf{v}_i^{s,\text{in}}\right]$$

Expanding using the velocity decomposition and grouping terms yields:

$$\boxed{\mathbf{L}_\text{robot}^{O_s,s,\text{in}} = \underbrace{\mathbf{L}_\text{com}^{s,\text{rel}} + \mathbf{r}_\text{com}^s \times m\,\mathbf{v}_\text{com}^{s,\text{rel}}}_{\mathbf{L}_\text{robot}^{O_s,s,\text{rel}}\;\text{(from Pinocchio)}} + \underbrace{\mathbf{I}_\text{robot}^{O_s,s}\,\boldsymbol{\omega}_s^s}_{\text{rotational drag}} + \underbrace{m\,\mathbf{r}_\text{com}^s \times \mathbf{v}_s^{s}}_{\text{translational drag}}}$$

where:

- $\mathbf{L}_\text{robot}^{O_s,s,\text{rel}} = \mathbf{L}_\text{com}^{s,\text{rel}} + \mathbf{r}_\text{com}^s \times m\,\mathbf{v}_\text{com}^{s,\text{rel}}$ is obtained by co-adjoint transport of Pinocchio's centroidal momentum to $O_p$
- $\mathbf{I}_\text{robot}^{O_s,s} = \sum_i \left[\mathbf{I}_i^s - m_i\,[\mathbf{r}_i^s]_\times^2\right]$ is the robot's locked (composite) inertia tensor about $O_p$ in platform frame (from Pinocchio's CRBA)
- $\mathbf{v}_s^{s} = \mathbf{R}_s^\top\,\mathbf{v}_s^w$ is the platform's linear velocity in platform-frame coordinates

**Physical interpretation:** the first term is the momentum the robot would have if the platform were stationary. The drag terms account for the angular and linear momentum the robot carries simply by riding on the moving platform.

---

### Conservation Law in Platform Frame

In the inertial frame, total angular momentum is conserved: $\mathbf{L}_\text{total}^w = \mathbf{L}_0 = \text{const}$.

Projecting into platform-frame coordinates ($\mathbf{L}^s = \mathbf{R}_s^\top \mathbf{L}^w$) gives the **Euler equation** in the rotating frame:

$$\frac{d}{dt}\!\left(\mathbf{L}_\text{total}^s\right) + \boldsymbol{\omega}_s^s \times \mathbf{L}_\text{total}^s = \mathbf{0}$$

where:

$$\mathbf{L}_\text{total}^s = \mathbf{L}_\text{robot}^{O_s,s,\text{in}} + \mathbf{h}_w^s + \mathbf{I}_s\,\boldsymbol{\omega}_s^s$$

with $\mathbf{h}_w^s = \sum_k I_{w,k}\,\omega_{w,k}^s$ the wheel angular momentum and $\mathbf{I}_s\,\boldsymbol{\omega}_s^s$ the platform body's angular momentum (excluding wheels and robot), all in platform-frame coordinates.

From the inertial conservation, expressed in platform-frame coordinates:

$$\mathbf{R}_s^\top\,\mathbf{L}_0 = \mathbf{L}_\text{robot}^{O_s,s,\text{in}} + \mathbf{h}_w^s + \mathbf{I}_s\,\boldsymbol{\omega}_s^s$$

Solving for the wheel momentum:

$$\mathbf{h}_w^s = \mathbf{R}_s^\top\,\mathbf{L}_0 - \mathbf{I}_s\,\boldsymbol{\omega}_s^s - \mathbf{L}_\text{robot}^{O_s,s,\text{in}}$$

Substituting the full expression for $\mathbf{L}_\text{robot}^{O_s,s,\text{in}}$ from §3.3:

$$\mathbf{h}_w^s = \mathbf{R}_s^\top\,\mathbf{L}_0 - (\mathbf{I}_s + \mathbf{I}_\text{robot}^{O_s,s})\,\boldsymbol{\omega}_s^s - m\,\mathbf{r}_\text{com}^s \times \mathbf{v}_s^s - \mathbf{L}_\text{com}^{s,\text{rel}} - \mathbf{r}_\text{com}^s \times m\,\mathbf{v}_\text{com}^{s,\text{rel}}$$

This is the **exact** expression — no approximations.

---

### Incremental Form for the NMPC

At each NMPC call, all platform-side quantities are **measured** from the current state and frozen over the prediction horizon:

| Measured quantity | Source | Symbol |
|---|---|---|
| Wheel angular momentum | $I_w \cdot \omega_\text{wheels}^s$ from MuJoCo | $\mathbf{h}_w^{s,0}$ |
| Relative centroidal mom. | Pinocchio `computeCentroidalMomentum()` | $\mathbf{L}_\text{com}^{s,\text{rel},0}$ |
| Robot CoM position | Pinocchio `rs.r_com` | $\mathbf{r}_\text{com}^{s,0}$ |
| Robot CoM relative velocity | Pinocchio `rs.v_com` | $\mathbf{v}_\text{com}^{s,\text{rel},0}$ |
| Robot locked inertia at $O_p$ | Pinocchio CRBA | $\mathbf{I}_\text{robot}^{O_s,s,0}$ |
| Platform angular velocity | `mj_data.qvel[3:6]` → platform body frame | $\boldsymbol{\omega}_s^{s,0}$ |
| Platform linear velocity | `mj_data.qvel[0:3]` → $\mathbf{R}_s^\top$ | $\mathbf{v}_s^{s,0}$ |

The initial inertial robot momentum is computed from these:

$$\mathbf{L}_\text{robot}^{O_s,s,\text{in},0} = \mathbf{L}_\text{com}^{s,\text{rel},0} + \mathbf{r}_\text{com}^{s,0} \times m\,\mathbf{v}_\text{com}^{s,\text{rel},0} + \mathbf{I}_\text{robot}^{O_s,s,0}\,\boldsymbol{\omega}_s^{s,0} + m\,\mathbf{r}_\text{com}^{s,0} \times \mathbf{v}_s^{s,0}$$

Define the **measured constant** (computed once per NMPC call):

$$\mathbf{c} = \mathbf{h}_w^{s,0} + \mathbf{L}_\text{robot}^{O_s,s,\text{in},0}$$

Over the prediction horizon, the platform's pose ($\mathbf{R}_s$), angular velocity ($\boldsymbol{\omega}_s^s$), linear velocity ($\mathbf{v}_s^s$), and inertia ($\mathbf{I}_s$) are frozen at their initial values. The structure is not modeled in the NMPC — its evolution is driven by the contact forces the NMPC plans, but the structure dynamics are treated as quasi-static over $T_h$.

Under this frozen-platform assumption, the change in $\mathbf{h}_w^s$ over the horizon is due to the change in robot relative momentum plus the change in drag terms:

$$\mathbf{h}_w^s(k) \approx \mathbf{c} - \mathbf{L}_\text{robot}^{O_s,s,\text{in}}(k)$$

where at knot $k$:

$$\mathbf{L}_\text{robot}^{O_s,s,\text{in}}(k) = \underbrace{\mathbf{L}_\text{com}^s(k) + \mathbf{r}_\text{com}^s(k) \times m\,\mathbf{v}_\text{com}^{s,\text{rel}}(k)}_{\text{from NMPC state (predicted)}} + \underbrace{\mathbf{I}_\text{robot}^{O_s,s,0}\,\boldsymbol{\omega}_s^{s,0} + m\,\mathbf{r}_\text{com}^s(k) \times \mathbf{v}_s^{s,0}}_{\mathbf{d}(k)\;\text{(drag, frozen platform, varying robot config)}}$$

Note that the drag $\mathbf{d}(k)$ is **not constant** over the horizon: the first term ($\mathbf{I}_\text{robot}^{O_s,s,0}\,\boldsymbol{\omega}_s^{s,0}$) is frozen, but the second term ($m\,\mathbf{r}_\text{com}^s(k) \times \mathbf{v}_s^{s,0}$) varies with the predicted CoM position, which is an NMPC state variable. This makes the constraint bilinear in the state (cross product of $\mathbf{r}_\text{com}^s$ with a constant), which CasADi handles natively.

#### Reduction to Neglected Drag (Option B)

For a massive structure ($m_s \gg m_\text{robot}$), the platform responds slowly to robot contact forces. The drag terms can be bounded:

$$\|\mathbf{I}_\text{robot}^{O_s,s,0}\,\boldsymbol{\omega}_s^{s,0}\| \leq \bar{I}_\text{robot}\,\|\boldsymbol{\omega}_s^{s,0}\|$$

$$\|m\,\mathbf{r}_\text{com}^s(k) \times \mathbf{v}_s^{s,0}\| \leq m\,\|\mathbf{r}_\text{com}^s\|_\max\,\|\mathbf{v}_s^{s,0}\|$$

For SpaceServicer on a realistic structure (mass ratio $\sim$14%, $\|\boldsymbol{\omega}_s\| \lesssim 0.05$ rad/s, $\|\mathbf{v}_s\| \lesssim 0.02$ m/s), these terms are small compared to $\mathbf{h}_\text{max}$. In this regime, the constraint simplifies to:

$$\mathbf{c}_\text{simple} - \mathbf{L}_\text{com}^s(k) - \mathbf{r}_\text{com}^s(k) \times m\,\mathbf{v}_\text{com}^{s,\text{rel}}(k) \in [-\mathbf{h}_\text{max},\;\mathbf{h}_\text{max}]$$

where $\mathbf{c}_\text{simple} = \mathbf{c} - \mathbf{I}_\text{robot}^{O_s,s,0}\,\boldsymbol{\omega}_s^{s,0} - m\,\mathbf{r}_\text{com}^{s,0} \times \mathbf{v}_s^{s,0}$ absorbs the initial drag into the constant. The drag contribution to constraint violation over the horizon is bounded by:

$$\epsilon_\text{drag} = m\,\Delta r_\text{com}^{\max}\,\|\mathbf{v}_s^{s,0}\|$$

where $\Delta r_\text{com}^{\max}$ is the maximum CoM displacement over the horizon. This can be compensated by tightening the box: $\mathbf{h}_\text{max}' = \mathbf{h}_\text{max} - \epsilon_\text{drag}\,\mathbf{1}$.

**Recommendation:** Use the full formulation (Option A) as the reference. Use the simplified formulation (Option B, neglected drag with tightened box) for the initial implementation, with the full formulation as a documented upgrade path.

---


### Power Balance and Passivity Property


The robot's equation of motion in platform frame is:

$$\mathbf{H}(\mathbf{q})\,\ddot{\mathbf{q}}_\text{full} + \mathbf{C}(\mathbf{q}, \dot{\mathbf{q}}_\text{full})\,\dot{\mathbf{q}}_\text{full} = \mathbf{B}_u\,\boldsymbol{\tau}_q + \mathbf{J}_c^\top\,\boldsymbol{\lambda}$$

where $\dot{\mathbf{q}}_\text{full} = [\mathbf{v}_t;\,\dot{\mathbf{q}}] \in \mathbb{R}^{18}$, $\mathbf{B}_u = [\mathbf{0}_{6\times 12};\,\mathbf{I}_{12}]$, and $\boldsymbol{\lambda}$ is the contact wrench.

The robot's kinetic energy in platform frame (relative to the platform) is:

$$T = \frac{1}{2}\dot{\mathbf{q}}_\text{full}^\top\,\mathbf{H}\,\dot{\mathbf{q}}_\text{full}$$

**Property (Skew-symmetry).** The matrix $\dot{\mathbf{H}} - 2\mathbf{C}$ is skew-symmetric for Euler-Lagrange systems: $\dot{\mathbf{q}}^\top(\dot{\mathbf{H}} - 2\mathbf{C})\dot{\mathbf{q}} = 0$ for all $\dot{\mathbf{q}}$ (Ortega et al., 1998, Ch. 2; Siciliano et al., 2009, §7.1.2; Spong et al., 2006, §6.5.1). Pinocchio constructs $\mathbf{C}$ to satisfy this by construction via `pin.computeCoriolisMatrix()`. This property is also stated as Property 1 in Mishra et al. (2022, eq. 4) in the identical context of a multi-arm robot with bilateral holonomic constraints for in-orbit assembly.

Taking the time derivative of $T$, using skew-symmetry, and noting that rigid bilateral contacts satisfy $\mathbf{J}_c\dot{\mathbf{q}}_\text{full} = \mathbf{0}$ (contact forces do no work — Mishra et al., 2022, eq. 12):

$$\boxed{\dot{T} = \dot{\mathbf{q}}^\top\,\boldsymbol{\tau}_q}$$

where $\dot{\mathbf{q}} \in \mathbb{R}^{12}$ is the joint velocity vector. **The joint torques are the sole energy port of the system.** In microgravity with rigid contacts, there is no gravity potential, no friction, no aerodynamic drag — the system is lossless except through the control torques.

**Critical caveat — relative kinetic energy.** $T$ is computed from relative velocities (robot w.r.t. platform). The platform's own kinetic energy is handled separately by the AOCS.

### Passivity Constraint During Double Support

During DS, we enforce exponential energy decay via a single linear inequality in the QP:

$$\dot{\mathbf{q}}^\top\,\boldsymbol{\tau}_q \leq -2\alpha\,T \quad \Rightarrow \quad T(t) \leq T(t_0)\,e^{-2\alpha(t - t_0)}$$

(Khalil, 2002, §4.5). Implementation: $\dot{\mathbf{q}}^\top\boldsymbol{\tau}_q + 2\alpha T \leq 0$ is a **single scalar linear inequality** in $\boldsymbol{\tau}_q$.

**DS exit condition** (energy-based): $T < T_\text{settle} = \frac{1}{2}\epsilon_v^2\lambda_\text{min}(\mathbf{H})$, guaranteeing $\|\dot{\mathbf{q}}_\text{full}\| < \epsilon_v$. Deterministic settling time: $t_\text{settle} = \frac{1}{2\alpha}\ln(T(t_0)/T_\text{settle})$.

**Relationship to Kalaycioglu & De Ruiter (2023).** Our constraint is a special case of their IF-OFP condition (eq. 15) with $\nu = 0$, but uses inertia-weighted kinetic energy $T = \frac{1}{2}\dot{\mathbf{q}}^\top\mathbf{H}\dot{\mathbf{q}}$ rather than the velocity norm $\dot{\mathbf{q}}^\top\dot{\mathbf{q}}$. The kinetic energy form is tighter — it accounts for configuration-dependent inertia.

**Relationship to Mishra et al. (2022).** Their $\dot{W} \leq 0$ (eq. 13) establishes passivity by impedance controller design. Our approach achieves the same dissipativity via a **QP constraint**, compatible with any controller structure (including task-prioritized QP). Trade-off: theirs is passivity-by-construction (no infeasibility); ours could make the QP infeasible if $\alpha$ is too aggressive.

**Critical caveats:** (1) Feasibility — if $\alpha$ is too large, the constraint conflicts with task objectives. Safe initial value: $\alpha = 1/t_\text{DS,target}$. (2) Discrete time — valid when $2\alpha\Delta t < 1$ ($\alpha < 50$ s⁻¹ at 100 Hz).

### Kinematic and Dynamic Singularities

#### The Generalized Jacobian Matrix

For a free-floating space manipulator, the end-effector velocity is not simply $\mathbf{J}_m\dot{\mathbf{q}}$ because the base moves in reaction to joint motion. Yoshida & Umetani (1989, 1993) introduced the **Generalized Jacobian Matrix** (GJM):

$$\mathbf{J}^* = \mathbf{J}_m - \mathbf{J}_b\,\mathbf{H}_b^{-1}\,\mathbf{H}_{bm}$$

where $\mathbf{J}_m$ is the kinematic manipulator Jacobian, $\mathbf{J}_b$ is the base Jacobian, $\mathbf{H}_b$ is the base (locked) inertia, and $\mathbf{H}_{bm}$ is the base-manipulator coupling inertia. The GJM maps joint velocities to EE velocities while accounting for momentum conservation at the base.

#### Dynamic Singularities

A **dynamic singularity** occurs when $\mathbf{J}^*$ loses rank even though the kinematic Jacobian $\mathbf{J}_m$ is full rank (Papadopoulos & Dubowsky, 1993). Physically: certain joint motions produce only base reaction and zero net EE displacement — the arm "pushes" the base instead of moving the end-effector. At a dynamic singularity, the EE position is no longer a function of the current joint angles alone but depends on the entire joint angle history (path-dependent, nonholonomic behavior).

Dynamic singularities depend on the **mass distribution** of the system, not just the geometry. They shift in the configuration space as the payload changes, which makes them harder to predict and avoid than kinematic singularities. For a detailed treatment, see Papadopoulos & Dubowsky (1993) and the tutorial by Flores-Abad et al. (2014, Frontiers in Robotics and AI).

#### Constrained Dynamic Singularities for Crawling Robots

VISPA is not classically free-floating — the stance arm is welded to the platform via bilateral holonomic constraints. The analog of the GJM for this constrained system is the **constraint-reduced task Jacobian** introduced by Mishra et al. (2022, eq. 3 and Table 1):

$$\tilde{\mathbf{T}}_{qj}^k = \mathbf{T}_{qj}^k - \mathbf{T}_{bj}^k\,\boldsymbol{\Psi}_{bq}$$

where:
- $\mathbf{T}_{qj}^k \in \mathbb{R}^{6 \times n}$ is the kinematic manipulator Jacobian of end-effector $j$ on arm $k$
- $\mathbf{T}_{bj}^k \in \mathbb{R}^{6 \times 6}$ is the base Jacobian
- $\boldsymbol{\Psi}_{bq} = -\boldsymbol{\Psi}_b^{-1}\boldsymbol{\Psi}_q$ is the constraint-induced base-to-joint coupling (from the holonomic constraint $\boldsymbol{\Psi}_b\mathbf{V}_b + \boldsymbol{\Psi}_q\dot{\mathbf{q}} = \mathbf{0}$)

The term $\mathbf{T}_{bj}^k\boldsymbol{\Psi}_{bq}$ accounts for the fact that joint motion moves the torso (via the stance arm kinematic chain), and this torso motion affects the swing EE position. A **constrained dynamic singularity** occurs when $\tilde{\mathbf{T}}_{qj}^k$ loses rank.

Unlike the free-floating case where dynamic singularities arise from the inertia coupling ($\mathbf{H}_b^{-1}\mathbf{H}_{bm}$), constrained dynamic singularities arise from the **kinematic coupling through the contact constraint** ($\boldsymbol{\Psi}_{bq}$). However, the QP operates at the acceleration level with the inertia matrix $\mathbf{H}$ weighting the solution, so the effective Jacobian in the QP is inertia-weighted. Configurations where the inertia coupling $\mathbf{H}_{tq}$ creates near-cancellations in the effective task Jacobian are dynamic singularities in the classical sense.

#### Conditions for Constrained Dynamic Singularity in the SpaceServicer

Three scenarios produce rank deficiency in $\tilde{\mathbf{T}}_{qj}^k$:

1. **Stance arm near kinematic singularity.** The constraint coupling $\boldsymbol{\Psi}_{bq}$ becomes poorly conditioned (the stance arm can't transmit torso motion to the platform effectively). This corrupts the coupling term and can cause $\tilde{\mathbf{T}}_{qj}^k$ to lose rank even if the swing arm is far from singular.

2. **Swing arm aligned with the constraint coupling direction.** Certain swing arm configurations cause the kinematic Jacobian $\mathbf{T}_{qj}^k$ and the coupling correction $\mathbf{T}_{bj}^k\boldsymbol{\Psi}_{bq}$ to cancel in specific task-space directions.

3. **Geometric coincidence at full extension.** During a crawling step, the stance arm progressively extends as the torso moves forward. Near the end of the step, the stance arm approaches full extension — **this is simultaneously the moment of highest singularity risk for the stance arm AND the moment when the swing arm needs maximum precision for docking**. The constrained dynamic singularity degrades the torso task (P1) right when the EE task (P2) needs the best null-space freedom.

#### Why 7-DOF Arms Are Required

With 6-DOF arms ($n_a = 6$), $\tilde{\mathbf{T}}_{qj}^k \in \mathbb{R}^{6 \times 6}$ is square. A rank-1 deficiency makes the system locally uncontrollable in that direction — there is no redundancy to compensate.

With 7-DOF arms ($n_a = 7$), $\tilde{\mathbf{T}}_{qj}^k \in \mathbb{R}^{6 \times 7}$ has a 1-dimensional null space. This enables:

1. **Self-motion for singularity avoidance.** Each arm can reconfigure without changing its EE position. The stance arm uses its null space to stay away from the extension singularity while maintaining its end-effector-to-structure weld.

2. **Manipulability optimization.** The posture task (P3) in the QP drives the joint configuration toward maximum manipulability. The manipulability measure $w(\mathbf{q}) = \sqrt{\det(\tilde{\mathbf{T}}_{qj}^k\,\tilde{\mathbf{T}}_{qj}^{k\top})}$ (Yoshikawa, 1985) provides a scalar index that the posture reference $\mathbf{q}_\text{nom}$ can be chosen to maximize.

3. **Singularity margin for the constrained Jacobian.** With 7 DOFs, a rank-1 kinematic deficiency in the stance arm still leaves the system with at least 1 DOF of redundancy per arm. The QP finds a degraded but feasible solution instead of becoming infeasible.

**Implementation in the QP:** the posture task reference $\mathbf{q}_\text{nom}$ should be configuration-dependent — not a fixed nominal pose, but a pose that maximizes manipulability for both arms:

$$\mathbf{q}_\text{nom}(\mathbf{q}) = \arg\max_{\mathbf{q}' \in \mathcal{N}(\tilde{\mathbf{T}})} w(\mathbf{q}')$$

where $\mathcal{N}(\tilde{\mathbf{T}})$ is the null space of the active task Jacobians. In practice, a gradient step in the null space is sufficient:

$$\mathbf{q}_\text{nom} = \mathbf{q} + \eta\,(\mathbf{I} - \tilde{\mathbf{T}}^\dagger\tilde{\mathbf{T}})\,\nabla_\mathbf{q} w$$

This is the classical gradient projection method (Liegeois, 1977), applied to the constrained Jacobian.

**Critical observation for the SpaceServicer:** singularity avoidance must consider **both arms' configurations simultaneously**, not just the swing arm. A singularity in the stance arm degrades the torso task (P1), which cascades into the EE task (P2) via the null-space projection $\mathbf{N}_\text{torso}\mathbf{J}_\text{ee}$.

#### 7th DOF Placement

The standard approach for adding redundancy to a 6-DOF anthropomorphic arm is a **redundant rotation in the upper arm** (between shoulder and elbow), creating a human-like elbow swivel DOF. This is the design used in the MIRROR project's 7-DOF VISPA arms (Mishra et al., 2022; Deremetz et al., 2021). The joint axis is along the upper arm segment (shoulder-to-elbow direction), allowing the elbow to trace a circle around the shoulder-to-wrist line without changing the shoulder or wrist pose.

#### Verified References

- Umetani, Y., Yoshida, K. (1989). "Resolved Motion Rate Control of Space Manipulators with Generalized Jacobian Matrix." *IEEE Trans. Robotics and Automation*, 5(3):303–314. — Original GJM.
- Yoshida, K., Umetani, Y. (1993). "Control of Space Manipulators with Generalized Jacobian Matrix." In *Space Robotics: Dynamics and Control*, Springer, KCIS vol. 188. — Extended GJM formulation.
- Papadopoulos, E., Dubowsky, S. (1993). "Dynamic Singularities in Free-Floating Space Manipulators." In *Space Robotics: Dynamics and Control*, Springer, KCIS vol. 188. — Original dynamic singularity definition and analysis.
- Mishra, H., De Stefano, M., Ott, C. (2022). "Dynamics and Control of a Reconfigurable Multi-Arm Robot for In-Orbit Assembly." *IFAC PapersOnLine*, 55-20, 235–240. — Constraint-reduced Jacobian for bilateral-contact crawling.
- Yoshikawa, T. (1985). "Manipulability of Robotic Mechanisms." *Int. J. Robotics Research*, 4(2):3–9. — Manipulability measure.
- Liegeois, A. (1977). "Automatic Supervisory Control of the Configuration and Behavior of Multibody Mechanisms." *IEEE Trans. Systems, Man, and Cybernetics*, 7(12):868–871. — Gradient projection for redundancy resolution.


---

## 4. Control Architecture

### NMPC Formulation (B2 — Reduced State)

Under the B2 formulation, $\mathbf{h}_w$ is eliminated from the state. The conservation law is enforced algebraically at each knot via the constraint derived in §3.5.

**State vector** ($n_x = 9$):

$$\mathbf{x} = [\mathbf{r}_\text{com}^s,\; \mathbf{v}_\text{com}^{s,\text{rel}},\; \mathbf{L}_\text{com}^{s,\text{rel}}] \in \mathbb{R}^{9}$$

All quantities are in platform frame $\mathcal{P}$, computed from relative velocities.

**Controls** ($n_u = 12$):

$$\mathbf{u} = [\mathbf{f}_1,\; \boldsymbol{\tau}_1,\; \mathbf{f}_2,\; \boldsymbol{\tau}_2] \in \mathbb{R}^{12}$$

**Dynamics** (platform frame, relative):

$$\dot{\mathbf{r}}_\text{com}^s = \mathbf{v}_\text{com}^{s,\text{rel}}$$

$$\dot{\mathbf{v}}_\text{com}^{s,\text{rel}} = \frac{\mathbf{f}_1 + \mathbf{f}_2}{m}$$

$$\dot{\mathbf{L}}_\text{com}^{s,\text{rel}} = \sum_j \left[(\mathbf{r}_{C_j}^s - \mathbf{r}_\text{com}^s) \times \mathbf{f}_j + \boldsymbol{\tau}_j\right]$$

Note: $\mathbf{r}_{C_j}^s$ are contact positions in platform frame — constants from the gait scheduler.

**Parameters** ($n_p = 15$ for Option B, $n_p = 21{+}$ for Option A):

| Parameter | Dim | Source |
|---|---|---|
| $\mathbf{r}_\text{ref}$ | 3 | CoM reference from TorsoPlanner |
| $\mathbf{v}_\text{ref}$ | 3 | CoM velocity reference |
| $\mathbf{r}_{C_1}^s$ | 3 | Contact A position (constant in $\mathcal{P}$) |
| $\mathbf{r}_{C_2}^s$ | 3 | Contact B position (constant in $\mathcal{P}$) |
| $\mathbf{c}$ | 3 | Measured constant $= \mathbf{h}_w^{s,0} + \mathbf{L}_\text{robot}^{O_s,s,\text{in},0}$ |
| *Option A adds:* $\boldsymbol{\omega}_s^{s,0}(3)$, $\mathbf{v}_s^{s,0}(3)$, $\mathbf{I}_\text{robot}^{O_s,s,0}(6)$ | 12 | Frozen structure state |

**Constraints (core contribution):**

1. **RWA momentum box** (every knot $k$):

*Option A (full):*

$$\mathbf{c} - \mathbf{L}_\text{com}^s(k) - \mathbf{r}_\text{com}^s(k) \times m\,\mathbf{v}_\text{com}^{s,\text{rel}}(k) - \mathbf{I}_\text{robot}^{O_s,s,0}\,\boldsymbol{\omega}_s^{s,0} - m\,\mathbf{r}_\text{com}^s(k) \times \mathbf{v}_s^{s,0} \in [-\mathbf{h}_\text{max},\;\mathbf{h}_\text{max}]$$

*Option B (simplified, neglected drag with tightened box):*

$$\mathbf{c}_\text{simple} - \mathbf{L}_\text{com}^s(k) - \mathbf{r}_\text{com}^s(k) \times m\,\mathbf{v}_\text{com}^{s,\text{rel}}(k) \in [-\mathbf{h}_\text{max}',\;\mathbf{h}_\text{max}']$$

2. **Momentum rate bound** (every knot $k$) — wheel torque limit:

$$\left|\sum_j \left[\mathbf{r}_{C_j}^s \times \mathbf{f}_j + \boldsymbol{\tau}_j\right]_i\right| \leq \tau_{w,\max} \quad \forall\, i \in \{1,2,3\}$$

Note: lever arms are from $O_p$ directly — no $\mathbf{r}_s$ needed.

3. **Force bounds:** $|f_{j,i}| \leq f_\max$, $|\tau_{j,i}| \leq \tau_\max$, zero on inactive contacts.

**Cost:**

$$J = \sum_k \left[ w_r \|\mathbf{r}_\text{com}^s - \mathbf{r}_\text{ref}\|^2 + w_v \|\mathbf{v}_\text{com}^{s,\text{rel}} - \mathbf{v}_\text{ref}\|^2 + w_L \|\mathbf{L}_\text{com}^{s,\text{rel}} - \mathbf{L}_\text{com}^\text{ref}(k)\|^2 + w_u \|\mathbf{u}\|^2 \right]$$

The reference $\mathbf{L}_\text{com}^\text{ref}(k)$ is provided by the TorsoPlanner (§4.3) and represents the expected centroidal angular momentum from the planned torso rotation. During DS (settling), $\mathbf{L}_\text{com}^\text{ref} = \mathbf{0}$. During SS, it is nonzero when the torso reorients between steps. This prevents the NMPC from treating intentional rotation as a disturbance.

Note: the $w_h\|\mathbf{h}_w\|^2$ penalty from the old formulation can be recovered by substituting the algebraic expression for $\mathbf{h}_w^s(k)$, making the cost nonlinear in the state.

**Outputs to Stage 2:** $\mathbf{r}_\text{com}^{s,\text{plan}}$, $\mathbf{v}_\text{com}^{s,\text{rel},\text{plan}}$, $\boldsymbol{\lambda}_\text{ref}$, $\mathbf{a}_\text{com}^\text{ff} = (\mathbf{f}_1+\mathbf{f}_2)/m$.

---


### NMPC Formulation (B3 — Full State with Algebraic Backup)

As a robust fallback, $\mathbf{h}_w^s$ is kept in the state and integrated via ODE, with the algebraic constraint from §3.5 as a consistency check.

**State vector** ($n_x = 12$):

$$\mathbf{x} = [\mathbf{r}_\text{com}^s,\; \mathbf{v}_\text{com}^{s,\text{rel}},\; \mathbf{L}_\text{com}^{s,\text{rel}},\; \mathbf{h}_w^s] \in \mathbb{R}^{12}$$

**Additional dynamics** (appended to the 9-state ODE):

$$\dot{\mathbf{h}}_w^s = -\sum_j \left[\mathbf{r}_{C_j}^s \times \mathbf{f}_j + \boldsymbol{\tau}_j\right]$$

This integrates the disturbance torque about $O_p$. The algebraic constraint from B2 is added as an additional path constraint to catch numerical drift:

$$\left\|\mathbf{h}_w^s(k) - \left(\mathbf{c} - \mathbf{L}_\text{com}^s(k) - \mathbf{r}_\text{com}^s(k) \times m\,\mathbf{v}_\text{com}^{s,\text{rel}}(k) - \mathbf{d}(k)\right)\right\| \leq \epsilon_\text{drift}$$

The box constraint is applied directly on the state: $\mathbf{h}_w^s(k) \in [-\mathbf{h}_\text{max},\;\mathbf{h}_\text{max}]$, which is linear and cheap for the solver.

---

### TorsoPlanner — Orientation and Momentum Feedforward

The TorsoPlanner bridges the gap between the centroidal NMPC (which has no concept of orientation) and the QP (which needs a 6D torso reference). It produces three outputs from the IK-computed dock orientation:

**Inputs:** $\mathbf{R}_\text{start}$, $\mathbf{R}_\text{goal}$ (from IK), $T_\text{step}$ (from coarse planner or heuristic), $\mathbf{I}_\text{torso}^\text{com}$ (approximate torso inertia about CoM).

**Computation:**

$$\Delta\boldsymbol{\theta} = \text{Log}(\mathbf{R}_\text{start}^\top\,\mathbf{R}_\text{goal}) \in \mathbb{R}^3$$

$$\sigma(s) : [0,1] \to [0,1], \quad \sigma(0) = 0,\; \sigma(1) = 1,\; \dot{\sigma}(0) = \dot{\sigma}(1) = 0$$

(e.g., $\sigma(s) = 6s^5 - 15s^4 + 10s^3$, a quintic with zero velocity and acceleration at endpoints)

$$\mathbf{R}_\text{ref}(t) = \mathbf{R}_\text{start}\,\text{Exp}\!\left(\Delta\boldsymbol{\theta}\cdot\sigma(t/T_\text{step})\right)$$

$$\boldsymbol{\omega}_\text{ref}(t) = \Delta\boldsymbol{\theta}\cdot\frac{\dot{\sigma}(t/T_\text{step})}{T_\text{step}}$$

$$\boldsymbol{\alpha}_\text{ref}(t) = \Delta\boldsymbol{\theta}\cdot\frac{\ddot{\sigma}(t/T_\text{step})}{T_\text{step}^2}$$

**Outputs:**

| Output | Destination | Purpose |
|---|---|---|
| $\mathbf{R}_\text{ref}(t)$, $\boldsymbol{\omega}_\text{ref}(t)$, $\boldsymbol{\alpha}_\text{ref}(t)$ | QP torso orientation task | Orientation reference for P1 |
| $\mathbf{L}_\text{com}^\text{ref}(t) = \mathbf{I}_\text{torso}^\text{com}\,\boldsymbol{\omega}_\text{ref}(t)$ | NMPC angular momentum reference | Prevents NMPC from fighting intentional rotation |

**During DS:** $\mathbf{R}_\text{ref} = \mathbf{R}_\text{current}$, $\boldsymbol{\omega}_\text{ref} = \mathbf{0}$, $\mathbf{L}_\text{com}^\text{ref} = \mathbf{0}$ (the passivity constraint drives settling; no rotation planned).

**Why not tangent-following:** aligning the torso with the velocity direction ($\hat{\mathbf{t}} = \mathbf{v}/\|\mathbf{v}\|$) was considered. It couples naturally to the NMPC (angular rate = curvature × speed). However: (1) singularity at zero velocity (start and end of each step, where precision matters most); (2) the tangent at $t = T_\text{step}$ has no reason to match the IK dock orientation; (3) the Frenet frame is ill-defined on straight lines. The SLERP from start to goal is the robust choice.

**Approximation quality:** $\mathbf{L}_\text{com}^\text{ref} = \mathbf{I}_\text{torso}^\text{com}\,\boldsymbol{\omega}_\text{ref}$ ignores the limb contribution to centroidal angular momentum during rotation. The error is $\sim(m_\text{arms}/m_\text{total})\cdot r_\text{arms}^2\cdot\omega$, typically $< 20\%$ of the torso term. The NMPC's feedback loop ($w_L\|\mathbf{L}_\text{com} - \mathbf{L}_\text{com}^\text{ref}\|^2$ tracking) absorbs this mismatch at each replanning cycle.

---

### Mapping Layer — Mathematical Derivation

**Position mapping:**

$$\mathbf{r}_b^\text{ref} = \frac{m_\text{total}}{m_b}\,\mathbf{r}_\text{com}^\text{ref} - \frac{1}{m_b}\,\boldsymbol{\delta}(\mathbf{q})$$

where $\boldsymbol{\delta}(\mathbf{q}) = \sum_{i \neq b} m_i\,\mathbf{r}_i(\mathbf{q})$ is computed from FK at the current configuration.

**Velocity mapping:**

$$\dot{\mathbf{r}}_b^\text{ref} = \frac{m_\text{total}}{m_b}\,\mathbf{v}_\text{com}^\text{ref} - \frac{1}{m_b}\,\dot{\boldsymbol{\delta}}(\mathbf{q}, \dot{\mathbf{q}})$$

The feedforward $\dot{\boldsymbol{\delta}}$ can be dropped initially (error ~20%, handled by PD gains).

**Orientation:** from TorsoPlanner directly ($\mathbf{R}_\text{ref}$, $\boldsymbol{\omega}_\text{ref}$). Decoupled from mass-weighted position mapping.

**Jacobian equivalence:** tracking $\mathbf{r}_b^\text{ref}$ with $\mathbf{J}_b^\text{pos}$ is exactly equivalent to tracking $\mathbf{r}_\text{com}^\text{ref}$ with $\mathbf{J}_\text{com}$:

$$\mathbf{J}_b^\text{pos} = \frac{m_\text{total}}{m_b}\,\mathbf{J}_\text{com} - \frac{1}{m_b}\sum_{i\neq b} m_i\,\mathbf{J}_i$$

---


### Mapping Layer — Algorithm

```
Algorithm: CoM-to-Torso Mapping (100 Hz)

Input:  r_com_ref, v_com_ref, a_com_ff (NMPC)
        R_torso_ref, omega_ref         (TorsoPlanner)
        q_current, dq_current          (Pinocchio FK)

1. δ = Σ_{i≠torso} m_i · r_i(q)
2. r_b_ref = (m_total/m_b) · r_com_ref − (1/m_b) · δ
3. v_b_ref_lin = (m_total/m_b) · v_com_ref  [drop δ_dot initially]
4. a_b_ff_lin = (m_total/m_b) · a_com_ff
5. Assemble: [r_b_ref, R_torso_ref], [v_b_ref_lin; omega_ref], [a_b_ff_lin; alpha_ref]
6. Soft CoM (SEPARATE): a_com_des = a_com_ff + Kp·(r_com_ref − r_com) + Kd·(v_com_ref − v_com)
```

---


### Stage 2 — Whole-Body QP (100 Hz)

**Decision variables** ($n = 42$):

$$\mathbf{z} = [\ddot{\mathbf{q}}_t(6),\; \ddot{\mathbf{q}}(12),\; \boldsymbol{\lambda}(12),\; \boldsymbol{\tau}_q(12)]$$

**Task stack (reworked):**

| Priority | Task | DOF | Jacobian |
|---|---|---|---|
| Hard | Dynamics: $\mathbf{H}\ddot{\mathbf{q}} + \mathbf{C} = \mathbf{B}_u\boldsymbol{\tau} + \mathbf{J}_c^\top\boldsymbol{\lambda}$ | 18 eq. | — |
| Hard | Contact: $\mathbf{J}_c\ddot{\mathbf{q}} + \dot{\mathbf{J}}_c\dot{\mathbf{q}} = 0$ | $6 n_c$ eq. | — |
| 1 | Torso 6D: $\mathbf{J}_\text{torso}\ddot{\mathbf{q}} = \ddot{\mathbf{x}}_\text{torso}^\text{des}$ | 6 | $\mathbf{J}_\text{torso} \in \mathbb{R}^{6\times 18}$ |
| 2 | EE 6D: $\mathbf{J}_\text{ee}\ddot{\mathbf{q}} = \ddot{\mathbf{x}}_\text{ee}^\text{des}$ (null-space) | 6 | $\mathbf{N}_\text{torso}\mathbf{J}_\text{ee}$ |
| 3 | Posture: $\ddot{\mathbf{q}} = K_p(\mathbf{q}_\text{nom} - \mathbf{q}) - K_d\dot{\mathbf{q}}$ | 12 | $\mathbf{I}$ |
| Soft | Wrench reg.: $\|\boldsymbol{\lambda} - \boldsymbol{\lambda}_\text{ref}\|^2$ | 12 | — |
| Soft | Torque reg.: $\|\boldsymbol{\tau}\|^2$ | 12 | — |

**DOF budget (single-support):**

- 14 free DOFs (20 − 6 weld)
- Torso 6D: 6 consumed
- EE 6D in null space: 6 consumed (from remaining 8)
- **2 DOF genuine redundancy** → posture task

**Momentum safety (backup):**

$$\mathbf{h}_w(k+1) \approx \mathbf{h}_w - \Delta t\,\mathbf{M}_\lambda\,\boldsymbol{\lambda} \in [\mathbf{h}_{w,\min},\;\mathbf{h}_{w,\max}]$$

---


### Soft CoM Residual


#### Rationale

Issue 5.2 (indirect cascade) is the most fundamental risk. The cleanest mitigation is to add a low-weight CoM tracking residual back into the QP — not as a competing task with its own priority level, but as a soft quadratic cost. This preserves the DOF budget fix (torso 6D as primary task) while restoring a weak form of the cascade guarantee between NMPC and QP.

#### QP Cost Structure

The total QP cost becomes:

$$J_\text{QP} = \underbrace{\alpha_\text{torso} \|\mathbf{J}_\text{torso}\ddot{\mathbf{q}} - \ddot{\mathbf{x}}_\text{torso}^\text{des}\|^2}_{\text{Priority 1: torso 6D}} + \underbrace{\alpha_\text{ee} \|\mathbf{J}_\text{ee}\ddot{\mathbf{q}} - \ddot{\mathbf{x}}_\text{ee}^\text{des}\|^2}_{\text{Priority 2: EE 6D}} + \underbrace{\alpha_\text{posture} \|\cdot\|^2}_{\text{Priority 3}} + \underbrace{\alpha_\text{com}^\text{soft} \|\mathbf{J}_\text{com}\ddot{\mathbf{q}} - \ddot{\mathbf{x}}_\text{com}^\text{des}\|^2}_{\textbf{Soft CoM residual (new)}} + \text{wrench/torque reg.}$$

#### Weight Hierarchy

| Term | Weight | Role |
|---|---|---|
| $\alpha_\text{torso}$ | 500–1000 | Primary DOF allocation |
| $\alpha_\text{ee}$ | 2000–5000 (phase-dep.) | End-effector tracking |
| $\alpha_\text{posture}$ | 5–100 (phase-dep.) | Joint regularization |
| $\alpha_\text{com}^\text{soft}$ | **1–10** | Momentum-consistent bias |

At this ratio, the soft CoM cost never overrides torso or EE tasks in the QP solution. It only acts within the residual freedom left by the higher-weighted tasks.

#### Geometric Interpretation

The torso task defines a 6-dimensional affine subspace of feasible $\ddot{\mathbf{q}}$. In single-support (14 free DOFs), this leaves an 8-dimensional null space. Without the soft CoM cost, the point within this subspace is chosen by EE + posture regularization, which have no momentum awareness. The soft CoM cost biases that choice toward the point that best tracks the NMPC's centroidal plan.

#### Critical Implementation Detail

The desired acceleration $\ddot{\mathbf{x}}_\text{com}^\text{des}$ in the soft cost **must come directly from the NMPC** (feedforward + PD on $\mathbf{r}_\text{com}$ error):

$$\ddot{\mathbf{x}}_\text{com}^\text{des} = \mathbf{a}_\text{com}^\text{ff} + K_p^\text{com}(\mathbf{r}_\text{com}^\text{ref} - \mathbf{r}_\text{com}) + K_d^\text{com}(\mathbf{v}_\text{com}^\text{ref} - \mathbf{v}_\text{com})$$

where $\mathbf{r}_\text{com}^\text{ref}$, $\mathbf{v}_\text{com}^\text{ref}$, $\mathbf{a}_\text{com}^\text{ff}$ are the NMPC outputs — **not** derived from the torso mapping. Deriving them from the torso reference would create a redundant copy of the torso task, defeating the purpose.

#### Effect on Cascade Guarantee

Even if the torso tracks imperfectly (error $\mathbf{e}_b$), the soft CoM residual provides a correction signal that keeps the actual CoM trajectory close to the NMPC's plan. The momentum constraint in the QP ($\mathbf{h}_w$ box on $\boldsymbol{\lambda}$) then operates on a solution closer to the NMPC's assumptions, reducing the momentum bias from §5.2.

#### Decision: Soft Cost vs. Priority Level

Two options were considered:

- **Soft cost (selected):** $\alpha_\text{com}^\text{soft}\|\cdot\|^2$ added to the single QP objective. Does not consume DOFs in the null-space projection sense. Simple to implement. Weight tuning is the only free parameter.
- **Third priority level:** CoM as P1, torso orientation as P2, EE as P3. Formally correct but increases the null-space cascade depth, which risks numerical conditioning issues and makes the interaction between CoM and torso orientation less transparent.

The soft cost is preferred for simplicity and because the torso position reference already encodes the CoM objective — the soft cost is a consistency check, not a primary control channel.

---


### AOCS Controller (Corrected)

The wheels must reject the full disturbance torque about $O_p$, not just the centroidal rate:

$$\boldsymbol{\tau}_w = -\dot{\mathbf{L}}_\text{com}^{s,\text{rel},\text{est}} - \mathbf{r}_\text{com}^s \times m\,\dot{\mathbf{v}}_\text{com}^{s,\text{rel},\text{est}} - K_{hw}(\mathbf{h}_w^s - \text{clip}(\mathbf{h}_w^s,\text{bounds}))$$

The orbital term $\mathbf{r}_\text{com}^s \times m\dot{\mathbf{v}}_\text{com}^{s,\text{rel},\text{est}}$ was missing in the current code — likely the cause of the 24° platform rotation at 14% mass ratio.


$$\boldsymbol{\tau}_w = -\dot{\mathbf{L}}_\text{est} - K_{hw}\,(\mathbf{h}_w - \text{clip}(\mathbf{h}_w,\,\text{bounds}))$$

Feedforward from centroidal rate estimate + feedback to center of the box.

---

## 5. Motion Planning


The control architecture (§3) assumes that CoM references and EE trajectories are provided. This section formalizes the planning layer that generates them.

### Position-Dependent Feasibility Envelope

The RWA momentum box constraint from §3.5 (Option B, simplified):

$$\mathbf{c}_\text{simple} - \mathbf{L}_\text{com}^s(k) - \mathbf{r}_\text{com}^s(k) \times m\,\mathbf{v}_\text{com}^{s,\text{rel}}(k) \in [-\mathbf{h}_\text{max}',\;\mathbf{h}_\text{max}']$$

Rearranging for the feasible velocity set at a given position $\mathbf{r}_\text{com}^s$ and centroidal momentum $\mathbf{L}_\text{com}^s$:

$$m\,[\mathbf{r}_\text{com}^s]_\times\,\mathbf{v}_\text{com}^{s,\text{rel}} \in [\mathbf{c}_\text{simple} - \mathbf{L}_\text{com}^s - \mathbf{h}_\text{max}',\;\mathbf{c}_\text{simple} - \mathbf{L}_\text{com}^s + \mathbf{h}_\text{max}']$$

The operator $[\mathbf{r}_\text{com}^s]_\times$ is skew-symmetric with rank 2 (null space along $\hat{\mathbf{r}}_\text{com}^s$). This induces a **position-dependent** feasibility geometry:

1. **Radial velocity** (along $\hat{\mathbf{r}}_\text{com}^s$): generates zero orbital momentum — always feasible regardless of distance from $O_p$.

2. **Transverse velocity** (perpendicular to $\hat{\mathbf{r}}_\text{com}^s$): generates orbital momentum proportional to $\|\mathbf{r}_\text{com}^s\|$. Maximum feasible transverse velocity:

$$v_{\perp,\max} \propto \frac{h_\text{max}'}{m\,\|\mathbf{r}_\text{com}^s\|}$$

3. **Feasible velocity set**: an elliptic cylinder in velocity space — unbounded along $\hat{\mathbf{r}}_\text{com}^s$, bounded in the two transverse directions with radii inversely proportional to $\|\mathbf{r}_\text{com}^s\|$.

**Consequence:** a robot operating at 2 m from $O_p$ can move transversely twice as fast as one at 4 m, for the same momentum budget. The feasible trajectory space varies across the structure. This means fixed step durations and geometric quintic trajectories are fundamentally inadequate — the motion planner must account for the position-dependent constraint.

### Coarse Pre-Planner

Before starting step $n$, a coarse trajectory optimization is solved over the full step horizon to produce a **momentum-feasible CoM reference**.

**Inputs:**

| Input | Source |
|---|---|
| $\mathbf{r}_\text{com}^{s,0}$, $\mathbf{v}_\text{com}^{s,0}$, $\mathbf{L}_\text{com}^{s,0}$ | Current state from Pinocchio |
| $\mathbf{c}$ | Measured constant (§3.5) |
| $\mathbf{r}_\text{com}^{s,\text{goal}}$ | From IK of next stance configuration |
| $\mathbf{r}_{C_\text{stance}}^s$ | Stance anchor (constant in $\mathcal{P}$) |
| $T_\text{step}$ | Decision variable or swept over candidates |

**Decision variables** at $M \approx 10{-}20$ collocation points ($\Delta t_c = T_\text{step}/M$):

- $\mathbf{r}_\text{com}^s(k)$, $\mathbf{v}_\text{com}^{s}(k)$, $\mathbf{L}_\text{com}^s(k)$ for $k = 0 \ldots M$
- $\mathbf{f}_\text{stance}(k)$, $\boldsymbol{\tau}_\text{stance}(k)$ for $k = 0 \ldots M-1$ (only stance contact active during SS)

**Dynamics:** centroidal ODE with one active contact:

$$\dot{\mathbf{r}}_\text{com}^s = \mathbf{v}_\text{com}^{s}, \qquad \dot{\mathbf{v}}_\text{com}^{s} = \mathbf{f}_\text{stance}/m$$

$$\dot{\mathbf{L}}_\text{com}^s = (\mathbf{r}_{C_\text{stance}}^s - \mathbf{r}_\text{com}^s) \times \mathbf{f}_\text{stance} + \boldsymbol{\tau}_\text{stance}$$

**Constraints:**

1. **Momentum box** (position-dependent, at every collocation point):

$$\mathbf{c} - \mathbf{L}_\text{com}^s(k) - \mathbf{r}_\text{com}^s(k) \times m\,\mathbf{v}_\text{com}^{s}(k) \in [-\mathbf{h}_\text{max}',\;\mathbf{h}_\text{max}']$$

2. **Rate bound:** $\left|[\mathbf{r}_{C_\text{stance}}^s \times \mathbf{f}_\text{stance} + \boldsymbol{\tau}_\text{stance}]_i\right| \leq \tau_{w,\max}$

3. **Force/torque bounds.**

4. **Boundary conditions:**

$$\mathbf{r}_\text{com}^s(0) = \mathbf{r}_\text{com}^{s,0},\quad \mathbf{v}_\text{com}^s(0) = \mathbf{v}_\text{com}^{s,0},\quad \mathbf{L}_\text{com}^s(0) = \mathbf{L}_\text{com}^{s,0}$$

$$\mathbf{r}_\text{com}^s(M) = \mathbf{r}_\text{com}^{s,\text{goal}},\quad \mathbf{v}_\text{com}^s(M) \approx \mathbf{0},\quad \mathbf{L}_\text{com}^s(M) \approx \mathbf{0}$$

5. **Terminal momentum margin** (multi-step budgeting):

$$\mathbf{c}_\text{terminal} - \mathbf{L}_\text{com}^s(M) - \mathbf{r}_\text{com}^{s,\text{goal}} \times m\,\mathbf{v}_\text{com}^s(M) \in [-\kappa\,\mathbf{h}_\text{max}',\;\kappa\,\mathbf{h}_\text{max}']$$

where $\kappa < 1$ (e.g., $\kappa = 0.7$) ensures the robot doesn't consume the full budget, preserving margin for the next step. This is the **multi-step momentum budgeting** mechanism — each step must leave enough margin for subsequent steps.

**Cost:**

$$J_\text{coarse} = w_T\,T_\text{step} + \sum_k \left[w_L\|\mathbf{L}_\text{com}^s(k)\|^2 + w_u\|\mathbf{f}_\text{stance}(k)\|^2\right]$$

$w_T\,T_\text{step}$ penalizes slow steps (if $T_\text{step}$ is a decision variable). $w_L$ keeps momentum small. $w_u$ smooths forces.

**Output:** a coarse momentum-feasible CoM trajectory $\{\mathbf{r}_\text{com}^s(k)\}_{k=0}^M$ and step duration $T_\text{step}$.

**How the output feeds into the architecture:**

- The TorsoPlanner interpolates the coarse trajectory with a smooth quintic/spline (no longer generates its own blind geometric path)
- The ContactScheduler uses $T_\text{step}$ for phase timing
- The NMPC tracks the coarse plan as its reference, refining it at 10 Hz with real state feedback

**Execution:** runs **once per step** before the step starts. With $M = 15$ collocation points and $n_x = 9$, the NLP has ~270 variables — solvable in ~100 ms with CasADi/IPOPT.

### Swing Trajectory with 6D Reference

The current SwingPlanner generates position + clearance bump only. For full 6D EE tracking, orientation is added via synchronized SLERP.

**Position** (unchanged):

$$\mathbf{p}_\text{ee}(t) = \text{quintic}(\mathbf{p}_\text{start},\;\mathbf{p}_\text{dock}) + \text{clearance\_bump}(t)$$

**Orientation** (new):

$$\mathbf{R}_\text{ee}(t) = \text{SLERP}\!\left(\mathbf{R}_\text{start},\;\mathbf{R}_\text{dock},\;\sigma(t)\right)$$

where:

- $\mathbf{R}_\text{start}$: EE orientation at weld release
- $\mathbf{R}_\text{dock}$: required dock orientation (from `dock_configuration` IK)
- $\sigma(t) \in [0,1]$: delayed cosine timing function

**Timing function** — concentrates rotation in the second half of the swing (approach phase) where orientation accuracy matters:

$$\sigma(t) = \begin{cases} 0 & t < t_\text{delay} \\ \frac{1}{2}\left[1 - \cos\!\left(\pi\,\frac{t - t_\text{delay}}{T_\text{swing} - t_\text{delay}}\right)\right] & t \geq t_\text{delay} \end{cases}$$

with $t_\text{delay} \approx 0.2\,T_\text{swing}$ (orientation stays at start during early clearance phase).

**Angular velocity reference** (from SLERP differentiation):

$$\boldsymbol{\omega}_\text{ee}^\text{ref}(t) = \dot{\sigma}(t)\,\frac{\theta}{\sin\theta}\,\text{Log}\!\left(\mathbf{R}_\text{start}^\top\,\mathbf{R}_\text{dock}\right)$$

where $\theta$ is the total rotation angle and $\text{Log}: SO(3) \to \mathfrak{so}(3)$ is the logarithmic map.

**Full 6D reference:**

$$\mathbf{x}_\text{ee}^\text{ref}(t) = \left[\mathbf{p}_\text{ee}(t),\;\mathbf{R}_\text{ee}(t),\;\dot{\mathbf{p}}_\text{ee}(t),\;\boldsymbol{\omega}_\text{ee}^\text{ref}(t)\right]$$

This feeds into the QP's priority-2 EE task. The null-space projection ensures it doesn't fight the torso task.

### Architecture Diagram

```
                    Handhold selection (future)
                           │
                    Coarse pre-planner (NEW, §4.2)
                    ├── momentum-feasible CoM traj
                    ├── step duration T_step
                    └── terminal momentum margin κ
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
   TorsoPlanner      SwingPlanner 6D    ContactScheduler
   (interpolates      (pos + SLERP       (uses T_step)
    coarse traj)       to dock ori)
          │                │                │
          ▼                ▼                ▼
    ┌─────┴────┐    ┌──────┴─────┐    ┌────┴─────┐
    │ CoM ref  │    │  6D EE ref │    │ Contact  │
    │→ mapping │    │            │    │ config   │
    │→ torso   │    │            │    │          │
    └────┬─────┘    └──────┬─────┘    └────┬─────┘
         └────────────┬────┘               │
                      ▼                    │
              CentroidalNMPC (10 Hz)  ◄────┘
                      │
              r_com, v_com, λ_ref
                      │
                      ▼
              Mapping Layer (§3.7)
                      │
              r_b_ref, R_b_ref
                      │
                      ▼
              WholeBodyQP (100 Hz)
                      │
                   τ_q, τ_w
```


---

## 6. Operational Design

### Two-Phase State Machine

The locomotion cycle mirrors terrestrial legged locomotion (Henze et al., 2016; Mishra et al., 2022) with exactly two phases per step:

#### Phase 1: DS (Double Support — both end-effectors welded)

- **Purpose:** Settle after dock; replan for next step
- **NMPC:** 2 active contacts, settling trajectory ($\mathbf{v}_\text{com} \to 0$, $\mathbf{L}_\text{com} \to 0$)
- **QP:** torso 6D only + passivity constraint $\dot{\mathbf{q}}^\top\boldsymbol{\tau}_q \leq -2\alpha T$
- **Exit:** $T < T_\text{settle}$ (energy-based, not time-based)
- **On exit:** (1) IK for next step, (2) optionally coarse pre-planner (§4.2), (3) prepare 6D swing trajectory (§4.3), (4) reset NMPC warm start, (5) record $\mathbf{c} = \mathbf{h}_w^{s,0} + \mathbf{L}_\text{robot}^{O_s,s,\text{in},0}$

#### Phase 2: SS (Single Support — swing arm free)

- **Purpose:** Move torso + swing arm to next stance and dock
- **On entry:** release weld, ramp EE task weight over $T_\text{ramp} \approx 0.3$ s (prevents momentum spike), switch NMPC to 1 contact
- **NMPC:** 1 active contact with momentum box (§3.6)
- **QP:** torso 6D (P1) + EE 6D (P2) + posture (P3) + soft CoM (§9). **No passivity constraint** (energy injection needed)
- **Gain scheduling** — distance-dependent: $K_p^\text{ee}(d) = K_p^\text{cruise} + (K_p^\text{final} - K_p^\text{cruise})\cdot\sigma((d_\text{approach} - d)/d_\text{approach})$. Eliminates hard EXT phase switch.
- **Exit (dock):** $\|\mathbf{p}_\text{ee} - \mathbf{p}_\text{dock}\| < r_\text{weld}$ AND $\|\text{Log}(\mathbf{R}_\text{ee}^\top\mathbf{R}_\text{dock})\| < \theta_\text{weld}$ → activate weld → DS
- **Failure:** tracking divergence → pause/resume; NMPC infeasible → $\boldsymbol{\lambda}_\text{ref} = 0$; RWA near saturation → slow swing


### Identified Risks and Mitigation


#### Mapping valid only at current configuration

$\boldsymbol{\delta}(\mathbf{q})$ uses current limb positions. Between NMPC ticks (100 ms), limbs move — the torso reference drifts from the true CoM-achieving position. Recomputed at 100 Hz QP rate, so error is bounded by swing arm velocity × arm mass / torso mass per QP tick.

**Mitigation:** Recompute $\boldsymbol{\delta}$ every QP tick (negligible cost, already doing FK).

#### Indirect cascade weakens momentum guarantee

NMPC optimizes $\mathbf{r}_\text{com}$; QP tracks $\mathbf{r}_b$. Torso tracking error $\mathbf{e}_b$ induces CoM error $\approx (m_b/m_\text{total})\,\mathbf{e}_b$ — attenuated but nonzero. Near the $\mathbf{h}_w$ box boundary, this bias may matter.

**Mitigation:** Add a low-weight soft CoM residual in the QP cost: $w_\text{com}^\text{soft}\|\mathbf{J}_\text{com}\ddot{\mathbf{q}} - \ddot{\mathbf{x}}_\text{com}^\text{des}\|^2$ — not a competing task, just a gentle incentive.

#### Loss of direct momentum observability in QP

With a torso task (not CoM), the QP no longer "sees" momentum in its objective. If the $\mathbf{h}_w$ constraint is inactive, there is no incentive to minimize momentum.

**Mitigation:** The NMPC plan is the primary momentum manager. The QP's $\mathbf{h}_w$ box constraint remains as a safety net.

#### Torso orientation reference inconsistency

TorsoPlanner generates orientation jointly with position (quintic + SLERP). If position now comes from NMPC mapping while orientation comes from TorsoPlanner, the two may be inconsistent (one momentum-optimal, the other geometric).

**Mitigation:** Regenerate orientation trajectory consistent with NMPC position plan, or accept small coupling errors.

#### Double-support phase degeneracy

In DS, 12 DOFs consumed by contacts, 8 free. $\boldsymbol{\delta}$ is nearly constant (limbs rigidly attached). The mapping is algebraically valid but the effective inertia changes dramatically.

**Mitigation:** Phase-dependent QP gains (already implemented in test code: `qp_ds`, `qp_ss`, `qp_fin`).

#### Singularity near full arm extension

During EE approach, J_torso and J_ee may become rank-deficient in similar directions. The null-space projection $\mathbf{N}_\text{torso}\mathbf{J}_\text{ee}$ could lose rank at dock.

**Mitigation:** Monitor condition number of projected Jacobian; damped pseudo-inverse in the null-space projection.

#### NMPC blind to orientation objective — RESOLVED

The NMPC has no notion of torso orientation. Large reorientations generate angular momentum not budgeted by the centroidal plan.

**Resolution:** The TorsoPlanner (§4.3) provides $\mathbf{L}_\text{com}^\text{ref}(t) = \mathbf{I}_\text{torso}^\text{com}\,\boldsymbol{\omega}_\text{ref}(t)$ to the NMPC. The cost $w_L\|\mathbf{L}_\text{com} - \mathbf{L}_\text{com}^\text{ref}\|^2$ accepts the planned rotation's momentum while still constraining unplanned disturbances. Residual approximation error ($\sim 20\%$ from ignored limb inertia) is corrected by the NMPC feedback loop.

---

## 7. Validation


| Component | Before | After |
|---|---|---|
| QP tasks | CoM (3D) + Torso (6D) + EE (3–6D) = 12–15 DOF | Torso (6D) + EE (6D) = 12 DOF |
| CoM tracking | Explicit $\mathbf{J}_\text{com}$ task in QP, priority 1 | Folded into torso position ref via mass mapping |
| NMPC outputs | $\mathbf{r}_\text{com}, \mathbf{v}_\text{com}$ → QP directly | $\mathbf{r}_\text{com}, \mathbf{v}_\text{com}$ → mapping → $\mathbf{r}_b^\text{ref}, \dot{\mathbf{r}}_b^\text{ref}$ |
| EE orientation | Sacrificed (DOF deficit) | Full 6D tracking in 8-dim null space |
| TorsoPlanner | Generates independent 6D torso ref | Generates orientation ref; position from NMPC mapping |
| Warm start | Carried across dock events | Reset to fresh rollout at new contact config |

---


### Momentum Algebra (unit, no simulation)

- **T1:** $\mathbf{L}_\text{robot}^{O_s,s,\text{in}}$ (§3.3) matches Pinocchio + transport + drag, 5 configs. Pass: error $< 10^{-10}$.
- **T2:** Conservation law algebraic vs. integrated h_w over 1 s. Pass: drift $< 10^{-6}$ Nm·s.
- **T3:** CoM-to-torso mapping Jacobian equivalence. Pass: error $< 10^{-10}$.

### NMPC Standalone (B2)

- **T4:** B2 from rest, 0.3 m displacement, $h_\text{max} = 5$ Nm·s. Pass: converges, box satisfied.
- **T5:** 0.5 m vs. 3.0 m from $O_p$. Pass: step time at 3.0 m longer (envelope verified).
- **T6:** Two consecutive steps, $\kappa = 0.7$. Pass: terminal margin respected.

### QP Standalone

- **T7:** Torso 6D + EE 6D null-space, SS. Pass: torso $< 5$ mm, EE pos $< 10$ mm, EE ori $< 5°$.
- **T8:** Soft CoM residual on vs. off. Pass: CoM tracking improves.
- **T9:** Dynamics residual $< 10^{-8}$.
- **T10:** DS passivity: $T(t)$ decays exponentially within 5% of theoretical bound.

### Closed-Loop Single Step (MuJoCo)

- **T11:** DS→SS→DOCK, 1% mass ratio. Pass: dock $< 5$ mm / $5°$, $h_w$ in box.
- **T12:** Same at 14%. Pass: dock succeeds, rotation $< 5°$.
- **T13:** AOCS feedforward accuracy $< 0.1$ Nm.
- **T14:** DS settling within $t_\text{settle} \pm 20\%$.

### Closed-Loop 3-Step (MuJoCo)

- **T15/16:** 3 docks at 1% and 14%. Pass: all docks, $h_w$ in box.
- **T17:** EE orientation at dock $< 5°$ (46° failure resolved).
- **T18:** NMPC $> 95\%$ solve rate within 50 ms.
- **T19/20:** Dynamics residual + zero QP failures across traversal.

---

## 8. Contribution and Open Questions


A centroidal NMPC plans momentum-feasible locomotion trajectories for a platform-mounted crawling robot, with constraints derived from the angular momentum conservation law expressed rigorously via SE(3) adjoint maps in the platform body frame. The RWA saturation constraint bounds both the centroidal (spin) and orbital components of the robot's angular momentum, yielding a position-dependent feasibility envelope that couples locomotion planning to the platform's AOCS capacity. The whole-body QP tracks these trajectories via a prioritized task stack where the NMPC's centroidal plan is mapped to a floating-base reference (Belvedere et al., 2022), adapted to the free-floating case where balance is maintained by RWA momentum management rather than ground reaction forces. A passivity constraint during double support guarantees exponential energy decay and deterministic settling (extending Mishra et al., 2022 to a QP framework with settling rate guarantees; Kalaycioglu & De Ruiter, 2023 to constrained locomotion with momentum management). The position-dependent feasibility envelope analytically characterizes the velocity bound identified by Rognant et al. (2019, 2025) and provides the closed-loop trajectory optimization they lack.

---


- [x] Soft CoM residual: priority level or regularization? → **Soft cost** (§9.7)
- [x] TorsoPlanner refactoring → SLERP + L_com feedforward (§4.3)
- [ ] CoM tracking error budget for momentum safety margin?
- [ ] NMPC torso orientation dynamics ($\geq 3$ additional states)?
- [ ] Warm-start reset strategy?
- [ ] $\alpha_\text{com}^\text{soft}$ optimal value (simulation sweep: 1, 5, 10)?
- [ ] Passivity constraint $\alpha$ tuning (simulation sweep)?
- [ ] Level 2 energy budget during SS — formalize and test?
- [ ] 7-DOF arm URDF/MJCF specification (gap E)?
- [ ] Numerical parameter table consolidation (gap K)?
