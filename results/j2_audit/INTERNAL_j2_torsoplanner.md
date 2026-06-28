# INTERNAL — TorsoPlanner audit: grounding a "DS-mobile" mode (`ae0673e`, READ-ONLY)

**Goal:** map the TorsoPlanner's mechanics so the Q2 "DS-mobile" design (a **moving** torso-orientation +
arm-posture reference in the reorientation null space, gait-triggered, **not** CoM transport) stands on
verified ground. Map only — no design, no implementation. Branch `j2/ds-active-rework`. Reproducer
`scripts/audit_torsoplanner.py` (**16/16**). The wiring-vs-building call per item, as the J2 audit did.

**TL;DR — DS-mobile is mostly WIRING for the torso-orientation half and BUILDING for the arm-posture half.**
The planner's time-varying trajectory machinery (`add_phase`/`reference_at`) already exists and is used in
SS; `set_hold` is just its degenerate static case. The QP already tracks `R_torso_ref` **per tick**, so a
moving torso-ori reference flows through `_step`→`qp.solve` unchanged. The reorientation freedom at a docked
config is **5-D (3 torso-ori + 2 arm-posture)**. The 3 torso-ori are interface-ready; the **2 arm-posture**
DOF are pinned by an existing posture task to a **fixed** `q_nominal` with **no per-tick target** — moving
them needs new plumbing.

| item | exists today | DS-mobile = | wiring / building |
|---|---|---|---|
| (i) moving torso-ori trajectory | `add_phase`/`set_from_waypoints` + `reference_at` interpolation (used in SS) | call it in DS instead of `set_hold` | **WIRING** |
| (ii) gait trigger | DS `set_hold(equilibrium)` call sites; gait context (anchors, step) in scope | build a moving phase at that hook | **WIRING** |
| (iii-a) QP track moving torso-ori | `_step`→`qp.solve(R_torso_ref=…)` per tick; torso-ori-3D task | nothing — already per-tick | **WIRING** |
| (iii-b) QP track moving arm-posture | posture task active in DS, but **fixed** `q_nominal`, no per-tick arg | per-tick posture target + a posture-trajectory source | **BUILDING** |
| (iv) reorientation null space | `null(Jc)`=8; at fixed CoM **=5** (3 ori + 2 posture) | command a moving ref within the 5-D space | (3 ori WIRING, 2 posture BUILDING) |

---

## Q1 — TorsoPlanner output + how `set_hold` works

**File/class:** `crawlbot/planning/torso_planner.py`, class `TorsoPlanner` (all quantities in the **structure
frame**).

**Output (three accessors, queried by `_step`/NMPC each tick):**
- `reference_at(t) → TorsoReference` = **`p(3)` + `R(3×3)` + `v(6)` twist + `a(6)` accel** — a full SE(3)
  pose **with derivatives** (`torso_planner.py:40`).
- `com_reference_at(t) → ComReference` = `r_com(3)` + `v_com(3)` (`:49`).
- `l_com_reference_at(t) → L_com_ref(3)` — the NMPC angular-momentum feedforward.

**`set_hold(p, R, r_com)` (`:125`)** stores `_hold_p/_hold_R/_hold_com` and `_hold_reference()` (`:543`)
returns them with **`v=0, a=0`** — a **pure static setpoint**, not a trajectory.

**Internal DOF/params that could be made time-varying:** `_phases` (the trajectory list),
`_hold_p/_hold_R/_hold_com` (static targets), `_I_torso_body`. **The time-varying machinery already exists:**
`add_phase`/`set_from_waypoints` (`:195`/`:140`) build piecewise quintic / trapezoidal / FK phases, and
`reference_at` returns the **interpolated `p,R,v,a`** when `t` is inside a phase, else falls through to the
static hold (`:439`). FK-mode phases can even carry a joint trajectory `q_seq` (`:205`) — but that feeds
CoM/`L_com` FK, **not** an arm-posture output (the planner has **no** posture output).

**What DS-mobile must add structurally:** for the **torso-ori half**, *nothing structural* — call
`add_phase` with a moving-orientation, constant-CoM phase over the DS window instead of `set_hold`
(**WIRING**). For the **arm-posture half**, the planner produces **no** posture reference → that is **new**
(BUILDING; see Q3).

## Q2 — Gait state-machine trigger (ContactScheduler ↔ planner)

- **SS swing** drives the planner **time-varying**: `self.torso_planner.add_phase(...)` (`sim_loop.py:1612`,
  built from the IK waypoint sequence over `[0, T_step]`).
- **DS** (initial / trailing / DWELL) drives it **static**: `self.torso_planner.set_hold(welded-equilibrium)`
  (`sim_loop.py:1551` initial; `:2315/2327/2333` trailing-DS, computing the both-tools-at-anchors
  equilibrium via IK or current state).
- `_step` calls `reference_at(t)` **every tick** (`sim_loop.py:2643`) → `qp.solve`. So whatever the planner
  holds is re-queried per tick; **a moving DS phase would flow through `_step` with no change**.

**Natural DS-mobile hook:** the DS `set_hold` call sites (`:2315` trailing-DS, the DWELL block, the
inter-step DS) — the gait event that starts a locomotion DS that needs reorientation. **The gait context is
already in scope there** (`last_sa`/`last_sb` anchors, `step_idx`, the scheduler `anchor_se3`), so the
trigger needs **no new plumbing** — replace/augment the `set_hold(equilibrium)` with an `add_phase(moving
reorientation, fixed CoM)`. **WIRING.** (ContactScheduler phases `SINGLE_A/B/DOUBLE` are consumed via the
scheduler in the gait loop that surrounds these sites; the DS blocks already know the phase.)

## Q3 — QP consumption of the reference

- **Interface (time-varying-capable):** `_step` passes `p_torso_ref`/`R_torso_ref` into `qp.solve` **per
  tick** (`sim_loop.py:2789`, values from `reference_at`). Not shaped for a static hold — it tracks whatever
  is current.
- **DS tracking tasks** (`wholebody_qp.py:1065`, gated `settle_mode ∧ ds_centroidal_mode ∧
  ds_centroidal_active`): **CoM-3D** (`ds_alpha_com`, P1) + **torso-ANGULAR-3D** (`ds_alpha_torso_ori`, P1,
  tracks `R_torso_ref`). ⇒ a **moving torso-ori target injects with zero interface change** (**WIRING**).
- **Arm-posture — there IS a third DS task, but it's fixed.** A **posture task** is active in DS-centroidal
  (`_posture_in_ds`, `wholebody_qp.py:987`; comment: *"posture is needed to constrain the 2 arm-null-space
  DOFs"*), but it tracks **`self._q_nominal`** (`:990`) — a **fixed** config, set **once** at construction
  (`set_q_nominal`, `sim_loop.py:298`). **`qp.solve` has no per-tick posture-target argument.** So the
  arm-posture half of Q2's reference needs **new plumbing**: a per-tick posture target into `qp.solve` (or a
  mutated `q_nominal`) **and** a posture-trajectory source (the planner has none today; FK `q_seq` is a
  candidate but is not wired to the posture task). **BUILDING.**

⇒ DS tracking tasks = **CoM-3D + torso-ori-3D + posture(q_nominal)**. Torso-ori is moving-ready; posture is
not.

## Q4 — Reorientation null space and its dimension

- **`P_int = I − G⁺G` (`wholebody_qp.py:1170`) is NOT this space.** It is the 12-D contact-**wrench**
  internal-stress null space — a **force** object (the hyperstatic *wrench* redundancy resolved by the
  internal-stress regularizer). The **motion** reorientation freedom is a different object: the null space of
  the **contact Jacobian** `Jc`.
- **Computed at a docked 2-contact config** (real robot, `models/VISPA_crawling_fixed.urdf`, `nv=20`, the
  `final` dock snapshot of `fixA_gate`):
  - `Jc = [J_tool_a ; J_tool_b]` (12×20), **rank 12 ⇒ dim null(Jc) = 8** — internal motions with **both
    welds active**.
  - `[Jc ; J_com]` (15×20), **rank 15 ⇒ dim null([Jc; J_com]) = 5** — reorientation + posture at **fixed
    CoM**: **the DS-mobile command space.**
  - Decomposition: 8 (welds) − 3 (CoM-transport) = **5 = ~3 torso-ori + ~2 arm-posture**. The "**2
    arm-null-space DOFs**" the QP posture comment pins are exactly the 2 left after the torso-ori-3D task
    commands 3. This **bounds a DS-mobile reference: at most 5 DOF**, of which 3 are torso-orientation and 2
    are arm-posture.
- **Existing task to extend:** the **torso-ori-3D** task already commands the 3 orientation DOF (just with a
  static target today → moving = WIRING); the **posture** task already pins the 2 arm DOF (to fixed
  `q_nominal` → moving = BUILDING). No genuinely new *task* is required for either half — the torso-ori task
  takes a moving target for free; the posture task needs a moving target fed in.

---

## Divergences / refinements vs the prior J2-audit facts

1. **"TorsoPlanner is held (`set_hold`) ⇒ NMPC tracks a static docked setpoint"** (J2 audit / Part B) —
   **CONFIRMED**. Refinement: the **time-varying machinery already exists** (`add_phase` + `reference_at`,
   used in SS); DS-mobile **reuses** it rather than building a trajectory generator. `set_hold` is the
   degenerate case.
2. **Centroidal-DS tasks = CoM-3D + torso-ori-3D** (J2 audit Bloc-3 / envelope audit) — **CONFIRMED**, but
   **incomplete**: there is **also a posture task** active in DS-centroidal (`_posture_in_ds`,
   `wholebody_qp.py:987`) tracking fixed `q_nominal`. The prior audit noted posture is "re-enabled in DS" but
   did not frame it as the **arm-posture hook** for a DS-mobile reference. Stated here.
3. **`P_int` is the wrench null space, not the reorientation freedom** — clarified (Q4 explicitly asked). The
   reorientation freedom is `null(Jc)` (motion), 8-D with welds / **5-D at fixed CoM**.
4. The **QP interface is already per-tick** for `R_torso_ref` (so torso-ori is moving-ready) but has **no
   per-tick posture target** (so arm-posture is not) — the wiring/building split is *within* the QP
   consumption, not at the planner alone.

**STOP — doc-first.** The DS-mobile design discussion follows the digest, then the implementation brief. No
design, no implementation, no `crawlbot/` change, no `main`, no PR.
