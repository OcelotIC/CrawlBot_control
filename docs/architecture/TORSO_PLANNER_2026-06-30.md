# TorsoPlanner — as-it-runs audit (2026-06-30)

Read-only root-cause audit of `crawlbot/planning/torso_planner.py` for the **torso orientation-reference
defect** (audit CASE B, `618ddcf`: the orientation reference is re-anchored on every FSM sub-phase entry, so the
torso accumulates ~0.72° of drift on the canonical run, unconstrained against any global reference). Goal: decide
whether the defect is in the **planner** (it defines a per-phase orientation reference) or in **`sim_loop`'s
sourcing** (it feeds the planner a per-phase pose), so the fix lands in the right place.

**Verdict: the defect is in `sim_loop`'s SOURCING, not the planner.** The TorsoPlanner is a neutral,
FSM-agnostic interpolation container; it faithfully holds/SLERPs whatever poses it is handed. It **can** express a
global flat orientation (feed it a constant `R`). The per-phase re-anchoring is entirely in `sim_loop`'s
`set_hold(R_now)` / `add_phase(R_t0,…)` / dock-IK `R_torso_fixed=R_t0` calls. So the orientation fix needs **no
planner change** — it is a change to which `R` `sim_loop` sources.

This doc supersedes, for the orientation question, the pre-J2 `results/j2_audit/INTERNAL_j2_torsoplanner.md`
(`ae0673e`, framed for a "DS-mobile" design). `docs/architecture/SS_TORSO_GUIDANCE_AUDIT_2026-06.md` covers only
the **position/CoM** reference (δ_com); `docs/architecture/IK_FORMULATION.md` (2026-06-23) does **not** mention the
torso orientation-hold choice (`R_torso_fixed`) at all. No existing `docs/` doc described the orientation path —
hence this one.

---

## 1. What the planner exposes as the orientation reference
Three query accessors, all in the **structure body frame** (`torso_planner.py`):
- `reference_at(t) → TorsoReference(p, R, v, a)` (`:439`) — the 6-DoF torso pose + twist + accel. **`R` is the
  orientation reference.**
- `com_reference_at(t) → ComReference(r_com, v_com)` (`:450`) — position/CoM (the SS_TORSO_GUIDANCE subject).
- `l_com_reference_at(t) → (3,)` (`:475`) — NMPC angular-momentum feedforward.

`reference_at(t)` resolves `R` by one of three paths:
1. **Legacy task-space SLERP** (`_interpolate_phase`, `:630-650`) — within an active phase:
   `R = R_start · exp3(s · log3(R_startᵀ R_end))`, i.e. **geodesic SLERP from the phase's `R_start` to its
   `R_end`**. With `R_start == R_end` this is **constant `R_start`, ω = α = 0** (verified: `log3(RᵀR)=0`).
2. **Hold** (`_hold_reference`, `:532-545`) — outside all phases: returns `self._hold_R` (set by `set_hold`), or
   the last phase's `R_end`, or identity.
3. **FK-on-smoothed-q** (`_reference_at_fk`, `:380-395`) — only when a phase carries `q_seq`
   (`reference_source='joint_space_fk'`): `R` is the torso frame orientation from FK on the smoothed geodesic
   `q(τ)`. **Dormant in the canonical run** (`reference_source` default `'task_space'`, `config.py:509`; the
   figure config does not override it).

**The planner does not define a mission orientation.** `R_start`/`R_end`/`_hold_R` are **inputs** supplied by the
caller; the planner only stores and interpolates them. It has **no notion of "flat", no global setpoint slot, and
no FSM/phase-type (DS/SS) concept** — `_phases` is a plain time-windowed list keyed on `[t_start, t_end]`.

⇒ This is the second sub-option from the brief's Q1, but with the nuance that the planner is **neutral, not
actively per-stance**: it re-defines orientation per phase **only because `sim_loop` re-seeds it per phase**.

## 2. Where the FSM coupling lives — `sim_loop`, not the planner
The planner has **zero** state-machine coupling. Every re-anchoring is a `sim_loop` call that sources the **live
current** torso orientation `R_now` (or a free-IK equilibrium), never a persistent global reference:

| site | `sim_loop.py` | sourced orientation |
|---|---|---|
| SS dock-IK target | `:1478` `dock_configuration_fixed_rotation(…, R_torso_fixed=R_t0)` where `R_t0 = rs_s.oMf_torso.rotation` (`:1411`) | **current** → `R_t1 ≈ R_t0` |
| SS hold (pre-swing) | `:1676` `set_hold(p_t0, R_t0, …)` | **current** |
| SS phase | `:1737-1740` `add_phase(…, R_t0, …, R_t1, …)` | **current → ≈current** |
| DS inter-step hold | `:2099-2102` `set_hold(…, rs_d.oMf_torso.rotation, …)` | **current** |
| DS terminal hold | `:2479-2500` `set_hold(…, rs_hold/rs_eq.oMf_torso.rotation, …)` | **current**, else **free-IK welded-equilibrium** (`dock_configuration`, `:2487`, **no** `R_torso_fixed`) |

The intent is explicit at `:1415-1416`: *"prefer to hold torso rotation at R_start ('crawl forward, don't
pirouette')."* So the per-stance hold is **deliberate** — but it holds *"wherever the torso is now"* at each entry,
not a global reference, which is exactly the CASE-B accumulation. The consumer `reference_at`→`R_err` at `:943-949`
then reports the geodesic error vs this just-re-seeded reference, so the logged `torso_ori_err_deg` re-zeros each
phase and **masks** the cumulative drift.

## 3. Is "flat, held globally, FSM-independent" expressible? — YES, in `sim_loop`, no planner change
Because `_interpolate_phase` with `R_start == R_end == R_flat` returns a **constant** `R_flat` (ω = α = 0), and
`_hold_reference` returns `_hold_R` verbatim, the planner already represents a global flat orientation **if fed a
constant `R_flat`**. The fix is to source a single stored `R_flat` (e.g. the t=0 flat orientation) at the five
sites above — the dock-IK `R_torso_fixed`, the SS `set_hold`/`add_phase`, and the two DS `set_hold`s — instead of
the live `R_now`. Cost is small: the drift to correct is ~0.72°, so pinning to `R_flat` adds a ≤~1° de-rotation
per step, not a "pirouette." **No TorsoPlanner change is required for the orientation fix.** (FK mode would need
the same idea applied to the `q_end` rotation constraint, but FK mode is dormant in the canonical run.)

## 4. Stale assumptions ("it dates") — confronted against the current architecture
The planner's **orientation logic is NOT stale** (SLERP via `log3`/`exp3` is correct and architecture-agnostic;
it carries no EXT-phase or prior-FSM residue — a point in its favour). The debt is in surrounding profile/momentum
machinery:

1. **DEAD CODE — `_trapezoidal_params` (`:547-603`).** Never called: `_profile_params` (`:605`) unconditionally
   returns `_quintic_params` (`:615`). Its body comments (`:570-574`) assert *"a_torso_ff ≡ 0 during cruise"* and
   *"planned-δ mapping (v19) continues to supply feedforward through v_b_ref"* — the **planned-δ mapping was
   reverted** to `δ(q_current)+F-SAT` (CLAUDE.md, `50a9e52`), so the comment describes an abandoned mechanism, and
   the code path is dead regardless.
2. **Stale version-churn comments — `_profile_params` (`:606-614`).** The v18→v20→v21 narrative claims *"the
   cruise-phase shaping is now handled at the preplanner level (CoM acceleration constraint) which flows through
   the mapping into the torso linear reference."* CoM shaping is **OFF** (`a_cruise_max = 0`, CLAUDE.md), so this
   describes an inactive mechanism.
3. **Legacy torso-only `l_com_reference_at` (`:475-528`) + `set_torso_inertia` (`:97-123`).** The torso-only
   `L = R·I_torso·Rᵀ·ω` formula carries a documented ~20 % limb-contribution error and is **active in the
   canonical (task_space) run** (FK mode, which uses exact `pin.computeCentroidalMomentum`, is dormant). The
   NMPC's `L_com_ref` therefore rides a ~20 %-approximate feedforward, absorbed by the `w_L‖L−L_ref‖²` feedback.
   *(Secondary: this is the momentum FF, not the orientation reference.)*
4. **Module docstring (`:1-31`)** documents only the legacy CoM-from-torso `r_com = p + R·δ_com` path as the
   design; it predates and does not mention FK mode (`_add_phase_fk` et al.). Accurate for the canonical
   task_space run; incomplete as a description of the file as a whole.

## 5. Secondary red flags (raised, NOT fixed — per brief scope)
- **Terminal-DS orientation source is inconsistent with SS:** `:2487` `dock_configuration(...)` is **free-rotation**
  (no `R_torso_fixed`), so the terminal hold orientation is the welded-double-stance equilibrium — a *third*
  non-global orientation source, different from the SS fixed-rotation choice. (Orientation-adjacent.)
- **`set_from_waypoints` hold comment (`:181-193`)** is self-contradictory ("We'll update `_hold` to the end after
  the trajectory is done") but never updates `_hold` to the end — it sets hold to the **first** waypoint and relies
  on `_hold_reference` falling back to the last phase's `p_end`. Confusing; works by accident of fallthrough.
- **Position/CoM reference coupling** (`com_reference_at`, δ_com interpolation) is the separate
  SS_TORSO_GUIDANCE_AUDIT subject — not re-audited here.

## Where the fix goes (for Idriss to decide — NOT done here)
Orientation: **`sim_loop`** — source a stored global `R_flat` at the five sites in §2 instead of `R_now`. The
TorsoPlanner is sound for this and needs no change. Separately, the dead trapezoidal code (§4.1) and the stale
comments (§4.2) are cleanup candidates, and the legacy `l_com` formula (§4.3) is a known approximation to retire
if/when FK mode becomes the canonical reference source. **No `crawlbot/` change in this audit.**
