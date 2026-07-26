# PORT-AUDIT — synthesis & decisions (2026-07-18)

Companion to `PORT_AUDIT.md` (read-only inventory, bins A–E + checklist). This note fixes the
effort estimate, the latent-bug tickets, and the turnkey roadmap. No code change now: the
`32aefaf` canonical stays frozen through submission.

## Effort estimate (three tiers)

- **Tier 1 — same topology** (2 arms on a free-flyer torso, 3 wheels, any per-arm DOF):
  author the two mutually-consistent model files (8 hand-duplicated quantities, A-a2), honor
  the naming contract (B-b1), edit ~5 code literals (armature, torso-mass assert, inertia
  fallback, per-script paths). **Days**, dominated by careful model authoring. Per-arm DOF ≠ 7
  already works (b3-note).
- **Tier 2 — different topology** (arm count ≠ 2, wheel count ≠ 3 / CMG, TCP offset or
  non-identity mating): structural change across NMPC (nx=9/nu=12) + AOCS + state conversions
  + the identity-anchor assumption spread over 4 files (E-1/2). **Weeks.**
- **Tier 3 — re-validation** (always due, code-independent): Add-5 weights, torque-min
  feasibility gate, force/torque/momentum budgets, PD gains, CoM-z standoff, capture gate —
  all tuned to the 1% mass ratio. This is a canonical-freeze campaign; **the dominant cost.**

## Latent-bug tickets (all PROVEN dormant on the canonical; fix post-submission)

1. `away_normal` sign duplication: swing_planner.py:44 `[0,0,-1]` vs ik.py:1402 `[0,0,-1]`
   but ik.py:1282 default `[0,0,+1]` with a comment (ik.py:1281) that wrongly claims agreement.
   Dormant: `use_path_feasibility_check=False` on the canonical path (QP-STACK audit).
2. `tau_max` scatter 20/10/50/±50: robot_interface.py:155 default 10 is NOT overridden at
   construction ⇒ `state.tau_max=10` disagrees with cfg 20, QP default 50, plant ±50.
   Dormant: sim_loop clips with cfg.tau_max and overrides the QP bound. Audit consumers of
   `state.tau_max` before fixing.
3. Anchor-index parser single char: sim_loop.py:1207 `int(parts[1][0])-1` breaks at index ≥ 10.
   Dormant: anchors 1–6. Same family: RWA parity detection (state_conversions.py:58) breaks on
   odd total arm-DOF.
4. Deprecated module constants (robot_interface.py:76-86) still imported by ik.py:17 —
   overwritten at init, fragile import path.
5. Stale comment sim_loop.py:13 (ctrl[12:15] → actual 14:17).

## Turnkey roadmap (post-submission, order of value)

1. **Consistency validator** before any refactor: a script that loads both models and diffs
   the 8 BIN-A-a2 quantities + the naming contract; run in CI/diagnostics. Cheapest first
   because it converts silent disagreement into an error, which is the whole game.
2. Promote BIN-B literals + model paths into one `robot.yaml` (names, paths, armature,
   asserts); BIN C shows there are exactly four scatter sites today.
3. Fix tickets 1–5.
4. Only then consider Tier-2 generalizations (wheel/CMG count, arm count) if a partner needs
   them; do not pre-generalize.

The DLR-facing document is `PORT_AUDIT.md`'s checklist (steps 1–10) as-is.
