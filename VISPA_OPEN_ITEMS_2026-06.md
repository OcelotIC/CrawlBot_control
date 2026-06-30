# VISPA — open items (2026-06, J2 DS-active-rework)

Verified-open items at the head of `j2/ds-active-rework`. Every number here is reproduced from a **committed**
artifact (`results/j2_figdata/runA_traversal.csv` — the locked-config run `--interstep-settle-alpha-wrench 3
--interstep-settle-epsilon-v 5e-3`, `legacy_pid_numerical`, exact box — and `results/j2_closure_curves/`), not
from memory. These are **limitations / not-done**, not results.

## C1 — terminal-settle window too short (not returned to rest)
The terminal DS is a fixed 20 s window. At sim end the system is **still relaxing**, not at rest:
- `‖h_w‖` end = **0.396 N·m·s**, slope over the last 50 ticks **−0.040 N·m·s/s** (falling).
- `‖θ_s‖` end = **0.098°**, slope last 50 ticks **−0.010°/s** (falling).
⇒ Open: lengthen the terminal settle (or run to an energy/momentum floor) so the end state is a true rest, and
report the settled `h_w`/`θ_s`.

## C2 — CoM position tracking loose (~15× the torso position error)
SS CoM tracking error `‖r_com − r_ref‖`: **median 76.6 mm, max 189.9 mm** (vs torso position ~mm). Design
consequence — the stack servos the torso 6-DoF pose + centroidal **momentum**, but **not** CoM **position**;
this is the trajectory-vs-path-following gap. ⇒ Open: document (it is not a tracked DoF) or add a CoM-position
objective. Do not present it as tracked.

## C3 — torso 6-DoF orientation hold is per-stance-relative, NOT global (audit CASE B, commit 618ddcf)
The 6-DoF torso-pose task **recaptures its orientation target at each phase entry**
(`sim_loop.py:1411,1478,1676` SS `R_torso_fixed=R_t0=current`; `:2099,:2479-2499` DS `set_hold(…, R_now)` —
"crawl forward, don't pirouette"). Position is held globally (mm); **orientation only per-stance**. The
per-phase-reset metric `torso_ori_err_deg` (≤1.25° within-phase) **hid** the real behavior. TRUE cumulative
orientation drift vs a single t=0 reference (verified, `results/j2_closure_curves/TORSO_ORI_RESET_AUDIT.md`):
- structure-frame: **peak 4.12°, final 1.80°**, growing ~monotonically across the 5 steps; structure attitude
  `θ_s ≤ 0.6°`, so it is the **torso drifting relative to its own start**, unconstrained.
- **1.8° is NOT a bound** (it grows; the controller does not constrain it). A cumulative drift is a
  **limitation, not a result.**
⇒ Open: characterize the **full 6-DoF** torso behavior (Option 1 — show all 6 DoF, including the orientation
drift) **or** fix the controller to a **global** orientation reference (Option 2). Showing only the 3
translational DoF would be fraudulent.

## D1 — MuJoCo contact-solver momentum leakage unmeasured
From rest, total system momentum should be ~0 to machine precision. The continuous `‖L_com‖` reaches
**~1.507 N·m·s** during swings (numerical leakage; median 0.056, returns to 0.000 at snapshots). The
per-tick conservation figure was correctly **dropped** (not trustworthy). ⇒ Open: quantify the leakage and
whether it inflates `h_w`/`θ_s`; it is currently **unmeasured**.

---
These items gate any "final J2 results" claim. This PR is **chatter fix + Stage-2 QP architecture + figure/plot
infrastructure**, with C1–C3 and D1 explicitly open.
