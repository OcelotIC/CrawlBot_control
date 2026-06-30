# VISPA — open items (2026-06, J2 DS-active-rework)

Verified-open items at the head of `j2/ds-active-rework`. Every number here is reproduced from a **committed**
artifact run on the **CANONICAL config** (`run_figdata.sh` COMMON `--ss-two-task --ss-alpha-mom 5000
--alpha-torso-pose 24000 --ss-kp-torso 3 --ss-kd-torso 2.5 --aocs_mode legacy_pid_numerical --K_omega 50
--qp-envelope-exact` **+** chatter fix `--interstep-settle-alpha-wrench 3 --interstep-settle-epsilon-v 5e-3`):
`results/j2_figdata/runA_traversal.csv` + `runA_meta.json`, `results/figA_canon/sim_log.json`,
`results/j2_closure_curves/canonical_curves.csv`. These are **limitations / not-done**, not results.

> **Correction note:** the prior version of this file (and PR #27) carried numbers from a **wrong (simplified)
> config** that dropped the SS-stack flags above. All numbers below are now the canonical re-run; the full
> diff vs the wrong-config run is in `results/j2_canonical_revalidation/INTERNAL_canonical_revalidation.md`.

## C5 — reaction-wheel storage exceeds the 4.5 soft gate on the canonical config (NEW; wrong-config hid it)
`h_w∞ = ‖h_w‖∞` (peak, `hw`≡`hw_physical`) = **4.930 N·m·s** (peak on the z-axis, hw_z = −4.930), versus the
C5 soft gate **≤ 4.5** (90 % of the ±5 N·m·s hardware budget). It is **within the ±5 hardware limit (98.6 %)** but
**over the 4.5 soft gate ⇒ C5 FAILS on canonical.** The wrong-config run reported 3.86 (PASS). The canonical SS
stack (`ss-alpha-mom 5000` + `alpha-torso-pose 24000`) commands more aggressive torso motion, which the AOCS wheels
absorb. ⇒ Open: either retune the SS momentum/pose weights to pull the peak under 4.5, or re-justify the gate at the
hardware ±5 limit, or add explicit h_w-budget shaping. Do not present C5 as passing on the figure config.

## C1 — terminal-settle window too short (not returned to rest)
The terminal DS is a fixed 20 s window. At sim end (t=50.94 s) the system is **still relaxing**, not at rest:
- `‖h_w‖` end = **0.338 N·m·s**, slope over the last 50 ticks **−0.034 N·m·s/s** (falling).
- `‖θ_s‖` end = **0.086°**, slope last 50 ticks **−0.0067°/s** (falling).
⇒ Open: lengthen the terminal settle (or run to an energy/momentum floor) so the end state is a true rest, and
report the settled `h_w`/`θ_s`.

## C2 — CoM position tracking loose (~6× the torso position error)
SS CoM tracking error `‖r_com − r_ref‖`: **median 47.1 mm, max 95.1 mm** (vs torso position peak ~23 mm). Design
consequence — the stack servos the torso 6-DoF pose + centroidal **momentum**, but **not** CoM **position**; this is
the trajectory-vs-path-following gap. (Smaller than the wrong-config 76.6/189.9 mm, but the same architectural gap.)
⇒ Open: document (it is not a tracked DoF) or add a CoM-position objective. Do not present it as tracked.

## C3 — torso 6-DoF orientation hold is per-stance-relative, NOT global (audit CASE B, commit 618ddcf)
The 6-DoF torso-pose task **recaptures its orientation target at each phase entry**
(`sim_loop.py:1411,1478,1676` SS `R_torso_fixed=R_t0=current`; `:2099,:2479-2499` DS `set_hold(…, R_now)` —
"crawl forward, don't pirouette"). Position is held globally (~mm); **orientation only per-stance**. The
per-phase-reset metric `torso_ori_err_deg` (SS-peak ≤ 0.523°, init-offset tick 5.157°) **hides** the cumulative
behavior. TRUE cumulative orientation drift vs a single t=0 reference (canonical, 905 ticks):
- structure-frame: **peak 0.721°, final 0.721°** (effectively flat by end); structure attitude `θ_s ≤ 0.578°`,
  so it is the **torso drifting relative to its own start**, unconstrained.
- world-frame: peak 0.992°, final 0.747°.
- **0.72° is NOT a bound** (the controller does not constrain it; it is simply small on this config). Much smaller
  than the wrong-config 1.80° final / 4.12° peak, but the same per-stance-relative architecture. A cumulative drift
  is a **limitation, not a result.**
⇒ Open: characterize the **full 6-DoF** torso behavior (Option 1 — show all 6 DoF, including the orientation
drift) **or** fix the controller to a **global** orientation reference (Option 2). Showing only the 3
translational DoF would be fraudulent.

## D1 — MuJoCo contact-solver momentum leakage unmeasured
From rest, total system momentum should be ~0 to machine precision. The continuous `‖L_com‖` reaches
**~0.782 N·m·s** during swings (numerical leakage; median 0.040, returns to ≤3e-3 at snapshots). The
per-tick conservation figure was correctly **dropped** (not trustworthy). ⇒ Open: quantify the leakage and
whether it inflates `h_w`/`θ_s`; it is currently **unmeasured**.

---
These items gate any "final J2 results" claim. This PR is **chatter fix + Stage-2 QP architecture + figure/plot
infrastructure**, with **C5 (soft-gate exceedance)**, C1–C3 and D1 explicitly open.
