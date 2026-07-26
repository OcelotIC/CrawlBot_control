# INTERNAL — CC BRIEF (close the chatter fix), POINT A: derived settle EXIT criterion (STOP-GATE A)

Branch `j2/ds-active-rework` (push, never merge, no PR). Touches `crawlbot/` (inter-step settle exit target) —
**gated, default-off (C6 byte-identical)**. Reproducer: `scripts/diag_cooperative_arms.py
--interstep-settle-epsilon-v <ε_v>` + `scripts/point1_analyze.py`. STOP-GATE A: report the derived ε_v +
durations + dock/swing justification BEFORE re-testing run B.

## DECISIVE OUTPUT
1. **Derived ε_v = 5 mm/s** (5× looser than the over-tight 1 mm/s) from the dock tolerance:
   per-tick (dt_nmpc=0.1 s) residual drift `ε_v·dt_nmpc ≤ (1/10)·dock_gate(5 mm)` ⇒ ε_v = 5 mm/s.
2. **Settle durations drop 22–51 %** (canonical 5-step, ε_wrench=3): 1.8–2.7 s → 1.3–2.0 s — all still
   `target_met`.
3. **Dock hold UNCHANGED** (0.011–0.018 mm, all ≪ 5 mm gate) and **residual joint velocity at exit UNCHANGED**
   (0.05–0.07 rad/s, *identical* to baseline) — the earlier exit injects **no** extra disturbance; the residual
   is **≤8 % of the p90 swing joint speed** (0.87 rad/s), corrected by the NMPC within the swing.
4. **Chatter still clean** (Σf-flip ≤0.04) and **5/5 traversal preserved** (docks [1.9,4.8,4.9,4.9,4.7] mm,
   identical). Default-off (ε_v=0) is **byte-identical** (C6).

## The root issue (why the old target burned 2+ s)
The settle exits at `T_kin < T_settle = ½·ε_v²·λ_min` with `ε_v = 1 mm/s` on the **softest** mode
(`λ_min≈0.051`). That demands the *most weakly-damped* direction fall below 1 mm/s — a per-tick drift of
**0.1 mm, i.e. 1/50 of the 5 mm dock gate**, on the most expensive mode. The data proves this is wasted work:

| settle (c0, ε_wrench=3) | baseline ε_v=1 mm/s | derived ε_v=5 mm/s | Δ | resid ‖dq_j‖@exit (base→fix) | dock hold |
|---|---|---|---|---|---|
| step1/b3 | 181 (1.81 s) | **142 (1.42 s)** | −22 % | 0.051 → 0.051 rad/s | 0.0176 mm (same) |
| step2/a3 | 258 (2.58 s) | **201 (2.01 s)** | −22 % | 0.072 → 0.072 | 0.0133 (same) |
| step3/b4 | 271 (2.71 s) | **133 (1.33 s)** | −51 % | 0.035 → 0.035 | 0.0120 (same) |
| step4/a4 | 233 (2.33 s) | **152 (1.52 s)** | −35 % | 0.062 → 0.062 | 0.0114 (same) |

**Key:** the residual joint velocity at exit is *identical* whether the settle runs 142 or 181 steps — the
extra baseline time drove only the **softest mode's** KE down (2.4e-8 vs the new 6.1e-7), with **zero** effect
on the joints or the dock. The old target was precision spent where nothing physical cares.

## Derivation (dock-tolerance basis, not a sweep)
- Exit fires when the softest-mode residual velocity ≈ ε_v ⇒ per-NMPC-tick drift ≈ ε_v·dt_nmpc.
- Require that drift **≤ 1/10 of the HOTDOCK capture gate** (5 mm) — a 10× margin on the operative dock
  tolerance: `ε_v ≤ (0.1 · 5 mm)/0.1 s = 5 mm/s`. ⇒ **ε_v = 5 mm/s.**
- **Swing non-perturbation:** the residual at exit (0.05–0.07 rad/s) is unchanged from baseline and is
  ≤8 % of the p90 swing joint speed (0.87 rad/s); the dock holds to 0.011–0.018 mm. So the settle releases
  into the next swing no more perturbed than before — the NMPC absorbs it.

## Implementation (gated, default-off)
- `cfg.interstep_settle_epsilon_v` (default **0.0 = off** → uses `settle_inter_epsilon_v` = 1 mm/s,
  byte-identical). When > 0 it overrides the inter-step settle exit target only (the SS/`_step` DWELL stepper
  is untouched). Wired at the inter-step `_run_ds_passivity_loop` call site
  (`epsilon_v = override if >0 else settle_inter_epsilon_v`) + CLI `--interstep-settle-epsilon-v`.
- **C6:** default-off run (ε_v=0) reproduces the POINT-1 c0 baseline exactly (settles 181/258/271/233, T_end
  2.4e-8, docks identical) and is byte-identical to clean HEAD (verified worktree diff).

## Implication for run B (to be TESTED in POINT B — not assumed)
The canonical settles now finish in 1.3–2.0 s. Run-B's DWELL fires only if `gp.duration − settle ≥ 1 s`
(dt-ds=2.5 s). The longest settle (step2-type, ~2.0 s) leaves ~0.5 s — so ε_v=5 mm/s **may not, by itself,
clear the 1 s DWELL threshold** on every step. POINT B measures this directly: if the shorter settle releases
the budget and run B completes, run B is viable; if not, run B's in-paper-vs-future-work status is a separate
call. Pushing ε_v higher to force the DWELL would trade away the dock-tolerance margin (drift → toward the
gate), so it is NOT done here — the exit criterion stays dock-derived.

**STOP-GATE A. Reporting the derived ε_v + durations + justification; STOP before re-testing run B (POINT B).**
No merge, no PR.
