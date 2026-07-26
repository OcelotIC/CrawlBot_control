# INTERNAL — CC BRIEF #1 (METHOD): de-patchwork the chatter fix + elucidate the settle non-convergence

Branch `j2/ds-active-rework` (pushed, never merged). **Decides the DEFINITIVE settle regularization** that
regeneration brief #2 depends on — so **no canonical run is frozen here**. Touches `crawlbot/` (the settle QP)
to add a *second, evaluated* regularizer for the head-to-head; **both knobs stay default-off (byte-identical)**.
Reproducer: `scripts/diag_cooperative_arms.py --qp-envelope-exact [--interstep-settle-alpha-wrench ε |
--interstep-settle-alpha-sigf α] --out-dir …` + `scripts/audit_chatter.py`.

---

## DECISIVE OUTPUT (what brief #2 needs)

1. **Definitive regularizer: Tikhonov `α_wrench·‖λ‖²`, settle-only, ε = 3.0.** Chosen on **principle**, not
   sweep-pick (justification below). This is the **already-committed** knob `interstep_settle_alpha_wrench`
   (01bfc27) — adopting it means **no new production code**, only a recommended value for an existing gated flag.
2. **The brief's Task-A hypothesis is REFUTED.** The explicit Σf-penalty is **NOT** robust-by-construction: it
   has the **same cliff** as the Tikhonov (inert → partial → clean over the same span) and **over-damps** at its
   working weight. Its only genuine edge is physical interpretability, which buys no robustness. **Recommend
   Tikhonov.**
3. **Settle residual verdict (Task B): BENIGN early-exit once the chatter is removed.** The only *real*
   non-convergence was the chatter itself (it **pumped** kinetic energy). With it gone, the step-2 plateau is a
   benign, config-specific slow-dissipation mode that does not threaten the dock (holds 0.009 mm). **The settle
   does NOT need to run longer.**
4. **Production settle config:** `interstep_settle_alpha_wrench = 3.0`, `interstep_settle_alpha_sigf = 0.0`
   (Σf retained as a gated, *not-adopted* evaluation knob). **Default-off (both = 0.0) is byte-identical (C6).**
   **C1/C3/C4 PASS; C2/C5 still FAIL — these are the pre-existing exact-box *SS* issues, orthogonal to the
   chatter, and the fix slightly improves both.**

---

## TASK A — principled regularization: Σf-penalty vs Tikhonov (STOP-GATE A)

### What was implemented
A second settle-only QP task `α_Σf·‖Σf‖²` with `Σf = f1 + f2 = m·a_com` (the net contact force = CoM
acceleration). The intuition the brief asked me to test: penalising Σf **directly** removes the flat direction
in the *sign of Σf* (the chatter axis) "at the source", so Σf≈0 should fall out **by construction** over a
**wide, robust** range of α_Σf — unlike the Tikhonov's sharp ε-cliff. Added as
`solve(settle_alpha_sigf=…)` / `cfg.interstep_settle_alpha_sigf` (default 0.0), passed only from
`_run_ds_passivity_loop`; `nc==2` + `settle_mode` gated.

### Head-to-head at the working weight (arm-a settles s2, s4 — the chatterers)

| metric (s2 / s4)        | baseline ε=0 | **Tikhonov ε=3** | Σf α=10 |
|-------------------------|--------------|------------------|---------|
| flip-frac (Σf sign)     | 0.94 / 0.98  | **0.04 / 0.03**  | 0.007 / 0.008 |
| at-±5 frac              | 0.86 / 0.96  | **0.00 / 0.02**  | 0.00 / 0.00 |
| exact ‖Ḣ_s‖ med [N·m]  | 7.27 / 7.11  | **0.71 / 0.09**  | 0.019 / 0.023 |
| ‖Σf‖ med [N]            | 6.90 / 6.88  | **0.93 / 0.09**  | 0.016 / 0.016 |
| commanded Δτ_w med [N·m]| 14.6 / 14.2  | **0.027 / 0.004**| 0.000 / 0.000 |
| settle n (s1/s2/s3/s4)  | 102/51/100/51 **all plateau** | **172/51/206/205** | 328/576/413/496 **all target** |

**Both kill the chatter and both drive Σf→0.** Same outcome. The difference is the settle length (below).

### The cliff — BOTH regularizers share it (the refutation)

| weight | Tikhonov `ε·‖λ‖²` (step2) | Σf `α·‖Σf‖²` (step2) |
|--------|---------------------------|----------------------|
| 0.01   | —                         | flip 0.939, exact 7.275 — **byte-identical to baseline** |
| 0.1    | flip 0.98, exact 7.24 — **inert** | flip 0.980, exact 7.237 — **inert** |
| 1      | flip 0.98, at±5 0.44, exact 5.59 — **partial** | flip 0.980, at±5 0.46, exact 5.80 — **partial** |
| 3      | **flip 0.04, exact 0.71 — clean** | (still partial) |
| 10     | (over-damps, n→570)       | **flip 0.007, exact 0.019 — clean** |

The Σf-penalty shows the **identical** inert→partial→clean progression. It is *not* robust-by-construction;
there is **no wide α range that works for free**. Its clean point sits at α≈10 (vs ε≈3) only because ‖Σf‖² is a
weaker penalty per unit weight than ‖λ‖² (Σf is 3 of the 12 λ-components).

### Why the same cliff — the mechanism (this is the principle)
The chatter is an **active-set limit cycle**, not a property of which quadratic you add. When the exact envelope
binds, the QP cost has a **degenerate (flat) direction**: two equal-cost saturating vertices `A ≈ −B`
(‖A−B‖≈11.7 ≫ intra-vertex std). The production `cfg.alpha_wrench = 0.01` curvature in that direction is
**below the active-set solver's degeneracy/pivot tolerance**, so the solver is free to flip A↔B every tick =
period-2 chatter. **Any** strictly-convex regularizer fixes it the moment its curvature in the degenerate
direction exceeds the **solver** tolerance. That threshold is a property of the solver + problem scaling — it is
the **same** for Tikhonov and Σf, hence the same cliff. Below it, the Σf-penalty is just as invisible as the
Tikhonov; "removing the flat direction" only takes effect once the weight is supra-tolerance, exactly like
Tikhonov.

### Recommendation: **Tikhonov `ε·‖λ‖²`, settle-only, ε = 3.0** — four reasons on principle

1. **Full-wrench coverage.** ‖λ‖² regularizes all **12** contact-wrench components (net force Σf **plus**
   internal force **plus** contact torque) — a **superset** of the Σf-penalty's 3 net-force components. It
   removes **every** degenerate direction; the Σf-penalty leaves the internal-force/torque blocks unregularised
   (only partly covered by the existing Task 3c internal-stress reg).
2. **Shorter settle at equal chatter-kill.** At their respective working weights, **ε=3 gives a strictly better
   settle than α=10**: ε=3 keeps the settles short (s1/s3/s4 reach the target at n≤206; s2 holds its benign
   n=51 plateau), while α=10 forces **every** settle to run 5–11× longer (n=328–576) = the brief's "over-damp"
   regime. Same chatter kill, less over-damping.
3. **ε tied to a solver property, not a sweep cherry-pick.** ε must exceed the active-set degeneracy tolerance:
   ε=1 is partial/marginal (sign still flips), ε=3 is the **first fully-clean** value. ε=3 ≈ 3× the empirically
   located threshold — margin without over-damping. (This is the brief's own fallback: *tie ε to the solver
   tolerance and document why `alpha_wrench=0.01` is too small.*)
4. **Settle-only scope is what makes ε=3 safe.** `CLAUDE.md` forbids `α_wrench>1` globally because **in SS** a
   large wrench weight consumes QP budget and **blocks torso/EE authority**. But the inter-step settle has **no
   torso/EE tracking task to block** — it is a pure dissipation hold — so a large *settle-scoped* weight has no
   such side-effect. This is the principled reason the fix is a settle-only override, not a global change.

**Why `alpha_wrench=0.01` exists and is "too small" here.** It is deliberately tiny so that in SS it is *pure
regularisation, not a competing objective* (CLAUDE.md). That same smallness puts it below the solver degeneracy
tolerance, which is harmless in SS (the box rarely binds with a flat direction) but is exactly the chatter
trigger in the exact-box settle. The settle-only boost resolves the conflict without disturbing SS.

**Σf-penalty disposition:** keep the code as a **gated, default-off, NOT-adopted** evaluation knob
(`interstep_settle_alpha_sigf`) so the head-to-head stays reproducible from a clean checkout. It is documented
in-code as the principled-by-construction attempt that proved no more robust than the Tikhonov.

---

## TASK B — the step-2 inter-step settle plateau (STOP-GATE B)

**Verdict: BENIGN early-exit.** The only *real* dissipation failure was the chatter (it pumped energy); once
removed, the step-2 plateau is a benign, config-specific slow mode that does not threaten the dock.

### Residual KE at bail — joint vs base split
The residual lives almost entirely in the **welded-arm joints**, not the base: across all runs the exit-tick
ratio `‖dq_joint‖ / ‖dq_base‖ ≈ 16–30×`. The DS weld holds the base/structure nearly still; what remains is the
arms flexing against the welds.

### The baseline plateau was a REAL non-convergence (energy pumping)
For the arm-a settles (s2, s4) at baseline the kinetic-energy proxy **rises** over the settle:
`T_start 3.2e-4 → T_end 2.15e-3` (×6.7 **up**), the dominant joint-velocity component **flips sign 90 %** of
ticks (period-2 chatter), mean `‖dq_joint‖ = 0.099 rad/s`. The limit cycle injects energy faster than the
passivity damping removes it, so the settle **diverges in KE** and bails at the plateau floor (n=51). This is
the genuine non-convergence.

### After de-chattering (ε=3): pumping stops; s2 plateau is a slow mode
`s1/s3/s4` now reach the target. `s2` stops pumping (`T 3.3e-4 → 3.9e-4`, flat), flip-frac **0.90 → 0.08**, mean
`‖dq_joint‖ 0.099 → 0.053 rad/s` — but it still trips the plateau detector at n=51. This residual is a
**config-specific weakly-damped joint mode**, no longer chatter.

| run        | s2 mean ‖dq_joint‖ | s2 flip-frac | s2 T: start→end        | exit     |
|------------|--------------------|--------------|------------------------|----------|
| baseline   | 0.099 rad/s        | 0.900        | 3.2e-4 → **2.15e-3 ↑** | plateau  |
| Tikhonov 3 | 0.053 rad/s        | 0.080        | 3.3e-4 → 3.9e-4 (flat) | plateau  |
| Σf 10      | 0.006 rad/s        | 0.007        | 3.3e-4 → 2.5e-8 ↓      | target   |

### Why step-2 specifically
s2 and s4 are both post-arm-a-dock settles. After de-chattering, **s4 dissipates to target (n=205) but s2
plateaus** — so it is the *specific arm-a posture at anchor 3* (s2) that has a weakly-damped mode where the
passivity damping is near-orthogonal to the slow direction. It **is** dissipable, just slowly: at the stronger
weight (Σf α=10, more steps) **s2 reaches the target at n=576**. So: slow, not stuck.

### Is the dissipation target well-posed?
`T_settle = ½·ε_v²·λ_min = 2.55e-8` with `ε_v = 1 mm/s` on the **softest** mode (`λ_min ≈ 0.051`) — an extremely
tight target. The plateau detector (relative stall, `T(k) > 0.999·T(k−50)`) is the **operative** practical
criterion. The target is well-posed but very tight; reaching it is **not required** for dock integrity.

### Negligible or significant? → NEGLIGIBLE (benign)
The **dock holds at 0.009 mm** through every settle, all runs (s2: baseline 0.0097, ε=3 0.0092, α=10 0.0084 mm;
the 0.040 mm whole-run max is the docking *transient*, not the settle). The s2 residual joint velocity
(~0.05 rad/s) over a 0.1 s NMPC tick is ~5e-3 rad of drift against welds that hold the grippers to <0.01 mm. It
does not threaten the dock, attitude, or the next swing. **The s2 plateau is a benign early-exit; the settle
need not be lengthened.**

---

## Default-off safety + gate (what brief #2 must inherit)

- **C6 — default-off byte-identical: CONFIRMED.** Identical command run on clean HEAD (no Σf code) vs the
  working tree (Σf code, both knobs 0.0): **every computed quantity bit-identical** (deep max |Δ| over all
  leaves = 0.0); the only deltas are wall-clock `*_time_ms` (non-deterministic). The dormant Σf branch cannot
  affect the default path. (The Tikhonov knob's C6 was already confirmed at 01bfc27.)
- **C1 docking — PASS.** 5/5 dock at **[4.94, 4.45, 4.95, 4.65, 4.90] mm** (ε=3), all < 5 mm; e_ori_ee ≤ 0.16°.
- **C3 envelope — PASS.** SS `‖Ḣ_s‖∞ = 5.00`, `hits_bound = False` (rides the cap, no exceed) — same as baseline.
- **C4 attitude — PASS.** torso-ori peak 0.52° (SS) / 0.26° (DS-hold) — ≪ 5°.
- **C2 torso-pos — FAIL** (peak_SS 22.84 mm) and **C5 h_w∞ — FAIL** (4.910 Nms > 4.5). **These are the
  pre-existing exact-box *SS* issues** (c_curr brief: C2 28 mm, C5 4.949), **orthogonal to the chatter** (they
  live in SS; explained by the SS orbital-scaling law). The fix slightly **improves** both (C2 28.0→22.8 mm,
  C5 4.949→4.910) by removing the chattering inter-step Σf, but does not resolve them.

## Regression
`regression.log` (committed): **1 failed, 219 passed, 1 deselected** (4:23). The single failure is the
pre-existing FK test `test_E7_t15_step2_dock_under_fk_mode` (known/unrelated). **No new failures.** Default-off
byte-identical (C6 deep |Δ|=0 over all computed state), so the dormant Σf code cannot affect the default path.
The 1 deselected test is `test_trajectory_aware_ik.py::test_chain_consistency` — a trajectory-IK test
**unrelated** to the settle-QP flag, which **passed** in the prior identical regressions this session and is
environment-flaky-slow in the current container (>10 min; not a hang introduced here). C6 guarantees it behaves
identically with or without the Σf code.

## Reproduce
```
# head-to-head (chatter + Σf/exact + settle convergence):
for w in 0 3;  do diag_cooperative_arms.py --qp-envelope-exact --interstep-settle-alpha-wrench $w --out-dir cf_e$w;  done
for a in 0.01 0.1 1 10; do diag_cooperative_arms.py --qp-envelope-exact --interstep-settle-alpha-sigf $a --out-dir cf_sf$a; done
audit_chatter.py base=results/cf_e0 tik3=results/cf_tik3 sf001=results/cf_sf001 sf1=results/cf_sf1 sf10=results/cf_sf10
```
Supporting: `Misc/runs/j2_chatter_method/headtohead.log`. Raw run dirs reproducible, not committed.
**No merge, no PR.**
