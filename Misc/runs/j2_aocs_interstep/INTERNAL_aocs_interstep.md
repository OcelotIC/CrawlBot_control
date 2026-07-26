# INTERNAL — J2 step 4a: AOCS re-activation in the inter-step DS loop (implement + characterize)

**Implementation brief, then characterize — NO success threshold.** Raw numbers; the h_w/C5 trade and whether
to keep the default-on are decided by Idriss + reviewing Claude. Branch `j2/ds-active-rework` (pushed, never
merged). Base `ae0673e`. Flag default-on (this fixes a bug) but gated so OFF is byte-identical for A/B.

---

## DECISIVE OUTPUT

**(a) Does θ_s stop drifting (invariant restored)?** The INVARIANT — the AOCS runs every DS tick — is
**structurally restored**: τ_w in the inter-step settles is now **mean 3.28 / max 8.66 Nm** (was identically
0). Whole-traversal structure-attitude regulation **improves**: peak |θ_s| 0.593°→0.531° (−11%), **final
|θ_s| 0.103°→0.046° (−55%, more than halved)**, mean 0.297°→0.271°. The *per-settle instantaneous* excursion
stays ~0.05° (mixed: settle 1 down 0.032→0.027, settle 3 up 0.046→0.053) — the heavy structure (~7111 kg) +
short settle (0.5–1.7 s) + h_w-capped rotate-back means the AOCS **cannot null a settle's drift within that
settle**, and it **saturates** (τ_w hits √3·5 = 8.66, all three axes at ±5) fighting the post-dock impact. So:
the invariant is restored and the *accumulated* attitude is better-regulated; the *instantaneous* per-settle
excursion is bounded but not zeroed (torque-limited against the impact).

**(b) Cost on h_w / C5?** C5 (per-axis h_w∞, the budget metric): **OFF 4.373 → ON 4.483 Nms, both ≤ 4.5
(PASS)** — cost **+0.11 Nms**, margin shrinks to 0.017. The inter-step window's own h_w∞ is small (1.56→1.72
Nms); most of h_w∞ is SS (unchanged), so the inter-step AOCS adds little. Despite the torque saturation, the
*momentum* stays bounded (no runaway). With the **exact box** (qp_envelope_exact, vs the default proxy):
C5 = **4.949 (FAIL, > 4.5)** — the exact box's own ~+0.5 Nms SS-binding cost (Piste-A FLAG 2) stacks with the
AOCS-on +0.11 to break the budget, and the exact box also fails C1 (dock 4.99, margin 0.01) and C2 (pos-peak
28.0 mm). With the **default proxy box, AOCS-on holds all criteria** (C5 4.483). So the FLAG-2 decision and
the inter-step AOCS both push h_w and **stack**.

**(c) Does the passivity settle still converge?** **YES — essentially unchanged.** Same exit reasons (settle 0
target_met, settles 1–4 plateau — pre-existing), T-evolution matching to noise, settle durations OFF
[11,173,51,102,51] vs ON [11,175,51,104,51] (+3 ticks / +0.03 s over the whole traversal). The cross-tick
AOCS↔passivity coupling is **benign**, exactly as the AOCS-FF audit predicted (different actuators; the QP is
solved before τ_w each tick).

---

## Part 1 — Implementation (committed `096a8b5`; default-on, gated, OFF byte-identical)

- **`cfg.aocs_active_in_interstep`** (default **True** — the correct invariant-restoring behaviour). When True,
  `_run_ds_passivity_loop` (sim_loop.py) replaces the `ctrl[n_j:n_j+3] = 0.0` hardcode with the canonical
  **`legacy_pid_numerical`** AOCS via a new `_interstep_aocs_command(rs, cc_ds, lambda_qp_sol, ω_s_prev)`
  helper. When False, the legacy hardcoded-zero path runs unchanged.
- **DS wrench feedforward** `τ_w_FF = −Σ_i (r_Ci × f_i + τ_i)` from the settle-QP wrench `lambda_qp_sol`
  (captured at the QP solve), anchor levers `r_Ci = cc_ds.r_contact_{A,B}` (struct frame) — the welded-loop
  internal-stress couple the FD-on-L_com feedforward misses (AOCS-FF audit). NOT the FD-FF.
- **Feedback** = attitude PID + desat, `θ_s` from the geometric SO(3) error (identical to `_step`), `ω_s`/`h_w`
  from MuJoCo state, `ω̇_s` from a **loop-local `ω_s_prev`** (init from entry ω_s ⇒ ω̇_s=0 first tick; the
  only new history — the wrench FF needs no L_com/v_com history).
- **No NMPC dependency** (the loop bypasses the NMPC; the FF source is the QP wrench, not the plan). The QP,
  passivity constraint, and envelope box are **untouched**. The inter-step logger now records the **actual
  applied τ_w** (was hardcoded zeros).
- Plumbing: `--no-aocs-in-interstep` (diag A/B); forced off under `--baseline_ds_rework` (reproduces
  pre-rework hardcoded-zero).

## Part 2 — Characterization (canonical 5-step working point = Misc/runs/fixA_gate flags), raw

### 1. θ_s — the invariant check

| metric | OFF (legacy hardcode) | ON (AOCS re-activated) | Δ |
|---|---|---|---|
| **per-settle Δθ_s (max-min)**, settles 1/2/3/4 [°] | 0.0319 / 0.0189 / **0.0459** / 0.0266 | 0.0273 / 0.0194 / **0.0534** / 0.0287 | mixed (s1 −14%, s3 +16%) |
| whole-traversal peak \|θ_s\| [°] | 0.5934 | **0.5306** | **−11 %** |
| whole-traversal **final** \|θ_s\| [°] | 0.1030 | **0.0463** | **−55 %** |
| whole-traversal mean \|θ_s\| [°] | 0.2971 | 0.2711 | −9 % |
| DS-phase peak \|θ_s\| [°] | 0.5549 | 0.4894 | −12 % |

The per-settle max-min is an **ambiguous local** metric (it captures the AOCS's own corrective slewing *and*
its reaction to the post-dock impact, not just drift); the **unambiguous global** metric (peak/final/mean
|θ_s| over the traversal) is **clearly improved** — final attitude error more than halved. The invariant
(wheels active every tick) is structurally restored.

### 2. τ_w during the inter-step settles

| | OFF | ON |
|---|---|---|
| mean \|τ_w\| [Nm] | 0.0000 | **3.2776** |
| max \|τ_w\| [Nm] | 0.0000 | **8.6603** (= √3·5 ⇒ all 3 axes clipped at ±5) |
| per-settle \|τ_w\|max (s1/2/3/4) | 0/0/0/0 | 8.02 / 8.66 / 8.66 / 8.66 |

Confirmed non-zero and doing real work. The AOCS **saturates** during the post-dock impact (the wrench FF —
the welded-loop couple — exceeds the ±5 Nm per-axis limit on the impact transient), which is why the
instantaneous per-settle excursion is bounded but not zeroed. Settle 0 (initial DS, at rest) ≈ 0 — nothing
to correct.

### 3. h_w / C5 (the flagged interaction)

| | OFF | ON | exact box (ON) |
|---|---|---|---|
| **C5 h_w∞ (per-axis, cap 4.5)** | **4.373** | **4.483** (+0.11) | **4.949 (FAIL)** |
| inter-step-window h_w∞ (2-norm) [Nms] | 1.56 | 1.72 | — |

With the default **proxy** box: C5 holds (4.483 ≤ 4.5), margin 0.017. Most of h_w∞ is from SS (unchanged by
the inter-step AOCS); the inter-step contribution is small. The momentum stays bounded despite the torque
saturation (no runaway). With the **exact** box the wheel load adds further — see (b).

### 4. Settle convergence (energy dissipation still works)

| step | OFF n / exit | ON n / exit |
|---|---|---|
| 0 | 11 / target_met | 11 / target_met |
| 1 | 173 / plateau | 175 / plateau |
| 2 | 51 / plateau | 51 / plateau |
| 3 | 102 / plateau | 104 / plateau |
| 4 | 51 / plateau | 51 / plateau |

Same exit reasons, T_start/T_end matching to noise (the dock impact is the same; settles 2/4's small T_end>T_start
is pre-existing and identical OFF), +3 ticks total. The settle is unchanged — the AOCS (wheels) and the
passivity QP (joints) act on different actuators and do not compete within a tick.

### 5. C1–C5 full (raw)

| crit | OFF | ON |
|---|---|---|
| **C1 dock** [mm] | [4.94, 4.51, 4.91, 4.61, 4.84] — 5/5 dock, PASS | [4.94, 4.51, 4.91, 4.72, 4.85] — 5/5 dock, PASS |
| **C2 torso** | ori_rms 0.089 / pos_peak 16.0 mm — PASS | ori_rms 0.098 / pos_peak 16.9 mm — PASS |
| **C3 envelope** | ‖Ḣ_s‖∞_SS 5.00 [3.30,4.46,5.0] — PASS | ‖Ḣ_s‖∞_SS 5.00 [3.28,4.64,5.0] — PASS |
| **C4 attitude** | peak 0.59 / final 0.10 — PASS | peak 0.53 / final 0.05 — **PASS (better)** |
| **C5 h_w∞** | 4.373 — PASS | 4.483 — PASS |
| **residual** (traversal-final) [N·m·s] | 0.003977 | 0.004022 (+1 %) |

The AOCS-on shifts the dock distances **< 0.12 mm** (steps 4–5: 4.61→4.72, 4.84→4.85; all still ≤ 5 mm, all
dock), torso track ~unchanged, envelope SS-bound and unchanged (the inter-step AOCS doesn't touch SS),
residual +1 %. The only criterion it materially moves is **C4 (better)** and **C5 (+0.11, the flagged cost)**.

### 6. C6 — flag-OFF determinism

**BIT-IDENTICAL.** New-code flag-off (`--no-aocs-in-interstep`) vs old-code (`6a718f2`) canonical, 5-step:
worst |Δ| = **0.000e+00** over all 330 physical leaf arrays; the only differing fields are the two
non-deterministic wall-clock timers (`qp_time_ms`, `nmpc_time_ms`). The gate also reports C6 OFF/test
BIT-IDENTICAL (worst|Δ|=0) and `test_reworked_qp` 8/8.

---

## Regression (`pytest tests/`)

**220 passed, 1 failed** (949 s, on the new code at `096a8b5`). The single failure is the **pre-existing**
FK test `test_E7_t15_step2_dock_under_fk_mode` (known/unrelated, identical count on clean `ae0673e`) — **no
NEW failures**. The flag-off path is byte-identical (C6 above ⇒ the default-equivalent path is provably
unchanged); the new flag-on path is exercised by the A/B run (5/5 dock, all gates PASS).

## Flags / divergences vs the AOCS-during-DS / AOCS-FF / Piste-A audit facts

1. **Confirms the AOCS-during-DS audit fix & quantifies it:** the invariant is restored (τ_w mean 3.28, was
   0). The audit's worst-case open-loop drift (0.046°/settle) is now actively regulated — final |θ_s| halved
   (0.103→0.046°). Nuance the audit didn't have: the per-settle *instantaneous* excursion is **not** zeroed,
   because the heavy structure + short settle + torque saturation limit within-settle correction; the gain is
   in the *accumulated* attitude.
2. **Confirms the AOCS-FF audit:** the same `legacy_pid_numerical` call is makeable from in-loop values with
   only `ω_s_prev` new; the FF source is `lambda_qp_sol` (no NMPC); and the cross-tick AOCS↔passivity coupling
   is **benign** (settle unchanged, ±3 ticks). All three predictions hold empirically.
3. **New finding — wheel saturation on the post-dock impact:** the inter-step AOCS saturates (τ_w = √3·5)
   fighting the welded-loop wrench couple right after the dock — unlike DWELL DS (`_step`), where τ_w stayed
   < 2.2 Nm. The post-dock impact is a much larger disturbance than the settled DWELL. The wrench FF dominates
   and exceeds the ±5 Nm limit on the transient; the momentum nonetheless stays bounded (C5 4.483 ≤ 4.5).
4. **C5/FLAG-2 connection (the flagged risk):** with the default **proxy** box the +0.11 Nms cost keeps C5 ≤
   4.5 (margin 0.017). With the **exact** box (which the Piste-A audit showed adds ~0.5–0.76 Nms from the SS
   binding) the combined load is **C5 = 4.949 (FAIL)** — the exact box dominates (its own ~4.9 from the SS
   binding, Piste-A) and the inter-step AOCS stacks ~+0.1 on top; the exact box also marginally fails C1
   (dock 4.99) and C2 (pos-peak 28 mm). The FLAG-2 decision and the inter-step AOCS **both push h_w and stack
   past the budget** — relevant only if FLAG 2 is turned on; the default (proxy box) keeps AOCS-on within
   budget (4.483).
5. **Design note (in scope: not changed):** the settle QP's `hw_current` is computed once at loop entry and
   not refreshed per-tick. With the wheels frozen (AOCS off) that was exact; with the AOCS on the wheels move,
   so the QP's hw safety constraint sees slightly stale state during a settle. Left unchanged (the brief
   scopes to AOCS re-activation only); flagged as the obvious follow-up if the C5 margin needs recovering.

## Reproduce
```
bash Misc/scripts/run_aocs_interstep_ab.sh    # AOCS on vs off, 5-step, analysis + gate
MUJOCO_GL=disabled PYTHONPATH=. python3 Misc/scripts/audit_aocs_interstep.py on=results/aocs_on off=results/aocs_off
```
Supporting: `Misc/runs/j2_aocs_interstep/{aocs_ab.log, aocs_gate.log}`. C6 diff: new-code flag-off vs old-code
`6a718f2` canonical (bit-identical on all physical fields).

**STOP after the report.** No success threshold; the h_w/C5 trade and default-on are decided by Idriss +
reviewing Claude on these numbers. No merge, no PR.
