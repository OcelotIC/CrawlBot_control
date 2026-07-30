# NMPC horizon N: 8 → 15

**Branch** `claude/com-gain-semantics-audit-j0u6yr`
**Parent** `53486d2` (COM-GAIN-AUDIT Phase 3)
**Change** `crawlbot/simulation/config.py` — `nmpc_N: 8 → 15`. One field.

Both arms of the A/B carry the Phase-3 gain fix, so this is a **single-variable**
comparison in the sense of Rule 12: N is the only thing that moved.

---

## 1. ⚠ N is not a pure horizon knob — read this before interpreting the result

The NMPC holds a **constant reference across the whole horizon**, so the
reference must be sampled in the future. Three sites are keyed on `nmpc_N`:

| site | expression | N=8 | N=15 |
|---|---|---|---|
| `sim_loop.py:2131` | `t_horizon = t + N·dt` → CoM reference query | +0.8 s | **+1.5 s** |
| `sim_loop.py:2148` | `tau_rel = t_horizon − t0` → coarse-preplanner query | +0.8 s | **+1.5 s** |
| `sim_loop.py:2210` | `t_mid = t + N·dt/2` → `L_com` reference query | +0.4 s | **+0.75 s** |

Raising N therefore lengthens the prediction **and** pushes the target the NMPC
chases nearly twice as far ahead. **These are not separable through this field.**
Every number below is the joint effect of both. Anyone reading this as "longer
horizon ⇒ X" is over-claiming; a clean horizon ablation needs the sampling
expressions decoupled from `nmpc_N` first.

Lookahead is 1.5 s against a nominal `T_step = 6 s`
(`coarse_preplanner.py:92`), i.e. 25 % of the step — still inside it, so the
reference query does not saturate against the terminal hold.

---

## 2. Result

`gate/dock_check.py` on the post-change replay. **Baseline column is N=8 *with*
the gain fix** (`53486d2`), not the frozen artifact, so the horizon is the only
delta.

| metric | N=8 | N=15 | delta |
|---|---|---|---|
| dock 1 [mm] | 4.04 | 4.40 | +0.36 |
| dock 2 | 4.89 | 4.53 | −0.36 |
| dock 3 | 4.98 | 4.93 | −0.05 |
| dock 4 | 4.97 | 4.48 | −0.49 |
| dock 5 | 4.94 | **2.12** | **−2.82** |
| dock 6 | 4.63 | 4.40 | −0.23 |
| **docks under 5 mm** | 6/6 | **6/6** | — |
| **worst margin** | 0.02 mm | **0.07 mm** | **+0.05 (3.5× better)** |
| θ_s peak [deg] | 0.539 | **0.554** | **+0.015 (worse)** |
| h_w peak axis [Nms] | 4.104 | 3.992 | −0.112 (better) |
| h_w peak norm [Nms] | 4.244 | 4.172 | −0.072 (better) |
| **e_com peak [m]** | 0.154 | **0.190** | **+0.036, +23 % (worse)** |
| qp_fail | 0 | 0 | 0 |
| traversal length [ticks] | 2077 | **1981** | −96 (≈0.96 s faster) |

**It is not a clean win.** Docking margin and wheel-momentum headroom improve
materially; attitude excursion and CoM tracking get worse.

`e_com` rising by 23 % is the §1 coupling showing up exactly where it should:
the QP's CoM task is a plan-follower (COM_GAIN_PHASE3_FIX §3.1), so a reference
sampled 1.5 s ahead instead of 0.8 s makes the *instantaneous* CoM error larger
by construction. This is evidence for the reference-lead half of the change, not
against the horizon half — and it cannot be separated without the decoupling in §1.

---

## 3. Cost — and the one real concern

`results/gate_run_scratch/nmpc_step_log.json`:

| | N=8 | N=15 |
|---|---|---|
| solves | 709 | 634 |
| solve time median [ms] | 22.0 | **34.3** |
| p95 [ms] | 31.3 | **60.5** |
| **max [ms]** | 61.9 | **117.9** |
| IPOPT iterations median / max | 11 / 18 | 11 / 19 |
| non-success statuses | 4 | 2 |
| **solves over the 100 ms budget** | **0 / 709** | **1 / 634** |

⚠ **One solve exceeds the NMPC period.** `dt_nmpc = 0.1 s`, so a 117.9 ms solve
overruns its own 10 Hz slot. The offline sim is unaffected — it is not real-time,
and `qp_fail` stays 0 — but for the sim-to-real track this is a **hard real-time
violation**, and the margin at p95 has gone from 3.2× to 1.65×. If N=15 is kept,
this needs either a solve-time cap, a warm-start improvement, or an explicit
acknowledgement that the 10 Hz rate is nominal rather than guaranteed.

Iteration count barely moved (11 median both), so the cost is per-iteration
problem size, not harder convergence — consistent with a horizon that nearly
doubled.

---

## 4. Verification

| gate | result |
|---|---|
| `gate/run_suite.py` (full) | **PASS** — 206 tests, 205 passed, 0 failed, 0 errors, 1 xfail |
| `gate/run_gate.py` | **FAIL** on artifact identity (row count 2077 → 1981) — expected |
| `gate/sync_docs.py --check` | PASS |
| `gate/verify_docs.py` | PASS |
| `gate/link_audit.py` | PASS |
| `gate/verify_params.py` | PASS (after fixing 5 refs this change drifted) |

The strict `xfail` (`test_far_infeasible_under_tight_rate`) **did not flip** — it
is still xfail, so the preplanner's envelope semantics are unchanged by the
horizon. That was the main suite risk and it did not materialise.

`verify_params.py` caught five CLAUDE.md refs shifted +12 by the comment block
added to `config.py` (`303→315`, `282→294`, `290→302`, `283→295`, `284→296`),
all fixed. The gate bit correctly.

---

## 5. Open questions — for Idriss

1. **Is the trade acceptable?** +0.05 mm dock margin and −0.07 Nms `h_w` peak,
   against +0.015° θ_s and +23 % `e_com` peak. The dock margin is the gating
   criterion at 5 mm, which argues yes; θ_s is a headline paper number, which
   argues it needs deciding rather than assuming.
2. **θ_s = 0.554° breaks the paper's propagated value.** The last session's
   remaining-work list has "propagate θ_s = 0.54° everywhere". At N=15 it is
   0.55°. Do not propagate until the horizon is settled.
3. **The 117.9 ms solve.** Acceptable offline; not acceptable for the real-time
   claim. Needs a position.
4. **The §1 confound.** If the intent was "more prediction", the reference-lead
   change came along uninvited. Decoupling the three sampling expressions from
   `nmpc_N` is a small, self-contained change and would make the horizon
   independently tunable — recommended before any further N sweep.

---

## 6. Artifacts

| path | what |
|---|---|
| `crawlbot/simulation/config.py:243-255` | the change + the coupling warning |
| `docs/crawlbot/simulation/config.md` §5 | Rule 15 prose |
| `CLAUDE.md` | horizon row + 5 drifted refs |
| `results/gate_run_scratch/sim_log.json` | the N=15 replay (scratch, not committed) |

No committed canonical artifact modified.
