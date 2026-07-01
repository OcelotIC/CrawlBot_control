# Phase 1e'-B — why the closed loop misses the dock gate at h_max=6 (feasible plan, clean QP)

Diagnostics only, ZERO `crawlbot/` change, no fix. Isolates the realized-frontier cause among
{NMPC solve failure, time-starvation, dock-gate tightness, swing-tracking gains, QP conditioning}.
The h_max=6/7 realized runs from Phase 1e are on disk; the h_max=6 cond-at-speed run and the
Task-C run are new instrumented reruns (monkeypatch only). Data: `hmax6_abort_anatomy.json`,
`cond_at_speed.json`, `qp_cond_raw_figC_qpcond6.json`; scripts `analyze_hmax6_abort.py`,
`analyze_cond_at_speed.py`, parameterized `diag_qp_conditioning.py` / `diag_run_hmax.py`.

**Result — the abort is an under-shoot + drift (tracking/settling in the fast regime), NOT any
of: NMPC solve failure, time-starvation, or QP conditioning.** It is only *partially* helped by
swing-EE gain. The canonical h_max=5 run is unaffected.

## TASK A — anatomy of the h_max=6 abort
| h_max | ticks | SS ticks | **real NMPC solve-fails** | dock evals | fired | dock d first/min/end (mm) | hold (s) |
|---|---|---|---|---|---|---|---|
| 5 | 1080 | 328 | **0** | 21 | 6 | 5.91 / **4.41** / 4.93 | 35.5 (6 steps) |
| 6 | 114 | 104 | **0** | 81 | 0 | 8.01 / **6.80** / 10.27 | 8.0 |
| 7 | 110 | 100 | **0** | 81 | 0 | 8.77 / **8.77** / 14.27 | 8.0 |

- **NMPC solve failure — RULED OUT.** The reported "NMPC fails 10x" are **`NMPC_BYPASSED`** ticks
  (DS/init, `sim_loop:1114` "not run; not a failure"), not IPOPT failures. Every NMPC solve that
  ran returned `Solve_Succeeded` (h_max=6: 104/104). **Real IPOPT solve failures = 0 at every h_max.**
- **Time-starvation — RULED OUT.** h_max=6 step-0 has an **8.0 s hold** window and **104 SS ticks**
  (more than h_max=5's ~55/step). The dock distance dips to **6.80 mm** (best) early in the hold,
  then **DRIFTS UP to 10.27 mm** (Δ = +2.26 mm) — it is not cut off on the way down; it plateaus
  above the gate and drifts away. Trajectory: `8.0→7.0→6.8(min)→…→10.27`.
- **So the closed loop reaches ~6.8 mm and cannot hold/close the last 1.8 mm** — an under-shoot of
  the fast swing plus a hold-drift, with plenty of time.

### Addendum — does the ~1e6 QP conditioning EXPRESS at speed? (h_max=6 vs h_max=5, step-0)
| metric | h_max=5 (step-0 SS) | h_max=6 (step-0 SS) |
|---|---|---|
| **cond(H)** median / p90 / max | 3.56M / 3.74M / 3.75M | **3.55M / 3.60M / 3.75M** |
| solver status | Successful return (0 fail) | **Successful return (0/1040 fail)** |
| Ḣ_s HF ratio realized/planned [x,y,z]† | [1.15, 2.32, 2.34] | [0.92, 2.83, 1.25] |
| k+1 residual \|Δr\|/\|Δv\|/\|ΔL\| | 0.63mm / 2.76mm/s / 0.198Nms | 0.32mm / 3.94mm/s / 0.190Nms |

† here the "planned" is the NMPC ΔL/dt (angular-momentum rate), a smoother reference than the
per-tick CSV Ḣ_s used in the QP-cond round — hence the absolute ratio differs from that round's
[0.8,1.0,0.9]; what is diagnostic is the **h_max=6-vs-h_max=5** comparison with the *same* method.

**cond(H) is IDENTICAL at speed (3.55M vs 3.56M), all 1040 solves succeed, the Ḣ_s HF ratio does
not systematically rise (x,z lower at h_max=6), and the k+1 residual is comparable.** Per the
stated test: cond stays ~1e6 **and** no HF climb ⇒ **the conditioning is genuinely inert even at
speed; the frontier cause is elsewhere (tracking/settling), not the QP.**

## TASK B — is the 5 mm gate the limiter? (from realized, zero solve)
Best (global-min) EE-to-anchor approach per fast run: **h_max=6 → 6.80 mm; h_max=7 → 8.77 mm**
(baseline h_max=5 docks at **4.41 mm**). So the fast swings **under-shoot the 5 mm gate by ~1.8 mm
(h6) / ~3.8 mm (h7)**. A physically-justified capture tolerance in the ~7 mm (h6) / ~9 mm (h7)
range would let the *best approach* cross — **but** the approach is not stable (it drifts to
10–14 mm over the hold), so a looser gate alone would still need the drift addressed to fire
reliably. The gap distribution is a small consistent under-shoot, not a large divergence.
(The canonical 5 mm h_max=5 run is untouched; this only reports what gap the fast runs achieve.)

## TASK C — is it a swing-tracking-gain issue? (rerun h_max=6 with α_ee ×10, NOT α_torso)
`--ss-alpha-ee` 3e3 → **3e4** (swing-EE weight; α_torso, α_mom unchanged):
| | baseline h_max=6 | h_max=6, α_ee=3e4 |
|---|---|---|
| dock best (min) | 6.80 mm | **5.97 mm** |
| dock end (drift) | 10.27 mm | 10.37 mm |
| docked? | no (TIMEOUT) | **no (TIMEOUT)** |
| θ_s peak | 0.165° | **0.165°** (unchanged) |
| QP fails / NMPC solve-fails | 0 / 0 | **0 / 0** (health clean) |

Tightening swing-EE tracking **improves the terminal approach (6.80 → 5.97 mm, tunable)** and does
**not** degrade torso (θ_s unchanged) or solver health — but it **still does not dock** (5.97 > 5 mm)
and **does not fix the hold-drift** (→10.37 mm). So the 6-wall is **not simply under-tuned swing
gain**: the swing terminal is partly tunable, but a **hold-drift / settling** component remains that
α_ee does not touch.

## Isolation summary
| candidate cause | verdict |
|---|---|
| NMPC solve failure | **RULED OUT** (0 real IPOPT failures; "10 fails" = BYPASSED ticks) |
| time-starvation | **RULED OUT** (8 s hold, 104 SS ticks; dips to 6.8 mm then drifts) |
| QP conditioning at speed | **RULED OUT** (cond 3.55M ≡ h_max=5, 0 fails, no HF climb) |
| swing-EE tracking gain | **PARTIAL** (α_ee ×10: best 6.8→5.97 mm, still aborts; drift untouched) |
| **actual cause** | **under-shoot + hold-drift (tracking/settling in the fast regime)** |

## Caveat
Canonical h_max=5 is FINAL and unaffected. α_torso was NOT touched (QP-cond showed it regularizes
the cond tail). The "publish h_max=6 as a realized wall vs a tunable settling issue" call is
cross-check (you + Idriss); a fix (e.g. hold-settling gains, a physically-justified gate, or a
slower terminal-approach segment) is not attempted here and needs GO. No paper text. Task 3 gated.
Raw sim dumps (`figC_qpcond6/`, `figC_hmax6_ee/`) gitignored (regenerable).
