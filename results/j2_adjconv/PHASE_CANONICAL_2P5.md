# Phase CANONICAL-2p5 — frozen τ_w,max = 2.5 canonical: C(2.5) feasible 6/6 (margin 0.01), U shows saturation-without-management

**Branch** `j2/ds-active-rework` · push-only, never merge (Idriss merges via GitHub UI).
**Frozen-config commit: `32aefaf`** (`freeze(canonical-2p5): tau_w_max 2.5 (controller + plant) + Add-5 weights`).
Data: `results/j2_adjconv/canonical2p5_result.json`; per-tick fulldiag CSVs `c25_fulldiag.csv` (C) and
`u25_fulldiag.csv` (U) + `*_fulldiag_meta.json`; runner `scripts/diag_canonical2p5_run.py`. Raw runs
`results/figC25_addfive`, `figU25_addfive` (gitignored).

## The frozen configuration (confirmed applied exactly — captured from the live QP, not asserted)

| value | frozen | file:line |
|---|---|---|
| τ_w,max controller (NMPC Ḣ_s constraint + WQP box source) | **2.5** (was 5) | `config.py:80` |
| aocs_tau_w_max (AOCS torque clip) | **2.5** (was 5) | `config.py:84` |
| **PLANT cap** — wheel motor `ctrlrange` | **±2.5** (was ±5) | `models/VISPA_crawling_rwa3.xml:324-326` |
| h_max (wheel momentum box) | **±5 UNCHANGED** | `config.py:71-72` |
| torso-pose | 2000 | `config.py:351` |
| swing-EE | 1000 | `config.py:319` |
| momentum (T-MOM) | 400 | `config.py:336` |
| hw-slack | 800 | `wholebody_qp.py:181` |
| posture | 20 | `config.py:320` |
| torque-min | 5 | `sim_loop.py:1145` (QP-construction literal) |
| wrench-track | 1.0 | `config.py:321` |
| accel-reg | 1.0 | `sim_loop.py:1145` |
| ε (Tikhonov) | 1e-6 | `hierarchical_qp.py:97` (constructor default) |
| weight_ratio | 1 (priorities inert) | `wholebody_qp.py:75` |

Both runs' effective in-solver weights were captured from `WholeBodyQP.config` at construction and **match the
frozen vector exactly** (`weights_match_frozen: true` in the JSON, C and U).

**Plant-cap mechanism:** `autolimits="true"` (MJCF:30) + `gear=1` ⇒ MuJoCo clamps the applied motor torque to
`ctrlrange` regardless of the commanded `ctrl`. The logged `tau_w` channel is the pre-plant COMMAND; the applied
torque was measured from `data.actuator_force[14:17]` at every `mj_step`.
**Override chain caveat:** `dca.main` (scripts/diag_cooperative_arms.py:302-303) sets `cfg.tau_w_max` AND
`cfg.aocs_tau_w_max` from its single `tau_w_max` kwarg (script default still 5.0) — the phase runner passes the
frozen values explicitly. Flag for Idriss: align the script default in a follow-up, or bare `dca.main()` runs
revert the controller caps to 5 (the MJCF plant cap stays 2.5 regardless).

## STEP 0 — MJCF plant cap is LIVE: **PASS**

- **Static smoke test:** `ctrl = 10` (4× cap) → applied actuator force = **2.500 exactly**, all three wheels;
  `ctrlrange` reads ±2.5.
- **Closed-loop U run:** the controller (AOCS clip lifted to 1e6) **demanded up to 26.90 Nm**; the measured
  applied wheel torque **never exceeded 2.500** (max over 7503 mj_steps = 2.500). The plant clips even when the
  controller demands 10.8× more. Gate: **PASS**.

## STEP 1 — FEASIBILITY GATE C(2.5): **DOCKS 6/6** — degraded margin, razor-thin

| | C(2.5) frozen | C(5) Add-5 reference |
|---|---|---|
| feasibility | **6/6 dock** | 6/6 dock |
| at-weld docks [mm] | 4.02, 4.89, **4.99**, 4.97, 4.95, 4.62 | 2.56, 4.59, 4.89, 4.39, 2.49, 4.49 |
| worst / margin | **4.99 / 0.01 mm** | 4.89 / 0.11 mm |

**Yes, degraded:** halving the momentum budget slows the swing and pushes four of six docks above 4.9 mm; the
worst (step 2) passes the 5 mm gate by **0.01 mm**. Feasible, but at the edge — a paper-honest caveat.
- **Verbatim IPOPT (C, SS solves):** `{'Solve_Succeeded': 503, 'Solved_To_Acceptable_Level': 5}` — 508/508
  success codes. `qp_fail = 0`. (`nmpc_fail` 1368 = intentional DS bypass, pre-existing.)
- **Does C hit the physical clamp?** Mostly under, as expected with management ON: applied |τ_w| at the 2.5
  clamp on **368/8458 mj_steps (4.4%)**; command at cap on 96 ticks. 95.6% of the time the actuator has margin.

## STEP 2 — SATURATION-WITHOUT-MANAGEMENT (U: management OFF, plant cap ACTIVE): **CONFIRMED with one honest exception**

| metric | **U (no management)** | C (management) | verdict |
|---|---|---|---|
| controller τ_w demand, max | **26.90 Nm** (10.8× cap) | 2.5 (clipped by AOCS) | actuator demand explodes |
| applied τ_w (measured) | ≤ **2.500** (plant clips; 397/7503 = **5.3%** of steps AT clamp) | ≤ 2.500 (4.4% at clamp) | **actuator saturates** |
| θ_s peak / settled [deg] | **1.194 / 1.194** | 0.540 / 0.540 | **attitude 2.2× worse** |
| h_w per-axis peak [Nms] | [0.68, 2.70, **4.55**] | [0.71, 2.02, 4.10] | headroom nearly exhausted |
| h_w norm peak | **5.08** (> 5) | 4.24 | norm crosses h_max level¹ |
| ticks with any axis > 5 | 0 | 0 | per-axis box holds |
| struct drift [mm] | 21.6 | 22.5 | **NOT degraded** (exception) |
| docks | 6/6 (worst 4.86) | 6/6 (worst 4.99) | docking unaffected |

¹ h_max is a per-axis box (`hw_max = ±5` componentwise); no per-axis violation occurs (max 4.55), but the norm
peak 5.08 crossing the per-axis limit level shows the wheel-momentum budget effectively exhausted.

**VERDICT (the paragraph's claim):** with management OFF and the actuator physically capped at 2.5, the actuator
**does saturate** (demand 26.9 vs applied 2.5; 5.3% of physics steps pinned at the clamp) **and** attitude
**does degrade** (θ_s 0.54° → 1.19°, 2.2×) **and** wheel momentum **approaches overflow** (per-axis 4.55/5,
norm 5.08). **Confirmed** — with one honest exception: **struct drift does not grow** (21.6 vs 22.5 mm) and
docking is unaffected at this 6-step horizon. The saturation cost lands in attitude and momentum headroom, not
(yet) in position/docking.

## STEP 3 — REALIZED ablation (net effect in the realized system)

| realized quantity | C(2.5) | U | net |
|---|---|---|---|
| Ḣ_s per-axis peak [Nm] | [0.84, 1.95, **2.500**] (z pinned AT the box — WQP box binds) | [1.03, 1.97, **3.451**] | management caps realized z at exactly 2.5; free it runs 38% over |
| Ḣ_s per-step max-axis | 1.96, 2.04, 2.19, 1.50, **2.50**, 1.47 | 1.94, 2.05, 2.17, 1.79, **3.45**, 1.79 | step-4 peak is the discriminator |
| θ_s settled [deg] | **0.540** | 1.194 | the attitude price of removing management |
| h_w peak (axis / norm) | 4.10 / 4.24 | 4.55 / 5.08 | momentum headroom −0.45 axis |
| e_com peak [m] | 0.154 | 0.150 | unchanged |
| drift [mm] | 22.5 | 21.6 | unchanged |

Note the realized C z-peak = **2.500 exactly**: unlike at cap 5 (realized 2.50 ≪ cap 5, box slack), at cap 2.5
the **WQP envelope box is an ACTIVE constraint on the realized wrench** — the realized system now presses
against the management limit, which is what makes the 2.5 canonical the interesting operating point.

## STEP 4 — PLANNED ablation (fig:planned_ablation)

| planned Ḣ_s | C(2.5) | U (rate-off) |
|---|---|---|
| per-axis peak | [2.13, **2.500**, **2.500**] — **y AND z pin the cap** | [1.70, 3.69, **10.88**] |
| per-step max-axis | **2.5 on ALL SIX steps** | 7.83, 3.79, 8.12, 3.67, **10.88**, 3.69 |
| SS ticks at cap | **297/508 (58%)** | (would exceed 2.5 on 254/421 = 60%) |

At cap 5, only the arm-a steps saturated the plan (z only, 25% of ticks). At cap 2.5 the constraint **binds on
every step and on two axes** (arm-b plans of ~3.7 now also clip), 58% of SS ticks at the cap — while the
unconstrained U-plan wants up to **10.88 Nm = 4.4× the envelope** on the same steps. The planned ablation is
starker at 2.5 than it was at 5: management is not a formality; it is actively reshaping every step's plan.
Verbatim IPOPT (U, SS): `{'Solve_Succeeded': 421}` — 100%; the 10.88 plan is IPOPT-optimal. `swing_ref_pk = 0`
in both runs (stance-only Ḣ_s = exact full NMPC quantity).

## Conditioning, solver, tests

- **κ_SS: C 7.48e3, U 7.71e3** (~the expected 7.6e3 — weights unchanged from Add-5; conditioning preserved,
  530× below the old canonical 3.6e6).
- qp_fail = 0 in both. e_com ≈ 0.15 both.
- **pytest:** baseline (pre-freeze) 219 passed / **2 pre-existing failures** (`test_far_infeasible_under_tight_rate`,
  `test_E7_t15_step2_dock_under_fk_mode`). Post-freeze targeted subset (momentum, nmpc_qp_consistency,
  coarse_preplanner, invariants): 58 passed / 1 failed = the same pre-existing failure. Full post-freeze suite
  running at report time; outcome appended below.

## Files

| artifact | path |
|---|---|
| result JSON (smoke + U + C, full detail) | `results/j2_adjconv/canonical2p5_result.json` |
| per-tick fulldiag CSV — C(2.5) | `results/j2_adjconv/c25_fulldiag.csv` (+ `c25_fulldiag_meta.json`) |
| per-tick fulldiag CSV — U | `results/j2_adjconv/u25_fulldiag.csv` (+ `u25_fulldiag_meta.json`) |
| runner | `scripts/diag_canonical2p5_run.py` |
| frozen config | commit `32aefaf` |

**STOP for cross-check.** Push only, never merge.
