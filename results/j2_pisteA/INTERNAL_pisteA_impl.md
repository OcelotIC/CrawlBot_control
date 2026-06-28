# INTERNAL — Piste A (J2 #3): envelope-coupled passivity budget + exact Ḣ_s box — implement + characterize

**Implementation brief, then characterize — NO success threshold.** Raw numbers; the β value / tracking /
box-correction trade are decided by Idriss + reviewing Claude → this feeds the β thread. Branch
`j2/ds-active-rework` (pushed, never merged). Base `ae0673e`. Both lots default-off ⇒ default byte-identical.

---

## Part 0 — Verification stop-gate: PASS (all injection points exist; no new plumbing)

1. **NMPC-exact ‖Ḣ_s(lambda_ref)‖∞ in `_step`:** the NMPC returns `lr` (planned `lambda_ref`,
   `sim_loop.py:2560`); the anchors (`sched.anchors_{a,b}[stance_{a,b}]`) and `r_com` (`rs.r_com`) are in
   scope. `Ḣ_s(lr) = Σ r_Cj×f_j_ref + τ_j_ref` (origin-referenced — the same quantity the NMPC envelope
   path constraint enforces, `centroidal_nmpc.py:279-282`) is a few lines, **no reconstruction**.
2. **`passivity_W_budget` RHS hook:** the joint-block row `A_pass[τ_q]=dq`, `b_pass = −2α·T_kin + W_budget`
   (`wholebody_qp.py`, dock-floor audit). Piste A drives it per-tick (new `qp.solve` kwarg overriding the
   constant).
3. **`r_com` + anchors at the QP tick:** `r_com=rs.r_com`, `contact_config` (with `r_contact_{A,B}`) — the
   orbital term is `M_exact = compute_momentum_map(0, cc)`, linear in λ (`r_com` a per-tick parameter).

## Part 1 — Implementation (LOT A + LOT B, both default-off — committed `a603c82`)

- **LOT A (envelope-coupled budget):** `config.ds_passivity_beta` (β, default 0). Per moving-DS tick,
  `W_budget = β·α·max(0, τ_w_max − ‖Ḣ_s(lambda_ref)‖∞)` (NMPC-exact planned Ḣ_s), passed via a new per-tick
  `qp.solve(passivity_W_budget=…)` override on the joint-block passivity RHS. β=0 ⇒ None ⇒ strict
  (byte-identical).
- **LOT B (FLAG 2, exact box):** `config.qp_envelope_exact` (default False). When True the momentum-rate box
  uses the exact origin-referenced `Ḣ_s`: `|M_exact·λ| ≤ τ_w_max`, `M_exact = compute_momentum_map(0, cc)`
  (levers from O_s) — adds the orbital term `r_com×Σf` the `|M_λ·λ|` proxy omits. **Numerically confirmed**
  exact = proxy + orbital (residual 9e-17; proxy omits **0.33 N·m** at the 0.35 m standoff).
- `κ = β·α`, β the single knob; β=0 recovers strict ≤0. Locomotion scope only. No NMPC-state change, no new
  decision variable. Trace extended with `W_budget` + `dq_tau` for the safety check.

**Functional verify (β=1, exact, mag 0.05, n=2):** W_budget exercised (max 4.19, mean 3.53 W); **positive
joint work occurs** (`dqⱼᵀτ_q>0` in 67 % of moving-DS ticks, max +0.0024 W); **budget NEVER violated**
(0 ticks with `pass_resid > W_budget`; max `pass_resid − W = −3.05`); envelope ‖Ḣ_s‖∞ ≤ 0.23 ≪ 5; feasible
(2/0 docks). LOT A is functional and the budget bound holds every tick.

## Part 2A — β-sweep (moving CoM mag 0.05, dt_ds 2.5, exact box ON), raw

MOVING-segment metrics (from `ds_mobile_trace`) + traversal residual + C1/C5/C6:

| β | CoM err max / final [m] | W_budget max [W] | passivity p_frac | positive-work frac | budget viol. | ‖Ḣ_s‖∞ (cap 5) / over-cap | residual [N·m·s] | C1 (worst d) | C5 h_w | C6 | docks/to |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 (strict) | 0.0484 / 0.0029 | 0.000 | **0.79** | 0.00 | 0 | 0.252 / 0 | 0.002864 | PASS (4.94) | FAIL (4.935) | BIT-ID | 5 / 0 |
| 0.25 | 0.0469 / 0.0075 | 1.048 | **0.00** | **0.67** | 0 | 0.233 / 0 | 0.003115 | FAIL (4.95) | FAIL (4.932) | BIT-ID | 5 / 0 |
| 0.5 | 0.0469 / 0.0075 | 2.097 | **0.00** | **0.67** | 0 | 0.233 / 0 | 0.003115 | FAIL (4.95) | FAIL (4.932) | BIT-ID | 5 / 0 |
| 1.0 | 0.0469 / 0.0075 | 4.194 | **0.00** | **0.67** | 0 | 0.233 / 0 | 0.003115 | FAIL (4.95) | FAIL (4.932) | BIT-ID | 5 / 0 |

- **SAFETY — the budget mechanism works and is bounded.** β>0 opens the budget (W max 1.05→4.19, linear in
  β), **positive joint work now occurs** (`dqⱼᵀτ_q>0` in **0.67** of moving ticks vs 0.00 at β=0), and the
  **budget is NEVER violated** (0 ticks with `pass_resid>W_budget`), and **the envelope is NEVER exceeded**
  (‖Ḣ_s‖∞ ≤ 0.25 ≪ 5, 0 over-cap). Passivity binding falls from 0.79 to **0.00** (the budget fully relieves
  the strict-passivity binding α reported).
- **BUT CoM tracking does NOT improve, and saturates at β=0.25.** com_max barely moves (0.0484→0.0469, −3 %);
  **com_final gets *worse*** (0.0029→0.0075). And β=0.25/0.5/1.0 are identical — because the positive work
  actually needed is **tiny** (max ≈ +0.0024 W, verify) vs the budget (1–4 W), so the budget is
  ~1000× over-provisioned and non-limiting beyond β=0.25. **⇒ at mag 0.05 the CoM lag is NOT
  passivity-limited; relieving passivity does not buy tracking.** (Refines the α reading — passivity *binds*
  during the moving CoM, but the lag is set by the reference/horizon dynamics + welded-CoM kinematics, not by
  the passivity inequality.)
- **Dock margin / C1:** β=0 (exact box) PASSES C1 (worst 4.94, margin = baseline); β>0 FAILS marginally
  (worst 4.95, margin 0.05 < baseline 0.06). The budget does **not** recover the dock margin — it slightly
  degrades it. **C5 h_w FAILs at every β here** — that is a LOT-B (exact-box) effect, isolated in 2B below
  (β does not change h_w: 4.935→4.932).
- **Residual:** β=0 exact **0.002864**; β>0 exact **0.003115** (+9 % — the positive work slightly raises the
  leak). All ≪ the proxy/strict 0.00396.

## Part 2B — FLAG-2 box effect, ISOLATED (β=0, proxy vs exact), raw

| box (β=0) | CoM err max/final | ‖Ḣ_s‖∞ (DWELL) | residual [N·m·s] | C1 (worst d) | C5 h_w∞ | C6 | docks/to |
|---|---|---|---|---|---|---|---|
| proxy (`qp_envelope_exact=False`) | 0.0484 / 0.0029 | 0.252 | **0.003963** | FAIL (4.98) | **PASS (4.179)** | BIT-ID | 5 / 0 |
| exact (`qp_envelope_exact=True`) | 0.0484 / 0.0029 | 0.252 | **0.002864** | PASS (4.94) | **FAIL (4.935)** | BIT-ID | 5 / 0 |

**The exact box is a real trade — both effects come from it alone (β=0, same scenario):**
- **+ residual −28 %** (0.003963 → 0.002864) and **+ better dock margin** (C1 proxy FAIL 4.98 → exact PASS
  4.94): correcting the envelope quantity (origin-referenced Ḣ_s, orbital term in) yields a cleaner
  conservation leak and a slightly closer dock.
- **− h_w +0.76 Nms** (4.179 → 4.935 → **C5 FAIL**). **The box is active in SS too** (the momentum-rate box
  runs every QP tick, and C3 shows ‖Ḣ_s‖∞_SS = 5.0 binds in swing), so the correction changes the **SS**
  solution, not just the moving-DS — and the corrected envelope loads the wheels more (h_w↑). The DWELL CoM
  tracking is unchanged (com identical) — the box's effect is on λ/h_w/residual, not on the CoM track.
- feasibility unchanged (5/5 dock, 0 timeouts, C6 bit-identical).

---

## Flags / divergences vs the α / envelope / passivity / dock-floor audit facts

1. **Refines α (the load-bearing one):** α found the moving-CoM conflict *passivity-dominated* (passivity
   binds 61–100 %) and inferred Piste A would help. This data **confirms the budget relieves the binding**
   (p_frac 0.79→0.00, positive work enabled, safely) **but shows it does NOT improve CoM tracking** at mag
   0.05 (saturates at β=0.25; the needed work ≪ budget). So *passivity binding ≠ tracking bottleneck* here —
   the lag is reference/kinematics-limited. β buys safe positive-work capacity, not (at this magnitude)
   tracking.
2. **Envelope audit (LOT B) confirmed + quantified:** the proxy omits 0.33 N·m at the 0.35 m standoff
   (numeric check 9e-17 identity); correcting it (exact box) cuts the residual 28 % **but** costs +0.76 Nms
   h_w (C5) because the box binds in SS. The box correction is **not** free at the canonical working point.
3. **Consistent with dock-floor:** the dock stays ~4.5–5.0 mm (kinematic) across all runs; β/box only shift
   it ~0.05 mm at the margin. No timeouts.
4. **C6 BIT-IDENTICAL in every run** — the flag-OFF determinism holds with both lots present (defaults
   dormant).

## Reproduce
```
bash scripts/run_pisteA_sweep.sh   # β{0,0.25,0.5,1.0} exact + β0 proxy; n=5; mag 0.05; dt_ds 2.5
MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/audit_pisteA.py b0=results/pa_b0_exact ...
```
Supporting: `results/j2_pisteA/{beta.log, box.log, residual.log, gate_C1-C5.log}`. Raw per-run dirs
reproducible from the script, not committed.

**Regression (`pytest tests/`):** defaults dormant (β=0 ⇒ W_budget=None ⇒ strict; `qp_envelope_exact=False`
⇒ proxy box) ⇒ byte-identical, and **C6 is BIT-IDENTICAL in every sweep run above** (direct evidence the
flag-OFF path is unchanged). Full-suite count **in flight**; expected 220 passed / 1 pre-existing FK fail
(`test_E7_t15_step2_dock_under_fk_mode`). [Updated below on completion.]

**STOP after the report.** No success threshold; β and the box-correction trade are dimensioned by Idriss +
reviewing Claude on these numbers. No merge, no PR.
