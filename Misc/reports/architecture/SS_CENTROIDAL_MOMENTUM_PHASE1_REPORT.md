# SS Centroidal-Momentum Task — Phase 1 Report

**Branch:** `feat/ss-centroidal-momentum-task`  **Base / Phase-1.1 baseline commit:** `dcda974`
**Memo (authoritative):** `Misc/reports/architecture/SS_CENTROIDAL_MOMENTUM_TASK_2026-06.md` (§4 Phase 1)
**Scope:** Phase 1 only (baseline re-establishment, T-MOM implementation behind a default-OFF flag, bit-identical verification, run-metadata logging). **STOP at end — do not begin Phase 2.**

---

## 0. Headline

- **Phase 1.1 — baseline re-established and is BIT-IDENTICAL to the 5bca42c reference** (stronger than the memo's sanity bar; zero deviation, SS *and* DS).
- **Phase 1.2 — T-MOM (linear rows) implemented behind `cfg.ss_centroidal_momentum_task` (default OFF); flag-OFF is bit-identical** to the Phase-1.1 baseline in every physical/control quantity.
- **Phase 1.3 — run metadata logged** (git HEAD, clean control code, full 130-field config dump, env versions + the env-fix note).
- Two **memo bookkeeping flags** for review (neither blocks Phase 1): the brief's "expect DS-side differences" does not apply to this run; and the Phase-3 gate's quoted h_w / τ_w-saturation numbers do not match the actual committed 5bca42c baseline.

---

## 1. §2.2 "Phase-1 check" — `ss_alpha_com` / `ss_Kp_com` plumbing findings

Required by memo §2.2 before building on or bypassing the existing CoM plumbing.

- **Exists.** `SimConfig.ss_alpha_com = 2e2`, `ss_Kp_com = ss_Kd_com = 3.0` (`config.py`), piped via `_build_qp` (`sim_loop.py`) into `WholeBodyQPConfig.alpha_com / Kp_com / Kd_com`.
- **It is A_G-based (linear rows), NOT a torso-proxy.** The task (`wholebody_qp.py` CoM-cost helper) is
  `J_com·q̈ = a_com_des − J̇_com·q̇`, `a_com_des = a_com_ff + Kp(r_com_ref−r̂) + Kd(v_com_ref−v̂)`,
  with `J_com = data.Jcom` and `J̇_com·q̇ = data.acom` (`robot_interface.py`). Since `J_com = [A_G]_lin/m`, this **equals T-MOM-linear up to the mass scalar `m`** (folded into the weight). → the memo's "reuse only if A_G-based and sound" condition is met.
- **Canonically disabled** (matches the 5bca42c audit "no α_com task"): the canonical config has `use_m2_stack=True` (forces `alpha_com=0` in `_build_qp`, and gates out the P1 CoM task) and `alpha_com_soft=0.0` (no soft-CoM residual). CoM placement currently flows through `CoMToTorsoMapping → torso-linear P2` — exactly the channel T-MOM replaces.
- **Reuse decision:** reuse the `A_com`/`b_com` construction, but activate it as a **P2 weighted** task at `α_mom`, wired to the NMPC plan, behind the new flag — **not** the legacy P1 activation, which is left untouched.

---

## 2. Phase 1.1 — baseline re-establishment

**Run:** `diag_cooperative_arms.py` canonical 5-step, `legacy_pid_numerical`, K_ω=50, τ_w,max=5 (defaults otherwise: K_θ=1.0, mass_ratio=0.01, anchor_dx=0.8, α_torso_lin=500, settle=20 s), `cooperative_arms_mode=True`. Output dir: `Misc/runs/ssmom_phase1_baseline_main_dcda974/` (memo §7 — new dir; the committed 5bca42c reference `Misc/runs/diag_cooperative_arms_legacy_pid_numerical/` was never touched).

### 2.1 Comparison table — main HEAD (dcda974) vs 5bca42c reference

| Metric | 5bca42c (committed) | main HEAD (dcda974) | Δ |
|---|---|---|---|
| Docks (5/5), per-step d [mm] | 1.86 / 4.94 / 4.96 / 4.77 / 5.00 | 1.86 / 4.94 / 4.96 / 4.77 / 5.00 | **0** |
| θ_s attitude peak / final [deg] | 1.8795 / 1.6186 | 1.8795 / 1.6186 | **0** |
| θ_s RMS [deg] | 1.1372 | 1.1372 | **0** |
| Torso-ori SS RMS / peak [deg] | 0.6807 / 1.5531 | 0.6807 / 1.5531 | **0** |
| Torso-pos SS RMS / peak [mm] | 3.930 / 17.623 | 3.930 / 17.623 | **0** |
| Swing-EE ori SS RMS / peak [deg] | 0.3489 / 1.9847 | 0.3489 / 1.9847 | **0** |
| Ḣ_s SS peak x/y/z [N·m] | 5.00 / 5.00 / 5.00 | 5.00 / 5.00 / 5.00 | **0** |
| τ_w SS peak; clip fraction | 5/5/5; 4.51 % | 5/5/5; 4.51 % | **0** |
| h_w peak [N·m·s] (SS, t=9.34 s) | 3.584 (see §2.3) | 3.584 | **0** |

**`postproc_F3F4.csv` is byte-identical** between the two runs (1080 rows × 23 cols; verified with `cmp`). All `postproc_metrics.json` scalars match. **Zero deviation — the memo §4 Phase-1.1 STOP condition is not triggered.**

### 2.2 Why it is bit-identical (provenance)

`git diff 5bca42c..dcda974` touches **only** `Misc/reports/architecture/SS_CENTROIDAL_MOMENTUM_TASK_2026-06.md` (the memo) and `docs/architecture/setup_env.sh`. **`crawlbot/` and `models/` are byte-identical** between the two commits. With identical control code + identical canonical config + a faithful environment, the simulation is deterministically identical.

### 2.3 Memo bookkeeping flags (do not block Phase 1; for review)

1. **"DS-side differences expected" does not apply here.** The brief notes main carries a DS rework so DS metrics may differ from 5bca42c. But `ds_centroidal_mode=True` is **already set at 5bca42c** (in the canonical runner), and no `crawlbot/` code changed `5bca42c→dcda974` — so DS metrics match exactly, as expected once this is understood.
2. **Phase-3 gate numbers vs the actual committed baseline.** Memo §4 Phase-3 quotes h_w peak "3.38 N·m·s" and τ_w saturation "≤ 2.95 %". The committed 5bca42c baseline (reproduced here) is **h_w peak = 3.584 N·m·s** (an SS event at t=9.34 s) and **τ_w SS saturation = 4.51 %**. (The reference run's `sim_log.json` is not committed — only postproc artifacts — but `postproc_F3F4.csv` byte-identity + identical `crawlbot/` code prove the reference h_w equals this run's 3.584.) The other gate anchors (docks, attitude, tracking RMS) match the committed baseline exactly. → the h_w / τ_w-sat gate thresholds should be reconciled to the true baseline before Phase 3.

### 2.4 §6 plot set

Four overlay figures (main-HEAD candidate over 5bca42c reference, §6 convention: reference grey/dashed, candidate colour, same axes, SS shaded, per-axis x/y/z, ±5 N·m envelope) in `Misc/runs/ssmom_phase1_baseline_main_dcda974/phase1_plots/`:
`f1_torso_tracking.png`, `f2_swing_ee.png`, `f3_attitude.png`, `f4_hdot_tau.png`. (Curves perfectly superimpose — the visual corollary of byte-identity.) Generated by the committed, re-runnable `Misc/scripts/plot_ssmom_phase1.py`.

---

## 3. Phase 1.2 — T-MOM implementation (behind default-OFF flag)

### 3.1 What / where
- **`config.py`** — `ss_centroidal_momentum_task: bool = False`, `ss_alpha_mom: float = 5e2`, `ss_alpha_tl_weak: float = 0.0`. Kp/Kd reuse the existing `ss_Kp_com`/`ss_Kd_com`.
- **`wholebody_qp.py`** — same three fields on `WholeBodyQPConfig`; in `_build`, the cooperative-arms **P2 torso-linear block** now branches: when the flag is ON (cooperative SS, non-settle), the linear CMM task (`A_com`/`b_com`, already built upstream) replaces the torso-linear channel at `α_mom`, projected through the **same `N_torso`** as the EE task; Variant B adds a weak torso-linear regulariser at `ss_alpha_tl_weak` (`>0`). The `else` (OFF) branch is the **verbatim original** torso-linear task.
- **`sim_loop.py`** — `_build_qp` pipes the three new fields into `WholeBodyQPConfig`.
- **`diag_cooperative_arms.py`** — backward-compatible `--out-dir` and `--ss-centroidal-momentum-task / --ss-alpha-mom / --ss-alpha-tl-weak` CLI (makes the baseline reproducible from a committed script and the flag reachable for Phase 2; defaults preserve canonical behaviour).

This realises the memo §2.2 hierarchy: P1 torso-ori (strict, null-space) · **P2 momentum-lin (α_mom) + EE (α_ee)** · P3 posture · P4 wrench. The ON path is dimensionally identical to the task it replaces (both 3×n through the same projector) so it cannot throw a dimension error; full ON behaviour is **Phase 2's** validation, not Phase 1's.

### 3.2 Bit-identical (flag OFF) — PROOF

Re-ran the identical 5-step scenario with the **new code, flag OFF** (`results/ssmom_phase1_flagOFF_verify/`):
- `postproc_F3F4.csv` — **byte-identical** to the Phase-1.1 baseline (`cmp`).
- `sim_log.json` — of 71 numeric arrays, the **only** two that differ are `qp_time_ms` and `nmpc_time_ms` (per-tick **wall-clock solve times** — non-deterministic measurement noise, not simulation state). Every physical/control array (poses, velocities, torques, momentum, contact λ, docks, …) is **exactly identical** (max |Δ| = 0).
- Docks identical (1.86 / 4.94 / 4.96 / 4.77 / 5.00 mm).

→ **Flag OFF is bit-identical to the Phase-1.1 baseline.** (Proof produced via the out_dir redirect; the committed `diag_cooperative_arms.py --out-dir … ` with the flag OFF builds the identical `SimConfig`, so the result transfers.)

---

## 4. Frame-consistency note (memo §2.2)

`A_G` (Pinocchio `ccrba`/`dccrba`) and the CoM Jacobian `J_com = data.Jcom` / drift `data.acom` are expressed **at the CoM, world-aligned**. The NMPC references (`r_com*`, `v_com*`, `a_com_ff`) live in the rotating structure frame `R_s`. At ω_s ~ 1 °/s (the canonical run peaks at ‖ω_s‖ ≈ 8.45 mrad/s ≈ 0.48 °/s) the world↔R_s transport terms are negligible, but they **must be handled consistently** — the same treatment as the H-estimator `include_transport` logic. The v1 task evaluates `A_com`/`b_com` at the current `q` (world-aligned) and applies the PD on the NMPC plan; the small transport mismatch is in the noise of the ω_s magnitude here, but the choice is explicit and is the natural place a v2 (with `L_com` rows) would need a rotation. **Action for Phase 2:** log the realised vs planned `Ḣ_s` (already in the §6 Phase-2 plot set) to quantify any residual frame error empirically.

---

## 5. Regression & environment

- **Regression suite** (`pytest tests/`, run after the core-module edits per Rule 7): **216 passed, 1 pre-existing failure** (646 s) — `test_E7_t15_step2_dock_under_fk_mode`, which asserts over a **stale committed** `sim_log.json` (`results/M7_1pct_3step_v22_t15_fk/`, last touched 2026-04-27 `e0ac231`, contains 3 aborted steps). Environment-independent (reads a static JSON), a different scenario from the cooperative-arms baseline, confirmed a known issue. No new failures from the Phase-1 edits (flag default OFF).
- **Environment fix (reproducibility-critical).** `setup_env.sh` pins only `pin==3.9.0`; this image carried too-new cmeel libs and Pinocchio failed to load. Fixed: `cmeel-urdfdom==4.0.0` (provides `liburdfdom_sensor.so.4.0`), `cmeel-tinyxml2==9.0.0.2` (provides `libtinyxml2.so.9`; the `9.0.0` release ships no `.so`), plus `ldconfig` over `cmeel.prefix/lib` (the older wheels have incomplete RUNPATHs). These restore pin 3.9.0's build-time deps — validated faithful by the byte-identical reproduction of the 5bca42c artifacts. Offscreen osmesa rendering is unavailable (no GL libs), so sims use `MUJOCO_GL=disabled`; the §6 plot set is matplotlib data plots, unaffected. *Recommendation: harden `setup_env.sh` to pin these cmeel versions (separate change).*

---

## 6. Reproduction

```bash
# Baseline (main HEAD, flag OFF):
MUJOCO_GL=disabled FRAMES_PER_STEP=0 PYTHONPATH=. python3 scripts/diag_cooperative_arms.py \
    --aocs_mode legacy_pid_numerical --K_omega 50 --tau_w_max 5 \
    --out-dir ssmom_phase1_baseline_main_dcda974
# Post-process + §6 plots:
SSMOM_RUN_DIR=ssmom_phase1_baseline_main_dcda974 MUJOCO_GL=disabled PYTHONPATH=. \
    python3 Misc/scripts/postprocess_results_figs.py
PYTHONPATH=. python3 Misc/scripts/plot_ssmom_phase1.py \
    --ref-dir Misc/runs/diag_cooperative_arms_legacy_pid_numerical --ref-label 5bca42c \
    --cand-dir Misc/runs/ssmom_phase1_baseline_main_dcda974 --cand-label "main HEAD (dcda974)"
```

---

## 7. Open items / for the review session (before Phase 2)
1. Reconcile the Phase-3 gate's h_w (3.38) and τ_w-sat (2.95 %) numbers against the true committed baseline (3.584 / 4.51 %).
2. The updated memo (with §6 plotting rules) is **carried on this branch** because `main` was not advanced; if you prefer it on `main`, push it there and I will rebase (drop the branch copy).
3. `MUJOCO_GL=disabled` (osmesa unavailable) — fine for Phase 1/2/3 data plots; flagged for awareness.

**STOP — Phase 1 complete. Awaiting review before Phase 2.**
