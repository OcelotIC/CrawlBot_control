# INTERNAL — Dock-Leak Part 3: Fix A (full-DOF momentum-consistent impact map) + re-gate

**Branch:** `fix/dock-impact-map` off canonical **`21cec74`** (diagnosis on `diag/dock-leak` is a
separate artefact). Push, **never merge**. **Mode:** first behaviour-changing in-plant edit — SINGLE
PASS, no iteration. **Only the inelastic-impact block changed** (no controller/NMPC/QP/AOCS/weight/
scenario edits). **Frozen reference (untouched):** `results/p3b_gate_w24000_kp3/`. **Metric:**
`subtree_angmom[0]`. **Tooling:** `scripts/{run_fixA_gate.sh, audit_fixA_leak.py, plot_dock_leak_part3.py}`;
figure `fixA_plots/dock_leak_fixA.png`.

---

## ✅ OUTCOME — leak collapses 51× (0.203 → 0.0040 N·m·s); C1–C5 PASS (≥ canonical); one expected C6 change

Fix A ports the Part-2 **A.1** projection in-plant. The leak collapses from 0.203 to **0.0040 N·m·s**
(≈0.08 % of the ±5 N·m·s wheel budget), all five steps dock cleanly, and **every controller criterion
(C1–C5) passes at or slightly better than canonical**. The only criterion that moves is **C6(a)
byte-identical-OFF**, which necessarily changes because Fix A is an **always-on plant change** (it alters
the impact map in every run, including flag-OFF) — this is by design, not a controller/feature regression.
Per the single-pass discipline I **STOP and report**; the merge call is Idriss's.

---

## 1. The edit (Fix A) — full-DOF, MuJoCo-native, all-DOF write-back

Replaced the robot-only partial impact projection (`sim_loop.py`, the "Inelastic impact" sub-block inside
`if docked:`) with the full-DOF momentum-consistent map validated offline in Part-2 A.1. The
`_activate_weld` + `mj_forward` above it are untouched (the weld must be active so its constraint is in
the set). The new block:
- mass = **`mj_fullM`** (nv×nv: structure base + wheels + torso + joints), not robot-only `rs_dock.H`;
- constraint = the **relative-site weld Jacobian** `J = [J_grip − J_anchor]` (linear + angular) over **all**
  `qvel`, for every **active** weld (read from `eq_active` + the weld map) — the same 6-DOF-per-weld
  relation A.1 validated;
- project `v⁺ = v − M⁻¹Jᵀ(J M⁻¹ Jᵀ)⁻¹ J v` (same operator + plain `np.linalg.solve` as A.1) and write back
  **all** `qvel`.

**B.1 hard requirement satisfied:** the new path **never** calls `pinocchio_to_mujoco` /
`mujoco_to_pinocchio` (verified). Part 2 showed that conversion assumes `v_struct ≈ 0` (false at a dock)
and drops the structure-coupling terms; Fix A stays in MuJoCo DOF throughout, so both Part-2 defects
(one-sided impulse **and** the lossy conversion) are eliminated by construction.

*Note on the A.1 route:* the Part-2 brief sanctioned the relative-site Jacobian as equivalent to the
`efc_J` rows; my validated A.1 used the relative-site route, so per "port, not redesign" Fix A ports
exactly that (not `efc_J`). Also flagged: the cited `mj_fullM(model, M, qM)` signature in
`jacobian_convention_audit.py:152` is **stale for this MuJoCo build** — the working binding is
`mj_fullM(model, data, dst)` (what A.1 and Fix A use).

## 2. (3.1) In-plant first-dock parity vs A.1 — CONFIRMED

Verbose `Impact(fullDOF)` prints (per-dock `|H_sys|` before→after the impact):

| dock | ‖dv‖ | ‖J·v⁻‖ | `|H_sys|` before→after | per-dock impact Δ | A.1 offline |
|---|---|---|---|---|---|
| 0 | 0.0210 | 0.0069 | 0.0000 → 0.0004 | 0.0004 | 0.0004 |
| 1 | 0.0155 | 0.0056 | 0.0015 → 0.0015 | ~0.0000 | 0.0001 |
| 2 | 0.0186 | 0.0065 | 0.0017 → 0.0014 | ~0.0003 | 0.0003 |
| 3 | 0.0144 | 0.0046 | 0.0031 → 0.0031 | ~0.0000 | 0.0001 |
| 4 | 0.0165 | 0.0060 | 0.0034 → 0.0031 | ~0.0003 | 0.0003 |

The in-plant impact injects ~0.0004 or less per dock — matching the A.1 offline residuals. The port is
faithful (vs the canonical partial map's 0.041/0.048/0.049/0.146/0.073). No port bug.

## 3. (3.2) Full-traversal leak — 0.203 → 0.0040 (51×)

`subtree_angmom[0]` at each snapshot, Fix A vs frozen canonical:

| snapshot | t [s] | ‖H_sys‖ Fix A | ‖H_sys‖ canonical |
|---|---|---|---|
| dock_step0 | 3.31 | 0.0000 | 0.0000 |
| release_step1 | 5.04 | 0.0015 | 0.0419 |
| release_step2 | 13.95 | 0.0017 | 0.0226 |
| release_step3 | 18.27 | 0.0031 | 0.0617 |
| release_step4 | 26.88 | 0.0034 | 0.1349 |
| **final** | 50.38 | **0.0040** | **0.2030** |

**51× reduction** (from ~4 % to ~0.08 % of the wheel budget). All 5 steps dock cleanly (d =
[4.94, 4.51, 4.91, 4.61, 4.84] mm, all <5 mm; no aborts). The residual **0.0040** is *not* the impact
(which Fix A drove to ~0.0004/dock) — it is the **DS weld-stabilisation gap-couple** that Fix A does
**not** touch: the O(gap×f) Baumgarte term Part 1 measured (~0.6 % / 0.0022) acting over the DS dwells.
It would only move with weld `solref`/`solimp` or zero-gap docking (the original Part-2 hypothesis,
separately deferred). So the impact artefact (99.4 % of the leak) is removed; what remains is the
genuine, expected soft-weld residual.

## 4. (3.3) Re-gate — six criteria, Fix A vs frozen canonical (same flags, same baseline dcda974)

| # | criterion | limit | **Fix A** | canonical | verdict |
|---|-----------|-------|-----------|-----------|---------|
| C1 | docking 5/5, <5 mm | 5 mm | [4.94,4.51,4.91,4.61,4.84] | [4.94,4.51,4.93,4.62,4.89] | ✅ PASS |
| C2 | torso pos peak | ≤17.6 mm | **16.0** | 16.5 | ✅ PASS (better) |
| C2 | torso ori RMS | ≤0.68° | 0.089 | 0.092 | ✅ PASS |
| C3 | envelope ‖Ḣ_s‖∞ | ≤5 N·m | 5.00 | 5.0 | ✅ PASS |
| C4 | θ_s peak / final | ≤1.9 / 1.65° | 0.59 / **0.10** | 0.59 / 0.15 | ✅ PASS (better) |
| C5 | h_w peak ∞-norm | ≤4.5 N·m·s | **4.373** | 4.405 | ✅ PASS (more margin) |
| C6a | flag-OFF vs dcda974 | Δ=0 | **DIFFERS 4.9e-2** | Δ=0 | ⚠ CHANGED (by design) |
| C6b | test_reworked_qp | 8/8 | 8/8 | 8/8 | ✅ PASS |

- **C1–C5 all pass**, with C2/C4/C5 slightly **better** than canonical (the cleaner post-dock state
  marginally helps tracking and trims h_w from 4.405 → 4.373, *more* C5 margin — no regression).
- **B15** (τ_w-sat @100 Hz) aggregate **11.92 %** (canonical 12.1 %) — slightly better.
- **C6(a) changed by design:** with Fix A applied, the flag-OFF run is no longer byte-identical to
  dcda974 (worst |Δ| = 4.9e-2) — because Fix A is an **always-on** impact-map change present in *every*
  run, including OFF. The OFF path still docks **5/5** (d=[1.86,4.94,4.87,4.92,4.94] mm, no aborts). C6(a)
  was designed to confirm the **`ss_two_task` feature** is cleanly gated (OFF recovers the baseline); that
  feature-gating is intact — it is the deliberate plant fix, not the feature, that moves the OFF baseline
  off dcda974. **C6(b) `test_reworked_qp` 8/8** (the QP is untouched).

## 5. Test suite (CLAUDE.md rule 7)

`pytest tests/` → **220 passed, 1 failed**. The single failure
(`test_E7_t15_step2_dock_under_fk_mode`) is the **same pre-existing failure** that fails identically on
clean `21cec74` (verified in Part 1: it validates a committed, stale FK-reference artefact
`results/M7_1pct_3step_v22_t15_fk/sim_log.json` that already carries aborted steps — the loop-free /
FK-mapping "Open" item). It does **not** read Fix-A output. So **Fix A introduces zero new test
failures** (220 pass, same as the clean tree) — notably, no traversal/dock test regressed despite the
impact-map change, and `test_reworked_qp` is 8/8 (QP untouched).

## 6. Verdict (STOP — single pass; merge call is Idriss's)

1. **Leak collapsed 51× (0.203 → 0.0040).** The dock impact artefact (the confirmed mechanism, Parts 1–2)
   is removed; the in-plant per-dock injection matches A.1 (~0.0004). The residual is the un-touched DS
   weld gap-couple (the ~0.6 % Part-1 soft-weld term), not the impact.
2. **Controller gate intact:** C1–C5 pass at or better than canonical; B15 slightly better.
3. **One by-design change (C6a):** Fix A, being always-on, moves the flag-OFF baseline off dcda974
   (Δ=4.9e-2); the OFF path still docks 5/5 and the QP/feature-gating is intact (C6b 8/8). This is the
   intended plant fix surfacing — not a controller regression — but it **is** a movement vs canonical, so
   per the single-pass rule I flag it for your call rather than absorbing it.
4. **No iteration, no tuning, no merge.** Pushed to `fix/dock-impact-map` only. Decision points for you:
   (i) accept the C6(a) by-design change (update the C6 definition for an always-on plant fix vs a gated
   feature); (ii) whether to also address the residual 0.0040 gap-couple via the weld-`solref`/zero-gap
   path; (iii) merge timing (Fix A now vs fold into the J2 dock rework).
