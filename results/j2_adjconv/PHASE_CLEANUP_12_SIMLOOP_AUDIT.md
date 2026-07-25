# PHASE CLEANUP-12 — `sim_loop.py` audit + cross-study vs the QP/NMPC cleanup

Audit of the last large file, and an explicit comparison with what CLEANUP-1…11 found in
`centroidal_nmpc.py`, `nmpc_solver.py`, `wholebody_qp.py` and `hierarchical_qp.py`.
**No code changed.**

Method unchanged: coverage of the full canonical replay for live/dead, AST for structure, and
the canonical flag values **measured by instrumenting `SimulationLoop.__init__` during a real
`dca` run** — not read off dataclass defaults (CLEANUP-2 F1 lesson).

---

## 1. Scale and shape

| file | statements | never executed | coverage |
|---|---|---|---|
| `sim_loop.py` | 1527 | **372** | 76 % |
| `planning/torso_planner.py` | 221 | **99** | **55 %** ⚠ |
| `planning/coarse_preplanner.py` | 242 | 45 | 81 % |

`sim_loop.py` is 3747 lines, one class, 27 methods, 97 % of the file inside callables.

| lines | span | stmts | **depth** | params | method |
|---|---|---|---|---|---|
| **1095** | 2601–3695 | 482 | **9** | 17 | `_step` |
| **630** | 1950–2579 | 261 | **7** | 1 | `run` |
| 445 | 1387–1831 | 165 | 3 | 7 | `_setup_torso_for_step` |
| 313 | 197–509 | 103 | 2 | 4 | `setup` |
| 236 | 613–848 | 64 | 4 | 16 | `_run_ds_passivity_loop` |
| 210 | 922–1131 | 113 | 3 | 9 | `_log_ds_tick` |

Everything else is ≤ 126 lines.

## 2. Where the 372 dead statements live

| dead | of stmts | method |
|---|---|---|
| 142 | 482 | `_step` |
| 84 | 165 | `_setup_torso_for_step` |
| 61 | 261 | `run` |
| **20** | **22** | `_planned_arm_config` — **effectively the whole method** |
| 15 | 113 | `_log_ds_tick` |
| 11 | 103 | `setup` |
| 11 | 43 | `_settle_setup` |

## 3. Canonical flag values (measured) → what they kill

| flag | canonical | dead code it gates (approx.) |
|---|---|---|
| `reference_source` | **`task_space`** | the `joint_space_fk` path: 1373–1385, 1758–1767, **plus all of `_planned_arm_config`** ≈ 43 lines |
| `use_mid_waypoint_reshape` / `mid_waypoint_force_on` | **False** | 1595–1616, 1775–1785 ≈ 37 |
| `use_path_feasibility_check` | **False** | 1580–1586 ≈ 11 |
| `ds_mobile_com_magnitude` | **0.0** | 2119–2128, 3096–3102, 3107–3114 = 25 |
| `n_settle_damping_steps` | **0** | 561–572 = 12 (loop body never entered) |
| `use_local_delta_mapping` | **False** | 2886–2889, 2894–2898 = 9 |
| `ds_passivity_beta` | **0.0** | 3049–3056 = 8 |
| `use_gmo_dock` | **False** | 3470–3477 = 8 |
| `aocs_use_H_estimator` | **False** | 3599–3604 = 6 |
| `use_trajectory_aware_ik` | **False** | — |
| `_diag_freeze_ref`, `_diag_lock_arm_joints` | off | 2978–2984, 3451–3454 = 11 |

Live and load-bearing: `use_m2_stack=True`, `ds_centroidal_mode=True`, `ss_two_task_mode=True`,
`enforce_hw_conservation=True`, `use_com_z_standoff=True`,
`aocs_mode='legacy_pid_numerical'`, `interstep_settle_alpha_wrench=3.0`,
`stop_on_failed_step=True`.

### Must be KEPT — dead only because the system is healthy

≈ 46 dead lines are failure/fallback handling: `if not info_n.success` (2742–2746),
`if not nmpc_ok` (2758–2773), `if not step_feasible` (2192–2199), `if q_end is None`
(1472–1476, 1496–1499), missing-site guards (1254–1257), `if result.success` else-path
(1935–1938). Same class as the NMPC's `get_shifted_fallback` and the QP's `_solve_qp_raw`
branches — uncovered *because* nothing fails on the canonical, which is the opposite of a
reason to delete.

---

## 4. Cross-study vs the completed cleanup

### 4.1 Same disease, different organ

| | QP / NMPC (CLEANUP-5/6) | `sim_loop` (here) |
|---|---|---|
| dead fraction | 34 % of `wholebody_qp`, 30 % of `hierarchical_qp` | **24 %** of `sim_loop` |
| how it presents | dead **task blocks** in one giant `solve()` | dead **behavioural branches** across `_step` / `_setup_torso_for_step` / `run` |
| what gates it | config flags (`r_tube`, `cooperative_arms_mode`, …) | config flags (`reference_source`, `use_mid_waypoint_reshape`, …) |
| root cause | incremental research retained behind flags | **identical** |

The diagnosis that motivated CLEANUP-6 — *"the canonical run is the two-task stack, the rest
was incremental research"* — applies verbatim here. `sim_loop` accumulated ~10 opt-in research
switches, all defaulted off on the canonical.

### 4.2 But the risk profile is inverted

This is the key difference and it changes the recommendation:

| | `WholeBodyQP.solve()` | `SimulationLoop._step` |
|---|---|---|
| lines | 597 (pre-refactor) | **1095** |
| statements | 219 | **482** |
| **max nesting depth** | **3** | **9** |
| parameter locality | 30/40 single-block | 17 params, not yet measured per-block |
| shape | linear assembly pipeline | branching state machine |

`solve()` was *long but linear*, which is why extraction was provably safe and came back
byte-identical four times. `_step` is long **and** deeply nested — depth 9 means genuinely
interleaved control flow, so the same "lift a block into a helper" move is **not** justified by
the same evidence. Any `_step` decomposition needs its own measurement pass first (cross-block
locals, mutation, early returns/continues).

### 4.3 Residue of what we already removed — clean

Grepping `sim_loop` for the architectures retired in CLEANUP-6/8: `cooperative` 0 code refs,
`r_tube`/tube 0, `ss_centroidal_momentum_task` 0, soft-CoM 0, null-space/`N_torso` 0, `strict`
0, EXT-phase 0 code refs. The CLEANUP-8 plumbing prune was complete — `sim_loop` carries **no
orphan references** to the deleted QP machinery. The only surviving link is `use_m2_stack`
(3 code refs), which is the documented trap below.

### 4.4 Cross-link: a PORT_SYNTHESIS latent bug is dormant *because of* one of these flags

`PORT_SYNTHESIS.md` ticket 1 records the `away_normal` sign duplication
(`swing_planner.py:44` vs `ik.py:1402/1282`) as "dormant: `use_path_feasibility_check=False`
on the canonical path". This audit **confirms that flag is False by measurement**, and that its
gated block (1580–1586) never executes. So the latent bug and this dead branch are the same
fact seen from two directions — and **removing the flag would remove the bug's only reachable
path**, while *enabling* it would wake the bug. Worth deciding deliberately rather than by
accident.

### 4.5 `use_m2_stack` — the trap, restated

Still the single most dangerous item. It looks dead (its QP twin was removed) but gates
torso-reference routing (~2871) and **DS passivity** (~3038). Documented in `config.py` and in
`CLEANUP_CARRYOVER.md` §A2.

### 4.6 New finding: `torso_planner.py` is 45 % dead

Not previously audited and worse than `sim_loop` proportionally (99 of 221 statements never
execute). It should be audited before `sim_loop` is refactored, since `_setup_torso_for_step`
(445 lines, 84 dead) is its main caller and the two are likely dead for the same reasons.

---

## 5. Recommendation

Ordered by value ÷ risk. **Nothing here is as safe as the QP cleanup was** — say so plainly.

1. **Flag-gated dead branches** (≈ 170 lines across ~10 features). Same character as the QP
   task blocks: opt-in research paths, off on the canonical. Removable one flag at a time,
   each gate-verified. Start with the fully-contained ones — `ds_mobile_com_magnitude`,
   `ds_passivity_beta`, `aocs_use_H_estimator`, `n_settle_damping_steps`,
   `use_local_delta_mapping` — before the FK-reference path, which also takes
   `_planned_arm_config` with it.
2. **Audit `torso_planner.py`** (45 % dead) before touching `_setup_torso_for_step`.
3. **Do NOT decompose `_step` on the strength of this audit.** Depth 9 with 482 statements is a
   different animal from `solve()`'s depth-3 pipeline; it needs its own coupling/mutation
   measurement first, exactly as CLEANUP-10 did for `solve()`.
4. **Keep every failure/fallback branch** (≈ 46 lines).
5. The `use_path_feasibility_check` decision (§4.4) should be made consciously — it is
   entangled with a known latent bug.
