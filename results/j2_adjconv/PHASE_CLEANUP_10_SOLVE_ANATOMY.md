# PHASE CLEANUP-10 — anatomy of `WholeBodyQP.solve()` (READ-ONLY)

Audit of the one structural problem left in `wholebody_qp.py`. **No code changed.**
Measured with AST analysis, not by reading: block spans, per-parameter block locality,
cross-block local coupling, and control-flow nesting depth.

## Headline

| metric | value |
|---|---|
| `solve()` span | lines 259–855 = **597 lines** |
| docstring | 54 lines |
| signature + body | **543 lines** |
| parameters | **40** |
| statements (AST) | 219 |
| **max control-flow nesting depth** | **3** |
| labelled blocks | 12 |

**The important result: `solve()` is long and linear, not complex.** 219 statements at
nesting depth 3 across 12 sequential blocks. It is a straight-line assembly pipeline, not a
knot — which is the easiest possible shape to decompose.

## Block anatomy

| lines | size | block |
|---|---|---|
| 385–609 | **225** | `--- Build QP ---` ← **41 % of the body** |
| 610–664 | 55 | SS: the two-task stack (the canonical SS controller) |
| 665–687 | 23 | Posture regulation |
| 688–704 | 17 | DS: joint-space settle |
| 705–738 | 34 | DS: centroidal tasks |
| 739–751 | 13 | Contact-wrench tracking |
| 752–792 | 41 | DS: internal-stress regularization |
| 793–799 | 7 | Joint-torque minimization |
| 800–807 | 8 | Acceleration regularization |
| 808–826 | 19 | h_w slack penalty |
| 827–832 | 6 | Extract solution |
| 833–855 | 23 | h_w-slack telemetry |

Every task block is already small (7–55 lines) and clearly labelled. **One block is the
problem**: `Build QP` at 225 lines. It is not monolithic either — it carries four internal
banner-labelled sections:

| lines | size | sub-section |
|---|---|---|
| 396–446 | ~51 | **EQUALITY CONSTRAINTS** — full dynamics + contact acceleration |
| 447–545 | ~99 | **INEQUALITY CONSTRAINTS** — momentum box (M5 slack), L_com box, Ḣ_s rate box, passivity |
| 546–587 | ~42 | **BOUNDS** — q̈, τ, contact-wrench, slack non-negativity |
| 588–609 | ~22 | **TASKS** preamble — the shared `A_com` / `b_com` CoM row |

## Parameter locality — the decomposition signal

**30 of the 40 parameters are read in exactly ONE block.**

| destination | count | examples |
|---|---|---|
| `Build QP` only | 17 | `H_robot`, `C_robot`, `J_com`, `J_contacts`, `Jdot_dq_com`, `Jdot_dq_contacts`, `L_com_current`, `a_com_ff`, `dq_t`, `hw_current`, `hw_min`, `hw_max`, `passivity_active`, `passivity_W_budget`, `r_com`, `r_com_ref`, `v_com_ref` |
| SS two-task block only | 11 | `J_ee`, `Jdot_dq_ee`, `R_ee`, `R_ee_ref`, `a_ee_ff`, `a_torso_ff`, `p_ee`, `p_ee_ref`, `p_torso`, `p_torso_ref`, `v_ee_ref` |
| contact-wrench only | 2 | `lambda_ref`, `settle_alpha_wrench` |

Only **10** parameters span multiple blocks, and they are exactly what you would expect —
shared state and mode flags: `settle_mode` (5 blocks), `dq` (4), `ds_centroidal_active` (3),
`q`, `J_torso`, `Jdot_dq_torso`, `R_torso`, `R_torso_ref`, `v_torso_ref`, `contact_config` (2 each).

That locality is unusually clean: each block is nearly self-contained with respect to its
inputs, so extraction would not produce helpers with sprawling argument lists.

## Cross-block coupling — light, and of one kind

Only **16** local names cross a block boundary, and they fall into three groups:

| group | names | note |
|---|---|---|
| setup constants | `cfg`, `idx`, `n`, `nq`, `n_robot` | read in 6–12 blocks; the obvious shared context |
| the accumulator | `qp` | read in 9 blocks — every task block calls `qp.add_task(...)` |
| computed-once values | `A_com`, `b_com`, `dq_robot`, `_two_task` | 1–2 downstream readers each |
| solution outputs | `z_opt`, `qdd_t_opt`, `qdd_opt`, `lambda_opt`, `tau_q_opt`, `info` | tail of the function |

There is **no hidden long-range state**: nothing is mutated in one task block and re-read by
another. The task blocks only ever *append* to `qp`.

## Assessment

The 597 lines are not hiding complexity — they are hiding a pipeline. Two properties make
this genuinely safe to decompose, and both were measured rather than assumed:

1. **depth 3, no cross-block mutation** — extraction cannot change evaluation order or alias
   shared state;
2. **30/40 parameters are single-block** — helpers get short, meaningful signatures.

### Recommended shape (highest value / lowest risk first)

1. **Split `Build QP` along its four existing banners** into
   `_add_equality_constraints`, `_add_inequality_constraints`, `_set_variable_bounds`,
   and the small `_com_task_rows` helper. Removes **225 lines** from `solve()` — the single
   biggest win — and touches no task logic. `solve()` drops to ≈320 body lines.
2. **Optionally extract the SS two-task block** (55 lines) as `_add_ss_two_task_stack`, giving
   ≈270 body lines and making the canonical controller a named, findable unit.
3. **Leave the remaining task blocks inline.** They are 7–41 lines each, already labelled, and
   reading them in sequence is precisely how one wants to read a task stack.
4. **Do NOT restructure the 40-parameter signature** in the same pass. Grouping parameters
   into a state object is an API change that touches `sim_loop` call sites and would obscure
   the diff that proves the extraction was inert. Separate decision, separate commit.

Each step is verifiable by the gate as byte-identical, since extraction preserves both
evaluation order and values.

## Not recommended

Merging or reordering task blocks. The current order encodes the cost-assembly sequence and
the `priority=` labels; reordering would be a behavioural change dressed as a refactor.
