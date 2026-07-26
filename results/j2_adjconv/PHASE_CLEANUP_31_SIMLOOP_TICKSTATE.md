# PHASE CLEANUP-31 — `_step`'s logging tail, and the record it needed

`sim_loop.py` is the largest module and `_step` was 1014 lines of it — 30 % of
the file in one method, 136 top-level statements, 178 distinct locals, nesting
depth 9, and **two** section comments across the whole thing.

This pass lifts the logging tail into `_log_ss_tick`, the single-support
counterpart of the long-standing `_log_ds_tick`. `_step` drops to **851 lines**.

The interesting part is not the extraction. It is that the obvious version of it
was wrong, and measuring said so before any code moved.

---

## 1. Choosing the cut by measurement

A 1014-line method with 178 locals cannot be split by eye: the cost of a cut is
the number of variables live across it, and that number becomes the helper's
signature. `WholeBodyQP.solve()` shows where guessing leads — 40 parameters, 30
read in exactly one block, deferred as `CLEANUP_CARRYOVER` §A1.

So: for every statement boundary in `_step`, count the locals assigned before it
and read after it.

```
:2333    0
:2397    9  #########
:2437   17  #################
:2507   25  #########################   <- NMPC + QP + integration core
:3181   27  ###########################
:3247   14  ##############
:3296    8  ########
:3318    3  ###
```

Three regions, and the shape names them: a rise into the solver core, a plateau
at ~25 through it, and a **monotone decay to 3** over the last ~190 lines. That
decay is the signature of a block that only *records*.

Confirmed structurally before touching it — the tail performs **zero** attribute
stores. Everything is `.copy()`, a read-only query, or `log.*.append`. Two calls
do mutate (`mujoco.mj_forward`, `robot.update`), both recomputing derived state
from unchanged `qpos`/`qvel`, so the block was moved **verbatim and in order**.

---

## 2. The naive extraction was a bad trade, and the count said so

Live-in set by the obvious test — assigned in the head, read in the tail — was
21 locals plus 13 of `_step`'s own arguments: a **29-parameter** helper. That is
the §A1 debt rebuilt somewhere else, and it would have made the module worse
while looking like progress.

Two corrections shrank it:

**A name the tail re-assigns before reading is not an input.** `L_dot_est` and
`R_err` look like inputs — the head assigns both — but the tail recomputes them
from `rs_f` at `:3176` and `:3187`. The naive test counts any name assigned in
the head and read in the tail; the correct test is whether a read can be reached
**before** a write. Five names dropped out. Same class of error as every other
one this chantier has logged: the instrument was narrower than the question.

**Two more are not tick state at all.** `cfg` is only ever `self.cfg`, and `log`
is the destination rather than something the tick produced.

What remained crosses as one record.

---

## 3. `TickState` — 27 fields, grouped by provenance

Flat, per Idriss's call, but organised by where each value comes from:

| group | fields | |
|---|---:|---|
| step context | 12 | `t`, `phase`, `step_idx`, `ss_end`, `settle_mode`, `swing_arm`, `stance_arm`, `stance_a`, `stance_b`, `target_anchor`, `hw`, `L_com_prev` |
| stage 1 — centroidal NMPC | 8 | `nmpc_ok`, `nmpc_status_code`, `nmpc_cost_val`, `t_nmpc_ms`, `nmpc_info`, `lambda_ref`, `v_com_ref`, `r_com_ref` |
| stage 2 — whole-body QP | 6 | `qp_ok`, `t_qp_ms`, `lambda_qp`, `tau_joints`, `tau_wheels`, `transport_term_mag` |
| torso reference tracked | 1 | `p_torso_ref_used` |

**Names come from where each value is logged, not from the head's
abbreviations.** `lr` is the NMPC contact-wrench reference (`log.lambda_ref`), so
it is `lambda_ref`. `vp` is the planned CoM velocity (`log.v_com_ref`), so it is
`v_com_ref`. `cref_r` is `r_com_ref`. Reading `_log_ss_tick` no longer requires
holding a glossary of two-letter names in your head.

`t_qp_start` became `t_qp_ms`: the boundary moved one line up so the record
carries a *result* rather than a stopwatch reading the logger has to finish.

### One behaviour made explicit

`p_torso_ref_used` is bound only when the QP sub-loop ran. The old code
discovered this by catching **`NameError`** inside the logging block. As a field
defaulting to `None` it is an ordinary `is None` test — identical behaviour, said
out loud, and no longer dependent on Python scoping rules to express control
flow.

---

## 4. Verification

Logging-only, so the gate settles it — this is the same argument the torso-export
fix (`b619ef4`) made, and the same proof:

```
gate/run_gate.py
  [1] canonical replay + export   rc=0 (142.3 s)
  [2] artifact identity           PASS — 2077 rows × 132 928 fields
  VERDICT: PASS

gate/dock_check.py
  at-weld docks 6/6 — 4.02 / 4.89 / 4.99 / 4.97 / 4.95 / 4.62 mm
  every step delta +0.0000     worst margin 0.01 mm
  θ_s 0.540   h_w 4.102 / 4.243   e_com 0.154   qp_fail 0
  CANONICAL RESULTS: MATCH frozen 2.5

gate/run_suite.py --fast   PASS   199 tests,  27 s
gate/run_suite.py          PASS   200 tests,  92 s
sync_docs --check / verify_docs / verify_roots / link_audit   all OK
```

**`verify_params` bit on this pass** — and on my own change. Inserting the
dataclass shifted `sim_loop.py`, so CLAUDE.md's `α torque-min` and `α accel-reg`
rows still cited `:1126` when the literals had moved to `:1197`. Caught
mechanically, before the commit, which is the entire reason that checker exists.

Also corrected: the routine's suite timings still said `~65 s` / `~11 min`,
figures from before CLEANUP-30 retired 644 s of tests. Now `~25 s` / `~90 s`.

---

## 5. Result, and where the next cut goes

```
_step          1014 -> 851 lines
_log_ss_tick        203 lines   (vs _log_ds_tick's 210 — the asymmetry was drift)
TickState            68 lines   (record + its rationale)
```

The DS path had its logging extracted long ago; the SS path had the identical
thing inlined. That was never a design decision.

**Remaining, and honestly priced:**

- **The 667-line core** — coupling plateau ~25, so it needs the same treatment:
  a state object, not a parameter list. `TickState` is the pattern; the fields
  differ.
- **`run()`, 600 lines in 28 top-level statements** — its problem is nesting
  depth, not sequence, so there is no cheap top-level seam at all. Extraction has
  to come from inside the loop body, which is a different and harder job.
- **`solve()`'s 40 parameters** (§A1) — the same record pattern applies, and
  doing it there would retire the debt this pass just avoided recreating.
