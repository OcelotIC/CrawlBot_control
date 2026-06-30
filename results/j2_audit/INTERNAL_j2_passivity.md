# INTERNAL — J2 passivity audit: CONFIRM/REFUTE the reviewing-Claude reading (`ae0673e`)

**Mandate:** the reviewing Claude read the passivity code on a pre-Fix-A clone (`5ada364`) and drew three
conclusions. Fix A should not have touched the passivity/QP code, so the reading should transfer to the
Fix-A canonical (`main = ae0673e`) — but that must be **verified**, not assumed. READ-ONLY archaeology +
a reproducer (`scripts/audit_j2_passivity.py`). No `crawlbot/` change, no `main` write, no PR, no
implementation. Branch `j2/ds-active-rework`.

**Bottom line: all three claims CONFIRMED on `ae0673e`; the premise (passivity/QP code unchanged by Fix A)
holds.** Reproducer: **20/20 checks pass**. Two nuances worth carrying into the J2 spec (DIVERGENCES §):
(a) the activation depends on `use_m2_stack`, which is `True` in the canonical only because
`_make_m7_config` sets it — **not** the `SimConfig` default; and (b) the "tracking under passivity" that
exists today is for a **held/static** reference (`set_hold`), so the **moving-reference** case — the heart
of J2 — is genuinely unproven, and Claim 3 is exactly the evidence it is hard.

---

## PREMISE — did Fix A touch the passivity/QP code? **NO (verified).**

| check | result |
|---|---|
| `wholebody_qp.py` diff `5ada364..ae0673e` | **byte-identical** (`git diff --quiet` → no diff) |
| `sim_loop.py` Fix-A diff | one hunk only, `@@ -2132,32 +2132,68 @@` (the inelastic-impact block); **0 passivity ± lines** |
| passivity anchors present at both commits | identical counts (`passivity_active = bool(` ×2, override ×1, SS-hold comment ×1, `_pass_override=True` ×1, `T_kin < T_settle` ×2) |

⇒ `wholebody_qp.py` (Claim 2's inequality, Claim 1's QP task-add) is **identical**; `sim_loop.py`'s
passivity logic is **unchanged in content**, only shifted ≈ +37 lines below the impact hunk. The reviewing
Claude's `5ada364` line numbers map cleanly: passivity_active `2735 → 2772`; the SS-hold comment sits
*above* the hunk and is essentially unmoved (`2052-2060 → 2054-2059`). **The `5ada364` reading transfers
to `ae0673e` verbatim.**

**Canonical config (built as the diag builds it — `_make_m7_config()` + diag overrides):**
`use_m2_stack=True`, `ds_centroidal_mode=True`, `alpha_passivity=1.0`.

---

## CLAIM 1 — passivity + centroidal tracking already coexist in terminal/DWELL DS → **CONFIRMED**

**The quoted expression matches** (`sim_loop.py:2771-2772`, wrapped across two lines vs the one-liner on
`5ada364` — same expression):
```python
passivity_active = bool(
    cfg.use_m2_stack and (phase == 'DS' or passivity_hold))
if passivity_override is not None:          # 2773-2774
    passivity_active = bool(passivity_override)
```
With the canonical `use_m2_stack=True`, the base clause gives `passivity_active=True` for `phase=='DS'` —
**the reviewing Claude's inference holds.**

**The `passivity_override` path (the reviewing Claude flagged to check) — exists, and the terminal/DWELL DS
uses it, ON:**
- **DWELL** (`sim_loop.py:1885-1893`): `_step(..., settle_mode=True, passivity_override=True,
  ds_centroidal_active=True)` — gated `_dwell_target > 1.0 and cfg.ds_centroidal_mode` (L1867).
- **Trailing/terminal DS** (`sim_loop.py:2292-2307`): `if cfg.ds_centroidal_mode: _pass_override = True`
  (L2292-2293) → `_step(..., passivity_override=_pass_override, ds_centroidal_active=cfg.ds_centroidal_mode)`.
  Comment L2285-2291: forced ON "regardless of the abort flag" because centroidal-DS uses the passivity
  inequality for dissipation.

In the canonical the override pins `True` **and** the base clause already gives `True` — they agree; the
override does not *change* the value here (it matters post-abort / if `use_m2_stack` were off). So passivity
is ON in terminal/DWELL DS by **both** routes.

**Both tasks land in the SAME QP solve** (traced through `_step → qp.solve` at `sim_loop.py:2785`):
- **centroidal-DS tracking** added at `wholebody_qp.py:1065`:
  `if settle_mode and cfg.ds_centroidal_mode and ds_centroidal_active:` → CoM-3D (`ds_alpha_com`, P1, L1068)
  + torso-angular-3D (`ds_alpha_torso_ori`, P1, L1090). (And L1053 *replaces* the joint-vel-damping settle
  cost with these — comment L1049-1052.)
- **passivity inequality** added at `wholebody_qp.py:549`: `if passivity_active and cfg.alpha_passivity > 0:`.

⇒ **CONFIRMED: tracking-under-passivity is the existing terminal/DWELL-DS architecture, not a new
addition.** **Caveat (→ DIVERGENCE 3):** the tracked reference is **held** (`set_hold`,
`sim_loop.py:1876-1879` / `2280-2283`) — a *static* docked setpoint. So the coexistence is demonstrated only
for a **static** reference; the **moving** reference is the open J2 case.

## CLAIM 2 — passivity is a QP inequality; the loop is only the iteration driver → **CONFIRMED**

**Inequality form matches verbatim** (`wholebody_qp.py:549-555`):
```python
if passivity_active and cfg.alpha_passivity > 0:
    H_jj = H_robot[6:, 6:]                       # joint block; base DOFs excluded
    T_kin = 0.5 * float(dq @ H_jj @ dq)
    A_pass = np.zeros((1, n))
    A_pass[0, idx['tau'][0]: idx['tau'][1]] = dq # row on the tau block = dq
    b_pass = np.array([-2.0 * cfg.alpha_passivity * T_kin])
    qp.add_inequality_constraint(A_pass, b_pass) # dq^T tau_q + 2α T_kin ≤ 0
```
i.e. `dqⱼᵀ τ_q + 2α T_kin ≤ 0`, `T_kin = ½ dqⱼᵀ H_jj dqⱼ`, `H_jj = H_robot[6:,6:]` (the joint block),
gated `passivity_active and alpha_passivity>0` — exactly the claim. Enforces `T(t) ≤ T(t₀)·e^{−2αt}`.
*(Minor: the joint-block restriction is unconditional in the code; "both EEs welded ⇒ only joint KE" is the
physical setting (DS) in which it's used, not a code gate — see comment L542-543.)*

**`_run_ds_passivity_loop` is the driver, not a separate law** (`sim_loop.py:601-793`): docstring (L626-644)
— "Run the M2 QP in settle_mode + passivity_active until T<T_settle … calls `self.qp_ss.solve(...)` directly
— NMPC is bypassed." Body: the `for k in range(max_steps)` loop calls `qp.solve(..., settle_mode=True,
passivity_active=True)` (L735-756), `mj_step`s, and exits when `T < T_settle` (`exit_reason='target_met'`,
L720-721). The dissipation **law** is the QP inequality; the loop just **iterates** it.

**"Used by the (legacy/non-centroidal) settle path" — CONFIRMED.** The loop's `qp.solve` call (L735-756)
does **not** pass `ds_centroidal_active` (defaults `False`), so it adds the **joint-vel-damping settle cost**
(`wholebody_qp.py:1053-1057`), **not** the centroidal tracking tasks. Per the docstring it is the shared
engine for **setup-stage-2 settle** (L569) and **inter-step DS settle** (L1819). The **centroidal**
terminal/DWELL DS does **not** use this loop — it runs `_step` (Claim 1). So the two DS regimes are distinct,
as the Bloc-1 cartography found.

## CLAIM 3 — the positive-work conflict is ALREADY observed → **CONFIRMED (verbatim, + a sharper rationale)**

`sim_loop.py:2047-2071`, the SS convergence-hold window:
```python
# … Because both terminal references are static with zero velocity, PD feedback is
# self-decelerating — no passivity constraint is needed. We ran a first pass with
# passivity_hold=True and it prevented the arm from doing the positive work required
# to close the last few mm of position error. Normal SS tracking (passivity_active=False)
# is the correct regime for the hold window.
if not docked:
    while t < t_hold_deadline and not docked:
        hw, L_com_prev = self._step(..., passivity_hold=False)   # 2065-2070 → passivity OFF
```
**Interpretation CONFIRMED:** the inequality `dqⱼᵀ τ_q ≤ −2α T_kin ≤ 0` forbids **net positive joint work**
whenever `T_kin>0`, so it blocks following a reference that requires energy injection — here, the last-mm
position closure. They observed it, and disabled passivity for SS-hold.

**Sharper rationale in the comment (matters for J2):** disabling passivity in SS-hold is *safe specifically
because the reference is static* (zero-velocity ⇒ PD is self-decelerating, so dissipation isn't needed). A
**moving** DS reference has **neither** property — it is **not** self-decelerating **and** passivity would
block the work it needs. So you cannot reuse the SS-hold "just turn passivity off" escape for a moving DS
reference. **This is the load-bearing evidence that the J2 moving-reference reconciliation is a real
conflict, not a formality.**

---

## DEFERRED (NOT run here — flagged as the J2 spec's central design+feasibility question)

Whether a QP with `passivity_active=True` stays **feasible** while tracking a **moving** centroidal DS
reference — and what relaxation of `α` or the constraint form (e.g. budgeted/one-sided passivity, a
work-credit term, scheduling α along the reference) is needed — requires **running** the QP with a moving
reference. That is an implementation experiment, **out of scope** for this read-only audit. Recorded as the
**central open question** for the J2 design: *passivity vs moving-reference feasibility.* The three claims
above bound it: the machinery to add both (tracking + passivity) to one QP exists (Claim 1); the constraint
that creates the conflict is precisely characterized (Claim 2); and the conflict is already empirically
observed for the analogous SS case (Claim 3).

---

## DIVERGENCES (where the `5ada364` reading needs a footnote on `ae0673e`)

1. **`use_m2_stack` provenance — the inference is config-dependent.** The quoted base expression fires in
   DS **only if `cfg.use_m2_stack` is True**. That is **True in the canonical** — but set by
   **`_make_m7_config()`** (`scripts/run_m7_single_step.py:40`), **not** by any `--ss-*` diag flag, and the
   bare `SimConfig` **default is `False`** (`config.py:129`). So the reviewing Claude's "passivity_active=True
   whenever phase=='DS'" is correct **for the canonical run**, but would be **false** if read against a
   default `SimConfig`. *J2 spec should cite `_make_m7_config`, not the dataclass default.*

2. **The `passivity_override` path was not in the quoted snippet.** In the canonical terminal/DWELL DS,
   `passivity_override=True` (DWELL hardcoded L1892; trailing-DS `_pass_override=True` under
   `ds_centroidal_mode`, L2293) is what *pins* `passivity_active=True`, via `2773-2774` — independent of
   `use_m2_stack`/phase. The base clause agrees in the canonical (no behavioural divergence), but the
   **switch is the override**, whose stated purpose (L2285-2291) is to keep passivity ON **even post-abort**.
   *Don't model `phase=='DS'` as the sole activation.*

3. **"Tracking under passivity" today = a HELD (static) reference.** Both terminal and DWELL DS call
   `set_hold(...)` before the settle loop, so the coexistence Claim 1 confirms is demonstrated for a
   **static** docked target only. The **moving** reference (locomotion across n=1↔n=2) is **not** exercised
   anywhere — it is exactly the deferred question, and Claim 3 shows the static-case escape hatch (disable
   passivity) does not transfer.

4. **Line drift, content identical.** `passivity_active` 2735→2772 (+37, below the Fix-A impact hunk at
   2132); SS-hold comment 2052-2060→2054-2059 (above the hunk, ~unmoved). Premise proves the **content** is
   byte-identical — this is locating, not a divergence.

5. **`5ada364` vs `ae0673e`:** `wholebody_qp.py` byte-identical; `sim_loop.py` changed **only** in the
   inelastic-impact block (0 passivity lines). **No divergence in the passivity reading between the two
   commits.**

---

## Reproducer

`scripts/audit_j2_passivity.py` — READ-ONLY. Builds the canonical config via `_make_m7_config()` + the diag
overrides; locates every anchor **by text** (line numbers track the live tree); shells out to `git` for the
premise diff `5ada364..ae0673e`. Asserts all 20 load-bearing facts (premise + config + the three claims).

```
MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/audit_j2_passivity.py
→ VERDICT: 20/20 checks confirmed.  (exit 0)
```

**STOP — doc-first.** Awaiting the digest before any design or implementation. No `crawlbot/` change, no
`main` write, no PR.
