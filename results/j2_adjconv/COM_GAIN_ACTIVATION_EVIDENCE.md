# COM_GAIN_ACTIVATION_EVIDENCE — Phase 2 of the CoM Gain Semantics Audit

**Brief:** CoM Gain Semantics Audit and Controlled Fix (Idriss Chelikh)
**Phase:** 2 — is the defective law *active* and *load-bearing* in the canonical run.
**No behavioural code changed.** Instrumentation is record-only (proof in §1.2).
**Audit branch:** `claude/com-gain-semantics-audit-j0u6yr`, code state `eecbf94`
**Human stop gate reached — Phase 3 NOT started.**

---

## 1. Method

### 1.1 The run

The **exact** canonical managed (C) replay: `gate/replay_canonical.py`'s kwargs and
its `HierarchicalQP._solve_weighted` regularization pin (`EPS = 1e-6`), driving
`dca.main`, with `out_dir_override='com_gain_audit_scratch'` so **no committed
artifact is touched**. `MUJOCO_GL=disabled`.

Wrappers were placed around `WholeBodyQP._com_task_rows` and `WholeBodyQP.solve`.
Each wrapper calls the original, records, and returns the original's value
**unmodified**.

### 1.2 Proof the instrumentation is inert

The replay reproduced the frozen canonical headline numbers exactly:

| metric | this instrumented replay | frozen canonical (CLAUDE.md) | match |
|---|---|---|---|
| at-weld docks [mm] | 4.02 / 4.89 / **4.99** / 4.97 / 4.95 / 4.62 | 4.02 / 4.89 / **4.99** / 4.97 / 4.95 / 4.62 | ✔ **exact** |
| docks | 6/6 | 6/6 | ✔ |
| `h_w` peak norm [Nms] | 4.24 | 4.24 | ✔ |
| `h_w` violations | 0 / 2077 | 0 | ✔ |
| solver failures | 0 (`qp_ok` True on all 8458 solves) | `qp_fail = 0` | ✔ |

Six docks agreeing to the last reported digit, including the 4.99 mm worst case
whose margin to the 5 mm gate is 0.01 mm, is sufficient evidence that observation
did not perturb control.

### 1.3 What was recorded

Per `solve()` call: `settle_mode`, `ds_centroidal_active`, whether the CoM task was
added and at what weight, `‖A_com‖`, `‖b_com‖`, `e_r`, `e_v`, `a_com_ff`, the
rank-one feedback **actually applied**, the diagonal feedback that the intended law
**would** have produced (counterfactual — computed for comparison, never fed back),
`a_com_des`, the achieved `J_com q̈ + J̇_com q̇`, and the post-QP task residual.

---

## 2. §6.1 — Activation evidence

**The task is assembled, and it is assembled on the large majority of ticks.**

| quantity | value |
|---|---|
| `solve()` calls instrumented | **8458** |
| CoM task **ADDED** | **7090 (83.8 %)** |
|   … in SS, `qp.add_task(A_com, b_com, cfg.ss_alpha_mom, priority=2)` | **5080** at `α = 400.0` |
|   … in DS, `qp.add_task(A_com, b_com, cfg.ds_alpha_com, priority=1)` | **2010** at `α = 100.0` |
| CoM task not added | 1368 |
| `cfg.Kp_com` stored shape, **every one of the 8458 ticks** | **`(3, 3)`** — i.e. the rank-one path, always |
| weights observed when added | `{100.0, 400.0}` — exactly the two documented consumers |
| `ss_two_task_mode` / `ds_centroidal_mode` | `True` / `True` |
| `settle_mode` | `False` on all 5080 SS ticks, `True` on all 2010 DS ticks |
| `‖A_com‖` (SS) | ≈ 1.77 – 1.79 — well-conditioned, non-degenerate rows |
| QP reported success | **all 8458 ticks** |

So: **the rank-one law was live on every tick of the canonical run**, and its output
entered the QP as a weighted task on 83.8 % of them. This is not a dormant code
path.

---

## 3. The error the law acts on is ~1 mm — and that is structural

| `|e_r|` in SS (n = 5080) | value |
|---|---|
| median | **0.165 mm** |
| P75 | 0.353 mm |
| P95 | 0.800 mm |
| P99 | 1.044 mm |
| **max** | **1.147 mm** |

This is three orders of magnitude below the `e_com` peak of **0.154 m** in
CLAUDE.md's canonical table. **The two are different quantities and both are
correct.** `e_com` is step-level tracking against the coarse/planner reference;
`e_r` here is what `_com_task_rows` actually differences — the WQP's error against
the **NMPC plan**, which is re-solved from the *measured* CoM every 100 ms.

Mechanism, and the measurement that proves it:

- `sim_loop.py:2241-2242` — the NMPC is solved with `r_com=rs.r_com`, the
  **measured** CoM.
- `sim_loop.py:2289-2292` — `rp_k0 = x_plan[0:3, 0]`, the initial-state knot.
- `sim_loop.py:2342-2343` — `rp_interp = (1−α)·rp_k0 + α·rp_k1`, `α = qs/10`.
- At `qs = 0`, `α = 0` ⇒ `rp_interp = rp_k0 =` measured CoM ⇒ `e_r ≡ 0`.

**Measured: exactly 508 of 5080 SS ticks have `e_r` identically zero — 10.0 %,
i.e. precisely one in every ten QP sub-steps.** That is the re-anchoring, observed
rather than argued. The CoM feedback can therefore only ever see one NMPC cycle's
worth of drift, which is why it is bounded at ~1.1 mm.

**Consequence for the audit:** the defect's magnitude is capped by the cascade
architecture, not by the gain semantics. This does not make the semantics correct —
it bounds the blast radius.

---

## 4. §6.1 — Per-tick evidence

Requested ticks. The first pass's "partial cancellation" pick degenerated to a
zero-error tick (`|sum e|/|e| = 0/0`); redone with a `|e_r| ≥ P75` floor.

### 4.1 High-error SS tick (record 1009) — worst `|e_r|` in the run

```
settle_mode / ds_centroidal_active : False / False
CoM task added / weight            : True / 400.0
||A_com|| / ||b_com||              : 1.781626 / 0.025010
e_r [mm]        : [-0.2591, -1.1174, 0.0284]     |e_r| = 1.1474 mm
e_v [mm/s]      : [-1.4405,  1.0583, 0.7629]     |e_v| = 1.9435 mm/s
sum(e_r) [mm]   : -1.3482        <-- the ONLY thing the applied law sees
e_par (common)  : [-0.4494, -0.4494, -0.4494]    |.| = 0.7784 mm
e_perp (diff.)  : [ 0.1903, -0.6680,  0.4778]    |.| = 0.8430 mm  (73.5 % of |e_r|)
a_com_ff        : [-0.021090,  0.006466, 0.011567]
a_fb APPLIED    : [-0.002902, -0.002902, -0.002902]   <-- identical on all 3 axes
a_fb if DIAGONAL: [-0.005099, -0.000177,  0.002374]   <-- counterfactual
difference      : [ 0.002197, -0.002725, -0.005276]   |diff| = 0.006332 m/s2
a_com_des       : [-0.023992,  0.003564,  0.008665]
achieved        : [-0.004992, -0.005510,  0.002706]
task residual   : [ 0.019000, -0.009074, -0.005958]   |res| = 0.021883
```

The signature is visible directly: the applied feedback is `[-0.002902] × 3` —
**the same number on all three axes**, which is what `𝟙𝟙ᵀ` produces and what a
diagonal law can never produce. The intended law would have pushed **+z**
(`+0.002374`); the applied law pushes **−z** (`−0.002902`). The z-axis correction
has the **wrong sign**.

### 4.2 Strongest partial cancellation (record 4572) — the rank-one law blinded

```
e_r [mm]        : [0.5552, -0.5106, -0.0445]      |e_r| = 0.7556 mm
sum(e_r) [mm]   : +0.0001                          <-- cancels to ~zero
e_perp          : [0.5552, -0.5106, -0.0446]      |.| = 0.7556  (100.0 % of |e_r|)
e_par           : [0.0, 0.0, 0.0]                 |.| = 0.0001  mm
a_fb APPLIED    : [0.012510, 0.012510, 0.012510]
a_fb if DIAGONAL: [0.018665, -0.003130, -0.003024]
difference      :                                  |diff| = 0.022887 m/s2
```

A real 0.76 mm CoM error lying **100 % in the differential subspace**. Its
proportional channel is annihilated exactly as Phase 1 predicted: `sum(e_r) =
+0.0001 mm`. The `+0.01251` that *is* applied comes entirely from the `K_d` channel
(`sum(e_v) = +4.17 mm/s`) — and is again the same value on all three axes. The
intended law would have commanded `+x` and `−y`; the applied law commands `+x, +y,
+z` equally.

### 4.3 Low-error SS tick (record 4414)

```
e_r [mm]        : [0.0082, 0.0067, 0.0072]        |e_r| = 0.0128 mm
e_perp          : 8.8 % of |e_r|                   <-- nearly pure common mode
a_fb APPLIED    : [-0.002592, -0.002592, -0.002592]
a_fb if DIAGONAL: [-0.001658, -0.001164,  0.000231]
```

Even here the two laws disagree: the applied common-mode magnitude carries the `3k_p`
inflation, and the z-component again has the wrong sign.

### 4.4 Worst-case divergence in the run (record 259)

```
e_r [mm]        : [0.3932, -0.2454, 0.0026]       |e_r| = 0.4635 mm
e_v [mm/s]      : [6.6476,  4.1716, 0.0305]       sum(e_v) = +10.85 mm/s
a_fb APPLIED    : [0.033001, 0.033001, 0.033001]
a_fb if DIAGONAL: [0.021122, 0.011779, 0.000100]
difference      : [0.011878, 0.021222, 0.032901]  |diff| = 0.040914 m/s2
a_com_des       : [0.106500, 0.046532, 0.030940]
task residual   :                                  |res| = 0.128196
```

Here the injected error is **0.041 m/s²** — comparable to `|a_com_ff|` itself
(0.0735). But note the residual: **0.128 m/s², larger than `|a_com_des|`**. The QP
did not deliver this command either way.

---

## 5. §6.2 — Mode decomposition: how much of the error is invisible

With `e_∥ = ⅓𝟙𝟙ᵀe` and `e_⊥ = e − e_∥`, the rank-one law acts **only** on `e_∥`.

| population | mean `|e_⊥|/|e_r|` | median | P95 |
|---|---|---|---|
| SS (n = 5080) | **75.4 %** | 85.6 % | 99.9 % |
| DS (n = 2010) | **87.0 %** | 98.7 % | 99.9 % |
| all added (n = 7090) | **78.7 %** | 92.3 % | 99.9 % |

**On average 78.7 % of the CoM error vector is in the subspace this feedback law
cannot see.** The maximum is 100.0 % — errors that are entirely invisible do occur,
not just in principle (§4.2).

Correspondingly, the applied feedback is not a small perturbation of the intended
one; it is a different vector:

| `|a_fb,applied − a_fb,diagonal|` relative to `|a_fb,diagonal|` | median | P95 | max |
|---|---|---|---|
| all added ticks | **1.206** | 1.652 | 1.950 |

A median relative difference of **121 %** means the two laws typically disagree by
more than the whole magnitude of the intended correction.

---

## 6. §6.2 — Load-bearing status

### 6.1 Feedback authority `η_fb = |a_fb| / (|a_ff| + |a_fb| + ε)`

| population | median | P95 | max |
|---|---|---|---|
| SS | 0.0892 | 0.3273 | **0.9062** |
| DS | 0.0632 | 0.1797 | 0.7211 |
| all added | **0.0796** | 0.3000 | **0.9062** |

`|a_fb|` applied: median **0.0027**, P95 0.0197, **max 0.0879 m/s²**.
Non-negligible (> 0.01 m/s²) on **16.9 %** of ticks. `η_fb > 0.5` on 0.38 % of
ticks; `> 0.9` on 0.01 %. So the feedback is usually a ~8 % minority term against
the NMPC feedforward, occasionally dominant.

### 6.2 The absolute size of the defect

| `|a_fb,applied − a_fb,diagonal|` | median | P95 | max |
|---|---|---|---|
| absolute [m/s²] | **0.00309** | 0.01588 | **0.05915** |
| ÷ `|a_com_des|` | 0.1049 | 0.2893 | **1.0043** |
| ÷ `|a_com_ff|` | 0.1117 | 0.3343 | 9.3712 |

The defect perturbs the commanded CoM acceleration by ~10 % typically, and on the
worst tick by 100 % of the command.

### 6.3 …but the task itself is largely not served

This is the decisive measurement. Projecting the achieved acceleration onto the
commanded direction, `delivered = (achieved · â_des) / |a_des|`:

| population | `|a_des|` med | `|achieved|` med | delivered median | ticks < 50 % | ticks **< 0** (wrong sign) | `|res|/|a_des|` med |
|---|---|---|---|---|---|---|
| **SS**, `α_mom = 400` | 0.0570 | 0.0246 | **+0.407** | **66.4 %** | **11.8 %** | 0.679 |
| **DS**, `α_com = 100` | 0.00733 | 0.00018 | **+0.0009** | **100.0 %** | **48.8 %** | 0.999 |

In SS the QP delivers a median **41 %** of the CoM command, and on 11.8 % of ticks
the achieved acceleration points *against* it. In DS the task is served essentially
**not at all** — median delivered fraction **0.0009**, residual ≈ 100 % of the
command, and the sign is wrong half the time. `α_com = 100` against torso-angular
and posture leaves the CoM task nowhere.

**Scale comparison:** the CoM task residual the QP *already* leaves unserved is
median **0.0263 m/s²** — **8.5×** the median magnitude of the defect (0.00309). The
error introduced by the wrong gain semantics is small compared with the error the
weighted stack introduces by not honouring the task in the first place.

### 6.4 Classification (brief §6.2, four options)

- ~~not active in the canonical path~~ — refuted: 7090 additions, `(3,3)` on all 8458 ticks.
- ~~active but numerically negligible~~ — not supportable: max 0.0879 m/s², reaching
  `η_fb = 0.906` and 100 % of `|a_com_des|`.
- **✔ active and assembled, but effectively suppressed by the other weighted tasks
  and by the cascade architecture.** Doubly bounded: (a) the error it differences is
  ≤ 1.15 mm because the NMPC re-anchors from the measured state every 100 ms
  (§3, proven by the 508/5080 = 10.0 % exact zeros); (b) the task is outweighted in
  SS (torso-pose 2000, EE 1000 vs `α_mom` 400 → 41 % delivered) and effectively
  unserved in DS (0.09 % delivered).
- ~~active and load-bearing~~ — not supported: the defect is 8.5× smaller than the
  residual the stack already leaves on this task.

---

## 7. Phase 2 conclusion

**The canonical paper run executed the rank-one sum-and-broadcast law, on every tick,
as a weighted QP task — and that law is a genuine semantic defect that a corrected
diagonal law would change (median 121 % relative difference in the feedback vector,
sign inversions on individual axes, 78.7 % of the error unseen). But it is not
load-bearing:** the error it operates on is architecturally capped near 1 mm, and
the task is only ~41 % served in SS and ~0 % served in DS, leaving a residual 8.5×
the defect.

This is brief **§7.5 Outcome 4** territory — *"affected feedback is not
load-bearing: still harden the interface and use the classical diagonal law, but
document why headline results remain nearly unchanged."* The prediction that
follows is that a corrected diagonal law should perturb the headline metrics only
slightly. **That prediction is untested and must be measured, not assumed** — the
worst-case tick perturbs the command by 100 %, and the 4.99 mm dock has only
0.01 mm of margin against the 5 mm gate, so a small perturbation is not
automatically a safe one.

**Phase 3 is NOT started. Awaiting Idriss's explicit approval** (brief §3: "No
behavioural code change before the Phase-2 human checkpoint"; §6: "Do not implement
a fix before explicit approval").

---

## 8. Artifact declaration

| item | value |
|---|---|
| Code state audited | `eecbf94` on `claude/com-gain-semantics-audit-j0u6yr` (identical on this path to `main` and to canonical freeze `32aefaf` — §4 of `COM_GAIN_STATIC_TRACE.md`) |
| Command | `MUJOCO_GL=disabled PYTHONPATH=. python3 <scratchpad>/com_gain_activation.py` |
| Run artifact | `results/com_gain_audit_scratch/sim_log.json` — scratch, deleted after export, reproducible from the command above. **No committed artifact modified**; `canonical2p5_result.json` / `c25_fulldiag.csv` untouched. (Same convention as `results/gate_run_scratch/`, which is likewise untracked.) |
| **Committed evidence** | **`results/j2_adjconv/com_gain_audit_ticks.csv`** — 7090 rows × 31 cols (2.95 MB), one row per tick where the CoM task was added: `e_r`, `e_v`, common/differential split, applied vs counterfactual-diagonal feedback, `η_fb`, `a_com_des`, achieved, residual, delivered fraction |
| Probe record (raw) | `<scratchpad>/com_gain_activation.json` — all 8458 solve() calls |
| Analysis | `<scratchpad>/com_gain_analyse.py`, `com_gain_picks.py`, `com_gain_export.py` |
| Key numbers | docks 4.02/4.89/4.99/4.97/4.95/4.62 mm (canonical, exact); 7090/8458 task additions; `Kp_com` shape `(3,3)` on 8458/8458; `|e_r|` max 1.147 mm; 508/5080 SS ticks with `e_r ≡ 0`; `η_fb` med 0.0796 / max 0.9062; defect med 0.00309 / max 0.05915 m/s²; `|e_⊥|/|e_r|` mean 78.7 %; SS delivered 0.407, DS delivered 0.0009 |

The probe scripts are deliberately left in the scratchpad rather than committed:
they are throwaway instrumentation, and the durable form of their assertions is the
`tests/` additions specified in brief §7.3 — which belong to Phase 3.
