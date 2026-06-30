# INTERNAL — WHERE/WHY the torso orientation-reference chain breaks continuity (read-only diagnosis)

Branch `j2/ds-active-rework`. Diagnosis only — **no `crawlbot/` change, no fix, no merge.** Locates the rupture
behind the CASE-B `torso_ori_err_deg` sawtooth (the error jumps to ~0 at every DS→SS, in a chain that is supposed
to be continuous). Numbers from the canonical run `results/figA_canon` (config = commit `a24db03` runA:
`run_figdata.sh` COMMON + chatter fix); per-tick `q_torso` (realized R), `q_torso_ref` (the controller reference
R that `reference_at` returned), `e_torso_ori` (the logged metric). The sawtooth itself is corroborated by the
committed `results/j2_figdata/runA_traversal.csv` `torso_ori_err_deg` column. Reproducer:
`scripts/diag_ori_chain.py results/<run>`.

## Point 1 — the metric uses the SAME reference accessor in DS and SS (NOT a yardstick-mismatch artifact)
Both error sites compute `angle(reference_at(t).R, R_torso_realized)`:
- **DS** (`sim_loop.py:943-949`): `tref = self.torso_planner.reference_at(t_abs); R_torso_ref = tref.R; R_err =
  R_torso_ref.T @ rs.oMf_torso.rotation`.
- **SS** (`sim_loop.py:3469,3497-3498`): `tref_log = self.torso_planner.reference_at(t_log); R_err =
  tref_log.R.T @ rs_f.oMf_torso.rotation`.

Same accessor (`reference_at`), same formula — and it is the **same object the QP tracks** (`:2581`). So the jump
is **not** "the metric compares against a different reference kind in DS vs SS." The metric faithfully reports the
**controller's** reference. The reference *value* differs by phase only because the planner returns `_hold_R` in
DS vs the SLERP pose in SS — both re-seeded by `sim_loop`.

## Point 2 — the reference JUMPS at every DS→SS seam; the realized torso is CONTINUOUS (gap = in-DS drift)
Per DS→SS transition (last-DS tick → first-SS tick):

| step | t (s) | reference jump `∠(R_ref⁻,R_ref⁺)` | torso jump `∠(R_torso⁻,R_torso⁺)` | `e_ori` last-DS → first-SS |
|---|---|---|---|---|
| 0 | 0.11 | **5.157°** | 0.007° | 5.157 → 0.007 |
| 1 | 4.61 | **0.142°** | 0.002° | 0.142 → 0.002 |
| 2 | 13.52 | **0.194°** | 0.013° | 0.194 → 0.013 |
| 3 | 18.45 | **0.214°** | 0.002° | 0.214 → 0.002 |
| 4 | 27.64 | **0.156°** | 0.005° | 0.156 → 0.005 |

The **realized torso orientation is physically continuous** across every seam (≤0.013° between adjacent ticks —
integration noise). The **controller reference jumps** (0.14–0.21°; 5.16° once at step 0), and the metric jump
**equals** the reference jump. So the sawtooth is the *reference* resetting, not the torso moving.

The jump magnitude = the **in-DS drift** (the realized torso rotates off the *fixed* DS-hold during the inter-step DS):

| DS block | span | torso drift entry→exit | `e_ori` entry→exit |
|---|---|---|---|
| inter-step 0→1 | 1.28 s | 0.112° | 0.033 → 0.142 |
| inter-step 1→2 | 0.49 s | 0.188° | 0.017 → 0.194 |
| inter-step 2→3 | 1.31 s | 0.158° | 0.100 → 0.214 |
| inter-step 3→4 | 1.17 s | 0.152° | 0.016 → 0.156 |
| terminal | 19.90 s | 0.175° | 0.056 → 0.206 |

ref-jump ≈ in-DS drift (0.14–0.21° ≈ 0.11–0.19°). The DS hold is fixed at DS entry; the torso drifts off it; **SS
entry re-captures the drifted current**, jumping the reference by that drift and resetting `e_ori` to ~0.

## Point 3 — the dock-IK outputs are SMOOTH (no jump); and the SS SLERP does not even advance
`R_dock(k)` = controller reference R at SS-block end (= `R_t1`, the dock-IK target):

| step k | `R_dock(k)` rpy (deg) | step `∠(R_dock(k-1),R_dock(k))` |
|---|---|---|
| 0 | (−0.00, 0.00, **−5.16**) | — |
| 1 | (−0.03, 0.03, −5.29) | 0.142° |
| 2 | (−0.14, 0.17, −5.37) | 0.194° |
| 3 | (−0.28, 0.21, −5.53) | 0.214° |
| 4 | (−0.38, 0.32, −5.58) | 0.156° |

Smooth, monotone-ish (yaw drifts −5.16→−5.58° over the run; roll/pitch grow sub-degree). **No IK jump.** The
step-to-step `R_dock` change equals the seam ref-jump — because `R_dock(k+1)` is just the live torso re-captured
after the intervening DS.

**Structural finding the premise missed:** the **SS SLERP span is ≡ 0.0000°** for all five steps (`R_ref` at SS
start == `R_ref` at SS end). The dock-IK pin `R_torso_fixed=R_t0` (`:1478`) forces `R_t1 = R_t0` exactly, so the
SS orientation reference is a **constant hold at `R_t0(k)`, not an advancing dock-to-dock SLERP.** The assumed
"SS SLERP `R_dock(k)→R_dock(k+1)`" does not exist — there is no dock-to-dock interpolation to be continuous.

## LOCATED VERDICT — the rupture is the SS-entry RE-CAPTURE of live torso state
**WHERE:** the **DS-hold → SS-entry re-capture seam** — `R_t0 = rs_s.oMf_torso.rotation` at `sim_loop.py:1411`,
fed to the hold (`:1676`), the dock-IK target (`:1478`), and the swing end (`:1737`). The reference is **re-sampled
from live torso state at each SS entry** rather than chained from the previous phase's reference endpoint.

**WHY:** the references are **not chained through a shared stored value**; each phase independently re-samples the
live (drifted) orientation. Within each inter-step DS the controller holds the *fixed* DS reference only to ~0.11–
0.19° (imperfect tracking), and the next SS entry **accepts that drift** by re-capturing current — jumping the
reference by the drift and re-zeroing the error. A reference chained from the previous endpoint (a stored value)
would be continuous **by construction**, regardless of tracking drift. (Step 0's 5.16° jump is the one-time initial
capture of the mounting offset `R_flat`=yaw −5.16° from the pre-capture identity reference.)

**It is NOT:**
- a metric-yardstick artifact in the "different reference kinds DS vs SS" sense (Point 1: same accessor); though
  it IS reference-side, not torso-side — the physical torso glides continuously (≤0.013°), only the reference resets;
- a dock-IK discontinuity (Point 3: `R_dock(k)` smooth, steps 0.14–0.21°).

**Maps to the brief's middle branch:** "Point 2 gap nonzero from in-DS drift → the torso drifts during DS and the
SS recapture accepts it; the chain breaks at the DS-hold→SS-recapture seam." Confirmed, with the added structural
fact that the SS SLERP never advances (span ≡ 0), so the chain is a sequence of independent per-phase re-captures,
not a SLERP-connected dock-to-dock chain.

**Implication for the (deferred, gated) fix:** the rupture is the *re-sampling of live state*, so the cure is to
source the orientation reference from a **stored value** (a single global `R_flat`, or the previous phase's
endpoint) rather than re-capturing current. The seam jumps are sub-degree (0.14–0.21°), so the correction is
cheap. **No fix applied here — diagnosis only.**
