# IK formulation — trajectory-aware manipulability IK for VISPA crawling

**Status:** Working specification, draft 1. Captures the
mathematical structure of the three IK variants currently in the
codebase, the pathologies identified in the Phase-3 / Phase-4 /
diagnostic chain, and the corrected formulation to be implemented.

This document is foundational reference for both:
- The next round of IK code changes (the path-dependence and
  multi-start fixes).
- The IK contribution chapter of the IFToMM submission (deferred).

The content here is mathematical, not implementation-level; for
file-line pointers see `T15_post5_pipeline_audit.md` §3.7 and
§5, and `IK_ANOMALY_REPORT.md`.

---

## §1 — System and configuration space

The robot is a free-floating dual-arm crawler: a torso body
treated as a free-flyer, with two 7-DOF serial arms attached.

Configuration coordinates:

$$
q \in \mathcal{Q} = \mathrm{SE}(3) \times \mathbb{R}^{14},
\qquad
q = (q_b, q_{J,a}, q_{J,b})
$$

with $q_b = (p_t, R_t) \in \mathrm{SE}(3)$ the torso pose (position
plus orientation; encoded in the implementation as 3 + 4
components: position plus unit quaternion, $n_q = 7$ for the base
block) and $q_{J,a}, q_{J,b} \in \mathbb{R}^7$ the joint angles of
arms A and B respectively. Total $n_q = 7 + 14 = 21$.

The reduced velocity space:

$$
\dot q \in \mathbb{R}^{n_v} = \mathbb{R}^{20},
\qquad
\dot q = (\dot q_b, \dot q_{J,a}, \dot q_{J,b})
$$

with $\dot q_b \in \mathbb{R}^6$ the torso twist (linear plus angular
velocity in the body frame). The map between $\dot q$ and the
time-derivative of $q$ is the SE(3) right-trivialization for the
base block, and the identity for the joint blocks.

Two end-effector frames $\mathcal{F}_a$ and $\mathcal{F}_b$ are
rigidly attached to the tip of each arm. Their poses are functions
of the configuration:

$$
T_a(q) = T_{a,0}(q_b, q_{J,a}) \in \mathrm{SE}(3),
\qquad
T_b(q) = T_{b,0}(q_b, q_{J,b}) \in \mathrm{SE}(3)
$$

Note that $T_a$ depends only on $(q_b, q_{J,a})$ — arm B's joints do
not move tool A. Symmetric for $T_b$.

---

## §2 — Tool-frame Jacobian decomposition

The spatial velocity at tool A, expressed in $\mathcal{F}_a$'s LOCAL
frame, is:

$$
{}^a v_a = J_a(q) \, \dot q,
\qquad
J_a(q) \in \mathbb{R}^{6 \times n_v}
$$

By the kinematic structure of §1, $J_a$ has zero columns for arm
B's joints. Splitting columns by velocity-coordinate block:

$$
J_a(q) = \begin{pmatrix} J_a^{(b)}(q) & J_a^{(J,a)}(q) & 0_{6 \times 7} \end{pmatrix}
$$

where:

- $J_a^{(b)} \in \mathbb{R}^{6 \times 6}$ is the *base-induced* Jacobian:
  the linear map from torso twist $\dot q_b$ to tool-A velocity, with
  arms held fixed.
- $J_a^{(J,a)} \in \mathbb{R}^{6 \times 7}$ is the *arm-internal*
  Jacobian: the linear map from arm-A joint rates $\dot q_{J,a}$
  to tool-A velocity, with the torso held fixed.

Symmetric decomposition for arm B:

$$
J_b(q) = \begin{pmatrix} J_b^{(b)}(q) & 0_{6 \times 7} & J_b^{(J,b)}(q) \end{pmatrix}
$$

The IK formulations below use the **arm-internal** blocks
exclusively. We define:

$$
\hat J_a(q) := J_a^{(J,a)}(q) \in \mathbb{R}^{6 \times 7},
\qquad
\hat J_b(q) := J_b^{(J,b)}(q) \in \mathbb{R}^{6 \times 7}
$$

**Justification for using $\hat J_*$ rather than the full $J_*$.**
During single support (SS), the stance arm is anchored: the contact
constraint fixes its tool pose, which couples torso motion to
stance-arm joint motion. The QP's swing-arm EE tracking task uses
the swing arm's arm-internal Jacobian to compute joint
accelerations from a desired tool acceleration. Even when the base
can move, the rank-deficiency that triggers tracking degradation is
in $\hat J_b$, not in $J_b$. Therefore $\sigma_{\min}(\hat J_b)$ is
the directly relevant quantity for predicting QP tracking failure
during swing.

A weighting that distinguished stance from swing arm would be more
faithful to the failure mode (since stance manipulability matters
less), but the symmetric product $\sigma_{\min}(\hat J_a)
\cdot \sigma_{\min}(\hat J_b)$ is the simplest baseline and is what
the current codebase implements. This choice is open for revision
(see §10).

---

## §3 — Anchor constraints

At each step, the IK is given two target tool poses:

$$
T_a^* \in \mathrm{SE}(3), \qquad T_b^* \in \mathrm{SE}(3)
$$

These are the docked anchor poses the swing-end-of-step
configuration must satisfy. The hard constraints common to all
three IK variants are:

$$
T_a(q) = T_a^*, \qquad T_b(q) = T_b^*
$$

In residual form using the SE(3) logarithm:

$$
\xi_a(q) := \log(T_a^{*-1} T_a(q)) \in \mathbb{R}^6,
\qquad
\xi_b(q) := \log(T_b^{*-1} T_b(q)) \in \mathbb{R}^6
$$

with $\xi_a(q) = 0$, $\xi_b(q) = 0$ as the constraints. Twelve
scalar equations on the 20-D velocity space leave an **8-D
feasible set** (assuming generic regularity), parameterized as
solutions of $\xi(q) = 0$.

The redundancy that the IK variants exploit differently is
contained in this 8-D set.

---

## §4 — Manipulability metrics

Two scalar metrics on the arm-internal Jacobians are used in the
codebase. They are mathematically distinct and not interchangeable.

### §4.1 Yoshikawa volumetric manipulability

For a single arm:

$$
w_{\text{Y}}^{(\alpha)}(q) := \sqrt{\det\bigl(\hat J_\alpha \hat J_\alpha^\top\bigr)}
= \prod_{i=1}^{6} \sigma_i(\hat J_\alpha),
\qquad
\alpha \in \{a, b\}
$$

This is the volume of the velocity ellipsoid in the 6-D
task-velocity space. It penalizes any near-zero singular value
multiplicatively, so it goes to zero as the manipulator approaches
*any* singularity.

For the bimanual system:

$$
w_{\text{Y}}^{\text{bi}}(q) := w_{\text{Y}}^{(a)}(q) \cdot w_{\text{Y}}^{(b)}(q)
= \prod_{i=1}^{6} \sigma_i(\hat J_a) \cdot \prod_{i=1}^{6} \sigma_i(\hat J_b)
$$

### §4.2 Worst-direction manipulability

For a single arm:

$$
w_\sigma^{(\alpha)}(q) := \sigma_{\min}(\hat J_\alpha),
\qquad
\alpha \in \{a, b\}
$$

This measures the worst-case linear-velocity gain across all task
directions: it answers "how slowly can this arm move in the most
constrained direction." Equal to the radius of the largest
inscribed sphere in the velocity ellipsoid.

For the bimanual system:

$$
w_\sigma^{\text{bi}}(q) := \sigma_{\min}(\hat J_a) \cdot \sigma_{\min}(\hat J_b)
$$

### §4.3 Relationship and comparison

For a single arm, the relationship between the two metrics is:

$$
w_{\text{Y}}^{(\alpha)} = \sigma_{\min}(\hat J_\alpha) \cdot \kappa_\alpha,
\qquad
\kappa_\alpha := \prod_{\substack{i=1 \\ \sigma_i \neq \sigma_{\min}}}^{6} \sigma_i
$$

so:

$$
\frac{w_{\text{Y}}^{\text{bi}}}{w_\sigma^{\text{bi}}} = \kappa_a \kappa_b
$$

This ratio is **configuration-dependent**: it depends on the full
spectrum of singular values, not just on $\sigma_{\min}$. At
"generic" non-degenerate configurations $\kappa_a \kappa_b \sim
\mathcal{O}(1)$ to $\mathcal{O}(10)$. Empirically (per the
diagnostic, §4.3), the ratio at fixed-rotation IK's converged
$q_{\text{end}}$ at T15 step 2 is $\approx 3.86$.

**Implication for cross-IK comparison.** The fixed-rotation IK
reports $w_{\text{Y}}^{\text{bi}}$. The endpoint and trajectory-aware
IKs report $w_\sigma^{\text{bi}}$. Comparing values from one IK
against the other is valid only after explicit conversion. The
Phase-4 §3.1 framing of "$1.55 \times 10^{-2}$ vs $4.09 \times 10^{-8}$
= 6 orders of magnitude" mixed the two metrics and therefore
overstated the difference.

**Recommended convention.** Standardize all IKs on $w_\sigma^{\text{bi}}$
as the primary diagnostic, since it more directly predicts
worst-case tracking failure during swing. Yoshikawa can be reported
as a secondary diagnostic but should not be the basis for
comparison or for binary gating.

---

## §5 — IK 1: `dock_configuration_fixed_rotation`

### §5.1 Formal problem

Adds the orientation hard constraint $R_t = R_t^*$ where
$R_t^* = R_t(q_{\text{start}})$ is the live torso orientation at SS
entry.

Decision variables: $p_t \in \mathbb{R}^3$ and $q_J \in
\mathbb{R}^{14}$, total 17.

Hard constraints:

$$
\xi_a(q) = 0, \qquad \xi_b(q) = 0, \qquad R_t = R_t^*
$$

The orientation constraint removes 3 of the 6 base-twist DOFs, so
the effective velocity space is 17-D and the feasible set has
dimension $17 - 12 = 5$.

The IK is a **feasibility problem**, not an optimization:

$$
\text{find } q = (p_t, R_t^*, q_J)
\quad \text{s.t.} \quad
\xi_a(q) = 0,\; \xi_b(q) = 0
$$

Solved by Gauss-Newton iteration from a seed $q_{\text{init}}$. The
output depends on the seed: different seeds reach different points
in the 5-D redundant null-space.

### §5.2 Diagnostic output

After convergence, IK 1 reports:

$$
w_{\text{fixed}} := w_{\text{Y}}^{\text{bi}}(q_{\text{end}})
$$

Used in `_setup_torso_for_step` as a binary gate: if $w_{\text{fixed}}
\geq 10^{-4}$, the fixed-rotation solution is accepted; otherwise
the IK falls back to one of the manipulability variants.

### §5.3 Properties

- **Solution non-unique within the 5-D redundant null-space.**
  Different seeds give different $q_{\text{end}}$.
- **No optimization over manipulability.** $w_{\text{fixed}}$ is
  observed, not maximized.
- **Cheap.** Single Gauss-Newton solve; ~$10$–$50$ ms.

---

## §6 — IK 2: `manipulability_config` (endpoint-only)

### §6.1 Formal problem

Drops the fixed-rotation constraint. Outer optimization over torso
xyz position only:

$$
p_t^* = \arg\max_{p_t \in \mathbb{R}^3} \, w_\sigma^{\text{bi}}\bigl(q^*(p_t)\bigr)
$$

where:

$$
q^*(p_t) := \Phi_{\text{ik}}\bigl(p_t,\; R_t^{(0)},\; q_J^{(0)}\bigr)
$$

is the inner Gauss-Newton solver applied at torso position $p_t$ with
seeds $(R_t^{(0)}, q_J^{(0)})$. The orientation $R_t$ is determined
implicitly by what the inner solve converges to from $R_t^{(0)}$.

Outer optimizer: Nelder-Mead with $M = 3$ multi-starts at:

$$
p_t^{(m,0)} = p_{\text{midpoint}} + (0,0,dz_m), \qquad dz_m \in \{0, -0.3, -0.6\}\;\text{m}
$$

where $p_{\text{midpoint}}$ is the midpoint of the two anchor positions.

### §6.2 Reported manipulability

$$
w_{\text{end}} := w_\sigma^{\text{bi}}\bigl(q^*(p_t^*)\bigr)
$$

### §6.3 Properties

- **Decision variable: torso xyz only.** Torso orientation is
  not directly optimized; it inherits whatever the inner solve
  converges to from $R_t^{(0)}$.
- **Inner seed variability.** $\Phi_{\text{ik}}$ is sensitive to its
  seed $(R_t^{(0)}, q_J^{(0)})$. Different seeds may land in
  different IK branches (elbow up/down, wrist orientation), giving
  different $q^*$ for the same $p_t$. **The cost is not a pure
  function of $p_t$ unless the seed is held fixed across cost
  evaluations.**
- **Multi-start coverage.** Three seeds on a single vertical axis at
  the anchor midpoint. Lateral basins (off the $dz$ line) not
  explored.

---

## §7 — IK 3: `manipulability_config_trajectory` (Candidate 1)

### §7.1 Formal problem

Extends IK 2 by sampling the *interior* of the planned SS trajectory.

Given $q_{\text{start}}$ at SS entry, define a temporal interpolation
from $q_{\text{start}}$ to a candidate $q_{\text{end}}$:

$$
q(\tau) = \mathrm{interp}\bigl(q_{\text{start}}, q_{\text{end}}; s(\tau)\bigr),
\qquad
s(\tau) = 10\tau^3 - 15\tau^4 + 6\tau^5,\quad \tau \in [0, 1]
$$

Here $s(\tau)$ is the standard quintic temporal scaling
(zero velocity and acceleration at endpoints), and
$\mathrm{interp}$ is Pinocchio's `interpolate` operator: SE(3)
geodesic on the floating-base block, scalar-linear on the joints.

Sample at $K = 5$ interior points $\tau_k \in
\{0.2, 0.4, 0.6, 0.8, 1.0\}$, and define the cost:

$$
\text{cost}(p_t) = - \min_{k = 1, \ldots, K} w_\sigma^{\text{bi}}\bigl(q(\tau_k)\bigr)
$$

Outer optimization:

$$
p_t^* = \arg\max_{p_t \in \mathbb{R}^3} \, \min_k w_\sigma^{\text{bi}}\bigl(q(\tau_k)\bigr)
$$

via Nelder-Mead with the same 3 seeds as IK 2. The candidate
$q_{\text{end}}$ at each cost evaluation is the inner Gauss-Newton
output at the candidate $p_t$.

### §7.2 Reported quantities

$$
w_{\text{worst}} := \min_k w_\sigma^{\text{bi}}\bigl(q(\tau_k^*)\bigr) = - \text{cost}(p_t^*)
$$

$$
w_{\text{end}} := w_\sigma^{\text{bi}}\bigl(q(\tau = 1)\bigr) = w_\sigma^{\text{bi}}(q_{\text{end}})
$$

### §7.3 Interpretation of the "trajectory" being optimized

The interpolation $q(\tau)$ is a **fictional** trajectory: it is
not the actual trajectory the controller executes during SS. The
actual SS trajectory is determined by:

- TorsoPlanner's quintic in 6D pose space (linear in
  $\mathbb{R}^3$, SLERP in $\mathrm{SO}(3)$, applied independently to
  each pose component).
- The QP's reactive resolution of swing-arm joint velocities given
  the EE task's reference and the contact constraints on the stance
  arm.

These two together do not produce the joint-space quintic
$q(\tau)$ that IK 3 samples. The IK's interpolation is therefore
a **proxy**: if the arms moved smoothly from $q_{\text{start}}$
to $q_{\text{end}}$ in joint space, the worst-case manipulability
along that smooth path would be $w_{\text{worst}}$.

This is a reasonable proxy for what the controller will actually
experience, but it is not a guarantee. A more faithful formulation
would sample the actual TorsoPlanner + SwingPlanner reference
trajectory, evaluate the QP's expected joint-space response (via a
linearized model or rollout), and compute manipulability along
that. This is more invasive and is not the proposed v1 fix.

---

## §8 — Pathologies of the current IK 3 implementation

The Phase-3, Phase-4, and diagnostic chain identified two
mechanism-level failures of IK 3 that are independent of the
mathematical formulation but compromise its closed-loop behavior.

### §8.1 Pathology (B): insufficient multi-start coverage

The Nelder-Mead optimizer is local. With three seeds confined to
a single vertical axis at the anchor midpoint, only basins
intersecting that axis are explored.

Diagnostic evidence (§3.2 of `IK_ANOMALY_REPORT.md`): a brute-force
grid search on $\pm 0.5$ m around the anchor midpoint at $0.1$ m
resolution found a global maximum of $w_{\text{worst}} \approx 4.92
\times 10^{-2}$ at torso xyz $[1.0, -0.3, -1.275]$ — 0.9 m from any
of the three seeds, in a lateral direction the seeds do not span.
None of the multi-starts reach this basin; the IK converges to a
suboptimum at $w_{\text{worst}} \approx 4.83 \times 10^{-2}$.

Mitigation (proposed §9): broaden the seed set to include lateral
perturbations and physically motivated alternatives (live torso
position, fixed-rotation IK output).

### §8.2 Pathology (C): path-dependent inner solve

The inner solver $\Phi_{\text{ik}}$ takes a seed configuration
$(R_t^{(0)}, q_J^{(0)})$. The current implementation seeds it from
an internal cache `_cache['q_prev']` populated by the most recent
cost call. Therefore:

$$
q^*(p_t) = \Phi_{\text{ik}}\bigl(p_t,\; R_t^{(0)}\bigl[\pi\bigr],\; q_J^{(0)}\bigl[\pi\bigr]\bigr)
$$

depends on the Nelder-Mead simplex path $\pi$ that led to the
current $p_t$. Different paths through the same $p_t$ produce
different seeds, which can cause the Gauss-Newton inner solve to
land in different IK branches. Different branches can have
manipulability values differing by many orders of magnitude.

Diagnostic evidence (§3.1 of `IK_ANOMALY_REPORT.md`): at the same
torso xyz $[1.354, -0.189, -2.549]$, two different cost evaluations
yielded:

- $w_\sigma^{\text{bi}} = 4.83 \times 10^{-2}$ with the warm-start cache
  populated by nearby simplex steps.
- $w_\sigma^{\text{bi}} = 3.51 \times 10^{-9}$ with a cold seed
  ($q_J^{(0)} = $ neutral, only $q[:3] = p_t$ set).

Same xyz, same anchor pair, same arm-joint task — but seven orders
of magnitude difference in cost value, depending on the inner-solve
seed.

This makes the optimization problem **ill-posed**: Nelder-Mead is
operating on a function that is not deterministic in its decision
variable. Convergence properties are not guaranteed; small
numerical reorderings can route the simplex through different paths
and produce different optima.

### §8.3 Combined effect

Pathologies (B) and (C) are not independent. (C) creates spurious
"basins" at $p_t$ where the inner-solve seed happens to drive
Gauss-Newton into a singular branch. Nelder-Mead can be driven into
these spurious basins, mistaking them for genuine local minima of
the cost. (B) ensures that the multi-start does not have enough
coverage to escape these spurious basins.

The Phase-4 closed-loop result of $w_{\text{end}} = 4.09 \times
10^{-8}$ at step 2 is consistent with the IK landing in such a
spurious basin under the specific Nelder-Mead path that the live
sim_loop run executed. The diagnostic was unable to reproduce this
exact value from a fresh standalone call (best reproduction
$\approx 4.83 \times 10^{-2}$), confirming that the pathology is
sensitive to numerical path details that sim_loop's call context
exercises but a clean fixture does not.

---

## §9 — Proposed corrected formulation

To make IK 3 well-posed, two changes:

### §9.1 Deterministic inner-solve seed

Replace the path-dependent cache `_cache['q_prev']` with a fixed
seed across all cost evaluations within a single IK invocation. The
recommended seed is $q_{\text{start}}$:

$$
q^*(p_t) = \Phi_{\text{ik}}\bigl(p_t,\; R_t(q_{\text{start}}),\; q_J(q_{\text{start}})\bigr)
$$

**Justification.** At SS entry the robot is at $q_{\text{start}}$.
The natural seed for finding a new configuration that satisfies
the anchor constraints is the current state. The seed is the same
across all cost calls (since $q_{\text{start}}$ is fixed for the
duration of a single IK invocation), so the cost becomes a
deterministic function of $p_t$.

This eliminates pathology (C). Trade-off: no warm-start across
Nelder-Mead simplex steps, so the inner solve may take more
iterations on average. The diagnostic measured the on-demand IK at
~$6$ s wall-clock with warm-start; without warm-start, expect
~$10$–$15$ s. Acceptable for the offline use case.

### §9.2 Broadened multi-start

Replace the three vertical-axis seeds with a more diverse set:

$$
\mathcal{S} = \bigl\{
p_{\text{start}}^{(b)},\;
p_{\text{midpoint}},\;
p_{\text{midpoint}} + (\pm 0.3, 0, 0),\;
p_{\text{midpoint}} + (0, \pm 0.3, 0),\;
p_{\text{fixed}}^{(b)}
\bigr\}
$$

where $p_{\text{start}}^{(b)} = q_{\text{start}}[:3]$ is the live
torso position (the "don't move" seed), and $p_{\text{fixed}}^{(b)}$
is the torso position from a fixed-rotation IK solve from
$q_{\text{start}}$ (the "fixed-rotation hybrid" seed).

This gives 7 seeds spanning all three Cartesian axes plus two
physically motivated alternatives. Adequate basin coverage for a
3-D decision space.

### §9.3 Optional: post-convergence safety check

After convergence, evaluate $w_\sigma^{\text{bi}}(q_{\text{end}})$.
If below threshold $\epsilon$ (e.g. $\epsilon = 10^{-3}$), reject
the trajectory IK output and fall back to fixed-rotation or
endpoint-only IK. This is a safety net catching residual pathologies
in unusual configurations.

### §9.4 Resulting formulation

**Decision variable:** $p_t \in \mathbb{R}^3$.

**Cost (deterministic in $p_t$):**

$$
\text{cost}(p_t) = - \min_{k = 1, \ldots, 5} w_\sigma^{\text{bi}}\bigl(q(\tau_k)\bigr)
$$

where:

$$
q(\tau_k) = \mathrm{interp}\bigl(q_{\text{start}},\; q_{\text{end}}(p_t);\; s(\tau_k)\bigr)
$$

$$
q_{\text{end}}(p_t) = \Phi_{\text{ik}}\bigl(p_t,\; R_t(q_{\text{start}}),\; q_J(q_{\text{start}})\bigr)
$$

**Outer optimization:** Nelder-Mead with multi-start over the seed
set $\mathcal{S}$ defined in §9.2. Best-of-7 local optimization.

**Reported quantities:** $w_{\text{worst}}$, $w_{\text{end}}$,
optionally $w_{\text{Y}}^{\text{bi}}(q_{\text{end}})$ for cross-comparison
with fixed-rotation IK output.

---

## §10 — Open questions

The following choices are made by convention or convenience in the
current formulation but are open for revision:

### §10.1 Symmetric vs asymmetric metric

The product $\sigma_{\min}(\hat J_a) \cdot \sigma_{\min}(\hat J_b)$
weights both arms equally. During SS, only the swing arm's
manipulability directly governs QP tracking failure (the stance
arm's tracking is a hard constraint, not a tracked task). A weighted
metric:

$$
w_\sigma^{\text{weighted}} = \sigma_{\min}(\hat J_a)^{w_a} \cdot \sigma_{\min}(\hat J_b)^{w_b}
$$

with $w_b > w_a$ when arm B is the swing arm would be more
faithful. Investigation deferred.

### §10.2 Yoshikawa vs σ_min as the primary metric

For the IFToMM contribution, this choice deserves explicit
discussion. σ_min predicts worst-direction failure;
Yoshikawa predicts isotropic motion-volume reduction. For the
locomotion task where any rank-deficient direction can break
tracking, σ_min is more directly relevant. But a hybrid:

$$
w_{\text{combined}} = \sigma_{\min}(\hat J_a) \cdot \sigma_{\min}(\hat J_b) \cdot \bigl[\text{Yoshikawa correction term}\bigr]
$$

might be better still. Open.

### §10.3 Fictional vs actual interpolation

The current $q(\tau)$ is a joint-space quintic that does not match
the controller's actual SS trajectory (TorsoPlanner pose-space
quintic + QP-resolved joint accelerations). A formulation that
sampled the actual reference and predicted joint-space response
would be more faithful but requires a forward model of the
QP. Tradeoff between formal correctness and computational cost is
open.

### §10.4 Multi-step lookahead

All current IKs are step-local: the IK at step K does not see step
K+1's anchor pair. Per the audit (T15_post5 §7), the planning
pipeline as a whole is single-step-local.

For the IFToMM contribution, a multi-step manipulability IK that
chooses each $q_{\text{end}}$ knowing the next step's anchor pair
would be a cleaner story. The Phase-3 chained-precompute approach
attempted something like this and failed because the chain
abstraction (each step's $q_{\text{start}}$ predictable from the
previous step's plan) does not match closed-loop reality. A real
multi-step lookahead would need to either (a) operate at planning
time with predicted closed-loop drift, or (b) re-solve at each
step's SS entry with knowledge of step K+1's targets. Open.

### §10.5 Stance/swing weighting in the trajectory cost

Beyond §10.1's symmetric/asymmetric question on the per-step metric:
during the trajectory $\tau \in [0, 1]$, the swing arm transitions
from being free (just released, $\tau$ near 0) to being about to
dock (constrained, $\tau$ near 1). The rank-deficiency risk is
highest mid-swing. A $\tau$-dependent weight $w(\tau)$ on the
manipulability cost might be appropriate. Open.

---

## §11 — Implementation plan

The corrected formulation per §9 maps to:

1. **`crawlbot/core/ik.py::manipulability_config_trajectory`:**
   - Remove the `_cache['q_prev']` warm-start mechanism.
   - Pass $q_{\text{start}}$ explicitly as the inner-solve seed.
2. **`crawlbot/core/ik.py::manipulability_config_trajectory`:**
   - Replace the 3-seed multi-start with the 7-seed set from §9.2.
   - Compute $p_{\text{fixed}}^{(b)}$ via a fixed-rotation IK call
     from $q_{\text{start}}$, before the outer optimization begins.
3. **`crawlbot/core/ik.py::manipulability_config_trajectory`:**
   - Optional: implement the §9.3 safety check.
4. **Test:** the diagnostic fixture
   `tests/fixtures/step2_ss_entry_fixture.npz` is the regression test
   (`tests/test_mid_waypoint_reshape.py`, `tests/test_ik_anomaly_regression.py`).
   The corrected IK on this fixture should yield
   $w_{\text{worst}} \geq 0.045$ (within the order of the grid
   maximum), with no order-of-magnitude variability across runs.

After the IK fix lands, re-run T15 with `use_trajectory_aware_ik=True`
in the on-demand mode (Phase-4 wiring). The expected outcome is:
- Step 2's closed-loop SS proceeds with $w_\sigma^{\text{bi}}$ bounded
  away from $10^{-4}$.
- Step 2 docks within timeout.
- N-step demonstration extends to 5+ steps, M=14% mass ratio.

If step 2 still fails after this fix lands, the failure mode is no
longer "IK pathology" but "controller cannot track through the
endpoint-determined SS trajectory" — which would point to issues in
the TorsoPlanner shape, the QP weighting, or the swing reference,
all of which are §10's open territory.

---

## §12 — Notation reference

| Symbol | Meaning |
|---|---|
| $q$ | Configuration; $q \in \mathrm{SE}(3) \times \mathbb{R}^{14}$ |
| $\dot q$ | Reduced velocity; $\dot q \in \mathbb{R}^{20}$ |
| $q_b$ | Floating-base configuration; $q_b = (p_t, R_t)$ |
| $q_{J,\alpha}$ | Joint configuration of arm $\alpha$; $q_{J,\alpha} \in \mathbb{R}^7$ |
| $\dot q_b$ | Base twist (linear plus angular velocity) |
| $T_a, T_b$ | Tool poses, $\mathrm{SE}(3)$ |
| $T_a^*, T_b^*$ | Anchor target poses |
| $\xi_a, \xi_b$ | SE(3) constraint residuals |
| $J_\alpha$ | Full $6 \times 20$ tool-frame Jacobian |
| $J_\alpha^{(b)}$ | Base-induced block of $J_\alpha$, $6 \times 6$ |
| $J_\alpha^{(J,\beta)}$ | Joint-block of $J_\alpha$ for arm $\beta$, $6 \times 7$ |
| $\hat J_\alpha$ | Arm-internal Jacobian, $\hat J_\alpha = J_\alpha^{(J,\alpha)}$ |
| $w_{\text{Y}}^{(\alpha)}$ | Single-arm Yoshikawa, $\sqrt{\det(\hat J_\alpha \hat J_\alpha^\top)}$ |
| $w_\sigma^{(\alpha)}$ | Single-arm σ_min, $\sigma_{\min}(\hat J_\alpha)$ |
| $w^{\text{bi}}$ | Bimanual product over $\alpha \in \{a, b\}$ |
| $w_{\text{end}}$ | Bimanual σ_min at IK endpoint |
| $w_{\text{worst}}$ | Bimanual σ_min worst over interior samples (IK 3 only) |
| $\Phi_{\text{ik}}$ | Inner Gauss-Newton solver (anchor satisfaction) |
| $s(\tau)$ | Quintic temporal scaling, $10\tau^3 - 15\tau^4 + 6\tau^5$ |
| $K$ | Number of interior samples in IK 3 (currently 5) |
| $\mathcal{S}$ | Multi-start seed set in IK 3 |

---

*End of formulation document, draft 1.*
