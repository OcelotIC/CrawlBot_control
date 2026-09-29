# `crawlbot.aocs`

**Reaction-wheel attitude control.**

| file | lines | canonical coverage | document |
|---|---:|---:|---|
| `force_estimator.py` | 657 | 39 % | [force_estimator.md](force_estimator.md) |

## Role

The robot crawling along the structure transfers angular momentum to it. This
package is what spends the wheel budget the NMPC promised not to exceed.

One control law: `legacy_pid_numerical`, the one the paper uses. The five
variants and the H_{r/O}-estimator law were retired in R1 (never run by the
canonical nor by Table 2).

## Two points worth knowing

**The feedforward has two branches, and both run.** In single support it is a
finite-difference estimate from centroidal momentum; in double support the welded
loop carries an internal stress whose couple `(r_CA - r_CB) x f` is *invisible in
`L_com`*, so the term is instead computed directly from the QP contact wrenches.

**The disturbance estimator is gone.** `MomentumDisturbanceEstimator` was
constructed but never updated (`H_rO`, `H_dot_est` identically zero); it was
retired in R1 and the two log channels are now written as zeros directly.
