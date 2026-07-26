# `crawlbot.aocs`

**Reaction-wheel attitude control.**

| file | lines | canonical coverage | document |
|---|---:|---:|---|
| `force_estimator.py` | 657 | 39 % | [force_estimator.md](force_estimator.md) |

## Role

The robot crawling along the structure transfers angular momentum to it. This
package is what spends the wheel budget the NMPC promised not to exceed.

Six control laws are implemented; the canonical run uses exactly one
(`legacy_pid_numerical`). The other five are selectable alternatives — not dead
code, but **not covered by the gate**.

## Two points worth knowing

**The feedforward has two branches, and both run.** In single support it is a
finite-difference estimate from centroidal momentum; in double support the welded
loop carries an internal stress whose couple `(r_CA - r_CB) x f` is *invisible in
`L_com`*, so the term is instead computed directly from the QP contact wrenches.

**The estimator object is not in the loop.** `MomentumDisturbanceEstimator` is
constructed and its outputs are logged every tick, but `update()` is never
called — so `H_rO` and `H_dot_est` are identically zero across all 2077 ticks.
