# `crawlbot.estimation`

**Sensorless contact detection.**

| file | lines | canonical coverage | document |
|---|---:|---:|---|
| `contact_estimator.py` | 261 | 69 % | [contact_estimator.md](contact_estimator.md) |

## Role

A De Luca (2006) generalized momentum observer. Its appeal is that the momentum
derivative `p_dot = S^T tau + C^T v + J_c^T f_ext` contains **no acceleration
term**, so contact force can be estimated without force sensors and without
differentiating velocity.

## What runs, and what does not

The observer runs (single support only) and produces a real residual — measured
max 8.088, mean 1.017, non-zero on 2067 of 2077 ticks.

The **contact state machine does not**: `update()` is never called, so
`gmo_contact_state` is constant `NO_CONTACT` over the whole traversal.

That is architecture rather than a bug — docking is decided geometrically
(`d < 5 mm AND ori < 5 deg`), never by the observer. But it means the
`ContactObserverConfig` thresholds are not canonical values: nothing reads
them.
