# `crawlbot.solvers`

**The controller itself.**

| file | lines | canonical coverage | document |
|---|---:|---:|---|
| `wholebody_qp.py` | 950 | **97 %** | [wholebody_qp.md](wholebody_qp.md) |
| `centroidal_nmpc.py` | 702 | 88 % | [centroidal_nmpc.md](centroidal_nmpc.md) |
| `nmpc_solver.py` | 649 | 95 % | [nmpc_solver.md](nmpc_solver.md) |
| `hierarchical_qp.py` | 529 | 70 % | [hierarchical_qp.md](hierarchical_qp.md) |
| `contact_phase.py` | 138 | 85 % | [contact_phase.md](contact_phase.md) |

## The two-stage architecture

```
        stage 1                                stage 2
  CentroidalNMPC (dt = 0.1 s)   ---->   WholeBodyQP (dt = 0.01 s)
  N = 8, state 9, control 12            14 free DOF in SS
  -> momentum-feasible CoM plan         -> joint accelerations + torques
```

Stage 1 decides **what is feasible** against the reaction-wheel envelope;
stage 2 decides **how to realise it** with the arms. The split exists because
those two questions have very different natural rates: feasibility is a
horizon-scale question (0.8 s ahead, 10 Hz), tracking is an instantaneous one
(100 Hz).

The two stages share one piece of mathematics — the momentum map in
`contact_phase.py` — so they cannot disagree about how a contact wrench turns
into momentum.

## What ties them to the AOCS

Neither stage models the reaction wheels. The coupling is a **conservation law**:
total angular momentum about the structure origin is constant, so the wheel
momentum can be *reconstructed* at any planned node and bounded there. That, plus
a linear cap on `H_s_dot`, is the entire decentralised contract with the AOCS.

See `centroidal_nmpc.md` section 1.3.

## Cross-cutting rules

- `weight_ratio = 1.0` means **the alpha magnitudes are the hierarchy**; the
  `priority` integers are inert (`hierarchical_qp.md` section 2).
- The dataclass defaults in this package are **not** the canonical values —
  `sim_loop` overrides them. Read CLAUDE.md.
- Unexercised failure branches (`get_shifted_fallback`, `_solve_qp_raw` error
  paths) are dead **because the system is healthy**. Keep them.
