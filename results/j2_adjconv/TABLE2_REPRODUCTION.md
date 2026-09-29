# Reproducing the paper's Table 2 from this branch

The Table 2 arms come from the three-way envelope ablation of `e37b9ca`
(branch `claude/review-closure-bloc-2-uwu1x7`, never merged) plus the published
unmanaged run. The `none` arm needs `SimConfig.nmpc_tau_w_max`, which existed only
on that branch; it was ported to `refactor/sim-loop-split` in `a44a3b6`
(`crawlbot/` hunks) and the next commit (`dca.main` kwargs). Every arm now runs
from this branch:

```bash
MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/run_table2.py full             # Managed (= the canonical)
MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/run_table2.py rate             # Rate-only
MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/run_table2.py none             # No envelope
MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/run_table2.py u25              # Rate/clip lifted (published unmanaged)
MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/run_table2.py station_keeping  # 900 s settle (T4b)
```

Each writes `results/table2/<arm>/` — the driver's files, `sim_log.json`
(git-ignored) and the 66-column fulldiag export `<arm>_fulldiag.csv`.

| arm | NMPC rate cap | NMPC storage box | QP box + AOCS clip | kwargs over the canonical |
|---|---|---|---|---|
| `full` | 2.5 | on | 2.5 | — |
| `rate` | 2.5 | off | 2.5 | `enforce_hw_conservation=False` |
| `none` | off | off | 2.5 (held) | `nmpc_tau_w_max=inf`, `enforce_hw_conservation=False` |
| `u25` | lifted | on | lifted | `tau_w_max=1e6` |
| `station_keeping` | 2.5 | on | 2.5 | `settle_seconds=900` |

The canonical kwargs are those of `gate/replay_canonical.py`, including its
regularization setting (ε = 1e-6).

## Checks

- `full` reproduces the host-local canonical reference bit-for-bit (see the
  commit that added this page).
- `rate`, `none`, `u25` are gate scenarios on a shortened traversal
  (`gate/scenarios.py`: `table2_rate`, `table2_none`, `table2_u25`), frozen on
  `36c44bf`. `table2_none` with the ported field is identical to the baseline
  produced by a harness injection of its single effect,
  `CentroidalNMPCConfig(tau_w_max=inf)`.
- The full-length arms of `e37b9ca` are committed on its branch under
  `results/review_closure/c4_ablation/{none,rate,full}/`.
- On the committed canonical baseline itself, see `gate/README.md`: it does not
  reproduce on a Python 3.11.16 host (a 1e-15 flip); the lock is 3.11.15.
