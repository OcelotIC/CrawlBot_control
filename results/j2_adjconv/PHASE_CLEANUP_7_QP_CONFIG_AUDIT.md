# PHASE CLEANUP-7 — QP config-surface audit (READ-ONLY)

Audit of the configuration surface left behind by CLEANUP-6, which removed the legacy
task-stack *implementations* but deliberately left their *declarations*. **No code changed.**

Method: for every `WholeBodyQPConfig` field, count genuine attribute reads
(`cfg.X` / `self.config.X` / `config.X`) inside `wholebody_qp.py`, excluding the declaration
line and comments. Cross-checked by raw-name occurrence count, and verified there is **no
`getattr(cfg, ...)` blind spot** (zero such calls in the file) — so the regex audit is complete.

## Headline

**12 of 53 `WholeBodyQPConfig` fields are now never read.** All 12 are still passed by
`sim_loop._build_qp()`, so the dead plumbing runs on both sides of the interface.

| dead field | canonical value | passed at `sim_loop:` |
|---|---|---|
| `alpha_com` | 0.0 | 1143 |
| `alpha_torso` | 500.0 | 1144 |
| `use_m2_stack` | **True** | 1177 |
| `alpha_com_soft` | 0.0 | 1179 |
| `ee_null_space` | **True** | 1178 |
| `r_tube` | 0.0 | 1183 |
| `w_tube_lin` | 50.0 | 1184 |
| `cooperative_arms_mode` | **True** | 1185 |
| `alpha_torso_ang` | 500.0 | 1186 |
| `alpha_torso_lin` | 0.0 | 1187 |
| `ss_centroidal_momentum_task` | False | 1188 |
| `ss_alpha_tl_weak` | 0.0 | 1190 |

Note the three set to **True/500** by the canonical: `use_m2_stack`, `ee_null_space`,
`cooperative_arms_mode`, `alpha_torso_ang`. The config advertises those features as *on* while
their implementations no longer exist — the most misleading state in the file.

## ⚠ F1 — `SimConfig.use_m2_stack` must NOT be pruned (load-bearing outside the QP)

The `WholeBodyQPConfig` copy is dead, but the **`SimConfig`** field is read twice in `sim_loop`
on paths that have nothing to do with the QP task stack:

| site | what it gates |
|---|---|
| `sim_loop.py:2871` | selects the torso-reference branch — whether the CoM→torso δ-mapping is used or the raw TorsoPlanner quintic is fed through |
| `sim_loop.py:3038` | `passivity_active = cfg.use_m2_stack and (phase == 'DS' or passivity_hold)` — **the DS passivity constraint** |

Pruning `SimConfig.use_m2_stack` along with its QP twin would silently disable DS passivity and
change torso-reference routing. The gate would catch it, but the trap is worth recording: the
two same-named fields have opposite fates. Only the `WholeBodyQPConfig` one is removable, plus
the `use_m2_stack=` / `ee_null_space=cfg.use_m2_stack` lines that feed it.

## F2 — `cooperative_arms_mode` is now a no-op that the canonical still sets True

Every reference in `crawlbot/` is either a comment, the declaration, or the `_build_qp` pass-through:

```
wholebody_qp.py:118,124  comments
wholebody_qp.py:126      declaration
sim_loop.py:1185         cooperative_arms_mode=cfg.cooperative_arms_mode
config.py:203            declaration
```

Zero readers. CLAUDE.md already records the cooperative split as superseded, but the config and
the `dca` CLI still present it as an active mode — anyone reading the canonical config would
reasonably conclude cooperative-arms control is running. Same class: `ee_null_space=True`.

## F3 — mirroring `SimConfig` fields that are now pure plumbing

These feed only the dead QP fields (read solely at the `_build_qp` call site or its signature):

`ss_alpha_com`, `ss_alpha_torso`, `alpha_com_soft`, `r_tube`, `w_tube_lin`,
`ss_alpha_torso_ang`, `ss_alpha_torso_lin`, `ss_centroidal_momentum_task`, `ss_alpha_tl_weak`,
`cooperative_arms_mode` — **10 fields**. (`use_m2_stack` explicitly excluded, per F1.)

## F4 — `dca` CLI still exposes inert flags

The canonical runner advertises knobs that can no longer change anything:
`--cooperative-arms-mode` (5 refs), `--ss-centroidal-momentum-task` (3), `--ss-alpha-torso-lin`
(2), `--r-tube` (1), `--ss-alpha-torso-ang` (1). Worse than dead code: a flag that accepts a
value and silently ignores it invites a false experiment.

## F5 — eight silent canonical values (Rule 5)

Fields `sim_loop` never overrides, so the dataclass default *is* the canonical value:

| field | canonical (= default) |
|---|---|
| `method` | `'weighted'` |
| `solver` | `'qpoases'` |
| `weight_ratio` | 1.0 |
| `w_hw_slack` | 800.0 |
| `alpha_settle` | 1000.0 |
| `Kd_settle` | 10.0 |
| `qdd_max` | 50.0 |
| `tau_contact_max` | 300.0 |

`w_hw_slack` is at least cited in CLAUDE.md's frozen table (`wholebody_qp.py:181`); the other
seven are undocumented. Same finding class as the NMPC's six weights (CLEANUP-3 F5), which were
hoisted into `SimConfig`.

## F6 — stale section banners in the dataclass

`wholebody_qp.py:103–160` still carries headers for removed implementations: "M2 stack: torso P1
+ EE null-space P2 + posture P3 + soft CoM", "Option D: torso linear soft tube",
"Cooperative-arms task stack", "SS centroidal-momentum task (T-MOM)". Each now documents code
that does not exist.

## Suggested pruning (not performed)

1. Remove the 12 dead `WholeBodyQPConfig` fields + their `_build_qp` pass-through lines + the
   `ac`/`at` parameters that become unused, and rewrite the F6 banners.
2. Remove the 10 `SimConfig` fields from F3 — **keeping `use_m2_stack`** (F1).
3. Remove the F4 `dca` CLI flags.
4. Hoist or at least document the F5 silent values.

All four are declaration-level and should be gate-verifiable as **byte-identical**, since none
of these values is read by any surviving code path. Step 2 is the one with a real trap in it.
