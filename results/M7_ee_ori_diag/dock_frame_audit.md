# M7 EE-orientation dock-frame audit

Single diagnostic, no simulation performed. Reads:
- MJCF: `/home/user/CrawlBot_control/models/VISPA_crawling_rwa3.xml`
- log : `/home/user/CrawlBot_control/results/M7_1pct_1step_v21_mapping_off/sim_log.json`

## 1. SwingPlanner._R_end (target EE orientation)

Value read from `sim.swing_planner._R_end` after `SimulationLoop.setup(n_steps=1, start_a=2, start_b=2)`.
(Never reassigned — SwingPlanner.__init__ sets it to `np.eye(3)`, and no setter exists.)

```
  +1.000000  +0.000000  +0.000000
  +0.000000  +1.000000  +0.000000
  +0.000000  +0.000000  +1.000000
```

## 2. Anchor 3 of arm b (scheduler + MJCF site)

`sim.sched.anchor_se3("b", 3)` returns `pin.SE3(np.eye(3), pos)`:

- position (structure frame): `[0.4, -0.3, 0.02499999999999991]`
- rotation (structure frame): identity by construction.

MJCF site `anchor_3b`: no `quat`/`euler`/`axisangle` attribute → identity in its parent body frame. Parent body (structure root) resolved as `structure`. Rotation in structure frame (from `site_xmat`):

```
  +1.000000  +0.000000  +0.000000
  +0.000000  +1.000000  +0.000000
  +0.000000  +0.000000  +1.000000
```

## 3. Dock-required R_tool at anchor_3b

MJCF weld `grip_b_to_3b` (line 338) links `site1="gripper_b"` and `site2="anchor_3b"`. Neither site carries an explicit orientation attribute; both inherit identity from their parent bodies. With the weld's default `relpose`, activation enforces `world_pose(gripper_b) == world_pose(anchor_3b)` in 6D. Converting to structure frame gives the required tool rotation at docking:

```
  +1.000000  +0.000000  +0.000000
  +0.000000  +1.000000  +0.000000
  +0.000000  +0.000000  +1.000000
```

For reference, gripper_b site's current world rotation in structure frame (at setup complete, before release):

```
  +1.000000  -0.000000  +0.000000
  +0.000000  +1.000000  -0.000000
  -0.000000  +0.000000  +1.000000
```

## 4. Actual EE orientation at SS abort

Log: `/home/user/CrawlBot_control/results/M7_1pct_1step_v21_mapping_off/sim_log.json`.
Last SS sample at `i = 112`, `t = 11.310 s`.
`q_ee[i]` (wxyz) = `[0.9942198226589578, 7.363107177173931e-05, 0.00016509676024187802, 0.10736345539109618]`.

Rotation matrix (structure frame):

```
  +0.976946  -0.213486  +0.000344
  +0.213486  +0.976946  -0.000111
  -0.000312  +0.000182  +1.000000
```

## Release rotation passed to `set_swing_orientation(...)` (for completeness)

`SwingPlanner.set_swing_orientation(oMf_release.rotation)` is called in `sim_loop.py` right after the swing weld is deactivated, at the first SS tick. Reading `q_ee` at the first SS log sample `i = 0`, `t = 0.110 s`:

`q_ee[i_first]` (wxyz) = `[0.9999981000011706, -0.001272477055077443, -0.0013189254185320915, 0.0006642529139729141]`

Rotation matrix (structure frame):

```
  +0.999996  -0.001325  -0.002640
  +0.001332  +0.999996  +0.002543
  +0.002636  -0.002547  +0.999993
```

Angular distance between release and `_R_end` (= the rotation the SwingPlanner SLERPs through during SS):

`||log3(R_release^T · R_end)|| = 0.2234°`

## Angular differences

All computed as `||log3(R_a^T · R_b)||` in degrees.

| label | R_a | R_b | value [°] |
|---|---|---|---|
| A | SwingPlanner._R_end | dock-required R_tool | 0.0000 |
| B | EE ori at SS abort  | SwingPlanner._R_end  | 12.3267 |
| C | EE ori at SS abort  | dock-required R_tool | 12.3267 |
