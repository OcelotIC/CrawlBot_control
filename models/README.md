# VISPA CrawlBot Model Files

## Canonical Model

**`VISPA_crawling_rwa3.xml`** is the single canonical MJCF for all development
and validation work. It contains:

- 2x 6-DOF robot arms (12 actuated joints)
- 3 orthogonal reaction wheels (rw_x, rw_y, rw_z)
- Structure body (ASTROHUB-class, 7110 kg)
- Robot mass ~71 kg (1% mass ratio)

### Key parameters (ground-truth: SimConfig)

| Parameter          | MJCF value      | SimConfig     | Spec ref |
|--------------------|-----------------|---------------|----------|
| RW spin inertia    | 0.01 kg.m2      | rwa_I_w=0.01  | sect 4.6 |
| RW torque limit    | ctrlrange +-5   | tau_w_max=5   | sect 5.1 |
| hw capacity        | (software-limited) | +-5 Nms    | sect 4.6 |
| Joint torque limit | ctrlrange +-50  | tau_max=20    | SimConfig|

## URDF (Pinocchio)

- **`VISPA_crawling_fixed.urdf`** -- fixed-base URDF for Pinocchio (no RW joints,
  12 arm DOF). Used by `RobotInterface` for kinematics/dynamics.

## Active Variants

| File | Purpose | Difference from canonical |
|------|---------|--------------------------|
| `VISPA_crawling.xml` | Legacy non-RWA model | No reaction wheels |
| `VISPA_crawling_rwa3_8pct.xml` | Fast test variant | Structure mass 888 kg (8% ratio) |
| `VISPA_crawling_rwa3_8pct_hw50.xml` | Comparison script | hw limits +-50, tau_w 10 |
| `VISPA_crawling_rwa3_8pct_hw100.xml` | Comparison script | hw limits +-100, tau_w 20 |

## Archived

`archive/` contains obsolete model variants (7-DOF, 0.1% ratio, pyramid RWA)
that are not referenced by any code. They are preserved in git history.

## Rules (anti-pattern A2)

Do NOT create new MJCF files with baked-in parameter changes. Parametric
variations (mass ratio, hw limits) should be applied programmatically via
SimConfig or at runtime. If a new MJCF is absolutely necessary, justify it
and document it here.
