# Claude Code Handoff — Reworked Architecture Implementation

**Date:** 2026-04-10
**Source document:** `brainstorming_reworked_architecture.md` (§0–§9)
**Codebase:** `CrawlBot_control/` (GitHub: `CrawlBot_control`)

---

## 0. Context & Ground Rules

### What this document is

A file-level implementation plan for the reworked two-layer control architecture described in `brainstorming_reworked_architecture.md`. Each milestone has:
- explicit files to create or modify
- pass/fail criteria with numerical thresholds
- mandatory diagnostic output (plots + metrics table)

### Ground rules for Claude Code

1. **Ground-truth is the code, not the paper.** Before modifying any module, read the current implementation. Do not assume the paper description matches the code.
2. **Milestone-by-milestone.** Do not proceed to M(n+1) until M(n) passes and Idriss validates.
3. **Every simulation produces diagnostics.** No simulation run is complete without calling `run_diagnostics()`. "It docked" is not a pass criterion.
4. **Write test scripts to disk first** (`cat > file.py`), then run them.
5. **PYTHONPATH:** always `PYTHONPATH=/home/claude/CrawlBot_control python3 <script>`.
6. **pip:** always `--break-system-packages`.

### Anti-patterns from past sessions — DO NOT REPEAT

These are failure modes that have caused wasted iterations in previous sessions. They are listed here as explicit prohibitions.

#### A1. "It docked" is not a metric

The previous evaluation loop was: run simulation → count docks → report "3/3 docks achieved." This hides catastrophic issues: 24° of platform rotation, hw at 95% saturation, 46° EE orientation error. A simulation that achieves all docks but with degraded tracking is **not passing**. Every simulation must produce the full diagnostic suite (M0) and every metric must be within threshold. If a metric is out of bounds, **that is a failure** even if all docks succeed.

#### A2. Do not copy-paste model files with parameter changes

The current codebase has 4+ MJCF variants with parameters baked into filenames (`_rwa3.xml`, `_8pct.xml`, `_hw50.xml`). This is the root cause of the `hw_min/max = ±50` bug — a parameter was wrong in one file and nobody noticed because there was no single source of truth.

**Rule:** There is ONE canonical MJCF. Parametric variations (mass ratio, hw limits, wheel count) are applied programmatically via `SimConfig` or a model-generation script, not by maintaining parallel XML files. If a test needs a different mass ratio, it modifies the model at runtime (e.g., `model.body_mass[structure_id] = desired_mass`), or generates the MJCF from a template. Any new MJCF file requires explicit justification.

#### A3. Do not patch — understand, then fix

When a bug is found, the temptation is to add a correction term, a clamp, or a special case. Example: instead of fixing the missing orbital term in the AOCS feedforward, one might add a gain on `omega_s` to damp the drift symptom. This creates layers of compensating hacks that are fragile and obscure the real physics.

**Rule:** Before writing a fix, state in a comment:
1. What is the root cause?
2. Where in the math (reference the brainstorming doc section) is this addressed?
3. What is the expected quantitative effect of the fix?

Then implement the fix, run diagnostics, and verify the expected effect matches reality. If it doesn't, the diagnosis was wrong — go back to step 1, do not add another patch.

#### A4. Do not silently change numerical values

Past sessions changed parameters (NMPC horizon, cost weights, tolerances, gains) mid-iteration without recording what changed and why. This makes it impossible to understand which change caused which effect.

**Rule:** All tunable parameters live in `SimConfig` (or the relevant `*Config` dataclass). Every parameter has:
- a default value with a comment explaining why
- a unit
- a reference to the spec section that justifies it

When a parameter is changed during development, the commit message (or session log) states: what changed, from what value to what value, and what diagnostic metric it was intended to affect. "Tweaked gains" is not acceptable.

#### A5. Do not ignore regression

Fixing module B may silently break module A if there is no regression test. Past sessions fixed the AOCS feedforward but didn't re-run the full pipeline to check that the QP still converged with the new torque profile.

**Rule:** After any change to a core module (`centroidal_nmpc.py`, `whole_body_qp.py`, `force_estimator.py`, `sim_loop.py`), re-run the full test suite (`pytest tests/ -v`). If a test breaks, it must be fixed before proceeding — not skipped, not marked `xfail`.

#### A6. Do not write wall-of-text explanations instead of showing data

When a simulation fails or a result is surprising, the correct response is: show the diagnostic plots, point to the specific subplot and time range where the problem is visible, and propose a hypothesis. The incorrect response is: write 3 paragraphs explaining why the result might be acceptable or why the metric doesn't apply. Data first, interpretation second.

#### A7. Read the file before editing it

Multiple bugs in past sessions came from editing code based on assumed content rather than actual content. The state vector ordering, the sign convention for contact wrenches, the qvel indexing for platform angular velocity — these are things that must be verified by reading the source, not assumed from the paper.

**Rule:** Before any `str_replace` or modification, `view` the file and confirm the current content matches assumptions. If it doesn't, update the understanding before editing.

#### A8. One canonical parameter set

There is exactly ONE set of physical parameters for the SpaceServicer + ASTROHUB system. It is defined in `SimConfig` and nowhere else. Any test, script, or notebook that needs parameters imports them from there. Hardcoded magic numbers in test files (`robot_mass=71.0`, `hw_max=5.0`) must reference `SimConfig` defaults or be explicitly documented as test-specific overrides.

### M-1. Codebase Cleanup (before M0)

Before building diagnostics, clean up the known debt so M0 starts on solid ground.

#### M-1.1 — Model consolidation

- Identify which MJCF is the ground-truth model
- Delete or archive obsolete variants
- Create `models/README.md` documenting the canonical model and how to derive variants
- Verify `hw_min/max` is correct (±5 Nms) in the canonical model — this was a known bug

#### M-1.2 — Update `requirements.txt`

Current `requirements.txt` specifies `pinocchio>=2.7` — update to match the pinned versions in the setup script:

```
pin==3.9.0
mujoco>=3.1,<4
casadi>=3.6,<4
numpy>=1.24,<2
matplotlib>=3.7
qpsolvers>=0.9
osqp>=0.6
Pillow>=10.0
pytest>=7.0
scipy>=1.10
```

Verify that `robot_interface.py` and all Pinocchio calls are compatible with pin 3.9 API. Fix any breaking changes before proceeding.

#### M-1.3 — Remove root-level shims

- Identify all root-level `.py` shim files
- Update any scripts/tests that import from them to use `crawlbot.*` imports
- Delete the shims
- Run `pytest tests/ -v` to confirm nothing breaks

#### M-1.4 — Parameter audit

- List every hardcoded numerical parameter in `crawlbot/` and `scripts/`
- Cross-reference with `SimConfig` — anything not in `SimConfig` either goes in or gets a justification comment
- Cross-reference with the brainstorming doc's parameter table (§0.4) — flag discrepancies

#### M-1.5 — Dead code removal

- Identify any modules, functions, or classes that are no longer called by any test or script
- Remove them (git history preserves everything)
- This reduces the surface area Claude Code has to understand

#### M-1 pass criteria

- Single canonical MJCF with correct parameters
- No root-level shim files
- `pytest tests/ -v` passes
- Parameter audit documented in `docs/parameter_audit.md`

### Codebase layout (current)

```
crawlbot/
├── core/           # robot_interface.py, state_conversions.py
├── solvers/        # centroidal_nmpc.py, nmpc_solver.py, whole_body_qp.py, contact_phase.py
├── planning/       # contact_scheduler.py, torso_planner.py, swing_planner.py, locomotion_planner.py
├── aocs/           # force_estimator.py
├── dynamics/       # constrained_dynamics.py
├── simulation/     # config.py, logging.py, sim_loop.py, plotting.py
models/             # MJCF + URDF files
tests/              # existing test suite
scripts/            # sim_torso6d.py, run_r6_full_sim.py, etc.
results/            # logs + figures
```

### Environment Setup

Claude Code sessions start with a clean container. The environment must be set up **at the start of every session** before any code runs. This is non-negotiable — importing `pinocchio` or `mujoco` before installation will fail silently or produce confusing errors.

#### Setup script

Run this verbatim at session start:

```bash
#!/bin/bash
# === CrawlBot environment setup ===
# Run ONCE at the start of every Claude Code session.

set -e

# 1. System dependencies (for MuJoCo offscreen rendering)
sudo apt-get update -qq && sudo apt-get install -y -qq \
    libosmesa6-dev libgl1-mesa-glx libglfw3 > /dev/null 2>&1

# 2. Python packages — PINNED VERSIONS
pip install --break-system-packages -q \
    "pin==3.9.0" \
    "mujoco>=3.1,<4" \
    "casadi>=3.6,<4" \
    "numpy>=1.24,<2" \
    "matplotlib>=3.7" \
    "qpsolvers>=0.9" \
    "osqp>=0.6" \
    "Pillow>=10.0" \
    "pytest>=7.0" \
    "scipy>=1.10"

# 3. MuJoCo rendering backend (headless)
export MUJOCO_GL=osmesa

# 4. Clone repo (if not already present)
if [ ! -d "/home/claude/CrawlBot_control" ]; then
    git clone https://github.com/<OWNER>/CrawlBot_control.git /home/claude/CrawlBot_control
fi

# 5. Verify critical imports
python3 -c "
import pinocchio as pin; print(f'Pinocchio {pin.__version__}')
import mujoco; print(f'MuJoCo {mujoco.__version__}')
import casadi; print(f'CasADi {casadi.__version__}')
import numpy as np; print(f'NumPy {np.__version__}')
import qpsolvers; print('qpsolvers OK')

# Verify Pinocchio 3.x API (breaking changes from 2.x)
assert hasattr(pin, 'computeCentroidalMomentum'), 'Missing centroidal API'
print('All imports OK')
"

# 6. Verify MuJoCo offscreen rendering
python3 -c "
import os; os.environ['MUJOCO_GL'] = 'osmesa'
import mujoco
spec = mujoco.MjSpec()
spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.1])
m = spec.compile()
d = mujoco.MjData(m)
r = mujoco.Renderer(m, 64, 64)
r.update_scene(d)
frame = r.render()
assert frame.shape == (64, 64, 3), f'Bad frame shape: {frame.shape}'
r.close()
print('MuJoCo offscreen rendering OK')
"

# 7. Run existing tests (quick smoke)
cd /home/claude/CrawlBot_control
PYTHONPATH=. MUJOCO_GL=osmesa python3 -m pytest tests/ -x -q --tb=short 2>&1 | tail -5

echo "=== Environment ready ==="
```

#### Critical notes on Pinocchio 3.x vs 2.x

Pinocchio 3.9 has API changes from 2.x that affect the codebase. If `robot_interface.py` was written for pin 2.x, these will need updating:

| pin 2.x | pin 3.x | Notes |
|---|---|---|
| `pin.forwardKinematics(model, data, q)` | Same | Unchanged |
| `pin.centerOfMass(model, data, q)` | Same, but `data.com[0]` → verify indexing | Check return |
| `pin.computeCentroidalMomentum(model, data)` | Same signature | Verify `data.hg` layout: `[angular; linear]` vs `[linear; angular]` — this is a known source of sign bugs |
| `pin.crba(model, data, q)` | Same | Returns upper-triangular; symmetrize if needed |
| `pin.computeCoriolisMatrix(model, data, q, v)` | Same | Verify skew-symmetry property holds |
| `pin.neutral(model)` | Same | |
| Quaternion convention | `[x, y, z, w]` | Pinocchio uses xyzw; MuJoCo uses wxyz — `state_conversions.py` handles this but **verify** |

**Rule:** At session start, after install, run `python3 -c "import pinocchio; print(pinocchio.__version__)"` and confirm it says `3.9.x`. If it doesn't, stop and fix before proceeding.

#### MUJOCO_GL

Set `export MUJOCO_GL=osmesa` in every shell session, or prefix every python command:

```bash
MUJOCO_GL=osmesa PYTHONPATH=/home/claude/CrawlBot_control python3 <script>
```

If osmesa is not available, fall back to `MUJOCO_GL=egl` (GPU) or `MUJOCO_GL=disabled` (no rendering — snapshots will be skipped). The diagnostic suite must handle `MUJOCO_GL=disabled` gracefully: skip Figure 8, log a warning, but produce all other figures and metrics.

---

## M0. Diagnostic Suite (BUILD FIRST)

**Rationale:** Every subsequent milestone needs quantitative evaluation. Without a standardized diagnostic module, iterations devolve into binary "docked / didn't dock" assessments, which hide tracking degradation, momentum margin erosion, and platform drift until they become catastrophic failures.

### M0.1 — Create `crawlbot/diagnostics/`

```
crawlbot/diagnostics/
├── __init__.py
├── metrics.py          # compute_metrics(log_data) → MetricsTable
├── plots.py            # generate_plots(log_data, output_dir)
├── snapshots.py        # capture_snapshots(model, data, sim_log, output_dir)
└── runner.py           # run_diagnostics(log_data, output_dir, thresholds=None)
```

### M0.2 — `SimLog` enrichment

The current `crawlbot/simulation/logging.py` (`SimLog`) must log the following time series. Check what already exists; add what's missing.

| Time series | Symbol | Source | Shape per step |
|---|---|---|---|
| Time | `t` | sim clock | (N,) |
| **Torso position ref** | `r_b_ref` | mapping layer output | (N, 3) |
| **Torso position actual** | `r_b` | `mj_data` torso body pos, projected into P | (N, 3) |
| **Torso orientation ref** | `R_b_ref` | TorsoPlanner | (N, 4) quat |
| **Torso orientation actual** | `R_b` | `mj_data` torso body quat | (N, 4) quat |
| **EE position ref** | `p_ee_ref` | SwingPlanner | (N, 3) |
| **EE position actual** | `p_ee` | `mj_data` EE site/body | (N, 3) |
| **EE orientation ref** | `R_ee_ref` | SwingPlanner (6D, new) | (N, 4) quat |
| **EE orientation actual** | `R_ee` | `mj_data` EE body quat | (N, 4) quat |
| **CoM ref (NMPC)** | `r_com_ref` | NMPC plan (first knot) | (N, 3) |
| **CoM actual** | `r_com` | Pinocchio `computeCenterOfMass` | (N, 3) |
| **v_com ref** | `v_com_ref` | NMPC plan | (N, 3) |
| **v_com actual** | `v_com` | Pinocchio | (N, 3) |
| **L_com ref** | `L_com_ref` | NMPC plan / TorsoPlanner feedforward | (N, 3) |
| **L_com actual** | `L_com` | Pinocchio `computeCentroidalMomentum` | (N, 3) |
| **h_w** | `h_w` | `I_w * omega_wheel` from MuJoCo | (N, 3) |
| **Platform angular velocity** | `omega_s` | `mj_data.qvel[3:6]` body frame | (N, 3) |
| **Platform attitude** | `R_s` | `mj_data.qpos[3:7]` → Euler (deg) | (N, 3) |
| **Platform position** | `p_s` | `mj_data.qpos[0:3]` | (N, 3) |
| **Joint torques** | `tau_q` | QP output | (N, n_u) |
| **AOCS torque command** | `tau_w` | AOCS output | (N, 3) |
| **Kinetic energy** | `T_kinetic` | 0.5 * dq^T H dq (relative) | (N,) |
| **Passivity LHS** | `passivity_lhs` | dq^T tau_q + 2*alpha*T | (N,) |
| **NMPC solve time** | `nmpc_solve_time` | solver wall time | (N_nmpc,) |
| **NMPC status** | `nmpc_status` | 0=converged, 1=max_iter, 2=infeasible | (N_nmpc,) |
| **NMPC cost** | `nmpc_cost` | objective value | (N_nmpc,) |
| **Phase label** | `phase` | 0=DS, 1=SS | (N,) |
| **Dock events** | `dock_times` | list of (t, arm_id, success_bool, pos_err, ori_err) | variable |
| **λ_ref (NMPC)** | `lambda_ref` | NMPC planned contact wrench | (N_nmpc, 12) |
| **λ actual** | `lambda_qp` | QP contact wrench solution | (N, 12) |
| **MuJoCo snapshots** | `snapshots` | list of `(t, qpos, qvel, label)` at key instants | variable |

### M0.3 — `metrics.py`: `compute_metrics(log_data, thresholds=None) → dict`

Computes scalar summary metrics from the logged time series. Returns a dict with pass/fail flags.

```python
DEFAULT_THRESHOLDS = {
    # --- Tracking ---
    'torso_pos_err_peak_mm':        10.0,     # mm
    'torso_ori_err_peak_deg':       5.0,      # deg
    'ee_pos_err_at_dock_mm':        5.0,      # mm
    'ee_ori_err_at_dock_deg':       5.0,      # deg
    'com_tracking_err_rms_mm':      15.0,     # mm

    # --- Momentum & AOCS ---
    'hw_saturation_ratio_peak':     1.0,      # |hw|/hw_max < 1
    'hw_saturation_ratio_rms':      0.7,      # RMS should stay well below limit
    'platform_rotation_total_deg':  5.0,      # cumulative |Δθ| over full sim
    'platform_omega_peak_deg_s':    2.0,      # peak |ω_s| (deg/s)
    'tau_w_peak_ratio':             1.0,      # |τ_w| / τ_w_max < 1

    # --- Energy & passivity ---
    'passivity_violations':         0,        # count of passivity_lhs > 0 during DS
    'ds_settling_time_rel_err':     0.20,     # |t_actual - t_predicted| / t_predicted

    # --- NMPC health ---
    'nmpc_solve_rate_50ms':         0.95,     # fraction solved in < 50ms
    'nmpc_infeasibility_rate':      0.02,     # fraction infeasible < 2%
}
```

**Per-metric computation:**

| Metric | Computation |
|---|---|
| `torso_pos_err_peak_mm` | `max(‖r_b - r_b_ref‖) * 1000` |
| `torso_ori_err_peak_deg` | `max(‖Log(R_b^T R_b_ref)‖) * 180/π` |
| `ee_pos_err_at_dock_mm` | `‖p_ee - p_ee_ref‖` at each dock event, take worst |
| `ee_ori_err_at_dock_deg` | `‖Log(R_ee^T R_ee_ref)‖` at each dock event, take worst |
| `com_tracking_err_rms_mm` | `rms(‖r_com - r_com_ref‖) * 1000` over SS phases only |
| `hw_saturation_ratio_peak` | `max(‖h_w‖ / h_max)` |
| `hw_saturation_ratio_rms` | `rms(‖h_w‖ / h_max)` |
| `platform_rotation_total_deg` | cumulative `Σ ‖Δθ_k‖` from Euler angles, or `‖Log(R_s(0)^T R_s(T))‖ * 180/π` |
| `platform_omega_peak_deg_s` | `max(‖ω_s‖) * 180/π` |
| `tau_w_peak_ratio` | `max(|τ_w_i|) / τ_w_max` per axis, take worst |
| `passivity_violations` | `count(passivity_lhs > ε)` where `ε = 1e-6`, during DS phases only |
| `ds_settling_time_rel_err` | compare measured settling to `(1/(2α)) ln(T0/T_settle)` |
| `nmpc_solve_rate_50ms` | `count(solve_time < 0.05) / N_nmpc` |
| `nmpc_infeasibility_rate` | `count(status == 2) / N_nmpc` |

**Output format:** Print table to console + save as CSV + return dict with `{metric_name: (value, threshold, pass_bool)}`.

### M0.4 — `plots.py`: `generate_plots(log_data, output_dir, dpi=150)`

Produces a **fixed set** of figures. Every simulation produces the same plots — no optional plots, no "if dock succeeded then plot X".

**Figure 1: Tracking overview (4 subplots, shared x-axis)**
- (a) Torso position error `‖r_b - r_b_ref‖` [mm] — with phase shading (DS=blue, SS=orange)
- (b) Torso orientation error `‖Log(R_b^T R_b_ref)‖` [deg]
- (c) EE position error `‖p_ee - p_ee_ref‖` [mm] — only during SS
- (d) EE orientation error `‖Log(R_ee^T R_ee_ref)‖` [deg] — only during SS
- Vertical dashed lines at dock events. Red dot on (c)/(d) at dock instant.

**Figure 2: Momentum & AOCS (4 subplots, shared x-axis)**
- (a) `h_w` 3 axes + `±h_max` bands (dashed red). Title shows peak saturation ratio.
- (b) `τ_w` 3 axes + `±τ_w_max` bands.
- (c) Platform angular velocity `ω_s` 3 axes [deg/s].
- (d) Platform attitude Euler angles [deg] — cumulative drift is immediately visible.
- Phase shading + dock lines as above.

**Figure 3: CoM tracking & centroidal momentum (3 subplots)**
- (a) CoM position: ref vs. actual, 3 axes [mm]. Compute error norm below.
- (b) CoM velocity: ref vs. actual, 3 axes [mm/s].
- (c) `L_com`: ref vs. actual, 3 axes [Nms]. Shows whether NMPC plan is tracked.

**Figure 4: Energy & passivity (2 subplots, DS phases only)**
- (a) Kinetic energy `T(t)` (log scale) + theoretical bound `T_0 e^{-2αt}` (dashed).
- (b) Passivity LHS `dq^T τ_q + 2αT` — should be ≤ 0. Red fill where violated.

**Figure 5: NMPC solver health (3 subplots)**
- (a) Solve time [ms] vs. call index. Horizontal line at 50 ms.
- (b) Solver status (color-coded scatter: green=converged, yellow=max_iter, red=infeasible).
- (c) Cost function value (log scale).

**Figure 6: Contact wrenches (2 subplots)**
- (a) `λ_ref` (NMPC) vs. `λ_qp` (QP) — force magnitude per contact.
- (b) Wrench tracking residual `‖λ_qp - λ_ref‖`.

**Figure 7: Joint-level (2 subplots)**
- (a) Joint torques `τ_q` (all joints, thin lines) + `±τ_max` bands.
- (b) Joint velocities `dq` — useful for spotting velocity spikes at phase transitions.

**Figure 8: MuJoCo rendered snapshots**

Offscreen-rendered frames at key simulation instants. Provides immediate visual sanity check without watching a video.

**Capture instants (automated, no manual selection):**
- `t = 0`: initial configuration
- Each DS→SS transition (weld release): 1 frame
- Mid-swing: 1 frame at `t = t_DS_exit + T_swing/2`
- Pre-dock approach: 1 frame at `t_dock - 0.5 s` (EE close to target)
- Each dock event: 1 frame at weld activation
- `t = T_final`: final configuration
- For a 3-step sim this produces ~12–15 frames

**Implementation:**

```python
import mujoco
import numpy as np
from PIL import Image

def capture_snapshots(model, data, sim_log, output_dir, 
                      width=1280, height=720, camera='trackcom'):
    """
    Render MuJoCo frames at key instants from a completed simulation.
    
    Requires MUJOCO_GL=osmesa (headless) or egl.
    Set env var before import: os.environ['MUJOCO_GL'] = 'osmesa'
    
    Args:
        model: mujoco.MjModel
        data:  mujoco.MjData (will be modified — restore state from log)
        sim_log: SimLog with .snapshots list of (t, q, v, label) tuples
        output_dir: where to save PNGs
    """
    renderer = mujoco.Renderer(model, height=height, width=width)
    
    for i, (t, qpos, qvel, label) in enumerate(sim_log.snapshots):
        # Restore MuJoCo state to this instant
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        mujoco.mj_forward(model, data)
        
        # Render
        renderer.update_scene(data, camera=camera)
        frame = renderer.render()  # (H, W, 3) uint8
        
        # Save
        img = Image.fromarray(frame)
        img.save(f"{output_dir}/snap_{i:02d}_t{t:.2f}_{label}.png")
    
    renderer.close()
```

**SimLog addition:** during simulation, `sim_loop.py` appends `(t, qpos.copy(), qvel.copy(), label_str)` to `sim_log.snapshots` at the trigger instants listed above. This stores the full state needed to reconstruct the frame, so rendering can happen post-hoc (no renderer needed during the sim loop itself).

**Camera setup:** define a named camera in the MJCF that tracks the robot or gives a good overall view. If no named camera exists, use the default free camera with `lookat` set to the platform CoM and `distance` ≈ 3 m.

**Environment:** requires `MUJOCO_GL=osmesa` for headless rendering (Claude Code has no display). Install `apt-get install libosmesa6-dev` if needed, and `pip install Pillow --break-system-packages`.

**Composite figure (optional):** after individual snapshots, assemble a contact sheet (4×3 grid) as `fig8_snapshots_grid.png` for quick overview in a single image.

**All figures:** saved as PNG to `output_dir/fig{N}_{name}.png`. `tight_layout()`. Consistent font size (10pt). Phase shading via `axvspan` with alpha=0.1.

### M0.5 — `runner.py`: `run_diagnostics(log_data, output_dir, thresholds=None)`

```python
def run_diagnostics(log_data, output_dir, thresholds=None):
    """
    Single entry point. Call at end of every simulation.

    1. Compute metrics
    2. Print summary table (with PASS/FAIL per row)
    3. Save metrics to output_dir/metrics.csv
    4. Generate all 8 figures to output_dir/
    5. Render MuJoCo snapshots (if model/data provided)
    6. Return metrics dict

    Usage:
        from crawlbot.diagnostics import run_diagnostics
        metrics = run_diagnostics(sim_log, 'results/M3_nmpc_standalone/')
    """
```

### M0 pass criteria

- `run_diagnostics()` runs without error on the **existing** simulation pipeline (current `sim_loop.py` with current controller).
- Produces all 8 figures and a metrics CSV.
- Metrics table prints to console with PASS/FAIL flags.
- If some time series are not yet logged (e.g., `R_ee_ref` before 6D swing planner exists), the corresponding plots show "N/A — not logged" and the metrics are marked "SKIPPED".

### M0 files to create/modify

| Action | File | Description |
|---|---|---|
| CREATE | `crawlbot/diagnostics/__init__.py` | re-export `run_diagnostics` |
| CREATE | `crawlbot/diagnostics/metrics.py` | `compute_metrics()` + `DEFAULT_THRESHOLDS` |
| CREATE | `crawlbot/diagnostics/plots.py` | `generate_plots()` — 8 fixed figures |
| CREATE | `crawlbot/diagnostics/snapshots.py` | `capture_snapshots()` — MuJoCo offscreen rendering |
| CREATE | `crawlbot/diagnostics/runner.py` | `run_diagnostics()` entry point |
| MODIFY | `crawlbot/simulation/logging.py` | Add missing time series to `SimLog` |
| MODIFY | `crawlbot/simulation/sim_loop.py` | Populate new `SimLog` fields during sim |
| CREATE | `tests/test_diagnostics.py` | Smoke test with synthetic log data |

---

## M1. Mapping Layer

**Spec reference:** §4.4, §4.5 (brainstorming doc)

### What to implement

A new module `crawlbot/core/com_to_torso_mapping.py` that converts NMPC centroidal outputs to torso references:

```python
class CoMToTorsoMapping:
    def __init__(self, robot_interface):
        """Stores robot_interface for FK/mass access."""

    def compute(self, r_com_ref, v_com_ref, a_com_ff, q_current, dq_current):
        """
        Returns:
            r_b_ref (3,): torso position reference
            v_b_ref (3,): torso linear velocity reference
            a_b_ff  (3,): torso linear acceleration feedforward
            delta   (3,): Σ m_i r_i for non-torso bodies (for monitoring)
        """
        # δ = Σ_{i≠torso} m_i · r_i(q)
        # r_b_ref = (m_total/m_b) · r_com_ref − (1/m_b) · δ
        # v_b_ref = (m_total/m_b) · v_com_ref   [drop δ_dot initially]
        # a_b_ff  = (m_total/m_b) · a_com_ff
```

### Pass criteria (T3 from validation suite)

**T3 — Jacobian equivalence test:**
- At 10 random configurations, verify that tracking `r_b_ref` with `J_torso_pos` is equivalent to tracking `r_com_ref` with `J_com`:
  - Compute `J_b_pos` and `(m_total/m_b) * J_com - (1/m_b) * Σ m_i J_i`
  - Error: `‖J_b_pos - J_com_mapped‖ < 1e-10`
- At 10 random configurations with random `r_com_ref`, verify the round-trip:
  - Compute `r_b_ref` from mapping
  - Compute CoM from FK at a configuration where torso is at `r_b_ref`
  - Error should be small (limited by linearization, not by the mapping itself)

### Files

| Action | File |
|---|---|
| CREATE | `crawlbot/core/com_to_torso_mapping.py` |
| CREATE | `tests/test_mapping_layer.py` |

---

## M2. Reworked QP Task Stack

**Spec reference:** §5.6, §5.7 (brainstorming doc)

### What to modify

`crawlbot/solvers/whole_body_qp.py` — change the task stack from:

**Current:** CoM (P1) + Torso 6D (P1) + EE 3D–6D (P2) + posture (P3)

**New:** Torso 6D (P1) + EE 6D (P2) + posture (P3) + soft CoM residual (cost term)

Key changes:
- Remove explicit CoM task from priority stack
- EE becomes full 6D (pos + ori) — requires orientation reference (stub until M5)
- Add soft CoM quadratic cost: `α_com_soft * ‖J_com * ddq - ddx_com_des‖²`
- `ddx_com_des` comes from NMPC (ff + PD), **not** from the mapping
- Passivity constraint `dq^T τ_q + 2αT ≤ 0` active only during DS
- Phase-dependent gains (`α_ee` higher during approach)

### Pass criteria

**T7** — Torso 6D + EE 6D null-space tracking (SS, standalone QP, no sim):
- Torso position error < 5 mm, EE position error < 10 mm, EE orientation error < 5°

**T8** — Soft CoM residual effect:
- Run same trajectory with `α_com_soft = 0` vs. `α_com_soft = 5`
- CoM tracking RMS improves with soft cost ON

**T9** — Dynamics residual:
- `‖H*ddq + C*dq - B*τ - J_c^T * λ‖ < 1e-8` at every QP step

**T10** — DS passivity:
- `T(t) ≤ T(t0) * exp(-2α(t-t0))` within 5% for 3 seconds
- Zero passivity violations

→ **Run diagnostics** after each test. Check Fig 1 (tracking) and Fig 4 (passivity).

### Files

| Action | File |
|---|---|
| MODIFY | `crawlbot/solvers/whole_body_qp.py` |
| CREATE | `tests/test_reworked_qp.py` |

---

## M3. NMPC with Corrected Conservation Law

**Spec reference:** §4.1 B2 formulation, §4.6 incremental form

### What to modify

`crawlbot/solvers/centroidal_nmpc.py` — the RWA momentum constraint must use the conservation law from §4.5–4.6, not the current (buggy) formulation.

Key changes:
- Compute `c = h_w^{s,0} + L_robot^{O_s,s,in,0}` at each NMPC call
- `L_robot^{O_s,s,in,0}` uses the full decomposition from §4.4 (spin + orbital + drag)
- Start with Option B (neglected drag, tightened box): `c_simple - L_com(k) - r_com(k) × m*v_com(k) ∈ [-h_max', h_max']`
- The cross-product `r_com(k) × m*v_com(k)` is bilinear in state — CasADi handles this natively
- Add `L_com_ref(k)` from TorsoPlanner to cost: `w_L * ‖L_com - L_com_ref‖²` (stub to zero until M5)
- **Reset warm start** at phase transitions (DS→SS, SS→DS)

### What to verify before modifying

READ `centroidal_nmpc.py` and `nmpc_solver.py` line by line. Document:
1. Current state vector — does it match B2 (9-state)?
2. Current constraint formulation — what exactly is the hw constraint?
3. Where is `h_w` computed? Is it a state variable or algebraic?
4. What are `hw_min/max` defaults? (Known bug: were ±50 Nms instead of ±5)
5. What CasADi variables exist? What solver options?

### Pass criteria

**T4** — B2 NMPC from rest, 0.3 m displacement, `h_max = 5 Nms`:
- Converges to target
- `h_w(k) ∈ [-h_max, h_max]` at all knots

**T5** — Position-dependent envelope: 0.5 m vs. 3.0 m from O_p:
- Step time at 3.0 m is longer (transverse velocity limited by larger lever arm)

**T6** — Two consecutive steps with terminal margin κ = 0.7:
- Terminal `h_w ∈ [-0.7*h_max, 0.7*h_max]` at end of each step

→ **Run diagnostics** (standalone NMPC: Fig 2a is the key figure — h_w vs. bounds).

### Files

| Action | File |
|---|---|
| MODIFY | `crawlbot/solvers/centroidal_nmpc.py` |
| MODIFY | `crawlbot/solvers/nmpc_solver.py` (if solve interface changes) |
| CREATE | `tests/test_nmpc_conservation.py` |

---

## M4. Corrected AOCS Feedforward

**Spec reference:** §5.8

### What to modify

`crawlbot/aocs/force_estimator.py` — add the missing orbital term:

```
τ_w = -dL_com_est - r_com × m * dv_com_est - K_hw * (h_w - clip(h_w, bounds))
```

The orbital term `r_com × m * dv_com_est` was missing — likely the cause of the 24° platform rotation at 14% mass ratio.

### Pass criteria

**T13** — AOCS feedforward accuracy:
- Run a single step at 14% mass ratio
- Compare platform rotation with and without orbital correction
- Feedforward torque error (compared to actual disturbance) < 0.1 Nm RMS

→ **Run diagnostics.** Key figure: Fig 2d (platform attitude). The 24° drift should drop to < 5°.

### Files

| Action | File |
|---|---|
| MODIFY | `crawlbot/aocs/force_estimator.py` |
| CREATE | `tests/test_aocs_orbital.py` |

---

## M5. TorsoPlanner + SwingPlanner 6D

**Spec reference:** §5.3 (TorsoPlanner), §6.3 (SwingPlanner 6D)

### TorsoPlanner refactor

`crawlbot/planning/torso_planner.py`:
- Position reference now comes from the NMPC via the mapping layer (M1), **not** from TorsoPlanner
- TorsoPlanner generates only: `R_ref(t)`, `ω_ref(t)`, `α_ref(t)` via SLERP with quintic timing
- Plus: `L_com_ref(t) = I_torso_com * ω_ref(t)` for NMPC feedforward

### SwingPlanner 6D

`crawlbot/planning/swing_planner.py`:
- Add orientation trajectory: `R_ee(t) = SLERP(R_start, R_dock, σ(t))`
- Delayed cosine timing: `σ(t)` concentrates rotation in second half
- Output full 6D reference: `[p_ee, R_ee, dp_ee, ω_ee]`

### Pass criteria

- TorsoPlanner: `L_com_ref` at peak angular velocity matches `I_torso * ω_peak` to < 5%
- SwingPlanner: `R_ee(T) = R_dock` exactly (to numerical precision)
- Angular velocity reference `ω_ee_ref` is smooth (no discontinuities at t_delay)

→ **Run diagnostics.** Fig 1c/d should now show EE orientation tracking (was N/A before).

### Files

| Action | File |
|---|---|
| MODIFY | `crawlbot/planning/torso_planner.py` |
| MODIFY | `crawlbot/planning/swing_planner.py` |
| CREATE | `tests/test_planners_6d.py` |

---

## M6. Coarse Pre-Planner

**Spec reference:** §6.2

### What to create

New module `crawlbot/planning/coarse_preplanner.py`:
- Solves a trajectory optimization over the full step horizon (M ≈ 15 collocation points)
- Centroidal ODE with one active contact (single-support)
- Momentum box constraint at every collocation point
- Terminal momentum margin κ
- Outputs: coarse CoM trajectory + step duration T_step
- Runs **once per step** before step starts

### Pass criteria

- Solves in < 500 ms for M = 15 collocation points
- Output trajectory satisfies momentum box at all collocation points
- Terminal constraint: `h_w ∈ [-κ*h_max, κ*h_max]`
- Position-dependent effect visible: trajectory at 3 m from O_p is slower than at 0.5 m

### Files

| Action | File |
|---|---|
| CREATE | `crawlbot/planning/coarse_preplanner.py` |
| CREATE | `tests/test_coarse_preplanner.py` |

---

## M7. Closed-Loop Integration — Two-Phase State Machine

**Spec reference:** §7 (Operational Design), §8 (Validation T11–T20)

### Architectural decision: eliminate the EXT phase

The legacy code has three phases: DS → SS → EXT. The EXT phase was a workaround for trajectory desynchronization — the torso was still moving when the swing arm reached the target, so a separate "close the gap" phase was added with special gains, threshold-based triggers, and reference freezing. All of these are patches for a coordination problem that shouldn't exist.

**The reworked architecture has exactly two phases: DS and SS.** This mirrors the terrestrial legged locomotion convention (Belvedere et al., Henze et al., Mishra et al.). No EXT phase. No gain scheduling. No approach thresholds. No torso freezing.

The key insight: in terrestrial bipedal walking, nobody has a special "foot landing" phase with gain scheduling. The swing foot trajectory is planned to arrive at the foothold at `t = T_step` with zero velocity — and it does, because the torso and swing trajectories are **synchronized over the same time horizon**.

### Synchronized trajectory planning

The root cause of the docking failure was trajectory desynchronization: the swing planned over 6s, the torso over 14.8s. The arm arrived first, the torso kept moving, and the arm got dragged away. The fix is not threshold-based freezing — it is trajectory synchronization.

**Per-step planning sequence (runs once at DS exit, before SS entry):**

```
1. CoarsePrePlanner(start_state, goal_com, contacts) → T_step, com_trajectory
   - T_step is set by the momentum envelope (position-dependent feasibility)
   - com_trajectory is momentum-feasible by construction

2. TorsoPlanner(R_start, R_goal, T_step)
   - Orientation: SLERP with quintic timing over [0, T_step]
   - Position: comes from NMPC via mapping layer (not planned here)
   - L_com_ref(t) = I_torso · ω_ref(t) for NMPC feedforward
   - Terminal: ω_ref(T_step) = 0, R_ref(T_step) = R_goal

3. SwingPlanner(p_start, p_dock, R_start, R_dock, T_step)
   - Position: quintic + clearance bump over [0, T_step]
   - Orientation: SLERP with quintic timing over [0, T_step]
   - Terminal: v_ee(T_step) = 0, ω_ee(T_step) = 0
   - Both position and orientation arrive at T_step simultaneously
```

**Critical: all three planners use the same `T_step`.** The torso and swing arm both arrive at their targets at the same time. No separate timing. No EXT phase needed.

### Two-phase state machine

#### Phase 1: DS (Double Support — both end-effectors welded)

- **Purpose:** Settle after dock; prepare next step
- **QP:** `passivity_active=True`, `dq^T τ_q + 2αT ≤ 0`
- **Exit condition:** energy-based: `T < T_settle = 0.5 · ε_v² · λ_min(H)`
  - NOT time-based. The passivity constraint guarantees exponential decay.
  - Safety cap: `t_settle_max` to prevent infinite hang if T_settle is unreachable
- **On exit:**
  1. Reset NMPC warm start (`nmpc.reset_warm_start()`)
  2. Compute `c = h_w_0 + L_robot_inertial_0` for the next NMPC horizon
  3. Run coarse pre-planner → get `T_step` and momentum-feasible CoM trajectory
  4. Plan torso orientation over `[0, T_step]`
  5. Plan swing 6D trajectory over `[0, T_step]`
  6. Release stance weld on the swing arm → enter SS

#### Phase 2: SS (Single Support — swing arm free)

- **Purpose:** Execute the locomotion step — move torso + dock swing arm
- **QP:** `passivity_active=False` (energy injection needed for locomotion)
  - Torso 6D (P1): position from NMPC→mapping, orientation from TorsoPlanner
  - EE 6D (null-space projected): position + orientation from SwingPlanner
  - Posture (null-space projected): manipulability optimization
  - Soft CoM residual: NMPC feedforward + PD
  - hw slack: soft momentum safety
- **NMPC:** 10 Hz, conservation law constraint, `L_com_ref` from TorsoPlanner
- **Duration:** `T_step` (from coarse pre-planner)
- **Exit conditions:**
  - **Dock success:** `‖p_ee - p_dock‖ < 5 mm` AND `‖Log(R_ee^T R_dock)‖ < 5°` → activate weld → DS
  - **Timeout:** `t > T_step + T_margin` without docking → hold position, attempt convergence for `T_hold_max` seconds, then abort step
  - **NMPC infeasible:** use warm-shifted fallback (shift previous plan by one step)
  - **Tracking divergence:** if `‖r_com - r_com_ref‖ > d_abort` → pause and report

### What to delete from sim_loop.py

- The entire EXT phase branch
- The `qp_approach` variant and its gain overrides
- The torso-freeze logic (`d < 10mm` threshold)
- The `t_ext_max` timeout parameter
- The gain scheduling / approach-band logic
- Any `phase == 'EXT'` conditionals

### What to modify in sim_loop.py

- Rewrite `_step()` to have exactly two branches: DS and SS
- SwingPlanner and TorsoPlanner both receive `T_step` from the coarse pre-planner
- The NMPC interpolation (10 QP steps per NMPC call) stays as implemented in M5
- The NMPC infeasibility fallback (warm-shift) stays as implemented in M5
- Inter-step DS settling uses the passivity-constrained QP with energy-based exit (same as setup settling)

### Pass criteria

**T11** — Single step, 1% mass ratio: dock < 5 mm / 5°, `h_w` in box, 7-DOF arms.
**T12** — Single step, 14% mass ratio: dock < 5 mm / 5°, platform rotation < 5°.
**T14** — DS settling within ±20% of theoretical `t_settle`.
**T15/T16** — 3-step traversal at 1% and 14%.
**T17** — EE orientation at dock < 5° (confirmed by the 7-DOF upgrade).
**T18** — NMPC > 95% solve rate within 50 ms.
**T19/T20** — Zero QP failures, dynamics residual < 1e-8 across full traversal.

→ **Run diagnostics** on every closed-loop sim. All 8 figures, full metrics table.

### Files

| Action | File |
|---|---|
| MODIFY | `crawlbot/simulation/sim_loop.py` — rewrite phase machine, delete EXT |
| MODIFY | `crawlbot/simulation/config.py` — remove EXT parameters, add T_step sync |
| MODIFY | `crawlbot/planning/swing_planner.py` — accept T_step, synchronize with torso |
| MODIFY | `crawlbot/planning/torso_planner.py` — accept T_step from pre-planner |
| MODIFY | `crawlbot/planning/coarse_preplanner.py` — output T_step as primary duration |
| CREATE | `scripts/run_reworked_single_step.py` |
| CREATE | `scripts/run_reworked_3step.py` |
| CREATE | `tests/test_closed_loop_reworked.py` |

---

## M8. Unit Tests for Momentum Algebra (T1–T2)

**Spec reference:** §8.1

These are pure math tests — no simulation, no controller. They validate the conservation law implementation before it's used in the NMPC.

**T1** — `L_robot^{O_s,s,in}` (§4.4) matches Pinocchio + transport + drag, at 5 random configs. Error < 1e-10.

**T2** — Conservation law: algebraic `h_w` vs. integrated `h_w` over 1 s of random joint motion. Drift < 1e-6 Nms.

### Files

| Action | File |
|---|---|
| CREATE | `tests/test_momentum_algebra.py` |

---

## Milestone Dependency Graph

```
M-1 (codebase cleanup)
 │
 └── M0 (diagnostics)
      │
      ├── M8 (momentum algebra — pure math, no deps)
      │
      ├── M1 (mapping layer)
      │    │
      │    └── M2 (QP rework) ←── needs mapping to provide r_b_ref
      │         │
      │         └── M3 (NMPC conservation law)
      │              │
      │              ├── M4 (AOCS correction)
      │              │
      │              ├── M5 (TorsoPlanner + SwingPlanner 6D)
      │              │
      │              └── M6 (Coarse pre-planner)
      │                   │
      │                   └── M7 (Closed-loop integration)
```

**M-1 then M0 are mandatory first.** M8 can run in parallel with M1. M4, M5, M6 can be developed somewhat independently after M3, but all feed into M7.

---

## Open Questions (carry forward from spec)

These are **not** blocking for implementation but need decisions before the paper:

- [ ] `α_com_soft` optimal value — simulation sweep: 1, 5, 10
- [ ] Passivity `α` tuning — simulation sweep
- [ ] Warm-start reset strategy — full reset vs. shifted rollout
- [ ] 7-DOF arm URDF/MJCF — **blocker for final validation** (current tests use 6-DOF)
- [ ] Numerical parameter table consolidation
- [ ] CoM tracking error budget for momentum safety margin
- [ ] Level 2 energy budget during SS

---

## Reference: Diagnostic Invocation Pattern

Every test script ends with:

```python
from crawlbot.diagnostics import run_diagnostics

# ... run simulation, collect sim_log ...

metrics = run_diagnostics(
    sim_log,
    output_dir='results/M3_nmpc_standalone/',
    thresholds=None  # uses defaults; override per-test if needed
)

# Optional: assert specific metrics for CI
assert metrics['hw_saturation_ratio_peak']['pass'], \
    f"hw saturation {metrics['hw_saturation_ratio_peak']['value']:.2f} exceeds limit"
```
