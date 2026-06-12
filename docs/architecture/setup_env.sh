#!/bin/bash
# === CrawlBot environment setup ===
# Run ONCE at the start of every Claude Code session.
# Referenced by: CLAUDE.md, docs/architecture/CLAUDE_CODE_HANDOFF.md
#
# Hardening notes (2026-06):
#   1. The PyPI package named "pinocchio" is a DEPRECATED 0.x project
#      UNRELATED to the real Pinocchio dynamics library. The codebase
#      requires Pinocchio 3.x, whose PyPI package is named "pin".
#      We aggressively uninstall any stale "pinocchio" 0.x and verify
#      post-install that `import pinocchio` resolves to 3.x.
#   2. Python packages are installed one at a time so a single source-
#      build failure (e.g. osqp 2.x source build) does not abort the
#      rest of the install.
#   3. Numpy upper-bound (<2) was relaxed — mujoco / pin 3.9.x both
#      support numpy 2.x and the older pin caused unnecessary rebuilds.
#   4. Repo root is auto-detected from the script's own location rather
#      than hard-coded to /home/claude/CrawlBot_control. Both
#      /home/user and /home/claude are supported as a fallback.

set -e

echo "=== CrawlBot environment setup ==="

# Auto-detect the repo root: this script lives at docs/architecture/,
# so the repo root is two parents up.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ ! -d "$REPO_ROOT/.git" ]; then
    # Fallback: scan well-known sandbox paths.
    for CANDIDATE in "/home/user/CrawlBot_control" "/home/claude/CrawlBot_control"; do
        if [ -d "$CANDIDATE/.git" ]; then
            REPO_ROOT="$CANDIDATE"
            break
        fi
    done
fi
echo "  REPO_ROOT=$REPO_ROOT"

# 1. System dependencies — best-effort; never aborts setup.
echo "[1/7] System dependencies (best-effort)..."
{
    sudo apt-get update -qq 2>/dev/null || true
    sudo apt-get install -y -qq \
        libosmesa6-dev libgl1-mesa-glx libglfw3 >/dev/null 2>&1 \
        || echo "  WARNING: osmesa deps unavailable — rendering may need MUJOCO_GL=disabled"
} || true

# 2. Python packages — installed one-at-a-time so a single failure
#    doesn't take out the rest (single atomic install had osqp source-
#    build failures cascade and leave matplotlib/pytest missing).
echo "[2/7] Python packages..."

# CRITICAL FIRST STEP — uninstall the deprecated `pinocchio` 0.x
# PyPI package if present. It provides a totally unrelated module of
# the same name and shadows the real Pinocchio 3.x (PyPI: `pin`),
# which the codebase requires.
echo "  uninstall deprecated 'pinocchio' 0.x (if present)..."
pip uninstall -y -q pinocchio 2>/dev/null || true

PIP="pip install --break-system-packages -q --upgrade-strategy only-if-needed"
echo "  install pin==3.9.0  (the REAL Pinocchio 3.x)"
$PIP "pin==3.9.0"
echo "  install mujoco, casadi, numpy, matplotlib, Pillow, qpsolvers, osqp, pytest, scipy"
$PIP "mujoco>=3.1,<4"
$PIP "casadi>=3.6,<4"
$PIP "numpy>=1.24"                # numpy 2.x is fine with mujoco + pin 3.9.x
$PIP "matplotlib>=3.7"
$PIP "Pillow>=10.0"
$PIP "qpsolvers>=0.9"
# osqp: pin to <2 to force the binary-wheel install (osqp 2.x source
# build fails on this image's pip + setuptools combo with
# FileNotFoundError: requirements.txt).
$PIP "osqp>=0.6,<2"
$PIP "pytest>=7.0"
$PIP "scipy>=1.10"

# 3. MuJoCo rendering backend (headless)
export MUJOCO_GL=osmesa

# 4. Repository — REPO_ROOT was set above. Only clone if completely
#    missing (e.g. brand-new sandbox with neither /home/user nor
#    /home/claude having the repo yet).
echo "[3/7] Repository..."
if [ ! -d "$REPO_ROOT/.git" ]; then
    echo "  Cloning CrawlBot_control to $REPO_ROOT..."
    git clone https://github.com/OcelotIC/CrawlBot_control.git "$REPO_ROOT"
else
    echo "  Repository already present at $REPO_ROOT"
fi

# 5. Architecture docs — copy from this script's dir into the repo
#    (no-op if already present, idempotent).
echo "[4/7] Architecture docs..."
ARCH_DIR="$REPO_ROOT/docs/architecture"
mkdir -p "$ARCH_DIR"
for f in CLAUDE_CODE_HANDOFF.md brainstorming_reworked_architecture.md; do
    if [ -f "$SCRIPT_DIR/$f" ] && [ ! -f "$ARCH_DIR/$f" ]; then
        cp "$SCRIPT_DIR/$f" "$ARCH_DIR/$f"
        echo "  Copied $f"
    fi
done
if [ -f "$SCRIPT_DIR/../CLAUDE.md" ] && [ ! -f "$REPO_ROOT/CLAUDE.md" ]; then
    cp "$SCRIPT_DIR/../CLAUDE.md" "$REPO_ROOT/CLAUDE.md"
    echo "  Copied CLAUDE.md to project root"
fi

# 6. Verify critical imports — including the deprecated-pinocchio guard.
echo "[5/7] Verifying Python imports..."
python3 - <<'PYEOF'
import pinocchio as pin
import mujoco, casadi, qpsolvers
import numpy as np

# CRITICAL: confirm we have the real Pinocchio (3.x), not the
# deprecated `pinocchio` 0.x PyPI package which provides an unrelated
# module of the same name. The 0.x package lacks pin.SE3 /
# pin.Quaternion / pin.computeFrameJacobian — all of which the
# codebase uses.
assert pin.__version__.startswith('3.'), (
    f'ERROR: need real Pinocchio 3.x (PyPI package: pin), got '
    f'{pin.__version__}. The deprecated `pinocchio` 0.x PyPI package '
    f'may be shadowing it. Try: pip uninstall pinocchio && '
    f'pip install pin==3.9.0')
for attr in ('SE3', 'Quaternion', 'computeFrameJacobian',
             'getFrameJacobian', 'log3', 'LOCAL_WORLD_ALIGNED'):
    assert hasattr(pin, attr), (
        f'ERROR: pin.{attr} missing — Pinocchio package mis-resolved? '
        f'Have version {pin.__version__}.')

print(f'  Pinocchio (pin)  {pin.__version__}')
print(f'  MuJoCo           {mujoco.__version__}')
print(f'  CasADi           {casadi.__version__}')
print(f'  NumPy            {np.__version__}')
print(f'  qpsolvers        OK')
print('  All imports OK')
PYEOF

# 7. Verify MuJoCo offscreen rendering (best-effort; non-fatal)
echo "[6/7] Verifying MuJoCo offscreen rendering..."
python3 - <<'PYEOF' 2>/dev/null || \
    echo "  WARNING: Offscreen rendering failed — snapshots will be disabled (MUJOCO_GL=disabled)"
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
print('  Offscreen rendering OK')
PYEOF

# 8. Smoke test
echo "[7/7] Smoke test (existing tests)..."
cd "$REPO_ROOT"
PYTHONPATH=. MUJOCO_GL=osmesa python3 -m pytest tests/ -x -q --tb=short 2>&1 | tail -5

echo ""
echo "==============================="
echo "  Environment ready"
echo "  PYTHONPATH=$REPO_ROOT"
echo "  MUJOCO_GL=osmesa"
echo "==============================="
