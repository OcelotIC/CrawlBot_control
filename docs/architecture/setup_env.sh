#!/bin/bash
# === CrawlBot environment setup ===
# Run ONCE at the start of every Claude Code session.
# Referenced by: CLAUDE.md, docs/architecture/CLAUDE_CODE_HANDOFF.md

set -e

echo "=== CrawlBot environment setup ==="

# 1. System dependencies (for MuJoCo offscreen rendering)
echo "[1/7] System dependencies..."
sudo apt-get update -qq && sudo apt-get install -y -qq \
    libosmesa6-dev libgl1-mesa-glx libglfw3 > /dev/null 2>&1 || \
    echo "WARNING: Some system deps failed — osmesa rendering may not work"

# 2. Python packages — PINNED VERSIONS
echo "[2/7] Python packages..."
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
echo "[3/7] Repository..."
if [ ! -d "/home/claude/CrawlBot_control" ]; then
    echo "  Cloning CrawlBot_control..."
    git clone https://github.com/OcelotIC/CrawlBot_control.git /home/claude/CrawlBot_control
else
    echo "  Repository already present"
fi

# 5. Copy architecture docs into repo if not already there
echo "[4/7] Architecture docs..."
ARCH_DIR="/home/claude/CrawlBot_control/docs/architecture"
mkdir -p "$ARCH_DIR"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
for f in CLAUDE_CODE_HANDOFF.md brainstorming_reworked_architecture.md; do
    if [ -f "$SCRIPT_DIR/$f" ] && [ ! -f "$ARCH_DIR/$f" ]; then
        cp "$SCRIPT_DIR/$f" "$ARCH_DIR/$f"
        echo "  Copied $f"
    fi
done
# Copy CLAUDE.md to repo root if not present
if [ -f "$SCRIPT_DIR/../CLAUDE.md" ] && [ ! -f "/home/claude/CrawlBot_control/CLAUDE.md" ]; then
    cp "$SCRIPT_DIR/../CLAUDE.md" "/home/claude/CrawlBot_control/CLAUDE.md"
    echo "  Copied CLAUDE.md to project root"
fi

# 6. Verify critical imports
echo "[5/7] Verifying Python imports..."
python3 -c "
import pinocchio as pin; print(f'  Pinocchio {pin.__version__}')
import mujoco; print(f'  MuJoCo    {mujoco.__version__}')
import casadi; print(f'  CasADi    {casadi.__version__}')
import numpy as np; print(f'  NumPy     {np.__version__}')
import qpsolvers; print('  qpsolvers OK')
assert pin.__version__.startswith('3.'), f'ERROR: Need Pinocchio 3.x, got {pin.__version__}'
print('  All imports OK')
"

# 7. Verify MuJoCo offscreen rendering
echo "[6/7] Verifying MuJoCo offscreen rendering..."
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
print('  Offscreen rendering OK')
" 2>/dev/null || echo "  WARNING: Offscreen rendering failed — snapshots will be disabled (MUJOCO_GL=disabled)"

# 8. Smoke test
echo "[7/7] Smoke test (existing tests)..."
cd /home/claude/CrawlBot_control
PYTHONPATH=. MUJOCO_GL=osmesa python3 -m pytest tests/ -x -q --tb=short 2>&1 | tail -5

echo ""
echo "==============================="
echo "  Environment ready"
echo "  PYTHONPATH=/home/claude/CrawlBot_control"
echo "  MUJOCO_GL=osmesa"
echo "==============================="
