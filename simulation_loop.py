"""Compatibility shim — imports from crawlbot.simulation and crawlbot.core.

All code has moved to:
    crawlbot/simulation/config.py      → SimConfig
    crawlbot/simulation/logging.py     → SimLog
    crawlbot/simulation/sim_loop.py    → SimulationLoop
    crawlbot/simulation/plotting.py    → plot_simulation
    crawlbot/core/state_conversions.py → mujoco_to_pinocchio, pinocchio_to_mujoco
"""

# Re-export everything that was previously importable from this module
from crawlbot.simulation.config import SimConfig
from crawlbot.simulation.logging import SimLog
from crawlbot.simulation.sim_loop import SimulationLoop
from crawlbot.simulation.plotting import plot_simulation
from crawlbot.core.state_conversions import (
    mujoco_to_pinocchio,
    pinocchio_to_mujoco,
    quat_wxyz_to_euler_deg,
)
