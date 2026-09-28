# `crawlbot.control`

**The controller as one block: measurements in, actuator commands out.**

Created by the `refactor/sim-loop-split` chantier to separate the control stack
from the simulation loop that used to host it, ahead of a ROS 2 port. The
solvers themselves stay in [`solvers/`](../solvers/solvers.md) and the AOCS laws
in [`aocs/`](../aocs/aocs.md); this package holds the blocks that *drive* them.

| file | document |
|---|---|
| `controller.py` | [controller.md](controller.md) |
| `attitude.py` | [attitude.md](attitude.md) |

## Role

Three blocks, per spec §4: the centroidal NMPC (10 Hz), the whole-body QP
(100 Hz) and the AOCS. They read the plant only through
[`SensorSuite`](../simulation/sensors.md) and act on it only through the command
they return, which the loop applies via [`MujocoPlant`](../simulation/plant.md).
