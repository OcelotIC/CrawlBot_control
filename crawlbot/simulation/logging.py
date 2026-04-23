"""Simulation data logger.

SimLog collects time-series data from each NMPC step for post-processing
and plotting. All fields are lists of scalars or numpy arrays.
"""

import contextlib
import io
import json
import os
import subprocess
import sys

import numpy as np
from dataclasses import dataclass, field


_ENV_VAR_NAMES = ('MUJOCO_GL', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')


def capture_environment() -> dict:
    """Snapshot versions + env vars + BLAS config at simulation start.

    Every probe is best-effort: any exception is recorded as an
    ``<error: ...>`` string so a transient introspection failure cannot
    abort a simulation. Returned keys are stable (always present):

    - ``python_version``           — ``sys.version``
    - ``mujoco_version``           — ``mujoco.__version__``
    - ``pinocchio_version``        — ``pinocchio.__version__``
    - ``numpy_version``            — ``numpy.__version__``
    - ``numpy_show_config``        — output of ``np.show_config()``
    - ``pip_freeze``               — ``python -m pip freeze`` stdout
    - ``env_vars``                 — dict with ``MUJOCO_GL``,
      ``OMP_NUM_THREADS``, ``OPENBLAS_NUM_THREADS`` (``None`` if unset).
    """
    def _safe(fn):
        try:
            return fn()
        except Exception as e:                    # noqa: BLE001
            return f'<error: {type(e).__name__}: {e}>'

    def _import_version(name):
        import importlib
        m = importlib.import_module(name)
        return getattr(m, '__version__', '<no __version__>')

    def _show_config():
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            np.show_config()
        return buf.getvalue()

    def _pip_freeze():
        out = subprocess.run(
            [sys.executable, '-m', 'pip', 'freeze',
             '--disable-pip-version-check'],
            capture_output=True, text=True, timeout=30, check=True)
        return out.stdout

    return {
        'python_version':    sys.version,
        'mujoco_version':    _safe(lambda: _import_version('mujoco')),
        'pinocchio_version': _safe(lambda: _import_version('pinocchio')),
        'numpy_version':     _safe(lambda: _import_version('numpy')),
        'numpy_show_config': _safe(_show_config),
        'pip_freeze':        _safe(_pip_freeze),
        'env_vars': {k: os.environ.get(k) for k in _ENV_VAR_NAMES},
    }


@dataclass
class SimLog:
    """Comprehensive logged data from a simulation run."""

    t: list = field(default_factory=list)
    phase: list = field(default_factory=list)
    step_idx: list = field(default_factory=list)

    # Torso tracking
    p_torso: list = field(default_factory=list)
    p_torso_ref: list = field(default_factory=list)
    e_torso_pos: list = field(default_factory=list)
    e_torso_ori: list = field(default_factory=list)
    q_torso: list = field(default_factory=list)       # actual torso quat (wxyz)
    q_torso_ref: list = field(default_factory=list)    # reference torso quat (wxyz)

    # End-effector
    d_grip_swing: list = field(default_factory=list)
    d_grip_stance: list = field(default_factory=list)
    swing_arm: list = field(default_factory=list)
    p_ee: list = field(default_factory=list)           # EE position actual (3,)
    p_ee_ref: list = field(default_factory=list)       # EE position ref (3,)
    q_ee: list = field(default_factory=list)           # EE orientation actual quat (wxyz)
    q_ee_ref: list = field(default_factory=list)       # EE orientation ref quat (wxyz)

    # Joint, EE, and torso velocities (T15-post-2 instrumentation).
    qvel_joints_a: list = field(default_factory=list)  # arm A joint rates (7,), rad/s
    qvel_joints_b: list = field(default_factory=list)  # arm B joint rates (7,), rad/s
    v_ee_a: list = field(default_factory=list)         # arm A tool linear vel (3,), m/s, struct frame
    v_ee_b: list = field(default_factory=list)         # arm B tool linear vel (3,), m/s, struct frame
    omega_ee_a: list = field(default_factory=list)     # arm A tool angular vel (3,), rad/s
    omega_ee_b: list = field(default_factory=list)     # arm B tool angular vel (3,), rad/s
    v_torso: list = field(default_factory=list)        # torso linear vel (3,), m/s, struct frame
    omega_torso: list = field(default_factory=list)    # torso angular vel (3,), rad/s

    # CoM
    r_com: list = field(default_factory=list)
    r_com_ref: list = field(default_factory=list)
    e_com: list = field(default_factory=list)
    v_com: list = field(default_factory=list)           # CoM velocity actual (3,)
    v_com_ref: list = field(default_factory=list)       # CoM velocity ref (3,)

    # Momentum
    L_com: list = field(default_factory=list)
    L_com_norm: list = field(default_factory=list)
    L_com_ref: list = field(default_factory=list)       # NMPC-planned L_com (3,)
    L_dot: list = field(default_factory=list)
    L_dot_norm: list = field(default_factory=list)
    hw: list = field(default_factory=list)

    # RWA physical
    hw_physical: list = field(default_factory=list)
    tau_w: list = field(default_factory=list)
    rw_speed: list = field(default_factory=list)

    # EE tracking error (vs planned trajectory, not just target distance)
    e_ee_pos: list = field(default_factory=list)
    e_ee_ori: list = field(default_factory=list)

    # GMO contact estimator
    gmo_residual_norm: list = field(default_factory=list)
    gmo_swing_residual: list = field(default_factory=list)
    gmo_contact_state: list = field(default_factory=list)

    # H_{r/O} estimator diagnostics
    H_rO: list = field(default_factory=list)
    H_dot_est: list = field(default_factory=list)
    omega_struct: list = field(default_factory=list)
    qfrc_constraint_torque: list = field(default_factory=list)

    # Joint torques
    tau: list = field(default_factory=list)
    tau_max_joint: list = field(default_factory=list)

    # Structure state
    struct_pos: list = field(default_factory=list)
    struct_quat: list = field(default_factory=list)
    struct_euler_deg: list = field(default_factory=list)
    omega_s: list = field(default_factory=list)         # platform angular velocity (3,)

    # Solver diagnostics
    nmpc_ok: list = field(default_factory=list)
    qp_ok: list = field(default_factory=list)
    lambda_ref_norm: list = field(default_factory=list)
    nmpc_time_ms: list = field(default_factory=list)
    qp_time_ms: list = field(default_factory=list)
    nmpc_status: list = field(default_factory=list)     # 0=ok, 1=max_iter, 2=infeasible
    nmpc_cost: list = field(default_factory=list)       # NMPC objective value
    nmpc_status_str: list = field(default_factory=list) # IPOPT return string
    nmpc_iterations: list = field(default_factory=list) # IPOPT iter count
    transport_term_mag: list = field(default_factory=list)
    # |ω_s × H_{r/O}| per tick, N·m. Diagnostic for Mode B
    # transport-term gap (see AOCS_CONCERN.md).

    # Contact wrenches
    lambda_ref: list = field(default_factory=list)      # NMPC planned wrench (12,)
    lambda_qp: list = field(default_factory=list)       # QP contact wrench (12,)

    # Energy / passivity
    T_kinetic: list = field(default_factory=list)       # 0.5 * dq^T H dq

    # Setup-phase settling (populated by _settle_setup)
    settling_t: list = field(default_factory=list)      # (n,) time [s]
    settling_T: list = field(default_factory=list)      # (n,) kinetic energy [J]
    settling_T_target: float = 0.0                      # T_settle threshold
    settling_stage1_steps: int = 0
    settling_stage2_steps: int = 0
    settling_exit_reason: str = ''

    # Inter-step DS passivity settling (spec §7.1.1). One entry per
    # step transition with energy-based exit. Each dict carries:
    #   step_idx, t_start, t_end, n_steps, T_start, T_end, T_settle,
    #   exit_reason ('target_met' | 'plateau' | 'max_steps')
    inter_step_settles: list = field(default_factory=list)

    # Dock events
    dock_events: list = field(default_factory=list)

    # M7: aborted steps (pre-planner infeasible or dock timeout).
    # Each dict carries: step_idx, t, reason ('preplanner_infeasible'
    # | 'dock_timeout'), and — for dock_timeout — d_mm, ori_deg.
    aborted_steps: list = field(default_factory=list)

    # M7: one T_step per SS phase, in the order they occur.  Used by
    # the per-axis tracking plots to draw τ=0.5 / τ=1.0 markers.
    preplanner_T_steps: list = field(default_factory=list)

    # MuJoCo snapshots for offline rendering
    snapshots: list = field(default_factory=list)       # [(t, qpos, qvel, label)]

    # Environment fingerprint populated by SimulationLoop.run() via
    # capture_environment(). Empty dict on legacy logs that predate this
    # field — SimLog.load() silently drops unknown keys and the
    # default_factory fills in {} for missing ones.
    environment: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            if k == 'snapshots':
                # Snapshots contain large arrays; serialize specially
                d[k] = [(t, q.tolist() if hasattr(q, 'tolist') else q,
                          v_.tolist() if hasattr(v_, 'tolist') else v_, lbl)
                         for t, q, v_, lbl in v]
            elif isinstance(v, list):
                d[k] = [x.tolist() if hasattr(x, 'tolist') else x for x in v]
            else:
                # Scalar fields (e.g. settling_T_target, settling_stage1_steps)
                d[k] = v
        return d

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f)

    @staticmethod
    def load(path: str) -> 'SimLog':
        with open(path) as f:
            d = json.load(f)
        log = SimLog()
        for k, v in d.items():
            if hasattr(log, k):
                setattr(log, k, v)
        return log
