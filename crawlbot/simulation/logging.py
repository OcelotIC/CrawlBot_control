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

    def _host():
        """CPU / memory / kernel of the machine producing this run (C2.2.4).

        Wall-clock solver timings move ~25 % between machines while iteration
        counts do not (review-closure C3.4 §3), so a timing number is only
        citable next to the hardware that produced it — and no artifact
        committed before this field existed records one.
        """
        import platform
        info = {
            'platform':     platform.platform(),
            'machine':      platform.machine(),
            'processor':    platform.processor(),
            'cpu_count_logical': os.cpu_count(),
        }
        # Model name and physical core count are not in `platform`; read the
        # two files that carry them. Both are best-effort and Linux-only.
        try:
            with open('/proc/cpuinfo') as fh:
                for line in fh:
                    if line.startswith('model name'):
                        info['cpu_model'] = line.split(':', 1)[1].strip()
                        break
        except Exception:                             # noqa: BLE001
            pass
        try:
            with open('/proc/meminfo') as fh:
                for line in fh:
                    if line.startswith('MemTotal'):
                        info['mem_total'] = line.split(':', 1)[1].strip()
                        break
        except Exception:                             # noqa: BLE001
            pass
        return info

    return {
        'python_version':    sys.version,
        'mujoco_version':    _safe(lambda: _import_version('mujoco')),
        'pinocchio_version': _safe(lambda: _import_version('pinocchio')),
        'numpy_version':     _safe(lambda: _import_version('numpy')),
        'numpy_show_config': _safe(_show_config),
        'pip_freeze':        _safe(_pip_freeze),
        'env_vars': {k: os.environ.get(k) for k in _ENV_VAR_NAMES},
        'host':              _safe(_host),
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

    # Phase-2.1: optional 100 Hz (QP-rate) SS logging of reaction-wheel torque
    # + stored wheel momentum. Populated ONLY when cfg.log_hifreq_ss is True;
    # empty by default ⇒ no effect on the existing 10 Hz logs and flag-OFF
    # bit-identical behaviour. Resolves the SS τ_w/h_w cadence ambiguity.
    t_ss_hifreq: list = field(default_factory=list)
    tau_w_ss_hifreq: list = field(default_factory=list)
    hw_ss_hifreq: list = field(default_factory=list)

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
    # Q1: actual contact wrench the QP delivered (full 12-vec
    # [f1, τ1, f2, τ2]) and its Euclidean norm. Logged per WBC tick.
    lambda_qp: list = field(default_factory=list)
    lambda_qp_norm: list = field(default_factory=list)
    nmpc_time_ms: list = field(default_factory=list)
    qp_time_ms: list = field(default_factory=list)
    nmpc_status: list = field(default_factory=list)     # 0=ok, 1=max_iter, 2=infeasible
    nmpc_cost: list = field(default_factory=list)       # NMPC objective value
    nmpc_status_str: list = field(default_factory=list) # IPOPT return string
    nmpc_iterations: list = field(default_factory=list) # IPOPT iter count

    # ── C2.2.1: true whole-body-QP solve statistics, per logged tick ──────
    # `qp_time_ms` above is NOT a QP solve time: its timer spans the whole
    # WBC block (ten QP solves plus ten Pinocchio updates, ten AOCS
    # evaluations, ten mj_step calls and this logging). These five are the
    # QP itself, aggregated over the `n_qp_per_nmpc` solves inside one tick,
    # from the `QPSolveInfo` that `HierarchicalQP.solve` already produced and
    # `sim_loop` used to discard.
    #
    # SENTINEL CONVENTION — these are measurements only where a QP ran under
    # a recorder that collects them. Where one did not, the value is a
    # sentinel and must be excluded from any statistic, exactly as
    # nmpc_time_ms/qp_time_ms are on NMPC-bypassed ticks:
    #     qp_solve_ms_sum / qp_solve_ms_max = 0.0
    #     qp_iter_sum = 0 ,  qp_n_solves = 0 ,  qp_n_failed = 0
    #     qp_status_worst = -1        (-1 = NOT MEASURED, never an outcome)
    # `qp_n_solves == 0` is the unambiguous test for "sentinel row"; prefer
    # it over testing the timers against 0.0.
    qp_solve_ms_sum: list = field(default_factory=list)
    qp_solve_ms_max: list = field(default_factory=list)
    qp_iter_sum: list = field(default_factory=list)
    qp_n_solves: list = field(default_factory=list)
    qp_n_failed: list = field(default_factory=list)
    # Worst backend outcome over the solves in this tick. Ordering, ascending
    # severity:  -1 not measured · 0 backend reported success · 1 backend
    # reported NOT success (stats()['success'] is False — this is the case
    # `qp_ok` structurally cannot see, because `error_on_fail: False` means an
    # infeasible QP returns normally instead of raising) · 2 the solve raised.
    # "Worst" = max over the tick, with -1 ranking below 0.
    qp_status_worst: list = field(default_factory=list)

    # ── C2.3: AOCS wheel-torque decomposition, per axis, per logged tick ──
    # `tau_w` records only the sum, so no artifact could say which term drove
    # a given wheel torque. These are the five contributions that form it in
    # compute_aocs_command_legacy_pid_numerical, plus the pre-clip total:
    #
    #   tau_w_preclip = tau_ff + tau_att_p + tau_rate_d + tau_accel_d
    #                   + tau_antiwindup
    #   tau_w         = clip(tau_w_preclip, ±aocs_tau_w_max)
    #
    # The identity holds by construction (the logged objects ARE the summands),
    # so a reader can check it rather than re-derive it. `tau_w_preclip` vs
    # `tau_w` is what makes the saturation fraction measurable — on the managed
    # canonical the clip is the difference between demand and applied torque.
    #
    # `tau_antiwindup` is K_hw·(sat(h_w) − h_w) and is identically zero
    # whenever |h_w| ≤ hw_max, which is every tick of the frozen canonical
    # (|h_w|_∞ = 4.1019 < 5) — a fact the paper asserts and this channel now
    # demonstrates per-tick rather than by inference from the h_w peak.
    #
    # Same sentinel discipline as the qp_* channels: on ticks where the AOCS
    # did not run under a recorder that collects them, all six are zeros and
    # `aocs_decomp_measured` is 0. Test THAT flag, not the values — zero is a
    # legitimate measurement for four of the six terms.
    aocs_tau_ff: list = field(default_factory=list)
    aocs_tau_att_p: list = field(default_factory=list)
    aocs_tau_rate_d: list = field(default_factory=list)
    aocs_tau_accel_d: list = field(default_factory=list)
    aocs_tau_antiwindup: list = field(default_factory=list)
    aocs_tau_w_preclip: list = field(default_factory=list)
    aocs_decomp_measured: list = field(default_factory=list)
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

    # Fix C (J2 #1): per-gate-evaluation trace of the 6-D weld-relative
    # twist ‖Jc·v⁻‖ + pose (d, ori) at each dock-gate check (once the
    # swing trajectory is done). Supports the Part-2 "does (b) drive the
    # gate" trajectory and the Part-3 firing/timeout characterization.
    # Each dict: {t, step, d_mm, ori_deg, twist, fired}. Empty on legacy
    # logs (default_factory []).
    dock_gate_trace: list = field(default_factory=list)

    # α (J2 #2): per-tick CoM-mobile DS conflict trace (DWELL only, gated on
    # cfg.ds_mobile_com_magnitude>0). Each dict: {t, com_err, pass_resid,
    # Hdot_inf, qp_ok, nmpc_status}. pass_resid≈0 ⇒ passivity binding;
    # Hdot_inf→τ_w_max ⇒ envelope binding; qp_ok False / nmpc_status 2 ⇒
    # feasibility loss. Empty by default.
    ds_mobile_trace: list = field(default_factory=list)

    # Dock-floor passivity audit: per-SS-tick trace (gated cfg.log_dock_work).
    # Each dict: {t, step, d_mm, dq_tau, pass_active}. dq_tau = dqⱼᵀτ_q (joint
    # mechanical power; >0 ⇒ the arm does positive work) — confirms whether
    # the relaxation is exercised in the dock-close window. Empty by default.
    dock_work_trace: list = field(default_factory=list)

    # M7: aborted steps (pre-planner infeasible or dock timeout).
    # Each dict carries: step_idx, t, reason ('preplanner_infeasible'
    # | 'dock_timeout'), and — for dock_timeout — d_mm, ori_deg.
    aborted_steps: list = field(default_factory=list)

    # M7: one T_step per SS phase, in the order they occur.  Used by
    # the per-axis tracking plots to draw τ=0.5 / τ=1.0 markers.
    preplanner_T_steps: list = field(default_factory=list)

    # C2.2.3: one record per coarse-pre-planner NLP solve (one per step), from
    # SimulationLoop._preplanner_stats. Each dict: {success, solve_ms,
    # iter_count, cost, status, t_plan_start, T_step}. These are IPOPT NLP
    # solves that gate every step; they were collected and printed but never
    # persisted, so no committed artifact carries them and a solver table
    # built from the logs alone omitted them entirely.
    preplanner_stats: list = field(default_factory=list)

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
