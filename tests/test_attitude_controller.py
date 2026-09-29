"""AttitudeController — history mechanics on synthetic inputs (no simulation).

Freezes what A0 (refactor/sim-loop-split) made explicit: the AOCS owns its
finite-difference history, restarts it every NMPC tick (the KNOWN DEFECT,
docs/crawlbot/control/attitude.md §3) and seeds it from the entry ω_s at an
inter-step settle. A future fix of the defect must change
``test_nmpc_tick_reset_is_the_known_defect`` deliberately.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from crawlbot.control.attitude import AttitudeController
from crawlbot.control.controller import DiagHooks
from crawlbot.simulation.config import SimConfig

DT = 0.01
K_D = 25.0


class _Sensors:
    """Measurement stub: fixed attitude, settable gyro, zero wheel momentum."""

    def __init__(self):
        self.omega = np.zeros(3)

    def omega_struct(self):
        return self.omega.copy()

    def omega_struct_view(self):
        return self.omega

    def struct_quat(self):
        return np.array([1.0, 0.0, 0.0, 0.0])

    def wheel_momentum(self):
        return np.zeros(3)


def _make(diag=None):
    cfg = SimConfig()
    cfg.aocs_mode = 'legacy_pid_numerical'
    cfg.aocs_use_legacy_corrected = False
    cfg.aocs_use_H_estimator = False
    cfg.dt_qp = DT
    # isolate the K_d·ω̇_s term
    cfg.aocs_K_theta = 0.0
    cfg.aocs_K_omega = 0.0
    cfg.aocs_K_hw = 0.0
    cfg.aocs_K_d = K_D
    cfg.aocs_tau_w_max = 1e9
    sensors = _Sensors()
    robot = SimpleNamespace(_total_mass=70.0)
    aocs = AttitudeController(cfg, robot, sensors, None,
                              np.array([1.0, 0.0, 0.0, 0.0]), np.ones(3),
                              diag or DiagHooks())
    return aocs, sensors


def _rs():
    """Constant centroidal state ⇒ zero FD feedforward after the first tick."""
    return SimpleNamespace(L_com=np.zeros(3), r_com=np.zeros(3),
                           v_com=np.zeros(3))


def _cmd(aocs, rs):
    cc = SimpleNamespace(active_contacts=[True, False])
    return aocs.command(phase='SS', rs=rs, lambda_qp_sol=np.zeros(12),
                        cc_nmpc=cc, stance_anchors=(np.zeros(3), np.zeros(3)))


def test_nmpc_tick_reset_is_the_known_defect():
    """First sub-step after reset_for_nmpc_tick sees ω_s,prev = 0 ⇒ the K_d
    term is K_d·ω_s/dt (2500·ω_s); the next sub-step uses the recorded ω_s."""
    aocs, sensors = _make()
    rs = _rs()
    sensors.omega = np.array([1e-4, 0.0, 0.0])
    aocs.reset_for_nmpc_tick(rs)
    tau0, _ = _cmd(aocs, rs)
    assert tau0[0] == pytest.approx(K_D * 1e-4 / DT)      # 0.25 N·m kick
    tau1, _ = _cmd(aocs, rs)                               # same ω_s
    assert np.allclose(tau1, 0.0)
    aocs.reset_for_nmpc_tick(rs)                           # next NMPC tick
    tau2, _ = _cmd(aocs, rs)
    assert tau2[0] == pytest.approx(K_D * 1e-4 / DT)      # kick again


def test_history_records_each_tick():
    aocs, sensors = _make()
    rs = _rs()
    aocs.reset_for_nmpc_tick(rs)
    sensors.omega = np.array([0.0, 2e-4, 0.0])
    tau, _ = _cmd(aocs, rs)
    h = aocs._hist
    assert np.array_equal(h.omega_s_prev, sensors.omega)
    assert h.omega_s_prev is not sensors.omega            # a copy, not the view
    assert np.array_equal(h.tau_w_prev, tau)


def test_disable_aocs_zeroes_command_and_history():
    diag = DiagHooks(disable_aocs=True)
    aocs, sensors = _make(diag)
    rs = _rs()
    aocs.reset_for_nmpc_tick(rs)
    sensors.omega = np.array([1e-4, 0.0, 0.0])
    tau, _ = _cmd(aocs, rs)
    assert np.array_equal(tau, np.zeros(3))
    assert np.array_equal(aocs._hist.tau_w_prev, np.zeros(3))


def test_settle_seeds_from_entry_omega():
    """reset_for_settle ⇒ ω_s,prev = entry ω_s (no kick on the first tick)."""
    aocs, sensors = _make()
    sensors.omega = np.array([3e-4, -1e-4, 2e-4])
    aocs.reset_for_settle()
    assert np.array_equal(aocs._hist.omega_s_prev, sensors.omega)
