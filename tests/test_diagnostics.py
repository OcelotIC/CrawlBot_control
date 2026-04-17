"""Smoke tests for the M0 diagnostic suite.

Uses synthetic SimLog data to verify that compute_metrics, generate_plots,
and run_diagnostics execute without error and produce expected outputs.
"""

import os
import tempfile
import numpy as np
import pytest

from crawlbot.simulation.logging import SimLog
from crawlbot.simulation.config import SimConfig
from crawlbot.diagnostics.metrics import (
    compute_metrics, print_metrics, save_metrics_csv, DEFAULT_THRESHOLDS)
from crawlbot.diagnostics.plots import generate_plots
from crawlbot.diagnostics.runner import run_diagnostics


def _make_synthetic_log(n=100):
    """Create a synthetic SimLog with all fields populated."""
    log = SimLog()
    dt = 0.1
    for i in range(n):
        t = i * dt
        log.t.append(t)
        log.phase.append('DS' if i < 5 else 'SS')
        log.step_idx.append(0)

        # Torso tracking
        p = np.array([0.8 * t / (n * dt), 0.0, -0.45])
        log.p_torso.append(p + np.random.randn(3) * 0.001)
        log.p_torso_ref.append(p)
        log.e_torso_pos.append(float(np.random.rand() * 0.005))
        log.e_torso_ori.append(float(np.random.rand() * 2.0))
        log.q_torso.append(np.array([1.0, 0, 0, 0]) + np.random.randn(4) * 0.01)
        log.q_torso_ref.append(np.array([1.0, 0, 0, 0]))

        # EE
        log.d_grip_swing.append(float(max(0.1 - t * 0.001, 0.003)))
        log.d_grip_stance.append(0.001)
        log.swing_arm.append('b')
        log.p_ee.append(np.random.randn(3) * 0.01)
        log.p_ee_ref.append(np.random.randn(3) * 0.01)
        log.q_ee.append(np.array([1.0, 0, 0, 0]))
        log.q_ee_ref.append(np.array([1.0, 0, 0, 0]))
        log.e_ee_pos.append(float(np.random.rand() * 0.003))
        log.e_ee_ori.append(float(np.random.rand() * 3.0))

        # CoM
        r_com = np.array([0.5, 0.0, -0.5]) + np.random.randn(3) * 0.005
        log.r_com.append(r_com)
        log.r_com_ref.append(r_com + np.random.randn(3) * 0.002)
        log.e_com.append(float(np.random.rand() * 0.01))
        log.v_com.append(np.random.randn(3) * 0.01)
        log.v_com_ref.append(np.random.randn(3) * 0.01)

        # Momentum
        L = np.random.randn(3) * 0.5
        log.L_com.append(L)
        log.L_com_norm.append(float(np.linalg.norm(L)))
        log.L_com_ref.append(L * 0.9)
        log.L_dot.append(np.random.randn(3) * 0.1)
        log.L_dot_norm.append(float(np.random.rand()))
        hw = np.random.randn(3) * 0.3
        log.hw.append(hw)
        log.hw_physical.append(hw)
        log.tau_w.append(np.random.randn(3) * 0.5)
        log.rw_speed.append(np.random.randn(3) * 10)

        # GMO
        log.gmo_residual_norm.append(float(np.random.rand()))
        log.gmo_swing_residual.append(float(np.random.rand()))
        log.gmo_contact_state.append(0)

        # H estimator
        log.H_rO.append(np.zeros(3))
        log.H_dot_est.append(np.zeros(3))
        log.omega_struct.append(np.zeros(3))
        log.qfrc_constraint_torque.append(np.zeros(3))

        # Joints
        tau = np.random.randn(12) * 2
        log.tau.append(tau)
        log.tau_max_joint.append(float(np.max(np.abs(tau))))

        # Structure
        log.struct_pos.append(np.random.randn(3) * 0.001)
        log.struct_quat.append(np.array([1, 0, 0, 0.0]))
        log.struct_euler_deg.append(np.random.randn(3) * 0.1)
        log.omega_s.append(np.random.randn(3) * 0.001)

        # NMPC solver
        log.nmpc_ok.append(True)
        log.qp_ok.append(True)
        log.lambda_ref_norm.append(float(np.random.rand() * 10))
        log.nmpc_time_ms.append(float(np.random.rand() * 30))
        log.qp_time_ms.append(float(np.random.rand() * 5))
        log.nmpc_status.append(0)
        log.nmpc_cost.append(float(np.random.rand() * 100))

        # Contact wrenches
        log.lambda_ref.append(np.random.randn(12))
        log.lambda_qp.append(np.random.randn(12))

        # Energy
        log.T_kinetic.append(float(np.random.rand() * 0.1))

    # Add a dock event
    log.dock_events.append({
        't': 5.0, 'step': 0, 'd_mm': 3.5,
        'arm': 'b', 'anchor': 3, 'method': 'kinematic'})

    return log


class TestComputeMetrics:
    """Test compute_metrics with synthetic data."""

    def test_returns_dict(self):
        log = _make_synthetic_log(50)
        results = compute_metrics(log)
        assert isinstance(results, dict)
        assert len(results) > 0

    def test_all_default_thresholds_covered(self):
        log = _make_synthetic_log(50)
        results = compute_metrics(log)
        for key in DEFAULT_THRESHOLDS:
            assert key in results, f"Missing metric: {key}"

    def test_metric_tuple_format(self):
        log = _make_synthetic_log(50)
        results = compute_metrics(log)
        for name, (val, thresh, passed) in results.items():
            assert passed in (True, False, 'SKIP'), \
                f"{name}: bad status {passed}"
            if passed != 'SKIP':
                assert isinstance(val, float), f"{name}: val not float"

    def test_with_config(self):
        log = _make_synthetic_log(50)
        cfg = SimConfig()
        results = compute_metrics(log, cfg=cfg)
        assert 'hw_saturation_ratio_peak' in results

    def test_empty_log_all_skip(self):
        log = SimLog()
        results = compute_metrics(log)
        for name, (val, thresh, passed) in results.items():
            assert passed == 'SKIP', f"{name} should be SKIP on empty log"


class TestPrintMetrics:
    def test_prints_without_error(self, capsys):
        log = _make_synthetic_log(30)
        results = compute_metrics(log)
        text = print_metrics(results)
        assert 'PASS' in text or 'FAIL' in text or 'SKIP' in text


class TestSaveMetricsCSV:
    def test_saves_csv(self, tmp_path):
        log = _make_synthetic_log(30)
        results = compute_metrics(log)
        path = str(tmp_path / 'metrics.csv')
        save_metrics_csv(results, path)
        assert os.path.exists(path)
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) > 1  # header + data


class TestGeneratePlots:
    def test_creates_all_figures(self, tmp_path):
        log = _make_synthetic_log(50)
        cfg = SimConfig()
        out = str(tmp_path / 'plots')
        generate_plots(log, out, cfg=cfg)
        expected = [
            'fig1_tracking.png',
            'fig2_momentum_aocs.png',
            'fig3_com_momentum.png',
            'fig4_energy_passivity.png',
            'fig5_nmpc_health.png',
            'fig6_contact_wrenches.png',
            'fig7_joints.png',
            'fig8_snapshots_grid.png',
        ]
        for fname in expected:
            assert os.path.exists(os.path.join(out, fname)), \
                f"Missing figure: {fname}"

    def test_empty_log_no_crash(self, tmp_path):
        log = SimLog()
        out = str(tmp_path / 'empty')
        generate_plots(log, out)
        assert os.path.exists(os.path.join(out, 'fig1_tracking.png'))


class TestRunDiagnostics:
    def test_full_pipeline(self, tmp_path):
        log = _make_synthetic_log(40)
        cfg = SimConfig()
        out = str(tmp_path / 'diag')
        results = run_diagnostics(log, out, cfg=cfg)
        assert isinstance(results, dict)
        assert os.path.exists(os.path.join(out, 'metrics.csv'))
        assert os.path.exists(os.path.join(out, 'fig1_tracking.png'))
        assert os.path.exists(os.path.join(out, 'fig5_nmpc_health.png'))
