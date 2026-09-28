"""
AttitudeController — the AOCS: reaction-wheel torque for the free-floating
platform (ASTROHUB structure), from measurements and the robot's centroidal
state.

One of the three blocks of the controller (NMPC, whole-body QP, AOCS). It reads
the platform gyro / attitude / wheel momentum through ``SensorSuite`` and never
touches MuJoCo; the caller applies the returned torque through the plant.

Two entry points, matching the two loops that exist today:

    command()            the NMPC-tracked tick (SS, DWELL, trailing DS) —
                         every ``cfg.aocs_mode`` branch
    command_interstep()  the inter-step DS passivity loop — canonical
                         ``legacy_pid_numerical`` with the DS wrench FF

Moved verbatim out of ``sim_loop.py`` (refactor/sim-loop-split, extraction
3a): same expressions, same order, so the floating-point sequence is
unchanged. See docs/crawlbot/control/attitude.md for the law and its traps.
"""

import numpy as np

try:
    import pinocchio as pin
except ImportError:
    pin = None

from crawlbot.aocs.force_estimator import compute_aocs_command


class AttitudeController:
    """Reaction-wheel torque law (spec §4, *AOCS Controller (Corrected)*)."""

    def __init__(self, cfg, robot, sensors, H_estimator,
                 struct_quat_init, struct_I):
        self._cfg = cfg
        self._robot = robot
        self._sensors = sensors
        self._H_estimator = H_estimator
        # Initial structure attitude (wxyz): θ_s = vee(R_init^T R_now).
        self._struct_quat_init = struct_quat_init
        # Structure principal inertia, for the *_model ω̇_s variants.
        self._struct_I = struct_I

    def command(self, *, phase, rs, lambda_qp_sol, cc_nmpc, stance_anchors,
                L_com_prev, v_com_prev, omega_s_prev, tau_w_prev):
        """Wheel torque for one QP tick of the NMPC-tracked loop.

        Parameters are the per-tick carry the loop threads through its
        sub-steps: ``L_com_prev`` / ``v_com_prev`` (previous sub-step's
        centroidal state, for the FD feedforward), ``omega_s_prev`` (for the
        numerical ω̇_s) and ``tau_w_prev`` (for the model ω̇_s).
        ``stance_anchors`` = (anchor A, anchor B) positions, structure frame,
        the levers of the DS wrench feedforward.

        Returns ``(tau_w_cmd, omega_s, transport_mag)`` — ``omega_s`` is the
        live gyro view the caller copies into its ``omega_s_prev`` carry;
        ``transport_mag`` = |ω_s × H_{r/O}| (diagnostic, not applied).
        """
        cfg = self._cfg
        hw_phys = self._sensors.wheel_momentum()
        # A live view, as the original read was (see sensors.py).
        omega_s = self._sensors.omega_struct_view()

        # Diagnostic: |ω_s × H_{r/O}| — magnitude of the transport
        # term missing from Mode B's feedforward. Computed but
        # NOT applied to tau_w_cmd (see AOCS_CONCERN_MEMO.md §7).
        H_rO_diag = (rs.L_com
                     + np.cross(rs.r_com,
                                self._robot._total_mass * rs.v_com))
        transport_mag = float(
            np.linalg.norm(np.cross(omega_s, H_rO_diag)))

        # DS-only per-contact wrench feedforward (legacy_pid_* only):
        # τ_w_FF = −Σ_i (r_Ci × f_i + τ_i) from λ_qp. Captures the
        # internal-stress couple at the welded loop that the FD on
        # L_com misses. r_Ci taken in struct body frame; λ_qp in
        # world frame — equivalent at small structure-frame rotation
        # (the regime we operate in; <5° transient).
        tau_struct_ff_aocs = None
        if (cfg.aocs_use_wrench_ff_in_ds
                and phase == 'DS'
                and cfg.aocs_mode in ('legacy_pid_numerical',
                                      'legacy_pid_model')):
            _r_C = stance_anchors
            _lam = np.asarray(lambda_qp_sol, dtype=float).ravel()
            _ff = np.zeros(3)
            for _ci in range(2):
                if not cc_nmpc.active_contacts[_ci]:
                    continue
                _f = _lam[6 * _ci: 6 * _ci + 3]
                _tq = _lam[6 * _ci + 3: 6 * _ci + 6]
                _ff -= np.cross(_r_C[_ci], _f) + _tq
            tau_struct_ff_aocs = _ff

        if cfg.aocs_off_in_ds and phase == 'DS':
            tau_w_cmd = np.zeros(3)
        elif cfg.aocs_mode == 'H_est' or cfg.aocs_use_H_estimator:
            # H_{r/O} estimator: feedforward on full robot angular
            # momentum about O (spin + orbital), with attitude
            # damping and desaturation feedback.
            H_dot_est = self._H_estimator.update(
                r_com=rs.r_com, v_com=rs.v_com,
                L_com=rs.L_com, omega_s=omega_s)
            tau_w_cmd = compute_aocs_command(
                H_dot_est=H_dot_est,
                omega_s=omega_s,
                hw_current=hw_phys,
                hw_target=cfg.aocs_hw_target,
                K_omega=cfg.aocs_K_omega,
                K_h=cfg.aocs_K_h,
                tau_w_max=cfg.aocs_tau_w_max)
        elif cfg.aocs_use_legacy_corrected:
            # M4: legacy formula + missing orbital term r × m·dv_com.
            from crawlbot.aocs.force_estimator import (
                compute_aocs_command_legacy_corrected)
            tau_w_cmd = compute_aocs_command_legacy_corrected(
                L_com=rs.L_com, L_com_prev=L_com_prev,
                r_com=rs.r_com, v_com=rs.v_com,
                v_com_prev=v_com_prev,
                hw_current=hw_phys, dt=cfg.dt_qp,
                robot_mass=self._robot._total_mass,
                K_hw=cfg.aocs_K_hw,
                hw_min=cfg.hw_min, hw_max=cfg.hw_max,
                tau_w_max=cfg.aocs_tau_w_max)
        elif cfg.aocs_mode == 'legacy_pd_numerical':
            # legacy_corrected + PD on ω_s (numerical ω̇_s).
            from crawlbot.aocs.force_estimator import (
                compute_aocs_command_legacy_pd_numerical)
            tau_w_cmd = compute_aocs_command_legacy_pd_numerical(
                L_com=rs.L_com, L_com_prev=L_com_prev,
                r_com=rs.r_com, v_com=rs.v_com,
                v_com_prev=v_com_prev,
                omega_s=omega_s, omega_s_prev=omega_s_prev,
                hw_current=hw_phys, dt=cfg.dt_qp,
                robot_mass=self._robot._total_mass,
                K_hw=cfg.aocs_K_hw, K_omega=cfg.aocs_K_omega,
                K_d=cfg.aocs_K_d,
                hw_min=cfg.hw_min, hw_max=cfg.hw_max,
                tau_w_max=cfg.aocs_tau_w_max)
        elif cfg.aocs_mode == 'legacy_pd_model':
            # legacy_corrected + PD on ω_s (model-based ω̇_s
            # from previous τ_w_cmd and structure inertia).
            from crawlbot.aocs.force_estimator import (
                compute_aocs_command_legacy_pd_model)
            tau_w_cmd = compute_aocs_command_legacy_pd_model(
                L_com=rs.L_com, L_com_prev=L_com_prev,
                r_com=rs.r_com, v_com=rs.v_com,
                v_com_prev=v_com_prev,
                omega_s=omega_s, tau_w_prev=tau_w_prev,
                I_struct=self._struct_I,
                hw_current=hw_phys, dt=cfg.dt_qp,
                robot_mass=self._robot._total_mass,
                K_hw=cfg.aocs_K_hw, K_omega=cfg.aocs_K_omega,
                K_d=cfg.aocs_K_d,
                hw_min=cfg.hw_min, hw_max=cfg.hw_max,
                tau_w_max=cfg.aocs_tau_w_max)
        elif cfg.aocs_mode in ('legacy_pid_numerical',
                               'legacy_pid_model'):
            # legacy_pd_* + attitude-tracking P term (drives
            # the structure back to its initial orientation).
            #
            # Geometric SO(3) attitude error (Lee–McClamroch):
            #   e_R = ½(R_init^T R_now − R_now^T R_init)^∨
            # where (·)^∨ extracts the 3-vector from a skew-
            # symmetric matrix. Properties vs log3:
            #   (1) frame-consistent with ω_s (both implicitly
            #       in current body frame at small angles, with
            #       only third-order discrepancy from R_now T R_init)
            #   (2) bounded |e_R| ≤ |sin θ| ≤ 1 — graceful
            #       saturation under any disturbance
            #   (3) no singularity at θ = π (log3 would have one)
            # For our sub-3° drift regime, e_R ≈ θ_s to within
            # 1% — the K_θ=10 result on log3 transfers directly.
            # Local import: see note at the H_est branch above.
            import pinocchio as _pin
            qw, qx, qy, qz = self._struct_quat_init
            R_init = _pin.Quaternion(qw, qx, qy, qz).toRotationMatrix()
            qw, qx, qy, qz = self._sensors.struct_quat()
            R_now = _pin.Quaternion(qw, qx, qy, qz).toRotationMatrix()
            R_err = R_init.T @ R_now
            # vee of (R_err − R_err^T) / 2 — geometric SO(3) error.
            theta_s = 0.5 * np.array([
                R_err[2, 1] - R_err[1, 2],
                R_err[0, 2] - R_err[2, 0],
                R_err[1, 0] - R_err[0, 1],
            ])
            if cfg.aocs_mode == 'legacy_pid_numerical':
                from crawlbot.aocs.force_estimator import (
                    compute_aocs_command_legacy_pid_numerical)
                tau_w_cmd = compute_aocs_command_legacy_pid_numerical(
                    L_com=rs.L_com, L_com_prev=L_com_prev,
                    r_com=rs.r_com, v_com=rs.v_com,
                    v_com_prev=v_com_prev,
                    omega_s=omega_s, omega_s_prev=omega_s_prev,
                    theta_s=theta_s,
                    hw_current=hw_phys, dt=cfg.dt_qp,
                    robot_mass=self._robot._total_mass,
                    K_hw=cfg.aocs_K_hw, K_omega=cfg.aocs_K_omega,
                    K_d=cfg.aocs_K_d, K_theta=cfg.aocs_K_theta,
                    hw_min=cfg.hw_min, hw_max=cfg.hw_max,
                    tau_w_max=cfg.aocs_tau_w_max,
                    tau_struct_ff=tau_struct_ff_aocs)
            else:
                from crawlbot.aocs.force_estimator import (
                    compute_aocs_command_legacy_pid_model)
                tau_w_cmd = compute_aocs_command_legacy_pid_model(
                    L_com=rs.L_com, L_com_prev=L_com_prev,
                    r_com=rs.r_com, v_com=rs.v_com,
                    v_com_prev=v_com_prev,
                    omega_s=omega_s, theta_s=theta_s,
                    tau_w_prev=tau_w_prev,
                    I_struct=self._struct_I,
                    hw_current=hw_phys, dt=cfg.dt_qp,
                    robot_mass=self._robot._total_mass,
                    K_hw=cfg.aocs_K_hw, K_omega=cfg.aocs_K_omega,
                    K_d=cfg.aocs_K_d, K_theta=cfg.aocs_K_theta,
                    hw_min=cfg.hw_min, hw_max=cfg.hw_max,
                    tau_w_max=cfg.aocs_tau_w_max,
                    tau_struct_ff=tau_struct_ff_aocs)
        else:
            # Legacy AOCS: L_dot feedforward only (spin component).
            # Desaturation sign matches compute_aocs_command_legacy_corrected
            # (+K_hw·hw_error). See that function's docstring for the
            # MuJoCo-convention derivation.
            L_dot_est = (rs.L_com - L_com_prev) / cfg.dt_qp
            hw_error = np.clip(hw_phys, cfg.hw_min, cfg.hw_max) - hw_phys
            tau_w_cmd = -L_dot_est + cfg.aocs_K_hw * hw_error
            tau_w_cmd = np.clip(tau_w_cmd, -cfg.aocs_tau_w_max, cfg.aocs_tau_w_max)


        return tau_w_cmd, omega_s, transport_mag

    def command_interstep(self, rs, cc_ds, lambda_qp_sol, omega_s_prev):
        """Canonical AOCS wheel-torque command for the inter-step DS loop.

        Re-activates the structure attitude controller inside
        ``SimulationLoop._run_ds_passivity_loop`` (J2 step 4a — the loop historically
        zeroed the wheels, violating the free-floating invariant; see the
        AOCS-during-DS audit). Mirrors the ``_step`` AOCS for the canonical
        ``legacy_pid_numerical`` mode, fed entirely from in-loop values:

          - DS WRENCH feedforward  τ_w_FF = −Σ_i (r_Ci × f_i + τ_i)  from
            ``lambda_qp_sol`` (the settle-QP wrench captured this tick),
            with anchor levers r_Ci = cc_ds.r_contact_{A,B} (struct frame).
            This is the welded-loop internal-stress couple the FD-on-L_com
            feedforward is blind to (force_estimator.py:582-592). Same
            construction as _step (sim_loop.py wrench-FF block).
          - attitude PID + desaturation via
            ``compute_aocs_command_legacy_pid_numerical`` — θ_s from the
            geometric SO(3) error ½ vee(R_errᵀ−R_err) (identical to _step),
            ω_s / h_w from MuJoCo state, ω̇_s from the loop-local
            ``omega_s_prev``.

        No NMPC dependency (the loop bypasses the NMPC): the feedforward
        source is the QP wrench, not the NMPC plan (AOCS-FF audit). The QP,
        passivity constraint, and envelope box are untouched. Returns the
        clipped τ_w command (±aocs_tau_w_max).
        """
        from crawlbot.aocs.force_estimator import (
            compute_aocs_command_legacy_pid_numerical)
        cfg = self._cfg
        omega_s = self._sensors.omega_struct()
        hw_phys = self._sensors.wheel_momentum()

        # Geometric SO(3) attitude error θ_s (same as _step).
        qw, qx, qy, qz = self._struct_quat_init
        R_init = pin.Quaternion(qw, qx, qy, qz).toRotationMatrix()
        qw, qx, qy, qz = self._sensors.struct_quat()
        R_now = pin.Quaternion(qw, qx, qy, qz).toRotationMatrix()
        R_err = R_init.T @ R_now
        theta_s = 0.5 * np.array([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ])

        # DS wrench feedforward from λ_qp (anchor levers in struct frame;
        # λ_qp in world frame — equivalent at the <5° structure-frame
        # rotation we operate in, as in _step).
        _lam = np.asarray(lambda_qp_sol, dtype=float).ravel()
        _r_C = (cc_ds.r_contact_A, cc_ds.r_contact_B)
        tau_struct_ff = np.zeros(3)
        for _ci in range(2):
            if not cc_ds.active_contacts[_ci]:
                continue
            _f = _lam[6 * _ci: 6 * _ci + 3]
            _tq = _lam[6 * _ci + 3: 6 * _ci + 6]
            tau_struct_ff -= np.cross(_r_C[_ci], _f) + _tq

        # L_com/v_com args are unused on the wrench-FF path (the FD branch
        # is `tau_struct_ff is None`), so current values are passed.
        return compute_aocs_command_legacy_pid_numerical(
            L_com=rs.L_com, L_com_prev=rs.L_com,
            r_com=rs.r_com, v_com=rs.v_com, v_com_prev=rs.v_com,
            omega_s=omega_s, omega_s_prev=omega_s_prev,
            theta_s=theta_s,
            hw_current=hw_phys, dt=cfg.dt_qp,
            robot_mass=self._robot._total_mass,
            K_hw=cfg.aocs_K_hw, K_omega=cfg.aocs_K_omega,
            K_d=cfg.aocs_K_d, K_theta=cfg.aocs_K_theta,
            hw_min=cfg.hw_min, hw_max=cfg.hw_max,
            tau_w_max=cfg.aocs_tau_w_max,
            tau_struct_ff=tau_struct_ff)
