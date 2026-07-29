# E3 — dataclass default vs canonical value

56 field(s) differ. Every one is a place where reading the dataclass would put a wrong number in the paper.

| config | field | canonical | default | source |
|---|---|---|---|---|
| `SimConfig` | `t_ss_margin` | 5 | 1 | `crawlbot/simulation/config.py:26` |
| `SimConfig` | `aocs_mode` | legacy_pid_numerical | legacy | `crawlbot/simulation/config.py:98` |
| `SimConfig` | `aocs_use_wrench_ff_in_ds` | True | False | `crawlbot/simulation/config.py:114` |
| `SimConfig` | `ds_torso_ref_from_state` | True | False | `crawlbot/simulation/config.py:123` |
| `SimConfig` | `interstep_settle_alpha_wrench` | 3 | 0 | `crawlbot/simulation/config.py:165` |
| `SimConfig` | `frames_per_step` | 5 | 0 | `crawlbot/simulation/config.py:181` |
| `SimConfig` | `use_m2_stack` | True | False | `crawlbot/simulation/config.py:189` |
| `SimConfig` | `enforce_hw_conservation` | True | False | `crawlbot/simulation/config.py:193` |
| `SimConfig` | `interstep_settle_epsilon_v` | 0.005 | 0 | `crawlbot/simulation/config.py:277` |
| `SimConfig` | `ss_alpha_lambda_int` | 1 | 0 | `crawlbot/simulation/config.py:285` |
| `SimConfig` | `log_hifreq_ss` | True | False | `crawlbot/simulation/config.py:294` |
| `SimConfig` | `ss_two_task_mode` | True | False | `crawlbot/simulation/config.py:302` |
| `SimConfig` | `qp_envelope_exact` | True | False | `crawlbot/simulation/config.py:338` |
| `SimConfig` | `ds_centroidal_mode` | True | False | `crawlbot/simulation/config.py:343` |
| `SimConfig` | `ss_Kp_torso` | 3 | 6 | `crawlbot/simulation/config.py:351` |
| `SimConfig` | `ss_Kd_torso` | 2.5 | 5 | `crawlbot/simulation/config.py:352` |
| `SimConfig` | `ik_level_axis` | [0.0, 0.0, 1.0] | None | `crawlbot/simulation/config.py:392` |
| `SimConfig` | `ik_q_nominal` | [0.297781, 1.526727, 0.842257, 1.147432, 0.75139, 0.… | None | `crawlbot/simulation/config.py:393` |
| `SimConfig` | `ik_w_posture` | 0.2 | 0 | `crawlbot/simulation/config.py:394` |
| `SimConfig` | `use_com_z_standoff` | True | False | `crawlbot/simulation/config.py:407` |
| `WholeBodyQPConfig` | `alpha_ee` | 1000 | 500 | `crawlbot/solvers/wholebody_qp.py:97` |
| `WholeBodyQPConfig` | `alpha_posture` | 20 | 100 | `crawlbot/solvers/wholebody_qp.py:98` |
| `WholeBodyQPConfig` | `alpha_wrench` | 1 | 10 | `crawlbot/solvers/wholebody_qp.py:99` |
| `WholeBodyQPConfig` | `alpha_torque` | 5 | 1 | `crawlbot/solvers/wholebody_qp.py:100` |
| `WholeBodyQPConfig` | `alpha_reg` | 1 | 0.01 | `crawlbot/solvers/wholebody_qp.py:101` |
| `WholeBodyQPConfig` | `alpha_lambda_int` | 1 | 0 | `crawlbot/solvers/wholebody_qp.py:102` |
| `WholeBodyQPConfig` | `ds_centroidal_mode` | True | False | `crawlbot/solvers/wholebody_qp.py:115` |
| `WholeBodyQPConfig` | `ss_two_task_mode` | True | False | `crawlbot/solvers/wholebody_qp.py:132` |
| `WholeBodyQPConfig` | `ss_alpha_mom` | 400 | 500 | `crawlbot/solvers/wholebody_qp.py:133` |
| `WholeBodyQPConfig` | `alpha_torso_pose` | 2000 | 1000 | `crawlbot/solvers/wholebody_qp.py:134` |
| `WholeBodyQPConfig` | `qp_envelope_exact` | True | False | `crawlbot/solvers/wholebody_qp.py:147` |
| `WholeBodyQPConfig` | `Kp_com` | [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]] | [100.0, 100.0, 100.0] | `crawlbot/solvers/wholebody_qp.py:162` |
| `WholeBodyQPConfig` | `Kd_com` | [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]] | [20.0, 20.0, 20.0] | `crawlbot/solvers/wholebody_qp.py:163` |
| `WholeBodyQPConfig` | `Kp_torso` | [3.0, 3.0, 3.0, 3.0, 3.0, 3.0] | [8.0, 8.0, 8.0, 5.0, 5.0, 5.0] | `crawlbot/solvers/wholebody_qp.py:166` |
| `WholeBodyQPConfig` | `Kd_torso` | [2.5, 2.5, 2.5, 2.5, 2.5, 2.5] | [6.0, 6.0, 6.0, 4.0, 4.0, 4.0] | `crawlbot/solvers/wholebody_qp.py:167` |
| `WholeBodyQPConfig` | `Kp_ee` | [10.0, 10.0, 10.0] | [80.0, 80.0, 80.0] | `crawlbot/solvers/wholebody_qp.py:170` |
| `WholeBodyQPConfig` | `Kd_ee` | [12.0, 12.0, 12.0] | [15.0, 15.0, 15.0] | `crawlbot/solvers/wholebody_qp.py:171` |
| `WholeBodyQPConfig` | `Kp_ee_ang` | [6.0, 6.0, 6.0] | [5.0, 5.0, 5.0] | `crawlbot/solvers/wholebody_qp.py:172` |
| `WholeBodyQPConfig` | `Kd_ee_ang` | [4.5, 4.5, 4.5] | [3.0, 3.0, 3.0] | `crawlbot/solvers/wholebody_qp.py:173` |
| `WholeBodyQPConfig` | `Kp_posture` | 1 | 25 | `crawlbot/solvers/wholebody_qp.py:176` |
| `WholeBodyQPConfig` | `Kd_posture` | 1.5 | 10 | `crawlbot/solvers/wholebody_qp.py:177` |
| `WholeBodyQPConfig` | `tau_max` | [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.… | [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.… | `crawlbot/solvers/wholebody_qp.py:184` |
| `WholeBodyQPConfig` | `dt_qp` | 0.01 | 0.008 | `crawlbot/solvers/wholebody_qp.py:190` |
| `WholeBodyQPConfig` | `L_max` | 10 | inf | `crawlbot/solvers/wholebody_qp.py:197` |
| `WholeBodyQPConfig` | `tau_w_max` | 2.5 | inf | `crawlbot/solvers/wholebody_qp.py:198` |
| `CentroidalNMPCConfig` | `robot_mass` | 71.056 | 90 | `crawlbot/solvers/centroidal_nmpc.py:81` |
| `CentroidalNMPCConfig` | `N` | 8 | 20 | `crawlbot/solvers/centroidal_nmpc.py:84` |
| `CentroidalNMPCConfig` | `dt` | 0.1 | 0.05 | `crawlbot/solvers/centroidal_nmpc.py:85` |
| `CentroidalNMPCConfig` | `f_max` | 300 | 3000 | `crawlbot/solvers/centroidal_nmpc.py:96` |
| `CentroidalNMPCConfig` | `tau_max` | 8 | 300 | `crawlbot/solvers/centroidal_nmpc.py:97` |
| `CentroidalNMPCConfig` | `L_max` | 10 | inf | `crawlbot/solvers/centroidal_nmpc.py:100` |
| `CentroidalNMPCConfig` | `tau_w_max` | 2.5 | inf | `crawlbot/solvers/centroidal_nmpc.py:101` |
| `CentroidalNMPCConfig` | `p_max` | 50 | inf | `crawlbot/solvers/centroidal_nmpc.py:102` |
| `CentroidalNMPCConfig` | `enforce_hw_conservation` | True | False | `crawlbot/solvers/centroidal_nmpc.py:110` |
| `CentroidalNMPCConfig` | `solver_opts` | {} | {} | `crawlbot/solvers/centroidal_nmpc.py:119` |
| `CoarsePrePlannerConfig` | `robot_mass` | 71.056 | 71 | `crawlbot/planning/coarse_preplanner.py:68` |
