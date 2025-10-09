function Run_benchmark(experiment_id)
%% ========================================================================
%  BENCHMARK LUTZE ET AL. (2023)
%  Reproduction IEEE Aerospace Conference 2023
%  ========================================================================

if nargin < 1
    experiment_id = 1;
end

%% Setup
script_dir = fileparts(mfilename('fullpath'));
cd(script_dir);

addpath(genpath('config'));
addpath(genpath('dynamics'));
addpath(genpath('control'));
addpath(genpath('simulation'));
addpath(genpath('visualization'));
addpath(genpath('utils'));
addpath(genpath('URDF_models'));

% Verify SPART
if ~exist('urdf2robot', 'file')
    error('SPART not found. Add SPART to path or download from: https://github.com/NPS-SRL/SPART');
end

fprintf('╔════════════════════════════════════════════════════════╗\n');
fprintf('║   BENCHMARK: Lutze et al. (2023)                      ║\n');
fprintf('╚════════════════════════════════════════════════════════╝\n\n');

%% Load robot
fprintf('→ Loading robot model...\n');
[robot, robot_keys] = urdf2robot('URDF_models/MAR_DualArm_6DoF.urdf');
fprintf('  ✓ Robot: %d DoF total\n\n', 6 + robot.n_q);

%% Environment
fprintf('→ Setup environment...\n');
env = setup_environment();
fprintf('  ✓ Config loaded\n\n');

%% Trajectory
fprintf('→ Generating trajectory (Exp #%d)...\n', experiment_id);
[traj, sim_params] = setup_trajectories(experiment_id);
fprintf('  ✓ %s\n\n', traj.description);

%% Controller
controller = LutzeQPController(env);

%% Simulate - NO OPT
fprintf('════════════════════════════════════════════════════════\n');
fprintf(' SIMULATION 1: NON-OPTIMIZED\n');
fprintf('════════════════════════════════════════════════════════\n');
tic;
results_no_opt = simulate_experiment(env, robot, traj, sim_params, controller, false);
fprintf('✓ Done in %.2fs\n\n', toc);

%% Simulate - OPT
fprintf('════════════════════════════════════════════════════════\n');
fprintf(' SIMULATION 2: OPTIMIZED (QP)\n');
fprintf('════════════════════════════════════════════════════════\n');
tic;
results_opt = simulate_experiment(env, robot, traj, sim_params, controller, true);
fprintf('✓ Done in %.2fs\n\n', toc);

%% Plots
fprintf('→ Generating figures...\n');
mkdir_safe('results/figures');

plot_satellite_rotation(results_no_opt, results_opt, experiment_id, env);
plot_contact_wrenches(results_no_opt, results_opt, experiment_id);
plot_tracking_error(results_no_opt, results_opt, experiment_id);
plot_momentum_saturation(results_no_opt, results_opt, experiment_id, env);
fprintf('  ✓ 4 figures saved\n\n');

%% Metrics
fprintf('════════════════════════════════════════════════════════\n');
fprintf(' PERFORMANCE METRICS\n');
fprintf('════════════════════════════════════════════════════════\n');
compute_performance_metrics(results_no_opt, results_opt, env);

%% Save
mkdir_safe('results/data');
save(sprintf('results/data/exp%d_no_opt.mat', experiment_id), 'results_no_opt');
save(sprintf('results/data/exp%d_opt.mat', experiment_id), 'results_opt');

fprintf('\n✓✓✓ BENCHMARK COMPLETE ✓✓✓\n');

end

function mkdir_safe(dir_path)
    if ~exist(dir_path, 'dir')
        mkdir(dir_path);
    end
end