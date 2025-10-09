function [traj, sim_params] = setup_trajectories(experiment_id)
% Trajectoires CoM robot (Lutze et al. 2023, Section 5)
%
% Trajectoires = position CoM robot (torso frame)
% Polynôme 5ème ordre avec v, a = 0 aux bornes

%% Temporal parameters
sim_params = struct();
sim_params.dt = 0.01;            % 100 Hz
sim_params.T_motion = 30;        % seconds
sim_params.T_stabilization = 40; % seconds
sim_params.T_total = sim_params.T_motion + sim_params.T_stabilization;
sim_params.time = 0:sim_params.dt:sim_params.T_total;

N = length(sim_params.time);
traj = struct();
traj.t = sim_params.time;

%% Select experiment
switch experiment_id
    case 1
        % Exp #1: Straight line crossing CoM
        traj.description = 'Straight line crossing satellite CoM';
        pos_start = [-7.5; 0; 1.0];
        pos_goal  = [ 7.5; 0; 1.0];
        
    case 2
        % Exp #2: Straight line with offset
        traj.description = 'Straight line with CoM offset';
        pos_start = [1.0; 1.0; 1.0];
        pos_goal  = [4.0; 2.0; 1.0];
        
    case 3
        % Exp #3: Circular arc (20 deg)
        traj.description = 'Circular arc (20 deg on edge)';
        r_structure = 7.5;
        theta_start = 0;
        theta_end   = deg2rad(20);
        
        pos_start = [r_structure*cos(theta_start); 
                     r_structure*sin(theta_start); 
                     1.0];
        pos_goal  = [r_structure*cos(theta_end); 
                     r_structure*sin(theta_end); 
                     1.0];
        
    otherwise
        error('Experiment ID must be 1, 2, or 3');
end

%% Generate smooth trajectory (5th order polynomial)
idx_motion = traj.t <= sim_params.T_motion;
t_motion = traj.t(idx_motion);
N_motion = length(t_motion);

% Normalized parameter τ ∈ [0, 1]
tau = t_motion / sim_params.T_motion;

% 5th order polynomial: s(τ)
% Boundary conditions: s(0)=0, s(1)=1, s'(0)=0, s'(1)=0, s''(0)=0, s''(1)=0
s       = 10*tau.^3 - 15*tau.^4 + 6*tau.^5;
s_dot   = (30*tau.^2 - 60*tau.^3 + 30*tau.^4) / sim_params.T_motion;
s_ddot  = (60*tau - 180*tau.^2 + 120*tau.^3) / (sim_params.T_motion^2);

if experiment_id ~= 3
    %% Linear trajectories (Exp #1, #2)
    direction = pos_goal - pos_start;
    
    traj.pos = pos_start + direction .* s;
    traj.vel = direction .* s_dot;
    traj.acc = direction .* s_ddot;
    
else
    %% Circular trajectory (Exp #3)
    theta = theta_start + s * (theta_end - theta_start);
    theta_dot = s_dot * (theta_end - theta_start);
    theta_ddot = s_ddot * (theta_end - theta_start);
    
    % Position
    traj.pos = [r_structure * cos(theta);
                r_structure * sin(theta);
                ones(1, N_motion)];
    
    % Velocity
    traj.vel = [-r_structure * sin(theta) .* theta_dot;
                 r_structure * cos(theta) .* theta_dot;
                 zeros(1, N_motion)];
    
    % Acceleration
    traj.acc = [-r_structure * sin(theta) .* theta_ddot ...
                -r_structure * cos(theta) .* theta_dot.^2;
                 r_structure * cos(theta) .* theta_ddot ...
                -r_structure * sin(theta) .* theta_dot.^2;
                 zeros(1, N_motion)];
end

%% Stabilization phase (hold final position)
idx_stab = traj.t > sim_params.T_motion;
N_stab = sum(idx_stab);

traj.pos(:, idx_stab) = repmat(traj.pos(:, N_motion), 1, N_stab);
traj.vel(:, idx_stab) = zeros(3, N_stab);
traj.acc(:, idx_stab) = zeros(3, N_stab);

%% Orientation (constant)
traj.quat  = repmat([1; 0; 0; 0], 1, N);
traj.omega = zeros(3, N);
traj.alpha = zeros(3, N);

%% Metadata
traj.pos_start = pos_start;
traj.pos_goal = pos_goal;
traj.experiment_id = experiment_id;

%% Validation
v_init = norm(traj.vel(:, 1));
v_final = norm(traj.vel(:, N_motion));

if v_init > 1e-8 || v_final > 1e-8
    warning('Initial/final velocities non-zero!');
end

fprintf('  ✓ v_max = %.3f m/s\n', max(vecnorm(traj.vel)));
fprintf('  ✓ Distance = %.2f m\n', norm(pos_goal - pos_start));

end