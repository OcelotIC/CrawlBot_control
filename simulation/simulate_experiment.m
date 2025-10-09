function results = simulate_experiment(env, robot, traj, sim_params, controller, use_optimization)
% Boucle simulation principale

fprintf('  Initializing...\n');

%% État initial
state = struct();
state.q_base = traj.pos(:,1);
state.quat_base = [0; 0; 0; 1];  % SPART: [q1,q2,q3,q4] avec q4=scalaire
state.qm = zeros(robot.n_q, 1);

state.qm = [
    deg2rad(15);   % arm1_joint1 (shoulder yaw)
    deg2rad(-30);  % arm1_joint2 (shoulder pitch)
    deg2rad(60);   % arm1_joint3 (elbow plié)
    deg2rad(0);    % arm1_joint4 (wrist yaw)
    deg2rad(-30);  % arm1_joint5 (wrist pitch)
    deg2rad(0);    % arm1_joint6 (wrist roll)
    deg2rad(-15);  % arm2_joint1
    deg2rad(-30);  % arm2_joint2
    deg2rad(60);   % arm2_joint3
    deg2rad(0);    % arm2_joint4
    deg2rad(-30);  % arm2_joint5
    deg2rad(0);    % arm2_joint6
];


state.omega_base = zeros(3,1);
state.u_base = zeros(3,1);
state.um = zeros(robot.n_q, 1);
state.omega_satellite = zeros(3,1);
state.quat_satellite = [0; 0; 0; 1];  % SPART convention
state.h_RW_stored = zeros(3,1);

%% Préallocation
N = length(traj.t);
results = struct();
results.t = traj.t;
results.alpha = zeros(1, N);
results.beta = zeros(1, N);
results.gamma = zeros(1, N);
results.Fc = zeros(6, N);
results.h_RW = zeros(3, N);
results.h_total = zeros(3, N);
results.pos_robot = zeros(3, N);
results.pos_error = zeros(3, N);

%% Boucle temporelle
for k = 1:N-1
    % Calcul état système
    sys = compute_system_state(state, robot, env);
    
    % Résolution QP
    [Fc_opt, tau_joints] = controller.solve(sys, traj, k, use_optimization);
    
    % Intégration dynamique
    state = integrate_dynamics(state, tau_joints, Fc_opt, sys, robot, sim_params.dt, env);
    
    % Sauvegarde
    results.Fc(:, k) = Fc_opt;
    
    % CORRECTION: Utiliser quat_Angles321 (SPART) 
    % Convention SPART: quaternion [q1,q2,q3,q4] où q4 est la partie scalaire
    % Séquence 321 (ZYX): alpha (roll-X), beta (pitch-Y), gamma (yaw-Z)
    angles_sat = quat_Angles321(state.quat_satellite);
    results.alpha(k) = angles_sat(1);  % Roll (rotation autour X)
    results.beta(k)  = angles_sat(2);  % Pitch (rotation autour Y)
    results.gamma(k) = angles_sat(3);  % Yaw (rotation autour Z)
    
    results.h_RW(:, k) = state.h_RW_stored;
    results.h_total(:, k) = sys.h_total;
    results.pos_robot(:, k) = state.q_base;
    results.pos_error(:, k) = traj.pos(:, k) - state.q_base;
    
    if mod(k, 1000) == 0
        fprintf('    %.1f%%\n', 100*k/N);
    end
end

% Dernier point (répétition du précédent)
results.Fc(:, N) = results.Fc(:, N-1);
results.alpha(N) = results.alpha(N-1);
results.beta(N) = results.beta(N-1);
results.gamma(N) = results.gamma(N-1);
results.h_RW(:, N) = results.h_RW(:, N-1);
results.h_total(:, N) = results.h_total(:, N-1);
results.pos_robot(:, N) = state.q_base;
results.pos_error(:, N) = traj.pos(:, N) - state.q_base;

fprintf('  Complete!\n');

end