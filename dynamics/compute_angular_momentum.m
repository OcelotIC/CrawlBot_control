function [h_total, h_satellite, h_robot, h_RW] = compute_angular_momentum(state, robot, env)
% Calcule composantes du moment angulaire
% Conservation: h_total = h_satellite + h_robot + h_RW (constant en orbite)
%
% ATTENTION:
%   - state.omega_satellite : vitesse angulaire SATELLITE (corps rigide porteur)
%   - state.u0 = [ω_torso; v_torso] : twist TORSE/base-link du robot (SPART)
%   - state.um : vitesses articulaires manipulateur

%% 1. Moment angulaire SATELLITE (corps rigide porteur)
% Le satellite est le corps principal qui porte le robot
omega_satellite = state.omega_satellite;  % Doit venir de l'état satellite
I_satellite = env.satellite.inertia;
h_satellite = I_satellite * omega_satellite;

%% 2. Moment angulaire ROBOT (torse + manipulateur)
% Utilise les matrices d'inertie SPART
% H encode l'inertie du système robot complet

% Contribution base (torse)
h_robot_base = state.H0(1:3, :) * state.u0;  % H0 : inertie base

% Contribution manipulateur
if isfield(state, 'H0m') && isfield(state, 'Hm')
    % H0m : couplage base-manipulateur
    % Hm : inertie manipulateur
    h_robot_manip = state.H0m(1:3, :) * state.um + state.Hm(1:3, :) * state.um;
else
    % Fallback si matrices couplées non disponibles
    h_robot_manip = zeros(3, 1);
    for i = 1:length(state.um)
        % Approximation simple
        h_robot_manip = h_robot_manip + robot.links(i).inertia(1:3,1:3) * state.um(i);
    end
end

h_robot = h_robot_base + h_robot_manip;

%% 3. Moment angulaire REACTION WHEELS
if isfield(state, 'h_RW_stored')
    h_RW = state.h_RW_stored;
else
    h_RW = zeros(3, 1);
end

%% 4. Total (doit être constant en orbite sans couples externes)
h_total = h_satellite + h_robot + h_RW;

end