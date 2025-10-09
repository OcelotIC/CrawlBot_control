function env = setup_environment()
% Configuration environnement MIRROR

env = struct();

%% Satellite
env.satellite.mass = 2040;
env.satellite.radius = 7.5;
env.satellite.height = 0.3;

r = env.satellite.radius;
h = env.satellite.height;
m = env.satellite.mass;
env.satellite.inertia = diag([
    (1/12)*m*(3*r^2 + h^2);
    (1/12)*m*(3*r^2 + h^2);
    (1/2)*m*r^2
]);

%% Standard Interconnect
env.SI.F_max = 3000;
env.SI.tau_max = 300;

%% Reaction Wheels
env.RW.n_wheels = 4;
env.RW.h_max = 50;
env.RW.h_total_max = env.RW.n_wheels * env.RW.h_max;
env.RW.tau_max = 10;

theta = 54.74 * pi/180;
env.RW.installation_matrix = [
    sin(theta)*cos(0),       sin(theta)*cos(pi/2),   sin(theta)*cos(pi),     sin(theta)*cos(3*pi/2);
    sin(theta)*sin(0),       sin(theta)*sin(pi/2),   sin(theta)*sin(pi),     sin(theta)*sin(3*pi/2);
    cos(theta),              cos(theta),             cos(theta),             cos(theta)
];

%% CORRECTION: Gains plus conservateurs
% Qb réduit pour éviter surpondération base
env.control.Qr = eye(6) * 1.0;      % Robot tracking
env.control.Qb = eye(6) * 10.0;     % Base stabilization (réduit de 100 → 10)
env.control.Qc = eye(6) * 0.1;      % Wrench regularization (augmenté 0.01 → 0.1)

% Gains PD plus conservateurs
env.control.Kr = diag([50, 50, 50, 25, 25, 25]);    % Réduit de moitié
env.control.Dr = diag([10, 10, 10, 5, 5, 5]);       % Réduit de moitié
env.control.Kb = diag([40, 40, 40]);                % Réduit de moitié
env.control.Db = diag([8, 8, 8]);                   % Réduit de moitié

end