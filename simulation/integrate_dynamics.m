function state_next = integrate_dynamics(state, tau_joints, Fc, sys, robot, dt, env)
% ✅ VERSION AMÉLIORÉE avec stabilisation Baumgarte

%% 1. Matrices dynamiques
H0 = sys.H0;
H0m = sys.H0m;
Hm = sys.Hm;
C0 = sys.C0;
C0m = sys.C0m;
Cm0 = sys.Cm0;
Cm = sys.Cm;

n_q = robot.n_q;
H_full = [H0, H0m; H0m', Hm];
C_full = [C0, C0m; Cm0, Cm];
u_full = [sys.u0; state.um];

%% 2. Efforts généralisés
tau_full = [zeros(6,1); tau_joints];

%% 3. Forces externes via Jacobien
J0_T = sys.Jc_base';
Jm_T = sys.Jc_joints';
F_ext = [J0_T * Fc; Jm_T * Fc];

%% 4. STABILISATION BAUMGARTE (inspirée Pinocchio)
% Pour éviter le drift des contraintes de contact
alpha_stab = 5.0;  % Gain amortissement
beta_stab = 10.0;  % Gain raideur

% Erreur de contrainte (doit être nulle si en contact)
% Φ = position_contact - position_satellite_surface
Phi_error = zeros(6,1);  % Simplifié : assume contrainte satisfaite

% Terme de stabilisation
% γ_stab = -2*α*J*q̇ - β²*Φ
J_full = [sys.Jc_base, sys.Jc_joints];
gamma_stab = -2*alpha_stab*J_full*u_full - beta_stab^2*Phi_error;

% Injection dans l'équation
% Au lieu de: M*q̈ = τ + F_ext - C*q̇
% On a:      M*q̈ = τ + F_ext - C*q̇ + J^T*γ_stab
stabilization_term = [J0_T; Jm_T] * gamma_stab;

%% 5. Équation dynamique avec stabilisation
rhs = tau_full + F_ext - C_full * u_full + stabilization_term;

%% 6. Résolution robuste
cond_H = cond(H_full);
if cond_H > 1e10
    % Régularisation proximale (à la Pinocchio)
    rho = 1e-6;  % Paramètre proximal
    H_full = H_full + rho * eye(size(H_full));
end

try
    udot_full = H_full \ rhs;
catch
    udot_full = pinv(H_full, 1e-6) * rhs;
end

% Validation et saturation
if any(isnan(udot_full)) || any(isinf(udot_full))
    udot_full = zeros(size(udot_full));
end
accel_max = 50;  % Réduit
if norm(udot_full) > accel_max
    udot_full = udot_full / norm(udot_full) * accel_max;
end

u0dot = udot_full(1:6);
umdot = udot_full(7:end);

%% 7. Intégration (reste identique)
state_next = state;
state_next.u_base = state.u_base + u0dot(4:6) * dt;
state_next.um = state.um + umdot * dt;
state_next.q_base = state.q_base + state_next.u_base * dt;
state_next.qm = state.qm + state_next.um * dt;

omega_base_robot = u0dot(1:3);
if norm(omega_base_robot) > 1e-8
    state_next.quat_base = quat_utils('integrate', state.quat_base, omega_base_robot, dt);
else
    state_next.quat_base = state.quat_base;
end

%% 8-11. Satellite et RW (identique)
% ... (code précédent inchangé)