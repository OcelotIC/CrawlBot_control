function output = quat_utils(operation, varargin)
% Utilitaires quaternions COMPATIBLES SPART
%
% Convention SPART: q = [q1, q2, q3, q4] où q4 est la partie SCALAIRE
%
% Usage:
%   q_next = quat_utils('integrate', q, omega, dt)
%   err = quat_utils('error', q_des, q_cur)

switch operation
    case 'integrate'
        output = integrate_quat_spart(varargin{:});
    case 'error'
        output = quat_error(varargin{:});
    otherwise
        error('Unknown operation: %s', operation);
end

end

function q_next = integrate_quat_spart(q, omega, dt)
% Intégration quaternion selon convention SPART
% Convention: q = [q1, q2, q3, q4] où q4 = partie scalaire
%
% Équation cinématique des quaternions:
%   q̇ = 0.5 * Ω(ω) * q
%
% où Ω(ω) = [ 0   -ω3   ω2   ω1 ]
%           [ ω3   0   -ω1   ω2 ]
%           [-ω2   ω1   0    ω3 ]
%           [-ω1  -ω2  -ω3   0  ]
%
% Intégration Euler: q(t+dt) = q(t) + q̇(t) * dt

omega_norm = norm(omega);

% Cas stationnaire
if omega_norm < 1e-12
    q_next = q;
    return;
end

%% Méthode 1: Intégration exacte via formule de Rodrigues pour quaternions
% Plus précise que Euler simple, utilisée dans SPART

% Angle de rotation
theta = omega_norm * dt;

% Axe de rotation normalisé
axis = omega / omega_norm;

% Quaternion de rotation (convention SPART: [qv; qs])
% Rotation de θ autour de axis:
%   q_rot = [sin(θ/2) * axis; cos(θ/2)]
q_rot = [sin(theta/2) * axis;  % q1, q2, q3 (partie vectorielle)
         cos(theta/2)];         % q4 (partie scalaire)

% Multiplication de quaternions (convention SPART)
q_next = quat_product_spart(q_rot, q);

% Normalisation (pour stabilité numérique)
q_next = q_next / norm(q_next);

end

function q_prod = quat_product_spart(q1, q2)
% Produit de quaternions selon convention SPART
% Convention: q = [q1, q2, q3, q4] où q4 = partie scalaire
%
% Formule: q1 ⊗ q2 = [q1_s*q2_v + q2_s*q1_v + q1_v × q2_v; 
%                     q1_s*q2_s - q1_v·q2_v]
%
% où qv = [q1, q2, q3]' (partie vectorielle)
%     qs = q4           (partie scalaire)

% Extraction parties vectorielle et scalaire
q1_v = q1(1:3);  % Partie vectorielle q1
q1_s = q1(4);    % Partie scalaire q1

q2_v = q2(1:3);  % Partie vectorielle q2
q2_s = q2(4);    % Partie scalaire q2

% Partie vectorielle du produit
qv_prod = q1_s * q2_v + q2_s * q1_v + cross(q1_v, q2_v);

% Partie scalaire du produit
qs_prod = q1_s * q2_s - dot(q1_v, q2_v);

% Quaternion résultat (convention SPART)
q_prod = [qv_prod; qs_prod];

end

function err = quat_error(q_des, q_cur)
% Erreur quaternion (partie vectorielle)
% Convention SPART: q = [q1, q2, q3, q4]

% Conjugué quaternion courant
q_conj_cur = [-q_cur(1:3); q_cur(4)];  % Conjugué: [-qv; qs]

% Erreur quaternion: q_err = q_des ⊗ conj(q_cur)
q_err = quat_product_spart(q_des, q_conj_cur);

% Approximation linéaire: erreur = 2 * partie vectorielle
err = 2 * q_err(1:3);

end