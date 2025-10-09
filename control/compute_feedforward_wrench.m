function [F_d_r, F_d_b] = compute_feedforward_wrench(sys, traj_des, t_idx, gains)
% Calcule wrenches désirés (éq. 12-16)
% ✅ AJOUT: Saturation pour éviter valeurs excessives

%% Validation entrées
if any(isnan(sys.r0)) || any(isinf(sys.r0))
    warning('[Feedforward] sys.r0 invalide');
    F_d_r = zeros(6,1);
    F_d_b = zeros(6,1);
    return;
end

%% Robot wrench (éq. 12-14)
r_des = traj_des.pos(:, t_idx);
v_des = traj_des.vel(:, t_idx);

r_cur = sys.r0;
v_cur = sys.t0(4:6);

% Erreurs
e_pos = r_des - r_cur;
e_vel = v_des - v_cur;

% ✅ SATURATION erreurs (éviter explosions numériques)
e_pos_max = 10.0;  % 10 mètres max
e_vel_max = 5.0;   % 5 m/s max

if norm(e_pos) > e_pos_max
    warning('[Feedforward] Erreur position excessive: %.2f m', norm(e_pos));
    e_pos = e_pos / norm(e_pos) * e_pos_max;
end
if norm(e_vel) > e_vel_max
    e_vel = e_vel / norm(e_vel) * e_vel_max;
end

% Feedforward (simplifié)
F_ff = zeros(6,1);

% Feedback translation
F_fb_trans = gains.Kr(4:6,4:6) * e_pos + gains.Dr(4:6,4:6) * e_vel;
F_fb_rot = zeros(3,1);
F_fb = [F_fb_rot; F_fb_trans];

F_d_r = F_ff + F_fb;

%  SATURATION wrench désiré robot
F_d_r_max = 1000;  % 1000 N max
if norm(F_d_r) > F_d_r_max
    F_d_r = F_d_r / norm(F_d_r) * F_d_r_max;
end

%% Base wrench (éq. 16)
omega_base = sys.t0(1:3);

% Erreur quaternion
if isfield(sys, 'quat_base')
    e_quat = 2 * sys.quat_base(2:4);
else
    e_quat = zeros(3,1);
end

% SATURATION erreur attitude
if norm(e_quat) > 1.0  % ~60° max
    e_quat = e_quat / norm(e_quat);
end

F_d_b_rot = gains.Kb * e_quat + gains.Db * (-omega_base);
F_d_b = [F_d_b_rot; zeros(3,1)];

% SATURATION wrench base
F_d_b_max = 500;  % 500 Nm max
if norm(F_d_b) > F_d_b_max
    F_d_b = F_d_b / norm(F_d_b) * F_d_b_max;
end

end