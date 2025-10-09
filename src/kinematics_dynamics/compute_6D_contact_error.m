function [Phi, Phi_dot, Jc] = compute_6D_contact_error(R0, r0, qm, u0, um, ...
                                                        contact_link_id, ...
                                                        r_contact_des, R_contact_des, ...
                                                        robot)
% Calcule l'erreur de contrainte 6D (position + orientation) pour un contact
%
% Inputs:
%   R0, r0, qm, u0, um : État actuel du robot
%   contact_link_id : ID du lien où se trouve le point de contact
%   r_contact_des : Position désirée du contact [3x1] dans le repère inertiel
%   R_contact_des : Orientation désirée du contact [3x3] dans le repère inertiel
%   robot : Structure SPART
%
% Outputs:
%   Phi : Erreur de contrainte [6x1] = [erreur_position; erreur_orientation]
%   Phi_dot : Dérivée temporelle de Phi [6x1]
%   Jc : Jacobien du contact [6 x n_dof]

%% 1. Cinématique directe avec SPART
[RJ, RL, rJ, rL, e, g] = Kinematics(R0, r0, qm, robot);

% Position actuelle du lien de contact (rL est [3 x n_links])
r_contact_actual = rL(:, contact_link_id);

% Orientation actuelle du lien de contact (RL est [3 x 3 x n_links])
R_contact_actual = RL(:, :, contact_link_id);

%% 2. Erreur de POSITION (3 composantes)
Phi_position = r_contact_actual - r_contact_des;

%% 3. Erreur d'ORIENTATION (3 composantes)
% Utiliser la représentation "rotation vector" (angle-axis)
R_error = R_contact_des' * R_contact_actual;

% Méthode robuste pour extraire l'erreur d'orientation
Phi_orientation = rotation_matrix_to_vector_robust(R_error);

%% 4. Erreur totale 6D
Phi = [Phi_position; Phi_orientation];

%% 5. Dérivée temporelle Phi_dot = Jc * qdot
% Calcul de la cinématique différentielle
[Bij, Bi0, P0, pm] = DiffKinematics(R0, r0, rL, e, g, robot);

% Calcul des twists (vitesses des liens)
[t0, tm] = Velocities(Bij, Bi0, P0, pm, u0, um, robot);

% Jacobien géométrique 6D au point de contact
[J0c, Jmc] = Jacob(r_contact_actual, r0, rL, P0, pm, contact_link_id, robot);
Jc = [J0c, Jmc];  % [6 x (6 + n_q)]

% Vitesse généralisée
qdot = [u0; um];

% Phi_dot = Jc * qdot
Phi_dot = Jc * qdot;

end


%% Fonction auxiliaire ROBUSTE : Conversion rotation matrix → rotation vector
function phi = rotation_matrix_to_vector_robust(R)
    % Extrait le vecteur de rotation (angle-axis scaled by angle) d'une matrice de rotation
    % Version robuste qui gère tous les cas (petites et grandes rotations)
    
    % S'assurer que R est bien une matrice de rotation
    R = (R + R') / 2;  % Forcer la symétrie si petite erreur numérique
    
    % Calcul de l'angle
    cos_theta = (trace(R) - 1) / 2;
    
    % Clip pour éviter des erreurs numériques
    cos_theta = max(-1, min(1, cos_theta));
    theta = acos(cos_theta);
    
    if abs(theta) < 1e-6
        % Cas 1 : Rotation très petite → approximation linéaire
        phi = 0.5 * [R(3,2) - R(2,3);
                     R(1,3) - R(3,1);
                     R(2,1) - R(1,2)];
        
    elseif abs(theta - pi) < 1e-6
        % Cas 2 : Rotation proche de 180° → cas singulier
        % Trouver l'axe comme vecteur propre associé à la valeur propre 1
        [V, D] = eig(R);
        [~, idx] = min(abs(diag(D) - 1));
        axis = V(:, idx);
        axis = axis / norm(axis);  % Normaliser
        
        % Choisir le bon signe (convention)
        if axis(3) < 0 || (abs(axis(3)) < 1e-10 && axis(2) < 0)
            axis = -axis;
        end
        
        phi = theta * axis;
        
    else
        % Cas 3 : Rotation générale (angle entre 0 et 180°)
        axis = 1/(2*sin(theta)) * [R(3,2) - R(2,3);
                                     R(1,3) - R(3,1);
                                     R(2,1) - R(1,2)];
        phi = theta * axis;
    end
end