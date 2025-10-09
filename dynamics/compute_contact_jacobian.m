function [Jc_base, Jc_joints] = compute_contact_jacobian(r_contact, r0, rL, P0, pm, robot)
% Calcule Jacobienne 6D du contact (utilise Jacob.m de SPART)
%
% Jacob.m retourne DÉJÀ un Jacobien 6D complet :
%   J0  : [6x6]   -> [ω; v] base
%   Jm  : [6xn_q] -> [ω; v] manipulateur
%
% Structure du twist retourné par Jacob:
%   t = [ω (3x1); v (3x1)] = J0*u0 + Jm*um

n = robot.n_links_joints;

%% Appel direct à Jacob.m pour le point de contact
[Jc_base, Jc_joints] = Jacob(r_contact, r0, rL, P0, pm, n, robot);

% Jc_base   : [6x6]   mapping u0 -> twist_contact in ine
% Jc_joints : [6xn_q] mapping um -> twist_contact in ine

end