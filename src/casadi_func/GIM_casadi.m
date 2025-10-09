function [H0, H0m, Hm] = GIM_casadi(M0_tilde,Mm_tilde,Bij,Bi0,P0,pm,robot)
% CasADi-compatible version of GIM function
% Computes the Generalized Inertia Matrix (GIM) H of the multibody vehicle.
%
% [H0, H0m, Hm] = GIM_casadi(M0_tilde,Mm_tilde,Bij,Bi0,P0,pm,robot)
%
% This version is compatible with CasADi symbolic variables
% Main changes:
% - Mm_tilde, Bij, Bi0 expected as cell arrays
% - Removed 'like' declarations
% - Compatible with both numeric and CasADi SX inputs

%=== LICENSE ===%
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU Lesser General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.

%=== CODE ===%

%--- Number of links and Joints ---%
n_q = robot.n_q;
n = robot.n_links_joints;

%--- Check if we're dealing with CasADi variables ---%
is_casadi = false;
try
    if isa(M0_tilde,'casadi.SX') || isa(M0_tilde,'casadi.MX') || isa(P0,'casadi.SX') || isa(P0,'casadi.MX')
        is_casadi = true;
    end
catch
    % CasADi not available
end

%--- H matrix ---%

%Base-link inertia matrix
H0 = P0' * M0_tilde * P0;

%--- Pre-allocate Hm ---%
if is_casadi
    import casadi.*
    Hm = SX.zeros(n_q, n_q);
else
    Hm = zeros(n_q, n_q);
end

%--- Manipulator inertia matrix Hm ---%
for j=1:n
    for i=j:n
        if robot.joints(i).type ~= 0 && robot.joints(j).type ~= 0
            % Compute the element
            Hm_ij = pm(1:6,i)' * Mm_tilde{i}(1:6,1:6) * Bij{i}{j}(1:6,1:6) * pm(1:6,j);
            
            % Assign to both (i,j) and (j,i) positions
            Hm(robot.joints(i).q_id, robot.joints(j).q_id) = Hm_ij;
            if i ~= j  % Avoid duplicate assignment on diagonal
                Hm(robot.joints(j).q_id, robot.joints(i).q_id) = Hm_ij;
            end
        end
    end
end

%--- Pre-allocate H0m ---%
if is_casadi
    import casadi.*
    H0m = SX.zeros(6, n_q);
else
    H0m = zeros(6, n_q);
end

%--- Coupling inertia matrix ---%
for i=1:n
    if robot.joints(i).type ~= 0
        H0m(1:6, robot.joints(i).q_id) = (pm(1:6,i)' * Mm_tilde{i}(1:6,1:6) * Bi0{i}(1:6,1:6) * P0)';
    end
end

end
