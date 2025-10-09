function [t0,tL]=Velocities_casadi(Bij,Bi0,P0,pm,u0,um,robot)
% CasADi-compatible version of Velocities function
% Computes the operational-space velocities of the multibody system.
%
% [t0,tL]=Velocities_casadi(Bij,Bi0,P0,pm,u0,um,robot)
%
% This version is compatible with CasADi symbolic variables
% Main changes:
% - Bij and Bi0 are expected as cell arrays from DiffKinematics_casadi
% - Removed 'like' declarations
% - Compatible with both numeric and CasADi SX inputs

%=== LICENSE ===%
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU Lesser General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.

%=== CODE ===%

%--- Number of links and joints ---%
n = robot.n_links_joints;

%--- Check if we're dealing with CasADi variables ---%
is_casadi = false;
try
    if isa(P0,'casadi.SX') || isa(P0,'casadi.MX') || isa(u0,'casadi.SX') || isa(u0,'casadi.MX')
        is_casadi = true;
    end
catch
    % CasADi not available
end

%--- Pre-allocate ---%
if is_casadi
    import casadi.*
    tL = SX.zeros(6,n);
else
    tL = zeros(6,n);
end

%--- Base-link ---%
t0 = P0 * u0;

%--- Forward recursion to obtain the twist ---%
for i=1:n
    
    if robot.joints(i).parent_link == 0
        %First link
        tL(1:6,i) = Bi0{i} * t0;
    else
        %Rest of the links
        parent_link_id = robot.joints(i).parent_link;
        tL(1:6,i) = Bij{i}{parent_link_id} * tL(1:6,parent_link_id);
    end
    
    %Add joint contribution
    if robot.joints(i).type ~= 0
        tL(1:6,i) = tL(1:6,i) + pm(1:6,i) * um(robot.joints(i).q_id);
    end
    
end

end