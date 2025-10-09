function [I0,Im]=I_I_casadi(R0,RL,robot)
% CasADi-compatible version of I_I function
% Projects the link inertias in the inertial CCS.
%
% [I0,Im]=I_I_casadi(R0,RL,robot)
%
% This version is compatible with CasADi symbolic variables
% Main changes:
% - RL expected as cell array from Kinematics_casadi
% - Im returned as cell array instead of 3D array
% - Removed 'like' declarations
% - Compatible with both numeric and CasADi SX inputs

%=== LICENSE ===%
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU Lesser General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.

%=== CODE ===%

%--- Check if we're dealing with CasADi variables ---%
is_casadi = false;
try
    if isa(R0,'casadi.SX') || isa(R0,'casadi.MX')
        is_casadi = true;
    end
catch
    % CasADi not available
end

%--- Base-link inertia ---%
I0 = R0 * robot.base_link.inertia * R0';

%--- Pre-allocate inertias as cell array ---%
Im = cell(robot.n_links_joints, 1);

if is_casadi
    import casadi.*
    for i=1:robot.n_links_joints
        Im{i} = SX.zeros(3,3);
    end
else
    for i=1:robot.n_links_joints
        Im{i} = zeros(3,3);
    end
end

%--- Inertias of the links ---%
for i=1:robot.n_links_joints
    Im{i}(1:3,1:3) = RL{i}(1:3,1:3) * robot.links(i).inertia * RL{i}(1:3,1:3)';
end

end