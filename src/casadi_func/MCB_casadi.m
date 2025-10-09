function [M0_tilde,Mm_tilde]=MCB_casadi(I0,Im,Bij,Bi0,robot)
% CasADi-compatible version of MCB function
% Computes the Mass Composite Body Matrix (MCB) of the multibody system.
%
% [M0_tilde,Mm_tilde]=MCB_casadi(I0,Im,Bij,Bi0,robot)
%
% This version is compatible with CasADi symbolic variables
% Main changes:
% - Im, Bij, Bi0 expected as cell arrays
% - Mm_tilde returned as cell array
% - Removed 'like' declarations
% - Compatible with both numeric and CasADi SX inputs

%=== LICENSE ===%
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU Lesser General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.

%=== CODE ===%

%Number of links and Joints
n = robot.n_links_joints;

%--- Check if we're dealing with CasADi variables ---%
is_casadi = false;
try
    if isa(I0,'casadi.SX') || isa(I0,'casadi.MX')
        is_casadi = true;
    end
catch
    % CasADi not available
end

%--- Pre-allocate Mm_tilde as cell array ---%
Mm_tilde = cell(n, 1);

if is_casadi
    import casadi.*
    for i=1:n
        Mm_tilde{i} = SX.zeros(6,6);
    end
else
    for i=1:n
        Mm_tilde{i} = zeros(6,6);
    end
end

%--- Backwards recursion ---%
for i=n:-1:1
    %Initialize M tilde
    Mm_tilde{i}(1:6,1:6) = [Im{i}(1:3,1:3), zeros(3,3); 
                            zeros(3,3), robot.links(i).mass*eye(3)];
    
    %Add children contributions
    child = find(robot.con.child(:,i))';
    for j=1:length(child)
        child_id = child(j);
        Mm_tilde{i} = Mm_tilde{i} + Bij{child_id}{i}' * Mm_tilde{child_id} * Bij{child_id}{i};
    end
end

%--- Base-link M tilde ---%
M0_tilde = [I0, zeros(3,3); zeros(3,3), robot.base_link.mass*eye(3)];

%Add children contributions
child = find(robot.con.child_base)';
for j=1:length(child)
    child_id = child(j);
    M0_tilde = M0_tilde + Bi0{child_id}' * Mm_tilde{child_id} * Bi0{child_id};
end

end
