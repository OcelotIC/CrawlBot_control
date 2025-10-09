function [t0dot,tLdot]=Accelerations_casadi(t0,tL,P0,pm,Bi0,Bij,u0,um,u0dot,umdot,robot)
% CasADi-compatible version of Accelerations function
% Computes the operational-space accelerations (twist-rate) of the multibody system.
%
% [t0dot,tLdot]=Accelerations_casadi(t0,tL,P0,pm,Bi0,Bij,u0,um,u0dot,umdot,robot)
%
% This version is compatible with CasADi symbolic variables
% Main changes:
% - Bij, Bi0 expected as cell arrays
% - Omegam uses cell array
% - Removed 'like' declarations
% - Compatible with both numeric and CasADi SX inputs

%=== LICENSE ===%
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU Lesser General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.

%=== CODE ===%

%--- Number of links and Joints ---%
n = robot.n_links_joints;

%--- Check if we're dealing with CasADi variables ---%
is_casadi = false;
try
    if isa(t0,'casadi.SX') || isa(t0,'casadi.MX') || isa(P0,'casadi.SX') || isa(P0,'casadi.MX')
        is_casadi = true;
    end
catch
    % CasADi not available
end

%--- Omega matrices ---%
%Base-link
Omega0 = [SkewSym_casadi(t0(1:3)), zeros(3,3);
          zeros(3,3), zeros(3,3)];

%Pre-allocate Omegam as cell array
Omegam = cell(n, 1);
if is_casadi
    import casadi.*
    for i=1:n
        Omegam{i} = SX.zeros(6,6);
    end
else
    for i=1:n
        Omegam{i} = zeros(6,6);
    end
end

%Compute Omega for manipulator
for i=1:n
    Omegam{i}(1:6,1:6) = [SkewSym_casadi(tL(1:3,i)), zeros(3,3);
                          zeros(3,3), SkewSym_casadi(tL(1:3,i))];
end

%--- Twist Rate ---%
%Base-link
t0dot = Omega0*P0*u0 + P0*u0dot;

%Pre-allocate
if is_casadi
    import casadi.*
    tLdot = SX.zeros(6,n);
else
    tLdot = zeros(6,n);
end

%Forward recursion
for i=1:n
    
    if robot.joints(i).parent_link == 0
        %First Link
        tLdot(1:6,i) = Bi0{i}*t0dot + ...
                       [zeros(3,6); SkewSym_casadi(t0(4:6)-tL(4:6,i)), zeros(3,3)]*t0;
    else
        %Rest of the links
        parent_link_id = robot.joints(i).parent_link;
        tLdot(1:6,i) = Bij{i}{parent_link_id}*tLdot(1:6,parent_link_id) + ...
                       [zeros(3,6); SkewSym_casadi(tL(4:6,parent_link_id)-tL(4:6,i)), zeros(3,3)]*tL(1:6,parent_link_id);
    end
    
    %Add joint contribution
    if robot.joints(i).type ~= 0
        tLdot(1:6,i) = tLdot(1:6,i) + ...
                       Omegam{i}*pm(1:6,i)*um(robot.joints(i).q_id) + ...
                       pm(1:6,i)*umdot(robot.joints(i).q_id);
    end
    
end

end

%--- Helper function ---%
function x_skew = SkewSym_casadi(x)
% CasADi-compatible version of SkewSym
% Computes the skew-symmetric matrix of a vector

x_skew = [0, -x(3), x(2); 
          x(3), 0, -x(1); 
          -x(2), x(1), 0];
end
