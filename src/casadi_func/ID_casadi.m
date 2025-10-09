function [tau0,taum] = ID_casadi(wF0,wFm,t0,tL,t0dot,tLdot,P0,pm,I0,Im,Bij,Bi0,robot)
% CasADi-compatible version of ID function
% This function solves the inverse dynamics (ID) problem (it obtains the
% generalized forces from the accelerations) for a manipulator.
%
% [tau0,taum] = ID_casadi(wF0,wFm,t0,tL,t0dot,tLdot,P0,pm,I0,Im,Bij,Bi0,robot)
%
% This version is compatible with CasADi symbolic variables
% Main changes:
% - Im, Bij, Bi0 expected as cell arrays
% - Mdot, wq, wq_tilde use cell arrays
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
    if isa(wF0,'casadi.SX') || isa(wF0,'casadi.MX') || isa(t0,'casadi.SX') || isa(t0,'casadi.MX')
        is_casadi = true;
    end
catch
    % CasADi not available
end

%--- Mdot ---%
%base-link Mdot
Mdot0 = [SkewSym_casadi(t0(1:3))*I0, zeros(3,3); zeros(3,3), zeros(3,3)];

%Pre-allocate Mdot as cell array
Mdot = cell(n, 1);
if is_casadi
    import casadi.*
    for i=1:n
        Mdot{i} = SX.zeros(6,6);
    end
else
    for i=1:n
        Mdot{i} = zeros(6,6);
    end
end

%Manipulator Mdot
for i=1:n
    Mdot{i}(1:6,1:6) = [SkewSym_casadi(tL(1:3,i))*Im{i}(1:3,1:3), zeros(3,3); 
                        zeros(3,3), zeros(3,3)];
end

%--- Forces ---%
%Base-link
wq0 = [I0, zeros(3,3); zeros(3,3), robot.base_link.mass*eye(3)]*t0dot + Mdot0*t0 - wF0;

%Pre-allocate wq as cell array
wq = cell(n, 1);
if is_casadi
    import casadi.*
    for i=1:n
        wq{i} = SX.zeros(6,1);
    end
else
    for i=1:n
        wq{i} = zeros(6,1);
    end
end

%Manipulator
for i=1:n
    wq{i}(1:6) = [Im{i}(1:3,1:3), zeros(3,3); 
                  zeros(3,3), robot.links(i).mass*eye(3)]*tLdot(1:6,i) + ...
                  Mdot{i}*tL(1:6,i) - wFm(1:6,i);
end

%Pre-allocate wq_tilde as cell array
wq_tilde = cell(n, 1);
if is_casadi
    import casadi.*
    for i=1:n
        wq_tilde{i} = SX.zeros(6,1);
    end
else
    for i=1:n
        wq_tilde{i} = zeros(6,1);
    end
end

%Backwards recursion
for i=n:-1:1
    %Initialize wq_tilde
    wq_tilde{i}(1:6) = wq{i}(1:6);
    %Add children contributions
    for j=find(robot.con.child(:,i))'
        wq_tilde{i} = wq_tilde{i} + Bij{j}{i}'*wq_tilde{j};
    end
end

%Base-link
wq_tilde0 = wq0;
%Add children contributions
for j=find(robot.con.child_base)'
    wq_tilde0 = wq_tilde0 + Bi0{j}'*wq_tilde{j};
end

%---- Joint forces ---%
%Base-link
tau0 = P0'*wq_tilde0;

%Pre-allocate
if is_casadi
    import casadi.*
    taum = SX.zeros(robot.n_q,1);
else
    taum = zeros(robot.n_q,1);
end

%Manipulator joint forces
for i=1:n
    if robot.joints(i).type ~= 0
        taum(robot.joints(i).q_id,1) = pm(1:6,i)'*wq_tilde{i};
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