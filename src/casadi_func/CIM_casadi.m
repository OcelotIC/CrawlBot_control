function [C0, C0m, Cm0, Cm] = CIM_casadi(t0,tL,I0,Im,M0_tilde,Mm_tilde,Bij,Bi0,P0,pm,robot)
% CasADi-compatible version of CIM function
% Computes the Generalized Convective Inertia Matrix C of the multibody system.
%
% [C0, C0m, Cm0, Cm] = CIM_casadi(t0,tL,I0,Im,M0_tilde,Mm_tilde,Bij,Bi0,P0,pm,robot)
%
% This version is compatible with CasADi symbolic variables
% Main changes:
% - Im, Mm_tilde, Bij, Bi0 expected as cell arrays
% - Omega, Mdot, etc. use cell arrays
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
    if isa(t0,'casadi.SX') || isa(t0,'casadi.MX') || isa(I0,'casadi.SX') || isa(I0,'casadi.MX')
        is_casadi = true;
    end
catch
    % CasADi not available
end

%--- Omega ---%
%Base-link Omega
Omega0 = [SkewSym_casadi(t0(1:3)), zeros(3,3);
          zeros(3,3), zeros(3,3)];

%Pre-allocate Omega as cell array
Omega = cell(n, 1);
if is_casadi
    import casadi.*
    for i=1:n
        Omega{i} = SX.zeros(6,6);
    end
else
    for i=1:n
        Omega{i} = zeros(6,6);
    end
end

%Compute Omega
for i=1:n
    Omega{i}(1:6,1:6) = [SkewSym_casadi(tL(1:3,i)), zeros(3,3);
                         zeros(3,3), SkewSym_casadi(tL(1:3,i))];
end

%--- Mdot ---%
%Base-link Mdot
Mdot0 = [Omega0(1:3,1:3)*I0, zeros(3,3); zeros(3,3), zeros(3,3)];

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

%Compute Mdot
for i=1:n
    Mdot{i}(1:6,1:6) = [Omega{i}(1:3,1:3)*Im{i}(1:3,1:3), zeros(3,3); 
                        zeros(3,3), zeros(3,3)];
end

%--- Mdot tilde ---%
%Pre-Allocate Mdot_tilde as cell array
Mdot_tilde = cell(n, 1);
if is_casadi
    import casadi.*
    for i=1:n
        Mdot_tilde{i} = SX.zeros(6,6);
    end
else
    for i=1:n
        Mdot_tilde{i} = zeros(6,6);
    end
end

%Backwards recursion
for i=n:-1:1
    %Initialize
    Mdot_tilde{i} = Mdot{i};
    %Add children contributions
    child = find(robot.con.child(:,i))';
    for j=1:length(child)
        child_id = child(j);
        Mdot_tilde{i} = Mdot_tilde{i} + Mdot_tilde{child_id};
    end
end

%Base-link
Mdot0_tilde = Mdot0;
%Add children contributions
child = find(robot.con.child_base)';
for j=1:length(child)
    child_id = child(j);
    Mdot0_tilde = Mdot0_tilde + Mdot_tilde{child_id};
end

%--- Bdot ---%
%Pre-allocate Bdotij as nested cell array
Bdotij = cell(n, 1);
for i=1:n
    Bdotij{i} = cell(n, 1);
    for j=1:n
        if is_casadi
            import casadi.*
            Bdotij{i}{j} = SX.zeros(6,6);
        else
            Bdotij{i}{j} = zeros(6,6);
        end
    end
end

%Compute Bdotij
for j=1:n
    for i=1:n
        if robot.con.branch(i,j) == 1
            %Links are in the same branch
            Bdotij{i}{j}(1:6,1:6) = [zeros(3,3), zeros(3,3); 
                                      SkewSym_casadi(tL(4:6,j)-tL(4:6,i)), zeros(3,3)];
        else
            %Links are not in the same branch (already initialized to zeros)
        end
    end
end

%--- Hij tilde ---%
%Pre-allocate Hij_tilde as nested cell array
Hij_tilde = cell(n, 1);
for i=1:n
    Hij_tilde{i} = cell(n, 1);
    for j=1:n
        if is_casadi
            import casadi.*
            Hij_tilde{i}{j} = SX.zeros(6,6);
        else
            Hij_tilde{i}{j} = zeros(6,6);
        end
    end
end

%Hij_tilde
for i=n:-1:1
    for j=n:-1:1
        Hij_tilde{i}{j} = Mm_tilde{i} * Bdotij{i}{j};
        %Add children contributions
        child = find(robot.con.child(:,i))';
        for k=1:length(child)
            child_id = child(k);
            Hij_tilde{i}{j} = Hij_tilde{i}{j} + Bij{child_id}{i}' * Hij_tilde{child_id}{i};
        end
    end
end

%Pre-allocate Hi0_tilde as cell array
Hi0_tilde = cell(n, 1);
if is_casadi
    import casadi.*
    for i=1:n
        Hi0_tilde{i} = SX.zeros(6,6);
    end
else
    for i=1:n
        Hi0_tilde{i} = zeros(6,6);
    end
end

%Hi0_tilde
for i=n:-1:1
    Bdot = [zeros(3,3), zeros(3,3); 
            SkewSym_casadi(t0(4:6)-tL(4:6,i)), zeros(3,3)];
    Hi0_tilde{i} = Mm_tilde{i} * Bdot;
    %Add children contributions
    child = find(robot.con.child(:,i))';
    for k=1:length(child)
        child_id = child(k);
        Hi0_tilde{i} = Hi0_tilde{i} + Bij{child_id}{i}' * Hij_tilde{child_id}{i};
    end
end

%--- C Matrix ---%
%Pre-allocate
if is_casadi
    import casadi.*
    Cm = SX.zeros(n_q, n_q);
    C0m = SX.zeros(6, n_q);
    Cm0 = SX.zeros(n_q, 6);
else
    Cm = zeros(n_q, n_q);
    C0m = zeros(6, n_q);
    Cm0 = zeros(n_q, 6);
end

%Cm Matrix
for j=1:n
    for i=1:n
        %Joints must not be fixed and links on the same branch
        if (robot.joints(i).type ~= 0 && robot.joints(j).type ~= 0) && (robot.con.branch(i,j) == 1 || robot.con.branch(j,i) == 1)
            %Compute Cm matrix
            if i <= j
                %Add children contributions
                if is_casadi
                    child_con = SX.zeros(6,6);
                else
                    child_con = zeros(6,6);
                end
                child = find(robot.con.child(:,j))';
                for k=1:length(child)
                    child_id = child(k);
                    child_con = child_con + Bij{child_id}{i}' * Hij_tilde{child_id}{j};
                end
                Cm(robot.joints(i).q_id, robot.joints(j).q_id) = pm(1:6,i)' * (Bij{j}{i}' * Mm_tilde{j} * Omega{j} + child_con + Mdot_tilde{j}) * pm(1:6,j);
            else
                Cm(robot.joints(i).q_id, robot.joints(j).q_id) = pm(1:6,i)' * (Mm_tilde{i} * Bij{i}{j} * Omega{j} + Hij_tilde{i}{j} + Mdot_tilde{i}) * pm(1:6,j);
            end
        end
    end
end

%C0 matrix
%Add children contributions
if is_casadi
    child_con = SX.zeros(6,6);
else
    child_con = zeros(6,6);
end
child = find(robot.con.child_base)';
for k=1:length(child)
    child_id = child(k);
    child_con = child_con + Bi0{child_id}' * Hi0_tilde{child_id};
end
C0 = P0' * (M0_tilde * Omega0 + child_con + Mdot0_tilde) * P0;

%C0m
for j=1:n
    if robot.joints(j).type ~= 0
        if j == n
            C0m(1:6, robot.joints(j).q_id) = P0' * (Bi0{j}' * Mm_tilde{j} * Omega{j} + Mdot_tilde{j}) * pm(1:6,j);
        else
            %Add children contributions
            if is_casadi
                child_con = SX.zeros(6,6);
            else
                child_con = zeros(6,6);
            end
            child = find(robot.con.child(:,j))';
            for k=1:length(child)
                child_id = child(k);
                child_con = child_con + Bi0{child_id}' * Hij_tilde{child_id}{j};
            end
            C0m(1:6, robot.joints(j).q_id) = P0' * (Bi0{j}' * Mm_tilde{j} * Omega{j} + child_con + Mdot_tilde{j}) * pm(1:6,j);
        end
    end
end

%Cm0
for i=1:n
    if robot.joints(i).type ~= 0  
        Cm0(robot.joints(i).q_id, 1:6) = pm(1:6,i)' * (Mm_tilde{i} * Bi0{i} * Omega0 + Hi0_tilde{i} + Mdot_tilde{i}) * P0;
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