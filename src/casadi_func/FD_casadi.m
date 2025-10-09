function [u0dot,umdot] = FD_casadi(tau0,taum,wF0,wFm,t0,tm,P0,pm,I0,Im,Bij,Bi0,u0,um,robot)
% CasADi-compatible version of FD function
% This function solves the forward dynamics (FD) problem (it obtains the
% acceleration from forces).
%
% [u0dot,umdot] = FD_casadi(tau0,taum,wF0,wFm,t0,tm,P0,pm,I0,Im,Bij,Bi0,u0,um,robot)
%
% This version is compatible with CasADi symbolic variables
% Main changes:
% - Im, Bij, Bi0 expected as cell arrays
% - M_hat, psi_hat, psi, mu use cell arrays
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
n_q = robot.n_q;

%--- Check if we're dealing with CasADi variables ---%
is_casadi = false;
try
    if isa(tau0,'casadi.SX') || isa(tau0,'casadi.MX') || isa(P0,'casadi.SX') || isa(P0,'casadi.MX')
        is_casadi = true;
    end
catch
    % CasADi not available
end

%---- Inverse Dynamics with 0 accelerations ---%
%Recompute Accelerations with u0dot=umdot=0
if is_casadi
    import casadi.*
    [t0dot,tmdot] = Accelerations_casadi(t0,tm,P0,pm,Bi0,Bij,u0,um,SX.zeros(6,1),SX.zeros(n_q,1),robot);
else
    [t0dot,tmdot] = Accelerations_casadi(t0,tm,P0,pm,Bi0,Bij,u0,um,zeros(6,1),zeros(n_q,1),robot);
end

%Use the inverse dynamics
[tau0_0ddot,taum_0ddot] = ID_casadi(wF0,wFm,t0,tm,t0dot,tmdot,P0,pm,I0,Im,Bij,Bi0,robot);

%--- Forward Dynamics ---%

%Initialize solution
phi0 = tau0 - tau0_0ddot;
phi = taum - taum_0ddot;

%--- M hat, psi hat and psi ---%
%Pre-allocate as cell arrays
M_hat = cell(n, 1);
psi_hat = cell(n, 1);
psi = cell(n, 1);

if is_casadi
    import casadi.*
    for i=1:n
        M_hat{i} = SX.zeros(6,6);
        psi_hat{i} = SX.zeros(6,1);
        psi{i} = SX.zeros(6,1);
    end
else
    for i=1:n
        M_hat{i} = zeros(6,6);
        psi_hat{i} = zeros(6,1);
        psi{i} = zeros(6,1);
    end
end

%Backwards recursion
for i=n:-1:1
    %Initialize
    M_hat{i}(1:6,1:6) = [Im{i}(1:3,1:3), zeros(3,3); 
                         zeros(3,3), robot.links(i).mass*eye(3)];
    
    %Add children contributions
    for j=find(robot.con.child(:,i))'
        M_hatii = M_hat{j} - psi_hat{j}*psi{j}';
        M_hat{i} = M_hat{i} + Bij{j}{i}'*M_hatii*Bij{j}{i};
    end
    
    if robot.joints(i).type == 0
        psi_hat{i}(1:6) = zeros(6,1);
        psi{i}(1:6) = zeros(6,1);
    else
        psi_hat{i}(1:6) = M_hat{i}*pm(1:6,i);
        denominator = pm(1:6,i)'*psi_hat{i};
        psi{i}(1:6) = psi_hat{i}/denominator;
    end
end

%Base-link
M_hat0 = [I0, zeros(3,3); zeros(3,3), robot.base_link.mass*eye(3)];
%Add children contributions
for j=find(robot.con.child_base)'
    M_hat0ii = M_hat{j} - psi_hat{j}*psi{j}';
    M_hat0 = M_hat0 + Bi0{j}'*M_hat0ii*Bi0{j};
end
psi_hat0 = M_hat0*P0;

%--- eta ---%
%Pre-allocate and initialize as cell arrays
eta = cell(n, 1);
if is_casadi
    import casadi.*
    phi_hat = SX.zeros(n,1);
    phi_tilde = SX.zeros(n_q,1);
    for i=1:n
        eta{i} = SX.zeros(6,1);
    end
else
    phi_hat = zeros(n,1);
    phi_tilde = zeros(n_q,1);
    for i=1:n
        eta{i} = zeros(6,1);
    end
end

%Backwards recursion
for i=n:-1:1
    %Initialize
    eta{i}(1:6) = zeros(6,1);
    %Add children contributions
    for j=find(robot.con.child(:,i))'
        eta{i} = eta{i} + Bij{j}{i}'*(psi{j}*phi_hat(j) + eta{j});
    end
    phi_hat(i) = -pm(1:6,i)'*eta{i};
    if robot.joints(i).type ~= 0
        phi_hat(i) = phi_hat(i) + phi(robot.joints(i).q_id);
        denominator = pm(1:6,i)'*psi_hat{i};
        phi_tilde(robot.joints(i).q_id) = phi_hat(i)/denominator;
    end
end

%Base-link
if is_casadi
    eta0 = SX.zeros(6,1);
else
    eta0 = zeros(6,1);
end
%Add children contributions
for j=find(robot.con.child_base)'
    eta0 = eta0 + Bi0{j}'*(psi{j}*phi_hat(j) + eta{j});
end
phi_hat0 = phi0 - P0'*eta0;

% Solve for phi_tilde0
% In CasADi, we need to use solve() or direct inversion
if is_casadi
    import casadi.*
    phi_tilde0 = solve(P0'*psi_hat0, phi_hat0);
else
    phi_tilde0 = (P0'*psi_hat0)\phi_hat0;
end

%--- Base-link acceleration ---%
u0dot = phi_tilde0;

%--- Manipulator acceleration (and mu) ---%

%Pre-allocate
mu = cell(n, 1);
if is_casadi
    import casadi.*
    umdot = SX.zeros(n_q,1);
    for i=1:n
        mu{i} = SX.zeros(6,1);
    end
else
    umdot = zeros(n_q,1);
    for i=1:n
        mu{i} = zeros(6,1);
    end
end

%Forward recursion
for i=1:n
    
    if robot.joints(i).parent_link == 0
        %First joint
        mu{i}(1:6) = Bi0{i}*(P0*u0dot);
    else
        %Rest of the links
        parent_link_id = robot.joints(i).parent_link;
        parent_joint = robot.links(parent_link_id).parent_joint;
        if parent_joint > 0 && robot.joints(parent_joint).type ~= 0
            mu_aux = pm(1:6,parent_joint)*umdot(robot.joints(parent_joint).q_id) + mu{parent_link_id};
        else
            mu_aux = mu{parent_link_id};
        end
        mu{i} = Bij{i}{parent_link_id}*mu_aux;
    end
    
    %Joint acceleration
    if robot.joints(i).type ~= 0
        umdot(robot.joints(i).q_id,1) = phi_tilde(robot.joints(i).q_id) - psi{i}'*mu{i};
    end
end

end