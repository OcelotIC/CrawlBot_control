function [Bij,Bi0,P0,pm]=DiffKinematics_casadi(R0,r0,rL,e,g,robot)
% CasADi-compatible version of DiffKinematics function
% Computes the differential kinematics of the multibody system.
%
% [Bij,Bi0,P0,pm]=DiffKinematics_casadi(R0,r0,rL,e,g,robot)
% 
% This version is compatible with CasADi symbolic variables
% Main changes:
% - 4D arrays replaced with nested cell arrays
% - 3D arrays replaced with cell arrays
% - Removed 'like' declarations
% - Compatible with both numeric and CasADi SX inputs

%=== LICENSE ===%
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU Lesser General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.

%=== CODE ===%

%--- Number of links ---%
n = robot.n_links_joints;

%--- Check if we're dealing with CasADi variables ---%
is_casadi = false;
try
    if isa(R0,'casadi.SX') || isa(R0,'casadi.MX') || isa(r0,'casadi.SX') || isa(r0,'casadi.MX')
        is_casadi = true;
    end
catch
    % CasADi not available
end

%--- Twist-propagation matrix ---%

% Bij is a 4D array in original, convert to nested cell array
% Bij{i}{j} will contain the 6x6 matrix for indices i,j
Bij = cell(n,1);
for i=1:n
    Bij{i} = cell(n,1);
    for j=1:n
        if is_casadi
            import casadi.*
            Bij{i}{j} = SX.zeros(6,6);
        else
            Bij{i}{j} = zeros(6,6);
        end
    end
end

% Compute Bij
for j=1:n
    for i=1:n
        if robot.con.branch(i,j) == 1
            % Links are in the same branch
            Bij{i}{j}(1:6,1:6) = [eye(3), zeros(3,3); 
                                  SkewSym_casadi(rL(1:3,j)-rL(1:3,i)), eye(3)];
        else
            % Links are not in the same branch
            % Already initialized to zeros
        end
    end
end

% Bi0 is a 3D array in original, convert to cell array
Bi0 = cell(n,1);
for i=1:n
    if is_casadi
        import casadi.*
        Bi0{i} = SX.zeros(6,6);
    else
        Bi0{i} = zeros(6,6);
    end
end

% Compute Bi0
for i=1:n
    Bi0{i}(1:6,1:6) = [eye(3), zeros(3,3); 
                       SkewSym_casadi(r0-rL(1:3,i)), eye(3)];
end

%--- Twist-propagation "vector" ---%

% Initialize P0 and pm
if is_casadi
    import casadi.*
    P0 = SX.zeros(6,6);
    pm = SX.zeros(6,n);
else
    P0 = zeros(6,6);
    pm = zeros(6,n);
end

% Base-link
P0 = [R0, zeros(3,3); zeros(3,3), eye(3)];

% Forward recursion to obtain the twist-propagation "vector"
for i=1:n
    if robot.joints(i).type == 1
        % Revolute joint
        pm(1:6,i) = [e(1:3,i); cross_product_casadi(e(1:3,i), g(1:3,i))];
    elseif robot.joints(i).type == 2
        % Prismatic joint
        pm(1:6,i) = [zeros(3,1); e(1:3,i)];
    elseif robot.joints(i).type == 0
        % Fixed joint
        pm(1:6,i) = zeros(6,1);
    end
end

end

%--- Helper functions ---%

function x_skew = SkewSym_casadi(x)
% CasADi-compatible version of SkewSym
% Computes the skew-symmetric matrix of a vector

x_skew = [0, -x(3), x(2); 
          x(3), 0, -x(1); 
          -x(2), x(1), 0];
end

function c = cross_product_casadi(a, b)
% CasADi-compatible cross product
% Computes c = a × b

c = [a(2)*b(3) - a(3)*b(2);
     a(3)*b(1) - a(1)*b(3);
     a(1)*b(2) - a(2)*b(1)];
end