function [RJ,RL,rJ,rL,e,g]=Kinematics_casadi(R0,r0,qm,robot)
% CasADi-compatible version of Kinematics function
% Computes the kinematics -- positions and orientations -- of the multibody system.
%
% [RJ,RL,rJ,rL,e,g]=Kinematics_casadi(R0,r0,qm,robot)
%
% This version is compatible with CasADi symbolic variables
% Main changes:
% - 3D arrays replaced with cell arrays
% - Removed 'like' declarations
% - Compatible with both numeric and CasADi SX inputs

%=== LICENSE ===%
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU Lesser General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.

%=== CODE ===%

%--- Number of links and joints ---%
n=robot.n_links_joints;

%--- Check if we're dealing with CasADi variables ---%
is_casadi = false;
try
    % Check if R0 is a CasADi type
    if isa(R0,'casadi.SX') || isa(R0,'casadi.MX') || isa(r0,'casadi.SX') || isa(r0,'casadi.MX') || isa(qm,'casadi.SX') || isa(qm,'casadi.MX')
        is_casadi = true;
    end
catch
    % CasADi not available, continue with numeric
end

%--- Homogeneous transformation matrices ---%
% Use cell arrays instead of 3D arrays for CasADi compatibility
TJ = cell(n,1);
TL = cell(n,1);

% Initialize each cell with 4x4 matrix
if is_casadi
    import casadi.*
    for i=1:n
        TJ{i} = SX.zeros(4,4);
        TL{i} = SX.zeros(4,4);
    end
else
    for i=1:n
        TJ{i} = zeros(4,4);
        TL{i} = zeros(4,4);
    end
end

%--- Base-link ---%
if is_casadi
    import casadi.*
    T0 = [R0, r0; SX.zeros(1,3), SX.ones(1,1)];
else
    T0 = [R0, r0; zeros(1,3), 1];
end 
%--- Forward kinematics recursion ---%
for i=1:n
    
    %Get child joint
    cjoint = robot.joints(i);
    
    %Joint kinematics (homogeneous transformation matrix)
    if cjoint.parent_link == 0
        %Parent link is the base-link
        TJ{cjoint.id}(1:4,1:4) = T0 * cjoint.T;
    else
        %Parent link is not the base-link
        TJ{cjoint.id}(1:4,1:4) = TL{cjoint.parent_link} * cjoint.T;
    end
    
    %Transformation due to current joint variable
    if cjoint.type == 1
        %Revolute joint
        T_qm = [Euler_DCM(cjoint.axis, qm(cjoint.q_id))', zeros(3,1); zeros(1,3), 1];
    elseif cjoint.type == 2
        %Prismatic joint
        T_qm = [eye(3), cjoint.axis*qm(cjoint.q_id); zeros(1,3), 1];
    else
        %Fixed joint
        T_qm = [eye(3), zeros(3,1); zeros(1,3), 1];
    end
    
    %Link Kinematics (homogeneous transformation matrix)
    clink = robot.links(cjoint.child_link);
    TL{clink.id}(1:4,1:4) = TJ{clink.parent_joint} * T_qm * clink.T;
end

%--- Rotation matrices, translation, position and other geometric quantities ---%

% Use cell arrays for rotation matrices
RJ = cell(n,1);
RL = cell(n,1);

% Initialize matrices
if is_casadi
    import casadi.*
    rJ = SX.zeros(3,n);
    rL = SX.zeros(3,n);
    e = SX.zeros(3,n);
    g = SX.zeros(3,n);
    for i=1:n
        RJ{i} = SX.zeros(3,3);
        RL{i} = SX.zeros(3,3);
    end
else
    rJ = zeros(3,n);
    rL = zeros(3,n);
    e = zeros(3,n);
    g = zeros(3,n);
    for i=1:n
        RJ{i} = zeros(3,3);
        RL{i} = zeros(3,3);
    end
end

%--- Format rotation matrices, link positions, joint axis and other geometric quantities ---%

%Joint associated quantities
for i=1:n
    RJ{i}(1:3,1:3) = TJ{i}(1:3,1:3);
    rJ(1:3,i) = TJ{i}(1:3,4);
    e(1:3,i) = RJ{i} * robot.joints(i).axis;
end

%Link associated quantities
for i=1:n
    RL{i}(1:3,1:3) = TL{i}(1:3,1:3);
    rL(1:3,i) = TL{i}(1:3,4);
    g(1:3,i) = rL(1:3,i) - rJ(1:3,robot.links(i).parent_joint);
end

end

%--- Helper function for Euler DCM (CasADi compatible) ---%
function DCM = Euler_DCM_casadi(e, alpha)
% CasADi-compatible version of Euler_DCM
% Provides the Direction Cosine Matrix (DCM) from a Euler axis e=[e1,e2,e3]
% and angle alpha.

% Check if we're dealing with CasADi variables
is_casadi = false;
try
    if isa(alpha,'casadi.SX') || isa(alpha,'casadi.MX')
        is_casadi = true;
    end
catch
    % CasADi not available
end

% Compute DCM using Rodrigues' formula (more CasADi-friendly than quaternions)
if is_casadi
    import casadi.*
    I = SX.eye(3);
else
    I = eye(3);
end

% Skew-symmetric matrix of e
e_skew = [0, -e(3), e(2); 
          e(3), 0, -e(1); 
          -e(2), e(1), 0];

% Rodrigues' rotation formula
c_a = cos(alpha);
s_a = sin(alpha);
DCM = I + s_a * e_skew + (1 - c_a) * (e_skew * e_skew);

end
