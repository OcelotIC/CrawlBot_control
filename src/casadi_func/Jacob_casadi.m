function [J0, Jm]=Jacob_casadi(rp,r0,rL,P0,pm,i,robot)
% CasADi-compatible version of Jacob function
% Computes the geometric Jacobian of a point `p`.
%
% [J0, Jm]=Jacob_casadi(rp,r0,rL,P0,pm,i,robot)
% 
% This version is compatible with CasADi symbolic variables
% Main changes:
% - Removed 'like' declarations
% - Uses SkewSym_casadi helper function
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
    if isa(rp,'casadi.SX') || isa(rp,'casadi.MX') || isa(r0,'casadi.SX') || isa(r0,'casadi.MX')
        is_casadi = true;
    end
catch
    % CasADi not available
end

%--- Base-link Jacobian ---%
J0 = [eye(3), zeros(3,3); SkewSym_casadi(r0-rp), eye(3)] * P0;

%--- Pre-allocate ---%
if is_casadi
    import casadi.*
    Jm = SX.zeros(6, robot.n_q);
else
    Jm = zeros(6, robot.n_q);
end

%--- Manipulator Jacobian ---%
%Iterate through all "previous" joints
for j=1:i
    %If joint is not fixed
    if robot.joints(j).type ~= 0
        if robot.con.branch(i,j) == 1
            Jm(1:6, robot.joints(j).q_id) = [eye(3), zeros(3,3); 
                                              SkewSym_casadi(rL(1:3,j)-rp), eye(3)] * pm(1:6,j);
        else
            % Already initialized to zeros
        end
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