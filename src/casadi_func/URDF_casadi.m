%SPART URDF Tutorial

%--- Clean and clear ---%
clc
close all
clear all
import casadi.*


%--- URDF filename ---%
%filename='kuka_lwr.urdf';
filename='SC_3DoF.urdf';
%filename='kuka_iiwa.urdf';
%filename='Simple_Spacecraft.urdf';
%--- Create robot model ad display---%
% [robot,robot_keys] = urdf2robot_flex_visu(filename);
% q_init=ones(robot.n_q,1);
% display=display_robot(robot,q_init);
[robot,robot_keys] = urdf2robot(filename);

%--- Parameters ---%
% R0=SX.eye(3);
% r0=SX.zeros(3,1);
% qm=0*SX.ones(robot.n_q,1);
% u0=SX.zeros(6,1);
% um=SX.zeros(robot.n_q,1);

p0 = SX.sym('p0',4,1);

r0=SX.sym('r0',3,1);
qm=SX.sym('q',robot.n_q,1);
u0=SX.sym('q0dot',6,1);
um=SX.sym('qdot',robot.n_q,1);

%R0=SX.sym('R',3,3);

R0 = quat_DCM(p0);
%--- Kinematics ---%
%Kinematics
[RJ,RL,rJ,rL,e,g]=Kinematics_casadi(R0,r0,qm,robot);

%test partial kinematics

%Diferential Kinematics
[Bij,Bi0,P0,pm]=DiffKinematics_casadi(R0,r0,rL,e,g,robot);