clear
close all
clc
%%
% syms q1(t) q2(t) l1 l2 m1 m2 t1 t2 c1(t) c2(t) c12(t) s1(t) s2(t) s12(t) g x1 x2 x3 x4 x5 x6
% assume(m1, 'real')
% assumeAlso(m1 > 0)
% assume(l1, 'real')
% assumeAlso(l1 > 0)
% assume(m2, 'real')
% assumeAlso(m2 > 0)
% assume(l2, 'real')
% assumeAlso(l2 > 0)
% assume(g > 0)
% e_x = [1, 0];
% e_y = [0, 1];
% c1(t) = cos(q1);
% c2(t) = cos(q2);
% c12(t) = cos(q1+q2);
% s1(t) = sin(q1);
% s2(t) = sin(q2);
% s12(t) = sin(q1+q2);
%%
% P01 = [l1*c1
%           l1*s1]
% 
% P02 = [l1*cos(q1)+l2*cos(q1+q2)
%           l1*sin(q1)+l2*sin(q1+q2)]
%%
% v1 = diff(P01,t);
% v2 = diff(P02,t);
% v1_sq = transpose(v1)*v1
% v2_sq = transpose(v2)*v2
%%
% k1 = simplify(1/2*m1*v1_sq);
% k2 = simplify(1/2*m2*v2_sq);
% K = simplify(k1 + k2)
%%
% u1 = -m1*g*e_y*P01;
% u2 = -m2*g*e_y*P02;
% U = u1+u2
%%
% f = 2*eye(2);
% F = 1/2*[diff(q1,t), diff(q2,t)]*f*[diff(q1,t); diff(q2,t)]
%%
% L = K - U;
% dLdq = [diff(L, q1)
%         diff(L, q2)];
% dLdqp = [diff(L, diff(q1))
%          diff(L, diff(q2))];
% ddt_dLdqp = diff(dLdqp, t);
% dFdqp = [diff(F, diff(q1))
%          diff(F, diff(q2))];
%%
% eq = ddt_dLdqp - dLdq - [t1; t2] + dFdqp;
% eq = eq(t);
% eq = subs(eq, diff(diff(q1,t),t), x5);
% eq = subs(eq, diff(diff(q2,t),t), x6);
% eq = subs(eq, diff(q1,t), x3);
% eq = subs(eq, diff(q2,t), x4);
% eq = subs(eq, q1, x1);
% eq = subs(eq, q2, x2);
% eq = simplify(eq);
% eq1 = eq(1);
% eq2 = eq(2);
% sols = solve([eq1 eq2], [x5 x6], 'ReturnConditions',true);
% x5_kuf = sols.x5
% x6_kuf = sols.x6
%%
% eq = ddt_dLdqp - dLdq - [t1; t2];
% eq = eq(t);
% eq = subs(eq, diff(diff(q1,t),t), x5);
% eq = subs(eq, diff(diff(q2,t),t), x6);
% eq = subs(eq, diff(q1,t), x3);
% eq = subs(eq, diff(q2,t), x4);
% eq = subs(eq, q1, x1);
% eq = subs(eq, q2, x2);
% eq = simplify(eq);
% eq1 = eq(1);
% eq2 = eq(2);
% sols = solve([eq1 eq2], [x5 x6], 'ReturnConditions',true);
% x5_ku = simplify(sols.x5)
% x6_ku = simplify(sols.x6)
%%
ti = 0;
tf = 60;
step = 0.0250;
num_points = (tf - ti) / step;
time = linspace(ti, tf, num_points);
m1_ = 3;
m2_ = 2;
L1_ = 2;
L2_ = 1;
%%
ti = 0;
x_k = zeros(1,4);
res = zeros(length(time), 4);
g_ = -9.81;
tau = [0, 0];
i = 1;
while ti <= tf
    % RK4 integration 
    k1 = dxdt_kuf(g_,L1_,L2_,m1_,m2_,tau(1),tau(2), x_k);
    k2 = dxdt_kuf(g_,L1_,L2_,m1_,m2_,tau(1),tau(2), x_k+step*k1/2);
    k3 = dxdt_kuf(g_,L1_,L2_,m1_,m2_,tau(1),tau(2), x_k+step*k2/2);
    k4 = dxdt_kuf(g_,L1_,L2_,m1_,m2_,tau(1),tau(2), x_k+step*k3);

    % compute next state
    x_k = x_k + step * (k1+2*k2+2*k3+k4) / 6;

    % saturate
    if x_k(1) < -pi
        x_k(1) = -pi;
    elseif x_k(1) > pi
        x_k(1) = pi;
    end
    if x_k(2) < -5/6*pi
        x_k(2) = - 5/6*pi;
    elseif x_k(2) > 5/6*pi
        x_k(2) = 5/6*pi;
    end
    % if x_k(3) < -1/2*pi/180
    %     x_k(3) = -1/2*pi/180;
    % elseif x_k(3) > 1/2*pi/180
    %     x_k(3) = 1/2*pi/180;
    % end
    % if x_k(4) < -1/2*pi/180
    %     x_k(4) = -1/2*pi/180;
    % elseif x_k(4) > 1/2*pi/180
    %     x_k(4) = 1/2*pi/180;
    % end


    res(i,:) = x_k;
    i = i+1;
    ti = ti + step;
end
[px1, py1, px2, py2] = fk(res(:,1), res(:,2), L1_, L2_);
%%
figure
subplot(1,3,1)
scatter(px2, py2, 1, 'o');
legend 'end effector'
%%
subplot(1,3,2)
plot(time, res(1:length(time),1), '-b', time, res(1:length(time),2), '-r' )
legend q1(t) q2(t)
%%
subplot(1,3,2)
plot(time, res(1:length(time),3), '-b', time, res(1:length(time),4), '-r' )
legend dq1(t) dq2(t)
%%
ti = 0;
x_k = zeros(1,4);
res = zeros(length(time), 4);
g_ = -9.81;
tau = [0, 0];
i = 1;
while ti <= tf
    % RK4 integration 
    k1 = dxdt_ku(g_,L1_,L2_,m1_,m2_,tau(1),tau(2), x_k);
    k2 = dxdt_ku(g_,L1_,L2_,m1_,m2_,tau(1),tau(2), x_k+step*k1/2);
    k3 = dxdt_ku(g_,L1_,L2_,m1_,m2_,tau(1),tau(2), x_k+step*k2/2);
    k4 = dxdt_ku(g_,L1_,L2_,m1_,m2_,tau(1),tau(2), x_k+step*k3);

    % compute next state
    x_k = x_k + step * (k1+2*k2+2*k3+k4) / 6;

    % saturate
    if x_k(1) < -pi
        x_k(1) = -pi;
    elseif x_k(1) > pi
        x_k(1) = pi;
    end
    if x_k(2) < -5/6*pi
        x_k(2) = - 5/6*pi;
    elseif x_k(2) > 5/6*pi
        x_k(2) = 5/6*pi;
    end
    % if x_k(3) < -1/2*pi/180
    %     x_k(3) = -1/2*pi/180;
    % elseif x_k(3) > 1/2*pi/180
    %     x_k(3) = 1/2*pi/180;
    % end
    % if x_k(4) < -1/2*pi/180
    %     x_k(4) = -1/2*pi/180;
    % elseif x_k(4) > 1/2*pi/180
    %     x_k(4) = 1/2*pi/180;
    % end
    
    % next step    
    res(i,:) = x_k;
    i = i+1;
    ti = ti + step;
end
[px1, py1, px2, py2] = fk(res(:,1), res(:,2), L1_, L2_);
%%
figure
subplot(1,3,1)
scatter(px2, py2, 1, 'o');
legend 'end effector'
%%
subplot(1,3,2)
plot(time, res(1:length(time),1), '-b', time, res(1:length(time),2), '-r' )
legend q1(t) q2(t)
%%
subplot(1,3,3)
plot(time, res(1:length(time),3), '-b', time, res(1:length(time),4), '-r' )
legend dq1(t) dq2(t)
%%
function xdot = dxdt_kuf(g, l1, l2, m1, m2, t1, t2, x)
    x1 = x(1);
    x2 = x(2);
    x3 = x(3);
    x4 = x(4);
    dx1 = x3;
    dx2 = x4;
    dx3 = (2*l2*t1 - 2*l2*t2 - 2*l2*x3 + 2*l2*x4 - 2*l1*t2*cos(x2) + 2*l1*x4*cos(x2) - g*l1*l2*m2*cos(x1 + 2*x2) + l1^2*l2*m2*x3^2*sin(2*x2) + 2*l1*l2^2*m2*x3^2*sin(x2) + 2*l1*l2^2*m2*x4^2*sin(x2) + 2*g*l1*l2*m1*cos(x1) + g*l1*l2*m2*cos(x1) + 4*l1*l2^2*m2*x3*x4*sin(x2))/(l1^2*l2*(2*m1 + m2 - m2*cos(2*x2)));
    dx4 = -(l2^2*m2*t1 - l1^2*m2*t2 - l1^2*m1*t2 - l2^2*m2*t2 + l1^2*m1*x4 + l1^2*m2*x4 - l2^2*m2*x3 + l2^2*m2*x4 + l1*l2^3*m2^2*x3^2*sin(x2) + l1^3*l2*m2^2*x3^2*sin(x2) + l1*l2^3*m2^2*x4^2*sin(x2) + g*l1*l2^2*m2^2*cos(x1) + l1^2*l2^2*m2^2*x3^2*sin(2*x2) + (l1^2*l2^2*m2^2*x4^2*sin(2*x2))/2 + l1*l2*m2*t1*cos(x2) - 2*l1*l2*m2*t2*cos(x2) - l1*l2*m2*x3*cos(x2) + 2*l1*l2*m2*x4*cos(x2) + l1^3*l2*m1*m2*x3^2*sin(x2) + 2*l1*l2^3*m2^2*x3*x4*sin(x2) + g*l1^2*l2*m2^2*sin(x1)*sin(x2) + g*l1*l2^2*m1*m2*cos(x1) - g*l1*l2^2*m2^2*cos(x1)*cos(x2)^2 + l1^2*l2^2*m2^2*x3*x4*sin(2*x2) + g*l1*l2^2*m2^2*cos(x2)*sin(x1)*sin(x2) + g*l1^2*l2*m1*m2*sin(x1)*sin(x2))/(l1^2*l2^2*m2*(- m2*cos(x2)^2 + m1 + m2));
    xdot = [dx1, dx2, dx3, dx4];    
end
function xdot = dxdt_ku(g, l1, l2, m1, m2, t1, t2, x)
    x1 = x(1);
    x2 = x(2);
    x3 = x(3);
    x4 = x(4);
    dx1 = x3;
    dx2 = x4;
    dx3 = (2*l2*t1 - 2*l2*t2 - 2*l1*t2*cos(x2) - g*l1*l2*m2*cos(x1 + 2*x2) + l1^2*l2*m2*x3^2*sin(2*x2) + 2*l1*l2^2*m2*x3^2*sin(x2) + 2*l1*l2^2*m2*x4^2*sin(x2) + 2*g*l1*l2*m1*cos(x1) + g*l1*l2*m2*cos(x1) + 4*l1*l2^2*m2*x3*x4*sin(x2))/(l1^2*l2*(2*m1 + m2 - m2*cos(2*x2)));
    dx4 = -(l2^2*m2*t1 - l1^2*m2*t2 - l1^2*m1*t2 - l2^2*m2*t2 + l1*l2^3*m2^2*x3^2*sin(x2) + l1^3*l2*m2^2*x3^2*sin(x2) + l1*l2^3*m2^2*x4^2*sin(x2) + g*l1*l2^2*m2^2*cos(x1) + l1^2*l2^2*m2^2*x3^2*sin(2*x2) + (l1^2*l2^2*m2^2*x4^2*sin(2*x2))/2 + l1*l2*m2*t1*cos(x2) - 2*l1*l2*m2*t2*cos(x2) + l1^3*l2*m1*m2*x3^2*sin(x2) + 2*l1*l2^3*m2^2*x3*x4*sin(x2) + g*l1^2*l2*m2^2*sin(x1)*sin(x2) + g*l1*l2^2*m1*m2*cos(x1) - g*l1*l2^2*m2^2*cos(x1)*cos(x2)^2 + l1^2*l2^2*m2^2*x3*x4*sin(2*x2) + g*l1*l2^2*m2^2*cos(x2)*sin(x1)*sin(x2) + g*l1^2*l2*m1*m2*sin(x1)*sin(x2))/(l1^2*l2^2*m2*(- m2*cos(x2)^2 + m1 + m2));
    xdot = [dx1, dx2, dx3, dx4];    
end
function [x1, y1, x2, y2] = fk(q1, q2, l1, l2)
    x1 = l1*cos(q1);
    y1 = l1*sin(q1);
    x2 = l1*cos(q1)+l2*cos(q1+q2);
    y2 = l1*sin(q1)+l2*sin(q1+q2);
end