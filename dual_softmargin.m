function [lambda, b] = dual_softmargin(K, y, beta)

[t, ~] = size(K);

f = -1*ones(t,1);

H = 1/beta*diag(y)*K*diag(y);

Aeq = y';
beq =0;
A = -1*eye(t,t);
b =zeros(t,1);
lb =zeros(t,1); 
ub =ones(t,1);

lambda = quadprog(H,f, A, b, Aeq, beq, lb, ub);

idx = 0<lambda & lambda<1;

b = 1/sum(idx) * sum(y(idx) - 1/beta * K(idx, :) * diag(y) * lambda);

end