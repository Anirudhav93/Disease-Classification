clear all;
close all;
clc;

%% reading the data
load('projdata.mat')
%%%%%%%%
% contains training examples X
% contains corresponding target values y
% contains test examples X_test
% contains corresponding targets y_test
%%%%%%%%

%% forming the quadratic program
H = X'*X;
[n, ~] = size(H);
H = 2*  [H, zeros(n, 1);
        zeros(1,n+1);];
    
f = [-2*y'*X, y'*y];

A = -1*eye(n+1);
b = 0.005*ones(n+1, 1);
q = quadprog(H, f, A, b);

w = q(1:n, 1);

%% computing the estimates
yhat_pre = X*w;
yhat = [];

yhat_pre_test = X_test*w;
yhat_test =[];

%% classification
for i =1 : length(y)
    if(yhat_pre(i)>0.5)
        yhat(i) = 1;
    else
        yhat(i) = 0;
    end
end

for i =1 : length(y_test)
    if(yhat_pre_test(i)>0.5)
        yhat_test(i) = 1;
    else
        yhat_test(i) = 0;
    end
end

%% computing accuracy
yhat = yhat';
yhat_test = yhat_test';

accuracy = sum(yhat == y)/length(y);
accuracy_test = sum(yhat_test == y_test)/length(y_test);    


