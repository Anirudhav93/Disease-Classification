clear all;
close all;
rng default;
clc;
load('projdata.mat')
load('features.mat')

%% Plotting LASSO with cross validation using inbuilt function 
fprintf('running lassoplot with cross validation..... \n');
[B, FitInfo] = lasso(X, y,'NumLambda', 90,'CV', 10);
lassoPlot(B, FitInfo, 'plottype', 'CV');
fprintf('press enter to continue. \n');
pause;


%% forming the quadratic program
fprintf('running LASSO using quadprog...\n');
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

accuracy_LASSO = sum(yhat == y)/length(y)
accuracy_test_LASSO = sum(yhat_test == y_test)/length(y_test)

fprintf('Program paused, press enter to continue. \n');
pause;

%% SVM using inbuilt function
fprintf('ploting svm results in 2D. \n');
a = 1;
b = 2;
svmStruct = svmtrain(X(:,a:b), y, 'ShowPlot', true);
hold on;
yhat = svmclassify(svmStruct, X(:,a:b));
yhat_test = svmclassify(svmStruct, X_test(:,a:b), 'Showplot', true);

svmStruct = svmtrain(X, y);% 'ShowPlot', true);
yhat = svmclassify(svmStruct, X);%(:,a:b));
yhat_test = svmclassify(svmStruct, X_test);%(:,a:b), 'Showplot', true);
hold on;
fprintf('Program paused. press enter to continue. \n');
pause;

fprintf('computing accuracy. \n');

accuracy_SVM = sum(y == yhat)/length(y)
accuracy_test_SVM = sum(y_test == yhat_test)/length(y_test)

fprintf('Program paused. press enter to continue. \n');
pause;
fprintf('Plot of number of features with error. \n');

%plots
plot(features, train, 'b-');
hold on;
plot(features, test, 'r-');
hold on;
xlabel('number of features');
ylabel('testing/training accuracy');
legend('Training accuracy', 'Testing accuracy');
fprintf('Program paused.\n press enter to continue. \n');
pause;

%% SVM with quadprog

Xtest = X_test;
ytest = y_test;
fprintf('running primal and dual svm with quadprog. \n');
%dual SVM
%training kernels
[K_G] = gauss_kernel(X, X, 10);
[K_L] = linear_kernel(X, X);
[K_P] = poly_kernel(X, X, 2);

%test Kernels
beta = 1;
[Ktest_G] = gauss_kernel(Xtest, X, 10);
[Ktest_P] = poly_kernel(Xtest, X, 2);
[Ktest_L] = linear_kernel(Xtest, X);

%dual training
[lambda_G, b_G] = dual_softmargin(K_G,y, beta);
[lambda_L, b_L] = dual_softmargin(K_L,y, beta);
[lambda_P, b_P] = dual_softmargin(K_P,y, beta);

fprintf('Training complete.Press enter to continue \n')
pause;
fprintf('computing errors for dual svm with different kernels... \n');
%dual training errors
%dual classify (train data)
yhat_G_train = dual_classify(K_G, lambda_G, b_G, y, beta);
yhat_L_train = dual_classify(K_L, lambda_L, b_L, y, beta);
yhat_P_train = dual_classify(K_P, lambda_P, b_P, y, beta);

%dual errors
accuracy_Gaussian_train = sum(yhat_G_train~=y)/length(y)
accuracy_Linear_train = sum(yhat_L_train~=y)/length(y)
accuracy_Polynomial_train = sum(yhat_P_train~=y)/length(y)

%dual test errors
%dual classify
yhat_G = dual_classify(Ktest_G, lambda_G, b_G, y, beta);
yhat_L = dual_classify(Ktest_L, lambda_L, b_L, y, beta);
yhat_P = dual_classify(Ktest_P, lambda_P, b_P, y, beta);

%dual errors
accuracy_Gaussian = sum(yhat_G~=ytest)/length(ytest)
accuracy_Linear = sum(yhat_L~=ytest)/length(ytest)
accuracy_Polynomial = sum(yhat_P~=ytest)/length(ytest)

fprintf('Program paused. press enter to continue. \n');
pause;


%primal SVM
fprintf('running primal svm. \n');
%primal training
[w , b] = primal_softmargin(X, y, beta);


%primal classification
[yhat_train] = margin_classify(X,w,b);
[yhat] = margin_classify(Xtest, w, b);

%primal training error
accuracy = sum(yhat~=ytest)/length(ytest)

%primal testing error
accuracy_test = sum(yhat_train~=y)/length(y)
fprintf('end of program. press enter to finish. \n');
pause;