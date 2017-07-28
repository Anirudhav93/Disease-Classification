clc;

load('projdata.mat');

Xtest = X_test;
ytest = y_test;
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

%dual training errors
%dual classify (train data)
yhat_G_train = dual_classify(K_G, lambda_G, b_G, y, beta);
yhat_L_train = dual_classify(K_L, lambda_L, b_L, y, beta);
yhat_P_train = dual_classify(K_P, lambda_P, b_P, y, beta);

%dual errors
accuracy_G_train = sum(yhat_G_train~=y)/length(y);
accuracy_L_train = sum(yhat_L_train~=y)/length(y);
accuracy_P_train = sum(yhat_P_train~=y)/length(y);

%dual test errors
%dual classify
yhat_G = dual_classify(Ktest_G, lambda_G, b_G, y, beta);
yhat_L = dual_classify(Ktest_L, lambda_L, b_L, y, beta);
yhat_P = dual_classify(Ktest_P, lambda_P, b_P, y, beta);

%dual errors
accuracy_G = sum(yhat_G~=ytest)/length(ytest);
accuracy_L = sum(yhat_L~=ytest)/length(ytest);
accuracy_P = sum(yhat_P~=ytest)/length(ytest);

%primal SVM
%primal training
[w , b] = primal_softmargin(X, y, beta);


%primal classification
[yhat_train] = margin_classify(X,w,b);
[yhat] = margin_classify(Xtest, w, b);

%primal training error
accuracy = sum(yhat~=ytest)/length(ytest);

%primal testing error
accuracy_test = sum(yhat_train~=y)/length(y);

