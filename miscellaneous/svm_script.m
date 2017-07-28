clc;
load('projdata.mat')
a = 1;
b = 10;
svmStruct = svmtrain(X(:,a:b), y, 'ShowPlot', true);%, 'kernel_function', 'linear');%, 'Showplot', true, 'kernel_function', 'rbf');
hold on;
yhat = svmclassify(svmStruct, X(:,a:b))%, 'Showplot', true);
yhat_test = svmclassify(svmStruct, X_test(:,a:b), 'Showplot', true);
hold on;

accuracy = sum(y == yhat)/length(y)
accuracy_test = sum(y_test == yhat_test)/length(y_test)

%plots
plot(features, train, 'b-');
hold on;
plot(features, test, 'r-');
hold on;
xlabel('number of features');
ylabel('testing/training accuracy');
legend('Training accuracy', 'Testing accuracy');