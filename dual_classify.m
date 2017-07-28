function [yhat] = dual_classify(Ktest, lambda, b, y, beta)

% [rows, ~] = size(Ktest);
size(lambda);

%for i =1:rows
yhat = sign(Ktest*diag(y)*lambda*(1/beta)+b);

end