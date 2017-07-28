function [yhat] = margin_classify(Xtest, w, b)

[m , ~] = size(Xtest);
yhat_pre = Xtest*w+ones(m,1)*b;
yhat = sign(yhat_pre);

