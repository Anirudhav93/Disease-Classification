clc;
load('projdata.mat')
mdl = fitcknn(X, y, 'Distance' , 'exhaustive');
mdl.parameter = mahalanobis;