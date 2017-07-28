clear all;
close all;
rng default;
clc;
load('projdata.mat')

[B, FitInfo] = lasso(X, y,'NumLambda', 90,'CV', 10);
lassoPlot(B, FitInfo, 'plottype', 'CV');