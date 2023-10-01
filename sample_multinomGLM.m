function [y1hot,pclass] = sample_multinomGLM(xx,wts)
% [y1hot,pclass] = sample_multinomGLM(xx,wts)
%
% Sample from multinomial logistic regression model
%
% INPUTS
% ------
%  xx [nsamp, nd] - design matrix
% wts [nd, k]     - per class weights
%
% OUTPUTS
% -------
%  y1hot [nsamp, k] - samples (1-hot representation, eg [0 1 0 0])
% pclass [nsamp, k] - probabilities of each class under the model

nsamp = size(xx,1); % number of samples
nclass = size(wts,2); % number of classes

% Compute projection of inputs onto the per-class weights
xproj = xx*wts;

% subtract off the max of each row (so max is zero), to avoid overflow
xproj = xproj - max(xproj,[],2);

% Compute probability of each class
pclass = exp(xproj)./sum(exp(xproj),2);

% Sample outcomes y from multinomial
pcum = cumsum(pclass,2); % cumulative probability
rvals = rand(nsamp,1);   % uniform random deviates
y1hot = (rvals<pcum) & (rvals >=[zeros(nsamp,1),pcum(:,1:nclass-1)]);
