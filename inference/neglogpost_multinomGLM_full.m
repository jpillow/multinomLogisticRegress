function [L,dL,ddL] = neglogpost_multinomGLM_full(wts,X,Y,Cinv)
% [negL,dnegL,H] = neglogli_multinomGLM_overp(wts,X,Y)
%
% Compute negative log-posterior of multinomial logistic regression model,
% with "full" or overparametrized weights,  so all k classes have a weight
% vector 
%
% Inputs:
%    wts [d*k,1] - weights mapping stimulus to each of k classes
%      X [T,d]   - design matrix of regressors
%      Y [T,k]   - output (one-hot vector on each row indicating class)
%   Cinv [d*(k-1),d*(k-1)] - prior inverse covariance 
%
% Outputs:
% --------
%     L [1,1]               - negative log-posterior (up to constant)
%    dL [d*k,1]         - gradient
%   ddL [d*k,d*k] - Hessian
%
% Details: 
% --------
% Classification model that maps vectors x to one of k classes
%
%     P(Y = j | x) = 1/Z exp(x * w_j)
%
% where normalizer Z = sum_i=1^k  exp(x*w_i)
%
% and prior 
%
%     vec(W) ~ N(0,C)
%
% Details: 
% --------
% Classification model mapping input vectors x to 1 of k classes
%     P(Y = j|x) = (1/Z) exp(x*w_j), where  Z = \sum_i=1^k  exp(x*w_i)
%
% - Note that the model is overparametrized, so we could add any vector to
%   columns of w and we would not change the log-likelihood
% 
% - Design matrix X should include a column of 1s to incorporate a constant
%
% - Output Y should be represented as a binary matrix of size N x k-1,
%   with '1' indicating the class 2 to k, or all-zeros for class 1.
%
% Notes:
% ------
%
% - Constant ('offset' or 'bias') not added explicitly, so regressors X
%   should include a column of 1's to incorporate a constant.

if nargout <= 1
    L = neglogli_multinomGLM_full(wts,X,Y);
    L = L + 0.5*wts'*Cinv*wts;

elseif nargout == 2
    [L,dL] = neglogli_multinomGLM_full(wts,X,Y);
    L = L + 0.5*wts'*Cinv*wts;
    dL = dL + Cinv*wts;

elseif nargout == 3
    [L,dL,ddL] = neglogli_multinomGLM_full(wts,X,Y);
    L = L + 0.5*wts'*Cinv*wts;
    dL = dL + Cinv*wts;
    ddL = ddL + Cinv;
end
