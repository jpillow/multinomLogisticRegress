function [L,dL,ddL] = neglogpost_multinomGLM_reduced(wts,X,Y,Cinv)
% [L,dL,ddL] = neglogpost_multinomGLM_reduced(wts,X,Y,Cinv)
%
% Negative log-posterior under multinomial logistic regression model with
% "reduced" parametrization so that weights are identifiable. 
% (Class-1 weights assumed to be the all-zeros vector)
%
% Inputs:
% -------
%    wts [d*(k-1),1] - weights mapping stimulus to classes 2 to k
%      X [T,d]       - design matrix of regressors
%      Y [T,k-1]     - class (1-hot vector on each row indicating class,
%                             or all-zeros representing class 1) 
%   Cinv [d*(k-1),d*(k-1)] - prior inverse covariance 
%
% Outputs:
% --------
%     L [1,1]               - negative log-posterior (up to constant)
%    dL [d*(k-1),1]         - gradient
%   ddL [d*(k-1),d*(k-1)] - Hessian
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
% Notes:
% ------
%
% - assumes weights for 1st class are 0.  So wts(:,j) are for class j+1
%
% - Design matrix X should include a column of 1s to incorporate a constant


if nargout <= 1
    L = neglogli_multinomGLM_reduced(wts,X,Y);
    L = L + 0.5*wts'*Cinv*wts;

elseif nargout == 2
    [L,dL] = neglogli_multinomGLM_reduced(wts,X,Y);
    L = L + 0.5*wts'*Cinv*wts;
    dL = dL + Cinv*wts;

elseif nargout == 3
    [L,dL,ddL] = neglogli_multinomGLM_reduced(wts,X,Y);
    L = L + 0.5*wts'*Cinv*wts;
    dL = dL + Cinv*wts;
    ddL = ddL + Cinv;
end
