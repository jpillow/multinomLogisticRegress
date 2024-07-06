function [negL,dnegL,df] = neglogli_multinomGLM_multibasis_HessMult(wts,X,Y,Bmat)
% [negL,dnegL,df] = neglogli_multinomGLM_multibasis(wts,X,Y,Bmat)
%
% Compute negative log-likelihood of multinomial logistic regression model
% using a unique basis for each row of the weight matrix, while returning
% info needed for HessMult
%
% Inputs:
%    wts [nbw,1]  - basis weights as a column vector
%      X [T,d]    - design matrix of regressors
%      Y [T,k]    - output (one-hot vector on each row indicating class)
%   Bmat [nbw,nwts] - basis for rows of weight matrix, nwts = k*d
%
% Outputs:
%    negL [1,1] - negative loglikelihood
%   dnegL [d,1] - gradient
%      df [d,k] - derivative of softmax function
%
% Details: 
% --------
% Classification model mapping input vectors x to 1 of k classes
%     P(Y = j|x) = (1/Z) exp(x*w_j), where  Z = \sum_i=1^k  exp(x*w_i)
%
% - Note that the model is overparametrized, so we could add any vector to
%   columns of w and we would not change the log-likelihood
% 
% - Output Y should be represented as a binary matrix of size N x k,
%   with '1' indicating the class 1 to k
%
% - Constant ('offset' or 'bias') not added explicitly, so regressors X
%   should include a column of 1's to incorporate a constant.

nX = size(X,2);  % # samples and # input dimensions
nclass = size(Y,2); % # classes

% Reshape GLM weights into a matrix
ww = reshape(Bmat*wts,nX,nclass); % make matrix of model weights

% Compute projection of stimuli onto weights
xproj = X*ww;

if nargout <= 1 % compute log-likelihood only
    negL = -sum(sum(Y.*xproj)) + sum(logsumexp(xproj)); % neg log-likelihood

elseif nargout >= 2  % compute gradient
    [f,df] = logsumexp(xproj); % evaluate log-normalizer & deriv
    
    negL = -sum(sum(Y.*xproj)) + sum(f); % neg log-likelihood
    dnegL = Bmat'*vec(X'*(df-Y));     % gradient

end