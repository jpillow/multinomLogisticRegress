function [negL,dnegL,df] = neglogli_multinomGLM_HessMult(wts,X,Y)
% [negL,dnegL,df] = neglogli_multinomGLM_HessMult(wts,X,Y)
%
% Compute negative log-likelihood of multinomial logistic regression model,
% with "full" or overparametrized weights,  so all k classes have a weight
% vector, with outputs passed for HessMult option
%
% Inputs:
%    wts [d*k,1] - weights mapping stimulus to each of k classes
%      X [T,d]   - design matrix of regressors
%      Y [T,k]   - output (one-hot vector on each row indicating class)
%
% Outputs:
%    negL [1,1] - negative loglikelihood
%   dnegL [d,1] - gradient
%      df [T,d] - gradients of softmax function, which is needed for
%                 Hessian Multiplication Function 
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


[nT,nX] = size(X); % number of predictors (dimensionality of input space)
nclass = size(Y,2); % number of classes
nw = nX*nclass; % total number of weights in the model

% Reshape GLM weights into a matrix
ww = reshape(wts,nX,nclass); 

% Compute projection of stimuli onto weights
xproj = X*ww;

if nargout <= 1
    negL = -sum(sum(Y.*xproj)) + sum(logsumexp(xproj)); % neg log-likelihood

elseif nargout >= 2
    [f,df] = logsumexp(xproj); % evaluate log-normalizer & deriv
    
    negL = -sum(sum(Y.*xproj)) + sum(f); % neg log-likelihood
    dnegL = reshape(X'*(df-Y),[],1);     % gradient
    
    end
end
