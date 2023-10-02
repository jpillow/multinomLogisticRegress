function [negL,dnegL,H] = neglogli_multinomGLM_full(wts,X,Y)
% [negL,dnegL,H] = neglogli_multinomGLM_full(wts,X,Y)
%
% Compute negative log-likelihood of multinomial logistic regression model,
% with "full" or overparametrized weights,  so all k classes have a weight
% vector 
%
% Inputs:
%    wts [d*k,1] - weights mapping stimulus to each of k classes
%      X [T,d]   - design matrix of regressors
%      Y [T,k]   - output (one-hot vector on each row indicating class)
%
% Outputs:
%    negL [1,1] - negative loglikelihood
%   dnegL [d,1] - gradient
%       H [d,d] - Hessian
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
    
    if nargout > 2  
        % compute Hessian
        
        % Compute stimulus weighted by df for each clas
        % Xdf = reshape(bsxfun(@times,X,reshape(df,[],1,nclass)),[],nw); % bsxfun version
        Xdf = reshape(X.*reshape(df,[],1,nclass),nT,nw);
        
        % Compute center block-diagonal portion 
        H = zeros(nw); 
        XXdf = X'*Xdf; % blocks along center 
        for jj = 1:nclass
            inds = (jj-1)*nX+1:jj*nX;
            H(inds,inds) = XXdf(:,inds);
        end
        % add off-diagonal part
        H = H-Xdf'*Xdf; 
        
        % % Equivalent version (w/o for loop; near-identical speed)
        % XXdf = mat2cell(X'*Xdf,nx,nx*ones(1,nclass));
        % H = blkdiag(XXdf{:}) - Xdf'*Xdf; % add off-diagonal part

    end
end
