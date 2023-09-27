function [negL,dnegL,H] = neglogli_multinomGLM_reduced(wts,X,Y)
% [negL,dnegL,H] = neglogli_multinomGLM_reduced(wts,X,Y)
%
% Negative log-likelihood under multinomial logistic regression model with
% "reduced" parametrization so that weights are identifiable. 
% (Class-1 weights assumed to be the all-zeros vector)
%
% Inputs:
% -------
%    wts [d*(k-1),1] - weights mapping stimulus to classes 2 to k
%      X [T,d]       - design matrix of regressors
%      Y [T,k-1]     - class (1-hot vector on each row indicating class,
%                             or all-zeros representing class 1) 
%
% Outputs:
% --------
%    negL [1,1] - negative loglikelihood
%   dnegL [d,1] - gradient
%       H [d,d] - Hessian
%
% Details: 
% --------
% Classification model that maps vectors x to one of k classes
%
%     P(Y = j | x) = 1/Z exp(x * w_j)
%
% where normalizer Z = sum_i=1^k  exp(x*w_i)
%      
% Notes:
% ------
%
% - assumes weights for 1st class are 0.  So wts(:,j) are for class j+1
%
% - Constant ('offset' or 'bias') not added explicitly, so regressors X
%   should include a column of 1's to incorporate a constant.

% Process inputs
[nT,nX] = size(X);      % # of trials, # of predictors (# input dims)
nClass = size(Y,2);    % # of observation classes minus 1
nwtot = nX*(nClass-1); % total number of weights in the model

% Reshape GLM weights into a matrix
ww = reshape(wts,nX,nClass-1); 

% Compute projection of stimuli onto weights
xproj = X*ww;

if nargout <= 1 % ========= compute neg log-likeli only ==================

    negL = -sum(sum(Y(:,2:end).*xproj)) + sum(multisoftplus(xproj)); 

elseif nargout >= 2 % ===== compute neg log-li and gradient ==============

    [f,df] = multisoftplus(xproj); % evaluate log-normalizer & deriv

    negL = -sum(sum(Y(:,2:end).*xproj)) + sum(f); % neg log-likelihood
    
    dnegL = X'*(df-Y(:,2:end));                   % gradient (matrix)
    dnegL = dnegL(:);                             % reshape to col vector
        
end

if nargout == 3 % ============ compute Hessian ===========================
    
    % Compute stimulus weighted by df for each clas
    dftensor = permute(df,[1,3,2]); % put df into 3-tensor
    Xdf = reshape(X.*dftensor,nT,nwtot); % X times df tensor
    
    % Multiply by stimulus (for center blocks)
    XXdf = X'*Xdf; 
    
    % Insert center-diagonal blocks of Hessian
    H = zeros(nwtot);
    for jj = 1:(nClass-1)
        inds = (jj-1)*nX+1:jj*nX;
        H(inds,inds) = XXdf(:,inds);
    end
    
    % add off-diagonal blocks of Hessian
    H = H-Xdf'*Xdf;
    
    % ----------------------------------------------------------
    % % Equivalent version (w/o for loop; near-identical speed)
    % ----------------------------------------------------------
    % Xdf = reshape(bsxfun(@times,X,reshape(df,[],1,nClass-1)),[],nwtot); % bsxfun version
    % XXdf = mat2cell(X'*Xdf,nX,nX*ones(1,nClass-1));
    % H = blkdiag(XXdf{:}) - Xdf'*Xdf; % add off-diagonal part
    % ----------------------------------------------------------
        
end
