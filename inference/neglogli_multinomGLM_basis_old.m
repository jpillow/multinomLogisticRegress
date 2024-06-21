function [negL,dnegL,H] = neglogli_multinomGLM_basis_old(wts,X,Y,B)
% [negL,dnegL,H] = neglogli_multinomGLM_basis(wts,X,Y,B)
%
% Compute negative log-likelihood of multinomial logistic regression model,
% with "full" or overparametrized weights,  so all k classes have a weight
% vector 
%
% Inputs:
%    wts [d*nb,1] - basis weights (rows) for each input dimension (cols)
%      X [T,d]   - design matrix of regressors
%      Y [T,k]   - output (one-hot vector on each row indicating class)
%      B [nb,k]  - basis for rows of W matrix
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
nbasis = size(B,1); % number of basis vectors
nbw = nX*nbasis; % number of basis weights in model

% Reshape GLM weights into a matrix
wb = reshape(wts,nX,nbasis);
ww = wb*B;

% Compute projection of stimuli onto weights
xproj = X*ww;

if nargout <= 1 % compute log-likelihood only
    negL = -sum(sum(Y.*xproj)) + sum(logsumexp(xproj)); % neg log-likelihood

elseif nargout >= 2  % compute gradient
    [f,df] = logsumexp(xproj); % evaluate log-normalizer & deriv
    
    negL = -sum(sum(Y.*xproj)) + sum(f); % neg log-likelihood
    dnegL = reshape(X'*(df-Y)*B',[],1);     % gradient
    
    if nargout > 2   % compute Hessian

%         % =================================================
%         % Compute full Hessian and then project onto basis
%         % =================================================        
%         % Compute stimulus weighted by df for each class
%         Xdf = reshape(X.*reshape(df,[],1,nclass),nT,nw);
%         
%         % Compute center block-diagonal portion 
%         H1 = zeros(nw); 
%         XXdf = X'*Xdf; % blocks along center 
%         for jj = 1:nclass
%             inds = (jj-1)*nX+1:jj*nX;
%             H1(inds,inds) = XXdf(:,inds);
%         end
%         % compute off-diagonal part
%         H2 = -Xdf'*Xdf;
%         H = H1 + H2;
% 
%         % map to space of W
%         Bkron = kron(B',eye(nX));
%         H = Bkron'*H*Bkron;

        % ====================================
        % Now attempt to do it more efficiently
        % ====================================

        % First term: sum(B*diag(df(ii,:))*B' \kron X_i X_i^T) 
        BdfBtrp = pagemtimes(df.*reshape(B',1,nclass,nbasis),B'); % compute B * diag(df) * B'
        BdfBtrp = reshape(BdfBtrp,nT,1,nbasis^2); % reshape into matrix
        H1 = X'*reshape(X.*BdfBtrp,nT,nX*nbasis^2); % left and right multiply by X
        H1 = reshape(H1,nX,nX,nbasis,nbasis); % reshape into 4-tensor
        H1 = permute(H1,[1,3,2,4]);  % flip 2nd and 3rd dimension 
        H1 = reshape(H1,nX*nbasis,nX*nbasis); % convert to a single matrix of correct shape
                
        % Second term: - (X * df *B')' (X df * B')   
        Xdf = reshape(X.*reshape((df*B'),[],1,nbasis),nT,nbw);
        H2 = -Xdf'*Xdf;
         
        H = H1 + H2;

    end
    
end