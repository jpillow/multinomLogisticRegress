function [negL,dnegL,H] = neglogli_multinomGLM_basis(wts,X,Y,Bmat)
% [negL,dnegL,H] = neglogli_multinomGLM_basis(wts,X,Y,Bmat)
%
% Compute negative log-likelihood of multinomial logistic regression model
% with a single shared basis for the rows of the weight matrix
%
% Inputs:
%    wts [d*nb,1] - basis weights (rows) for each input dimension (cols)
%      X [T,d]   - design matrix of regressors
%      Y [T,k]   - output (one-hot vector on each row indicating class)
%   Bmat [nb,k]  - basis for rows of W matrix
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
% - Output Y should be represented as a binary matrix of size N x k,
%   with '1' indicating the class 1 to k
%
% - Constant ('offset' or 'bias') not added explicitly, so regressors X
%   should include a column of 1's to incorporate a constant.


[nT,nX] = size(X);  % # samples and # input dimensions
nclass = size(Y,2); % # classes
nw = nX*nclass;     % # total weights in model
nbasis = size(Bmat,1); % # basis vectors
nbw = nX*nbasis;       % # of basis weights

% Reshape GLM weights into a matrix
wb = reshape(wts,nX,nbasis); % reshape basis weights into a matrix
ww = wb*Bmat;  % make matrix of model weights

% Compute projection of stimuli onto weights
xproj = X*ww;

if nargout <= 1 % compute log-likelihood only
    negL = -sum(sum(Y.*xproj)) + sum(logsumexp(xproj)); % neg log-likelihood

elseif nargout >= 2  % compute gradient
    [f,df] = logsumexp(xproj); % evaluate log-normalizer & deriv
    
    negL = -sum(sum(Y.*xproj)) + sum(f); % neg log-likelihood
    dnegL = reshape(X'*(df-Y)*Bmat',[],1);     % gradient
    
    if nargout > 2   % compute Hessian

        % =================================================
        % Compute full Hessian and then project onto basis
        % =================================================        
        
        % First term: sum(B*diag(df(ii,:))*B' \kron X_i X_i^T)
       
        % Compute stimulus weighted by df for each class
        Xdf = reshape(X.*reshape(df,[],1,nclass),nT,nw);

        % Compute center block-diagonal of full Hessian (not in basis)
        H1submatrices = cell(nclass,1);  % pre-allocate cell array
        XXdf = sparse(X'*Xdf); % blocks along center 
        for jj = 1:nclass
            inds = (jj-1)*nX+1:jj*nX; % indices for each block
            H1submatrices{jj} = XXdf(:,inds); % insert each block
        end

        % Project onto basis
        Bkron = kron(Bmat',speye(nX)); % basis matrix
        H1 = Bkron'*blkdiag(H1submatrices{:})*Bkron; % first term
                
        % Second term: - (X * df *B')' (X df * B')   
        XdfB = reshape(X.*reshape((df*Bmat'),[],1,nbasis),nT,nbw);
        H2 = -XdfB'*XdfB;  % second term
         
        % Sum together to get full Hessian
        H = H1 + H2;

    end
    
end