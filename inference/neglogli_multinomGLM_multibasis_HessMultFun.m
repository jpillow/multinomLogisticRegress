function Hv = neglogli_multinomGLM_multibasis_HessMultFun(df,Vvec,X,Bmat)
% Hv = neglogli_multinomGLM_HessMultFun(df,Vvec,X)
%
% Hessian Multiplication for multinomial logistic regression model: 
% computes the Hessian times an arbitrary vector v (which is the same size
% as basis wts)
%
% Inputs:
%      df [T,k]  - matrix of 1st derivatives of soft-max function
%    Vvec [d*k,1] - vector (length ndims * nclasses) to multiply by Hessian
%       X [T,d]  - design matrix of regressors
%
% Outputs:
%    Hv [d*k,1] - Hessian times v
%
% Note: for use with neglogli_multinomGLM_multibasis_HessMult function
        
% get sizes
nX = size(X,2); % number of predictors (dimensionality of input space)
nclass = size(df,2); % number of classes

if size(Vvec,2) == 1
    % If only one vector v passed in
        vvB = reshape(Bmat*Vvec,nX,nclass); % v projected into full weight space
        
        Xv = X*vvB; % stimuli times v
        pj_Xv = df.*Xv; % probability of class j times Xv
        alpha = pj_Xv - df.*sum(pj_Xv,2); % weights for each row of X
        
        % compute Hessian * v and project down onto basis
        Hv = Bmat'*reshape(X'*alpha,[],1);
else
    [nw,nv] = size(Vvec);
    Hv = zeros(nw,nv);

    for jj = 1:nv

        vvB = reshape(Bmat*Vvec(:,jj),nX,nclass);

        Xv = X*vvB; % stimuli times v
        pj_Xv = df.*Xv; % probability of class j times Xv
        alpha = pj_Xv - df.*sum(pj_Xv,2); % weights for each X

        % compute Hessian * v and project down onto basis
        Hv(:,jj) = Bmat'*reshape(X'*alpha,[],1);
    end
end
