function Hv = neglogli_multinomGLM_HessMultFun(df,Vvec,X)
% Hv = neglogli_multinomGLM_HessMultFun(df,Vvec,X)
%
% Hessian Multiplication for multinomial logistic regression model: 
% computes the Hessian times an arbitrary vector v (which is the same size
% as wts)
%
% Inputs:
%    M.X  [T,d]  - design matrix of regressors
%    M.df [T,k]  - matrix of 1st derivatives of soft-max function
%      v [d*k,1] - vector (length ndims * nclasses) to multiply by Hessian
%
% Outputs:
%    Hv [d*k,1] - Hessian times v
%
% Note: for use with neglogli_multinomGLM_HessMult function
        
% get sizes
nX = size(X,2); % number of predictors (dimensionality of input space)
nclass = size(df,2); % number of classes

if size(Vvec,2) == 1
    % If only one vector v passed in
    vv = reshape(Vvec,nX,nclass);

    Xv = X*vv; % stimuli times v
    pj_Xv = df.*Xv; % probability of class j times Xv
    alpha = pj_Xv - df.*sum(pj_Xv,2); % weights for each X
    
    % compute Hessian * v
    Hv = reshape(X'*alpha,[],1);
else
    [nw,nv] = size(Vvec);
    Hv = zeros(nw,nv);

    for jj = 1:nv

        vv = reshape(Vvec(:,jj),nX,nclass);

        Xv = X*vv; % stimuli times v
        pj_Xv = df.*Xv; % probability of class j times Xv
        alpha = pj_Xv - df.*sum(pj_Xv,2); % weights for each X

        Hv(:,jj) = reshape(X'*alpha,[],1);
    end
end
