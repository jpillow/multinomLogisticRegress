function [f,df] = multisoftplus(x)
% [f,df] = multisoftplus(x)
% 
% Computes the function:
%     f(x) = log(sum(1+exp(x)))
% and its 1st derivative, where sum is along rows of x
%
% Note: uses log-sum-exp trick for numerical stability

rowmax = max(0,max(x,[],2)); % get max along each row 
f = log(exp(-rowmax)+sum(exp(x-rowmax),2))+rowmax; % compute log-sum-exp

% Compute 1st derivative:
if nargout > 1
    df = exp(x-rowmax)./(exp(-rowmax)+sum(exp(x-rowmax),2));
end
