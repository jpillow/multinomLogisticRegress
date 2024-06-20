% demo_multinomLogisticRegress.m
%
% Multinomial logistic regression observations (multinomial GLM) with GP
% prior over rows of weight matrix

setpaths; 

% set up weights
nxdim = 55;  % number of input dimensions 
nclass = 72;  % number of output classes
nsamp = nclass*50; % number of samples to draw
lambda_ridge = 100; % ridge parameter (inverse prior variance)

% Sample weights from a Gaussian
wtrue = randn(nxdim,nclass)/sqrt(nxdim);

% make inputs 
xinput = randn(nsamp,nxdim); % inputs by time

% Sample class labels from model
[yy,pclass] = sample_multinomGLM(xinput,wtrue);

%  --------- Make plots --------
subplot(331);
imagesc(1:nclass, 1:nxdim, wtrue);
ylabel('input dimension'); xlabel('class');
title('true weights');

subplot(332); imagesc(pclass); 
title('p(class)'); ylabel('trial #'); 
xlabel('class');

subplot(333); imagesc(yy); 
title('observed class'); ylabel('trial #');
xlabel('class');

%% 2a. Compute ML estimate of weights, reduced parametrization

% Set initial weights
w0full = randn(nxdim,nclass); % full weights
w0red = w0full(:,2:end)-w0full(:,1); % reduced weights

lfun1 = @(w)(neglogli_multinomGLM_reduced(w,xinput,yy)); % neglogli function handle (reduced)

% Check accuracy of Hessian and Gradient (if desired)
% HessCheck(lfun1,w0red(:));

% Set optimization parameters and perform optimization
opts = optimoptions('fminunc','algorithm','trust-region','SpecifyObjectiveGradient',true,...
    'HessianFcn','objective','display','iter');

fprintf('\n-------------------------------------------------------------------------\n');
fprintf('Computing "reduced" ML estimate of multinomial logistic regression weights...\n');
fprintf('-------------------------------------------------------------------------\n');

tic;
wML1 = fminunc(lfun1,w0red(:),opts); % optimize negative log-posterior
toc;

%% 2b. Compute MAP weights, reduced parametrization

% Set inverse prior covariance
Cinv_red = speye(nxdim*(nclass-1))*lambda_ridge/3; % Note: not yet sure how to make two MAP versions equivalent!

% Set loss function
lfun1_map = @(w)(neglogpost_multinomGLM_reduced(w,xinput,yy,Cinv_red)); % neglogli function handle (reduced)

% Check accuracy of Hessian and Gradient (if desired)
% HessCheck(lfun1_map,w0red(:));

fprintf('\n-------------------------------------------------------------------------\n');
fprintf('Computing "reduced" MAP estimate of multinomial logistic regression weights...\n');
fprintf('-------------------------------------------------------------------------\n');

tic;
wMAP1 = fminunc(lfun1_map,w0red(:),opts); % optimize negative log-posterior
toc;

%% 3a. Compute ML estimate of weights, full parametrization

lfun2 = @(w)(neglogli_multinomGLM_full(w,xinput,yy)); % neglogli function handle (full)

% Check accuracy of Hessian and Gradient (if desired)
% HessCheck(lfun2,w0full(:));

fprintf('\n-------------------------------------------------------------------------\n');
fprintf('Computing "full" ML estimate of multinomial logistic regression weights...\n');
fprintf('-------------------------------------------------------------------------\n');

tic;
wML2 = fminunc(lfun2,w0full(:),opts); % optimize negative log-posterior
toc;

%% 3b. Compute MAP weights, full parametrization

% Set inverse prior covariance
Cinv_full = speye(nxdim*nclass)*lambda_ridge;

% Set loss function
lfun2_map = @(w)(neglogpost_multinomGLM_full(w,xinput,yy,Cinv_full)); % neglogli function handle (reduced)

% Check accuracy of Hessian and Gradient (if desired)
% HessCheck(lfun2_map,w0full(:));

fprintf('\n-------------------------------------------------------------------------\n');
fprintf('Computing "reduced" MAP estimate of multinomial logistic regression weights...\n');
fprintf('-------------------------------------------------------------------------\n');

tic;
wMAP2 = fminunc(lfun2_map,w0full(:),opts); % optimize negative log-posterior
toc;

%% 4.Compare fits to true weights

% Convert true weights to reduced form (for plotting purposes only)
wtrue_reduced = wtrue-wtrue(:,1); % substract off class-1 weights

% Reshape fitted weights into matrices
wMLred = [zeros(nxdim,1), reshape(wML1,nxdim,nclass-1)]; % reshape reduced weights into matrix
wMAPred = [zeros(nxdim,1), reshape(wMAP1,nxdim,nclass-1)]; % reshape reduced weights into matrix

wMLfull = reshape(wML2,nxdim,nclass); % reshape full weights into matrix
wMLfull = wMLfull - wMLfull(:,1); % substract off class-1 weights

wMAPfull = reshape(wMAP2,nxdim,nclass); % reshape full weights into matrix
wMAPfull = wMAPfull - wMAPfull(:,1); % substract off class-1 weights

%  --------- Make plots --------
nw = nxdim*nclass; % number of total weights
xlim = [nxdim+1, nxdim*2];

subplot(3,1,2);
plot(1:nw, wtrue_reduced(:), 1:nw, wMLfull(:), 1:nw, wMLred(:),'--');
legend('true', 'ML-reduced', 'ML-full', 'location', 'northwest');
set(gca,'xlim', xlim);
title('ML estimates')

subplot(3,1,3);
plot(1:nw, wtrue_reduced(:), 1:nw, wMAPfull(:), 1:nw, wMAPred(:),'--');
legend('true', 'MAP-reduced', 'MAP-full', 'location', 'northwest');
title('MAP estimates');
set(gca,'xlim', xlim);

% ---  Report results (R^2) ------------------
r2fun = @(w)(1-sum((wtrue_reduced(:)-w(:)).^2)/sum(wtrue_reduced(:).^2));
R2mlred = r2fun(wMLred);
R2mapred = r2fun(wMAPred);
R2mlfull = r2fun(wMLfull);
R2mapfull = r2fun(wMAPfull);

fprintf(' ML estimate R^2: [reduced=%7.3f, full=%7.3f]\n',R2mlred,R2mlfull);
fprintf('MAP estimate R^2: [reduced=%7.3f, full=%7.3f]\n',R2mapred,R2mapfull);


