% demo_multinomLogisticRegress.m
%
% Max likelihood estim for multinomial logistic regression (multinomial
% GLM) using a HessMult instead of full Hessian

setpaths; 
clf; clear;

% set up weights
nxdim = 40;  % number of input dimensions 
nclass = 60;  % number of output classes
nsamp = 5e4; % number of samples to draw

% Sample weights from a Gaussian
wtrue = 5*gsmooth(randn(nxdim,nclass)',3)'/sqrt(nxdim);
nwts = nxdim*nclass;  % total number of weights;

% make inputs 
xinput = randn(nsamp,nxdim); % inputs by time

% Sample class labels from model
[yy,pclass] = sample_multinomGLM(xinput,wtrue);

%  --------- Make plots --------
subplot(231);
imagesc(1:nclass, 1:nxdim, wtrue);
ylabel('input dimension'); xlabel('class');
title('true weights');

subplot(232); imagesc(pclass); 
title('p(class)'); ylabel('trial #'); 
xlabel('class');

subplot(233); imagesc(yy); 
title('observed class'); ylabel('trial #');
xlabel('class');

%% 2. Compute ML estimate of weights, full parametrization

% Set initial weights
w0 = .1*randn(nxdim,nclass); % full weights

lfun1 = @(w)(neglogli_multinomGLM_full(w,xinput,yy)); % neglogli function handle (full)

% Set optimization parameters and perform optimization
opts1 = optimoptions('fminunc','algorithm','trust-region','SpecifyObjectiveGradient',true,...
    'HessianFcn','objective','display','iter');

% Compute ML estimate
fprintf('\n-------------------------------------------------------------------------\n');
fprintf('Computing ML estimate of multinomial logistic regression weights...\n');
fprintf('-------------------------------------------------------------------------\n');
tic;
wML1 = fminunc(lfun1,w0(:),opts1); % optimize negative log-posterior
toc;

%% 3. Compute ML estimate of weights using HessMult

% Set loss function
lfun2 = @(w)(neglogli_multinomGLM_HessMult(w,xinput,yy)); % neglogli function handle (full)
% Set HessMult Function
HessMultFun = @(df,v)neglogli_multinomGLM_HessMultFun(df,v,xinput);

% Set optimization parameters and perform optimization
opts2 = optimoptions('fminunc','algorithm','trust-region','SpecifyObjectiveGradient',true,'display','iter',...
    'HessianMultiplyFcn',HessMultFun);

% % % Check that Hmult function is correct
% % % -------------------------------------
% [negL,dnegL,H] = lfun1(w0);
% [negL2,dnegL2,Hinfo] = lfun2(w0(:));
% vtest = randn(nwts,1);
% Hv = HessMultFun(Hinfo,vtest);
% [H*vtest, Hv]

% Compute ML estimate using HessMult trick
fprintf('\n-------------------------------------------------------------------------\n');
fprintf('Computing ML estimate using HessMult\n');
fprintf('-------------------------------------------------------------------------\n');
tic;
wML2 = fminunc(lfun2,w0(:),opts2); % optimize negative log-posterior
toc;

%% 4.Compare fits to true weights

% Convert true weights to reduced form (for plotting purposes only)
wtrue_reduced = wtrue-wtrue(:,1); % substract off class-1 weights

% Reshape fitted weights into matrices
wMLfull1 = reshape(wML1,nxdim,nclass); % reshape full weights into matrix
wMLfull1 = wMLfull1 - wMLfull1(:,1); % substract off class-1 weights

wMLfull2 = reshape(wML2,nxdim,nclass); % reshape full weights into matrix
wMLfull2 = wMLfull2 - wMLfull2(:,1); % substract off class-1 weights


%  --------- Make plots --------
subplot(231);
imagesc(1:nclass, 1:nxdim, wtrue_reduced);
ylabel('input dimension'); xlabel('class');
title('true weights');

subplot(234);
imagesc(1:nclass, 1:nxdim, wMLfull1);
title('ML inferred weights'); 

subplot(2,3,5:6);
nw = nxdim*nclass; % number of total weights
plot(1:nw, vec(wtrue_reduced'), 1:nw, vec(wMLfull1'),1:nw,vec(wMLfull2'));
legend('true', 'ML', 'ML-Hmult','location', 'northwest');
title('true and recovered weights')

% ---  Report results (R^2) ------------------
R2a = 1-sum((wtrue_reduced(:)-wMLfull1(:)).^2)/sum(wtrue_reduced(:).^2);
fprintf('reduced weight recovery R^2: %0.3f\n',R2a);
R2b = 1-sum((wtrue_reduced(:)-wMLfull2(:)).^2)/sum(wtrue_reduced(:).^2);
fprintf('   full weight recovery R^2: %0.3f\n',R2b);
