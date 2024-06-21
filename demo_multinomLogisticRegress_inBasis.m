% demo_multinomLogisticRegress.m
%
% Multinomial logistic regression observations (multinomial GLM) with GP
% prior over rows of weight matrix

setpaths; 

% set up weights
nxdim = 50;  % number of input dimensions 
nclass = 72;  % number of output classes
nbdim = 5;    % number of dimensions in basis
nsamp = nclass*100; % number of samples to draw

% Make Basis across classes
B = gsmooth(randn(nclass,nbdim),3);
B = orth(B)';  % basis for the rows of wts

%% Sample weights from a Gaussian
wbasis = 3*randn(nxdim,nbdim)/sqrt(nxdim);  % basis weights
wtrue = wbasis*B;  % full matrix of true weights

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


%% 2. Compute ML estimate of weights, no basis

% Set initial weights
wbasis0 = 0.1*randn(nxdim,nbdim); % initial basis weights
w0 = wbasis0*B; % initial full weights

% Set optimization parameters and perform optimization
opts = optimoptions('fminunc','algorithm','trust-region','SpecifyObjectiveGradient',true,...
    'HessianFcn','objective','display','iter');

lfun1 = @(w)(neglogli_multinomGLM_full(w,xinput,yy)); % neglogli function handle (full)

% % Check accuracy of Hessian and Gradient (if desired)
% HessCheck(lfun1,w0(:));
 
fprintf('\n-------------------------------------------------------------------------\n');
fprintf('Computing ML estimate...\n');
fprintf('-------------------------------------------------------------------------\n');

tic;
wML1 = fminunc(lfun1,w0(:),opts); % optimize negative log-posterior
toc;


%% 3b. Compute ML weights using basis

% Set loss function
lfun2 = @(wb)(neglogli_multinomGLM_basis(wb,xinput,yy,B)); % neglogli function handle w/ basis

% Check accuracy of Hessian and Gradient (if desired)
HessCheck(lfun2,wbasis0(:));

fprintf('\n-------------------------------------------------------------------------\n');
fprintf('Computing ML estimate in basis.....\n');
fprintf('-------------------------------------------------------------------------\n');

tic;
wML2basis = fminunc(lfun2,wbasis0(:),opts); % optimize negative log-posterior in basis
toc;

wML2 = reshape(wML2basis,nxdim,nbdim)*B; % reconstruct full weights using basis

%% 4.Compare fits to true weights
 
% Convert true weights to reduced form (for plotting purposes only)
wtrue_reduced = wtrue-wtrue(:,1); % substract off class-1 weights

wML1 = reshape(wML1,nxdim,nclass); % reshape full weights into matrix
wML1 = wML1- wML1(:,1); % substract off class-1 weights

wML2 = reshape(wML2,nxdim,nclass); % reshape full weights into matrix
wML2 = wML2- wML2(:,1); % substract off class-1 weights

% %  --------- Make plots --------
nw = nxdim*nclass; % number of total weights
xlim = [nxdim+1, nxdim*2];

subplot(3,1,2:3);
pltinds = 1:nclass;
plot(pltinds, wtrue_reduced(2,:), pltinds, wML1(2,:), pltinds, wML2(2,:));
legend('true', 'ML', 'ML w basis', 'location', 'northwest');
%set(gca,'xlim', xlim);
title('ML estimates')


% ---  Report results (R^2) ------------------
r2fun = @(w)(1-sum((wtrue_reduced(:)-w(:)).^2)/sum(wtrue_reduced(:).^2));
R2ml1 = r2fun(wML1);
R2ml2 = r2fun(wML2);

fprintf('      ML estimate R^2: %7.3f\n',R2ml1);
fprintf('ML-basis estimate R^2: %7.3f\n',R2ml2);


