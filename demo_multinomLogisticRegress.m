% demo_multinomLogisticRegress.m
%
% Multinomial logistic regression observations (multinomial GLM) with GP
% prior over rows of weight matrix

setpaths; 

% set up weights
nxdim = 10;  % number of input dimensions 
nclass = 7;  % number of output classes
nsamp = 5e3; % number of samples to draw

% Sample weights from a Gaussian
wtrue = randn(nxdim,nclass)/nxdim;

% make inputs 
xinput = randn(nsamp,nxdim); % inputs by time

% Sample class labels from model
[yy,pclass] = sample_multinomGLM(xinput,wtrue);

%  --------- Make plots --------
subplot(231);
imagesc(1:nclass, 1:nxdim, wtrue);
ylabel('input dimension'); xlabel('class');
title('weights');

subplot(232); imagesc(pclass); 
title('p(class)'); ylabel('trial #'); 
xlabel('class');

subplot(233); imagesc(yy); 
title('observed class'); ylabel('trial #');
xlabel('class');

%% 2. Compute ML estimate of weights, reduced parametrization

% Set initial weights
w0 = randn(nxdim,nclass); % full weights
w0red = w0(:,2:end)-w0(:,1); % reduced weights

lfun1 = @(w)(neglogli_multinomGLM_reduced(w,xinput,yy)); % neglogli function handle (reduced)

% Check accuracy of Hessian and Gradient (if desired)
HessCheck(lfun1,w0red(:));

% Set optimization parameters and perform optimization
opts = optimoptions('fminunc','algorithm','trust-region','SpecifyObjectiveGradient',true,...
    'HessianFcn','objective','display','iter');

fprintf('\n-------------------------------------------------------------------------\n');
fprintf('Computing "reduced" ML estimate of multinomial logistic regression weights...\n');
fprintf('-------------------------------------------------------------------------\n');

wML1 = fminunc(lfun1,w0red(:),opts); % optimize negative log-posterior

%% 3. Compute ML estimate of weights, full parametrization

lfun2 = @(w)(neglogli_multinomGLM_full(w,xinput,yy)); % neglogli function handle (full)

% Check accuracy of Hessian and Gradient (if desired)
HessCheck(lfun2,w0(:));

fprintf('\n-------------------------------------------------------------------------\n');
fprintf('Computing "full" ML estimate of multinomial logistic regression weights...\n');
fprintf('-------------------------------------------------------------------------\n');

wML2 = fminunc(lfun2,w0(:),opts); % optimize negative log-posterior

%% 4.Compare fits to true weights

% Convert true weights to reduced form (for plotting purposes only)
wtrue_reduced = wtrue-wtrue(:,1); % substract off class-1 weights

% Reshape fitted weights into matrices
wMLred = [zeros(nxdim,1), reshape(wML1,nxdim,nclass-1)]; % reshape reduced weights into matrix
wMLfull = reshape(wML2,nxdim,nclass); % reshape full weights into matrix
wMLfull = wMLfull - wMLfull(:,1); % substract off class-1 weights


%  --------- Make plots --------

subplot(234);
imagesc(1:nclass, 1:nxdim, wMLfull);
title('ML inferred weights (reduced)'); 

subplot(2,3,5:6);
nw = nxdim*nclass; % number of total weights
plot(1:nw, wtrue_reduced(:), 1:nw, wMLfull(:), 1:nw, wMLred(:),'--');
legend('true', 'ML-reduced', 'ML-full', 'location', 'northwest');
title('true and recovered weights')

% ---  Report results (R^2) ------------------
R2a = 1-sum((wtrue_reduced(:)-wMLfull(:)).^2)/sum(wtrue_reduced(:).^2);
fprintf('reduced weight recovery R^2: %0.3f\n',R2a);
R2b = 1-sum((wtrue_reduced(:)-wMLred(:)).^2)/sum(wtrue_reduced(:).^2);
fprintf('   full weight recovery R^2: %0.3f\n',R2b);
