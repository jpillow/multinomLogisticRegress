% demo_multinomLogisticRegress.m
%
% Multinomial logistic regression observations (multinomial GLM) with GP
% prior over rows of weight matrix

% if contains(pwd,'/demos')
%     cd ..; setpaths; % move up a directory and call setpaths
% end

% set up weights
nxdim = 10;  % number of input dimensions 
nclass = 5;  % number of output classes
nsamp = 2e3; % number of samples to draw

% Sample weights from a Gaussian
wtrue = randn(nxdim,nclass)/2;

% make inputs 
xinput = randn(nsamp,nxdim); % inputs by time

% Sample class labels from model
[yy,pclass] = sample_multinomGLM(xinput,wtrue);

%  --------- Make plots --------
xtck = 0:25:100;
subplot(231);
imagesc(1:nclass, 1:nxdim, wtrue);
ylabel('input dimension'); xlabel('class');
title('weights');
set(gca,'xtick',xtck);

subplot(232); imagesc(pclass); 
title('p(class)'); ylabel('trial #'); 
xlabel('class');

subplot(233); imagesc(yy); 
title('observed class'); ylabel('trial #');
xlabel('class');

%% 2. Compute ML estimate of weights 

lfun = @(w)(neglogli_multinomGLM_reduced(w,xinput,yy)); % neglogli function handle

% Set initial weights
w0 = randn(nxdim,nclass-1); 

% Check accuracy of Hessian and Gradient (if desired)
HessCheck(lfun,w0(:));

% Set optimization parameters and perform optimization
opts = optimoptions('fminunc','algorithm','trust-region','SpecifyObjectiveGradient',true,...
    'HessianFcn','objective','display','iter');

fprintf('\n-------------------------------------------------------------------------\n');
fprintf('Computing ML estimate of multinomial logistic regression model weights...\n');
fprintf('-------------------------------------------------------------------------\n');

wMLreduced = fminunc(lfun,w0(:),opts); % optimize negative log-posterior

%% 3. Reshape weights and compare to true weights
wML = [zeros(nxdim,1), reshape(wMLreduced,nxdim,nclass-1)]; % reshape into matrix

% normalize each column to have zero mean, just for plotting purposes
wtrue_reduced = wtrue-wtrue(:,1);

%  --------- Make plots --------
xtck = 0:25:100;
subplot(231);
imagesc(1:nclass, 1:nxdim, wtrue_reduced);
ylabel('input dimension'); xlabel('class');
title('true weights');
set(gca,'xtick',xtck);

subplot(234);
imagesc(1:nclass, 1:nxdim, wML);
title('inferred weights'); 


subplot(2,3,5:6);
plot(1:nxdim*nclass, wtrue_reduced(:), 1:nxdim*nclass, wML(:));
legend('true', 'estim', 'location', 'northwest');
title('true and recovered weights')

% Report results:
R2 = 1-sum((wtrue_reduced(:)-wML(:)).^2)/sum(wtrue_reduced(:).^2);
fprintf('\nWeight recovery R^2: %0.3f\n',R2);
