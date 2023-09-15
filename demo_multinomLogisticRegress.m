% demo_multinomLogisticRegress.m
%
% Multinomial logistic regression observations (multinomial GLM) with GP
% prior over rows of weight matrix

if contains(pwd,'/demos')
    cd ..; setpaths; % move up a directory and call setpaths
end

% --- Set weights for multinomial LR model  --------

% set up weights
nx = 10; % number of input dimensions 
nclass = 50; % number of output classes
nsamples = 2e4;


% Sample weights from a Gaussian
wTrue = randn(nx,nclass);


% -- Make inputs and generate class label data -----
xinput = 2*(randn(nsamples,nx)); % inputs by time
xproj = xinput*wTrue; % output of linear weights

% Compute probability of each class
pclass = exp(xproj)./sum(exp(xproj),2);

% Sample outcomes y from multinomial
pcum = cumsum(pclass,2); % cumulative probability
rvals = rand(nsamples,1); % random variables
yout = (rvals<pcum) & (rvals >=[zeros(nsamples,1),pcum(:,1:nclass-1)]);

%  --------- Make plots --------
xtck = 0:25:100;
subplot(231);
imagesc(1:nclass, 1:nx, wTrue);
ylabel('input dimension'); xlabel('class');
title('weight matrix');
set(gca,'xtick',xtck);

subplot(232); imagesc(pclass); 
title('p(class)'); ylabel('trial #'); 
xlabel('class');

subplot(233); imagesc(yout); 
title('observed class'); ylabel('trial #');
xlabel('class');

%% 2. Compute ML estimate of weights (no GP prior)

lfun = @(w)(neglogli_multinomGLM(w,xinput,yout)); % neglogli function handle
 
% Set optimization parameters and perform optimization
opts = optimoptions('fminunc','algorithm','trust-region','SpecifyObjectiveGradient',true,...
    'HessianFcn','objective','display','iter');
w0 = randn(nx,nclass-1); % initial weights

fprintf('\n-------------------------------------------------------------------------\n');
fprintf('Computing ML estimate of multinomial logistic regression model weights...\n');
fprintf('-------------------------------------------------------------------------\n');

tic;
wMLreduced = fminunc(lfun,w0(:),opts); % optimize negative log-posterior
toc;

%% 3. Reshape weights and compare to true weights
wML = [zeros(nx,1), reshape(wMLreduced,nx,nclass-1)]; % reshape into matrix

% normalize each column to have zero mean, just for plotting purposes
wTrueShifted = wTrue-wTrue(:,1);

%  --------- Make plots --------
xtck = 0:25:100;
subplot(231);
imagesc(1:nclass, 1:nx, wTrueShifted);
ylabel('input dimension'); xlabel('class');
title('weight matrix');
set(gca,'xtick',xtck);

subplot(234);
imagesc(1:nclass, 1:nx, wML);

subplot(2,3,5:6);
plot(1:nx*nclass, wTrueShifted(:), 1:nx*nclass, wML(:));
legend('true', 'estim', 'location', 'northwest');

% Report results:
R2 = 1-sum((wTrueShifted(:)-wML(:)).^2)/sum(wTrueShifted(:).^2);
fprintf('\nWeight recovery R^2: %0.3f\n',R2);

