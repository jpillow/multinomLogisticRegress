% demo_multinomLogisticRegress.m
%
% Multinomial logistic regression observations (multinomial GLM) with GP
% prior over rows of weight matrix

% if contains(pwd,'/demos')
%     cd ..; setpaths; % move up a directory and call setpaths
% end

% --- Set weights for multinomial LR model  --------

% set up weights
nX = 10; % number of input dimensions 
nClass = 5; % number of output classes
nSamp = 5e3;

% Sample weights from a Gaussian
wTrue = randn(nX,nClass);

% -- Make inputs and generate class label data -----
xinput = 1*(randn(nSamp,nX)); % inputs by time
xproj = xinput*wTrue; % output of linear weights

% Compute probability of each class
pclass = exp(xproj)./sum(exp(xproj),2);

% Sample outcomes y from multinomial
pcum = cumsum(pclass,2); % cumulative probability
rvals = rand(nSamp,1); % random variables
yout = (rvals<pcum) & (rvals >=[zeros(nSamp,1),pcum(:,1:nClass-1)]);
yout2 = sparse(yout);

%  --------- Make plots --------
xtck = 0:25:100;
subplot(231);
imagesc(1:nClass, 1:nX, wTrue);
ylabel('input dimension'); xlabel('class');
title('weights');
set(gca,'xtick',xtck);

subplot(232); imagesc(pclass); 
title('p(class)'); ylabel('trial #'); 
xlabel('class');

subplot(233); imagesc(yout); 
title('observed class'); ylabel('trial #');
xlabel('class');

%% 2. Compute ML estimate of weights (no GP prior)

lfun = @(w)(neglogli_multinomGLM_reduced(w,xinput,yout)); % neglogli function handle
lfun2 = @(w)(neglogli_multinomGLM_reduced(w,xinput,yout2)); % neglogli function handle

% Set initial weights
w0 = randn(nX,nClass-1); 

% Check accuracy of Hessian and Gradient (if desired)
HessCheck(lfun,w0(:));

% Set optimization parameters and perform optimization
opts = optimoptions('fminunc','algorithm','trust-region','SpecifyObjectiveGradient',true,...
    'HessianFcn','objective','display','iter');

fprintf('\n-------------------------------------------------------------------------\n');
fprintf('Computing ML estimate of multinomial logistic regression model weights...\n');
fprintf('-------------------------------------------------------------------------\n');

tic;
wMLreduced = fminunc(lfun,w0(:),opts); % optimize negative log-posterior
toc;

%% 3. Reshape weights and compare to true weights
wML = [zeros(nX,1), reshape(wMLreduced,nX,nClass-1)]; % reshape into matrix

% normalize each column to have zero mean, just for plotting purposes
wTrueShifted = wTrue-wTrue(:,1);

%  --------- Make plots --------
xtck = 0:25:100;
subplot(231);
imagesc(1:nClass, 1:nX, wTrueShifted);
ylabel('input dimension'); xlabel('class');
title('true weights');
set(gca,'xtick',xtck);

subplot(234);
imagesc(1:nClass, 1:nX, wML);
title('inferred weights'); 


subplot(2,3,5:6);
plot(1:nX*nClass, wTrueShifted(:), 1:nX*nClass, wML(:));
legend('true', 'estim', 'location', 'northwest');
title('true and recovered weights')

% Report results:
R2 = 1-sum((wTrueShifted(:)-wML(:)).^2)/sum(wTrueShifted(:).^2);
fprintf('\nWeight recovery R^2: %0.3f\n',R2);

