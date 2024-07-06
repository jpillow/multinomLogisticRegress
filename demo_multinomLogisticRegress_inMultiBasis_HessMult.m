% demo_multinomLogisticRegress_inMultiBasis.m
%
% Multinomial logistic regression observations (multinomial GLM) with a
% different basis for each row of the weight matrix, using HessMult instead
% of creating the entire Hessian matrix

clear; clf;
setpaths;  % add necessary paths

% Set up model dimensions
nxdim = 200;  % number of input dimensions (neurons)
nclass = 72;  % number of output classes
nbasis = [25,13,3*ones(1,nxdim-2)];  % number of dimensions in each basis
nbasis = min(nbasis,nclass);  % make sure # basis vectors  < # classes
nsamp = nclass*100; % number of samples to draw
nwts = sum(nbasis);  % total number of basis coefficients
ntotwts = nxdim*nclass; % number number of weights

% Make smooth basis for each class and sample weights
Bases_all = cell(nxdim,1);  % cell array for weights
basiswts_all = cell(nxdim,1);  % weights in basis
wtrue_mat = zeros(nxdim,nclass); % allocate space for true weights
for jj = 1:nxdim
    B1 = gsmooth(randn(nclass,nbasis(jj)),3);  % make a smooth random basis
    B1 = orth(B1)';  % basis for the rows of wts
    Bases_all{jj} = sparse(B1);
    basiswts_all{jj} = 3*randn(nbasis(jj),1);
    wtrue_mat(jj,:) = basiswts_all{jj}'*B1;
end

% We can also constructed these weights with single matrix mult
Bmat = blkdiag(Bases_all{:});  % block-diagonal matrix with 
basiswts_vec = cell2mat(basiswts_all); % vector of all basis weights
wtrue_vec = basiswts_vec'*Bmat;  % vector of all weights

% Massage into a basis for the columns of the weights
P = makeRowColPermMatrix(nclass,nxdim);  % column-row permutation matrix
Bmat_cols = P*Bmat';  % basis matrix for columns of weight matrix


%% Sample inputs and outputs

% Sample inputs from a Gaussian
xinput = randn(nsamp,nxdim); % inputs by time

% Sample class labels from model
[yy,pclass] = sample_multinomGLM(xinput,wtrue_mat);

%  --------- Make plots --------
subplot(331);
imagesc(1:nclass, 1:nxdim, wtrue_mat);
ylabel('input dimension'); xlabel('class');
title('true weights');

subplot(332); imagesc(pclass); 
title('p(class)'); ylabel('trial #'); 
xlabel('class');

subplot(333); imagesc(yy); 
title('observed class'); ylabel('trial #');
xlabel('class');


%% 2. Compute ML estimate of weights in basis

% Set initial weights
w0 = 0.1*randn(nwts,1); % initial basis weights

% Set optimization parameters and perform optimization
opts = optimoptions('fminunc','algorithm','trust-region','SpecifyObjectiveGradient',true,...
    'HessianFcn','objective','display','iter');

% Set loss function
lfun1 = @(wb)(neglogli_multinomGLM_multibasis(wb,xinput,yy,Bmat_cols)); % neglogli function handle w/ basis

fprintf('\n-------------------------------------------------------------------------\n');
fprintf('Computing ML estimate in basis.....\n');
fprintf('-------------------------------------------------------------------------\n');
tic;
wML1basis = fminunc(lfun1,w0,opts); % optimize negative log-posterior in basis
toc;

% multiply by basis to obtain weights in weight space
wML1 = reshape(Bmat_cols*wML1basis,nxdim,nclass); % reconstruct full weights using basis


%% 3. Compute ML estimate of weights using HessMult

% Set loss function
lfun2 = @(w)(neglogli_multinomGLM_multibasis_HessMult(w,xinput,yy,Bmat_cols)); % neglogli function handle (full)

% Set HessMult Function
HessMultFun = @(df,v)neglogli_multinomGLM_multibasis_HessMultFun(df,v,xinput,Bmat_cols);

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
wML2basis = fminunc(lfun2,w0,opts2); % optimize negative log-posterior
toc;

% multiply by basis to obtain weights in weight space
wML2 = reshape(Bmat_cols*wML2basis,nxdim,nclass); % reconstruct full weights using basis


%% 4.Compare fits to true weights

% Convert true weights to reduced form (for plotting purposes only)
wtrue_reduced = wtrue_mat-wtrue_mat(:,1); % substract off class-1 weights

wML1 = reshape(wML1,nxdim,nclass); % reshape full weights into matrix
wML1 = wML1- wML1(:,1); % substract off class-1 weights

wML2 = reshape(wML2,nxdim,nclass); % reshape full weights into matrix
wML2 = wML2- wML2(:,1); % substract off class-1 weights

% %  --------- Make plots --------
nw = nxdim*nclass; % number of total weights
xlim = [nxdim+1, nxdim*2];

subplot(3,1,2:3);
pltinds = 1:nclass;
plot(pltinds, wtrue_reduced(2,:), pltinds, wML1(2,:), pltinds, wML2(2,:),'--');
legend('true', 'ML', 'ML w basis', 'location', 'northwest');
title('ML estimates')


% ---  Report results (R^2) ------------------
r2fun = @(w)(1-sum((wtrue_reduced(:)-w(:)).^2)/sum(wtrue_reduced(:).^2));
R2ml1 = r2fun(wML1);
R2ml2 = r2fun(wML2);

fprintf('      ML estimate R^2: %7.3f\n',R2ml1);
fprintf('ML-basis estimate R^2: %7.3f\n',R2ml2);


