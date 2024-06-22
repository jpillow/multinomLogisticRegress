% demo_multinomLogisticRegress_inMultiBasis.m
%
% Multinomial logistic regression observations (multinomial GLM) with a
% different basis for each rows of the weight matrix

setpaths;  % add necessary paths

% Set up model dimensions
nxdim = 60;  % number of input dimensions (neurons)
nclass = 25;  % number of output classes
nbasis = [5,13,3*ones(1,nxdim-2)];  % number of dimensions in each basis
nbasis = min(nbasis,nclass);  % make sure # basis vectors  < # classes
nsamp = nclass*100; % number of samples to draw
nbasiswts = sum(nbasis);  % total number of basis coefficients
nwts = nxdim*nclass; % number number of weights

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

% % Convert to matrix and check match (if desired)
% wtrue_mat2 = reshape(wtrue_vec,nclass,nxdim)';
% max(max(abs(wtrue_mat-wtrue_mat2)))

% Massage into a basis for the columns of the weights
P = makeRowColPermMatrix(nclass,nxdim);  % column-row permutation matrix
Bmat_cols = P*Bmat';  % basis matrix for columns of weight matrix

% % Check that these match
% wtrue_mat2 = Bmat_cols*basiswts_vec;
% max(abs(wtrue_mat(:)-wtrue_mat2(:)))


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


%% 2. Compute ML estimate of weights, no basis

% Set initial weights
wbasis0 = 0.1*randn(nbasiswts,1); % initial basis weights
w0 = vec(reshape(wbasis0'*Bmat,nclass,nxdim)'); % initial full weights as vector

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
lfun2 = @(wb)(neglogli_multinomGLM_multibasis(wb,xinput,yy,Bmat_cols)); % neglogli function handle w/ basis

% % Check accuracy of Hessian and Gradient (if desired)
HessCheck(lfun2,wbasis0(:));

fprintf('\n-------------------------------------------------------------------------\n');
fprintf('Computing ML estimate in basis.....\n');
fprintf('-------------------------------------------------------------------------\n');

tic;
wML2basis = fminunc(lfun2,wbasis0(:),opts); % optimize negative log-posterior in basis
toc;

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
plot(pltinds, wtrue_reduced(2,:), pltinds, wML1(2,:), pltinds, wML2(2,:));
legend('true', 'ML', 'ML w basis', 'location', 'northwest');
title('ML estimates')


% ---  Report results (R^2) ------------------
r2fun = @(w)(1-sum((wtrue_reduced(:)-w(:)).^2)/sum(wtrue_reduced(:).^2));
R2ml1 = r2fun(wML1);
R2ml2 = r2fun(wML2);

fprintf('      ML estimate R^2: %7.3f\n',R2ml1);
fprintf('ML-basis estimate R^2: %7.3f\n',R2ml2);


