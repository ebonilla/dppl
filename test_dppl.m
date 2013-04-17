function test_dppl()
% Tests Discriminative Probabilistic Prototype Learning (DPPL) framework
% of Bonilla and Robles-Kelly (ICML, 2012)
%
% Edwin V. Bonilla
rand('seed', 12);
randn('seed', 24);

%% Add all necessary paths
add_all_path();


%% Parameters specific to synthetic data generation
C            = 2;    % Number of classes
N            = 500;  % Number of observations per class
Ntrain       = N/5;  % Proportion of training data-points
MIN_NSUPPORT = 100;  % Minimum number of support points per data point
MAX_NSUPPORT = 200;  % Maximum number of support points per data point
% Functions that generate toy data
%ptrf_gen_data = @generate_synthetic_data; % single datapoint per entity
ptrf_gen_data = @generate_synthetic_data2; % several data-points per entity

%% General settings
LAMBDA_W     = 0.01; % regularization for softmax (in real settings use xvalidation)
SINGLE_BETA  = 1;    % One Single beta for all "clusters"
K            = 10;   % Number of prototypes
MAX_ITER     = 100;  % Maximum number of iterations for softmax models
TOL          = 1e-3; % Tolerance for softmax models
TOTAL_ITER   = 1;    % Total number of global iterations
VERBOSE      = 1;    % Show progress of various things
%
if (SINGLE_BETA) 
    NBETA = 1; 
else
    NBETA = K;  
end


%% Compile mex files (if not existent)
compile_mex_files('scripts', 'cpp');
compile_mex_files('external/gpml', 'c');

 
%% Generates and preprocess data
[xtrain, xtest, ytrain, ytest, low, up, low_test, up_test] =  ...
    generate_data(ptrf_gen_data, N, Ntrain, C, MIN_NSUPPORT, MAX_NSUPPORT );
weights             = ones(length(ytrain),1); % weights on each datapoint

%% Does usual codebook approack: K-means + quantization
% We can use these centers for initialization
centers = get_centers_kmeans(xtrain, K, MAX_ITER, TOL, [], VERBOSE);
ztrain  = get_hard_codebook(xtrain, low, up, centers);



%% trains softmax and classifying: Fitting W
% We can use these weights to initialize our model
NW = sum(C*(size(ztrain,2)+1)); % number of parameters
w0 = ones(1,NW);
fprintf('Fitting W ...\n');
[w0, flog0_train] = fit_softmax_class(w0, ztrain, ytrain, C, [], @linear_map, ...
				       @grad_linear_map, MAX_ITER, TOL, LAMBDA_W, VERBOSE);  
w = w0;
fprintf('Fitting W done\n');


%% Initialization of parameters
% Mu   = centers;
% Mu   = rand(size(centers));
idx_centers = randperm(size(xtrain,1));
Mu          = xtrain(idx_centers(1:K),:);
BETA0       = rand(1, NBETA);
logbeta_x   = log(BETA0);
beta_x      = exp(logbeta_x);


%% Learns dppl model
[Mu, beta_x, w] = learn_dppl(xtrain, ytrain, weights, low, up, C, Mu, beta_x, w, LAMBDA_W, ...
               MAX_ITER, TOTAL_ITER, TOL, VERBOSE);
           
%% Predictions with learned dppl model          
[theta_pred, ypred, zpred] = predict_dppl(C, Mu, beta_x, w, xtest, low_test, up_test);


%% Performance of dppl model
acc = sum(ypred==ytest)/length(ytest);    % accuracy
%
% log-likelihood on test data --- should not be regularized
weights              = ones(length(ytest),1);
loglike = loglikelihood_softmax(w, zpred, ytest , C , weights, ...
                 @linear_map, @grad_linear_map, 0);
fprintf('\n *********\n');    
fprintf('TestLogL= %.2f\n',  loglike);            
fprintf('Test Accuracy=%.2f\n', acc);
fprintf('*********\n');



return;




%% Generates and preprocess data
function  [xtrain, xtest, ytrain, ytest, low, up, low_test, up_test] =  ...
    generate_data(ptrf_gen_data, N, Ntrain, C, MIN_NSUPPORT, MAX_NSUPPORT )
%
[xtrain ytrain xtest ytest cell_x cell_xtest] = ... 
        feval(ptrf_gen_data, N, Ntrain, C, MIN_NSUPPORT,  MAX_NSUPPORT);
[xtrain mu dev] = normalise(xtrain);
Ntrain          = length(cell_x);
Ntest           = length(cell_xtest);
[low, up]       = get_cell_limit(cell_x);
% converting back to cell after normalization
for i = 1 : Ntrain 
    cell_x{i} = xtrain(low(i):up(i),:);
end
% converting back to cell after normalization
for i = 1 : Ntest 
    cell_xtest{i} = normalise(cell_xtest{i}, mu, dev);
end
xtest = cell2mat(cell_xtest);

[low_test, up_test]       = get_cell_limit(cell_xtest);



return;
 




%% generate_synthetic_data2(N, Ntrain)
function [xtrain ytrain xtest ytest cell_xtrain cell_xtest] = ...
    generate_synthetic_data2(N, Ntrain, C, MIN_NSUPPORT, MAX_NSUPPORT)

for j = 1 : C
    n(j) = ceil(N/C);  % number of observations per class
end
N = sum(n);

S{1} = eye(2); 
S{2} = [1 0.95; 0.95 1];  
S{3}  = [1 0.5; 0.5 1];


Mu = zeros(3,2);
Mu(1,:)  = [0.75, 0]; 
Mu(2,:)  = [-0.75, 0];     
Mu(3,:)  = [3, 0];


if (C==2)
    Mu = Mu(1:2,:);
end

count = 1;
y      = zeros(N,1);
cell_x = cell(N,1);
for j = 1 : C 
    % vsamples  = randi(20,n(j),1); % n(j) integeters in [1 20]
    vsamples     = randi([MIN_NSUPPORT, MAX_NSUPPORT], n(j),1); 
    %vsamples = 50*ones(n(j),1);
    mu = Mu(j,:);
    for i = 1 : n(j)
        nsamples = vsamples(i);
        x = ( chol(S{j})'*randn(2,nsamples)+repmat(mu',1,nsamples) )';
        cell_x{count} = x;
        y(count) = j;
        count = count + 1;
    end
end


% shuffles things around and partitions train/test
idx = randperm(N);
idx_train = idx(1:Ntrain);
idx_test  = idx(Ntrain+1:N);

cell_xtrain = cell_x(idx_train); ytrain = y(idx_train);
cell_xtest  = cell_x(idx_test);  ytest  = y(idx_test);

xtrain  = cell2mat(cell_xtrain);
xtest   = cell2mat(cell_xtest);

return;


%% generate_synthetic_data2, this is restricted to a 2D problem so far
function [xtrain ytrain xtest ytest cell_x cell_xtest] = generate_synthetic_data(N, Ntrain, C)

for j = 1 : C
    n(j)=N/C;  % number of observations per class
end
N = sum(n);


S{1} = eye(2); S{2} = [1 0.95; 0.95 1];  S{3}  = [1 0.5; 0.5 1];
m(:,1) = [0.75; 0]; m(:,2) = [-0.75; 0];     m(:,3) = [3; 0];

x = []; y = [];
for j = 1 : C
    xx{j} = chol(S{j})'*randn(2,n(j))+repmat(m(:,j),1,n(j));
    x = [x; xx{j}'];
    y = [y; repmat(j, n(:,j), 1)];
end


samples_train = cell(C,1);
samples_test  = cell(C,1);
for i = 1 : C
    idx = find(y==i); % indices of current class
    L = length(idx);
    v = randperm(L);
    samples_train{i} = idx(v(1:Ntrain));
    samples_test{i}  = idx(v(Ntrain+1:L));
 end
idx_train = cell2mat(samples_train(:));
idx_test  = cell2mat(samples_test(:));
xtrain    = x(idx_train,:);
xtest     = x(idx_test,:);
ytrain    = y(idx_train,:);
ytest     = y(idx_test, :);


cell_x                      =  mat2cell(xtrain, repmat(1, size(xtrain,1),1));
cell_xtest                  = mat2cell(xtest, repmat(1, size(xtest,1),1));

return;








