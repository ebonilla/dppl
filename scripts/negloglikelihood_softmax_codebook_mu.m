function negloglike = negloglikelihood_softmax_codebook_mu(all_mu, ...
                            beta_x, all_w, all_x, low, up, y, nstates, ...
                            weights, ptr_func, ptr_gradfunc, K, lambda_w)
% Computes the negative log likelihood of the softmax codebook model for all_mu
%
% INPUT:
%   - all_mu: stacked vector of parameters of all centers
%   - beta_x: initial inverse temperature parameter o prototype function
%   - all_w: (C(D+1)x1) Vector of all initial weights for softmax model 
%   - all_x: (n x D) Matrix of all training vectors stacked together
%   - low: (Nx1) vector of indices indicating where the corresponding
%           input vectors of each training point start
%   - up:  (Nx1) vector of indices indicating where the corresponding
%           input vectors of each training point end
%   - y: vector of labels
%   - nstates: The number of classes
%   - weights: Nx1 weight values on each data-point
%   - ptr_func: Pointer to evaluation function of sotfmax 
%   - ptr_gradfunc: Pointer to the gradient of the evaluation function of sotfmax 
%   - K: The number of prototypes
%   - lambda_w: Regularization parameter
% OUTPUT:
%   - negloglike: The negative log likelihood
%
% Edwin V. Bonilla

N       = size(y,1);
D_x = size(all_x,2);
L       = size(y,2);  % Number of target variables
Nparam  = K + 1; % Number of parameters per class
logL    = zeros(L,1);
low_var = zeros(L,1);

Mu = reshape(all_mu', D_x, K )'; 
z = get_soft_codebook(all_x, low, up, Mu, beta_x); 

% for each target variable it figures out the point where the
% weight parameters start
low_var(1) = 1;
for j = 2 : L
  Cmax = nstates(j-1); % maximum number of classes for current variable
  low_var(j) = Cmax*(Nparam) + low_var(j-1);
end

% For every dimension of the target variables
for j = 1 : L % loop over target variavles
  Cmax = nstates(j); % Maximum number of current class
  classes = y(:,j);
  low_pos = low_var(j) + ((1:Cmax)-1)*(Nparam); 
  high_pos = low_pos + K; 
  W = zeros(Cmax,Nparam);
  for i = 1 :Cmax
    W(i,:) = all_w(low_pos(i):high_pos(i)); % extracts param vectors 
  end
  P = softmax_func(ptr_func,W,z); % Computing the probabilities for all clases
  P = P'; 
  idx_P = (Cmax)*((1:N)'- 1) + classes;
  % selects the corresponding log probabilites for each data_point
  % And sums over the weighted log-likelihoods
  logL(j)  = sum(weights.*log(P(idx_P)));  
end
loglike = sum(logL);


% We add penalty term that discourages large weight parameters
loglike = loglike - lambda_w*(all_w*all_w');

% also penalizing beta: 28/05/2011
% this may hinder optimization, actually
% loglike = loglike - LAMBDA*(beta_x^2);


negloglike = - loglike;


return;


 