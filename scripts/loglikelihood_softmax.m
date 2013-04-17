function loglike = loglikelihood_softmax(all_w,x,y,nstates,weights, ...
                    ptr_func, ptr_gradfunc, lambda_w)
% Computes log_likelihood of softmax model for the data given 
% INPUT:
%   - all_w: vector of all parameters of the softmax model
%       The format is the followinng:
%       [w_var1_class1, ..., w_var1_classC(var1), ...,
%        w_varL_class1, ..., w_varL_classC(varL) ]
%   - x: NxD features for training
%   - y: NxL(possible multidimensional) target variables. Each column
%       corresponds to a multi-classed variabled
%   - nstates: L-vector with the number of states of each class in N
%         nstates(j) determines the number of classes for target variable j
%   - weights: Nx1 weight values on each data-point
%   - ptr_func: Pointer to evaluation function of sotfmax 
%       it asssumes that the target variable y(:,j) comes with values 
%       in the range 1 : nstates(j)
%   - ptr_gradfunc: Pointer to gradient of evaluation function within
%           softmax (not used here, but necessary for optimization)
%   - lambda_w: regularizer for w
% OUTPUT:
%   - loglike: The log likelihood value
%
% Edwin V. Bonilla


[N D]  = size(x);
L      = size(y,2);  % Number of target variables
Nparam = D + 1; % Number of parameters per class
logL    = zeros(L,1);
low_var = zeros(L,1);


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
  high_pos = low_pos + D; 
  W = zeros(Cmax,Nparam);
  for i = 1 :Cmax
    W(i,:) = all_w(low_pos(i):high_pos(i)); % extracts param vectors 
  end
  P = softmax_func(ptr_func,W,x); % Computing the probabilities for all clases
  P = P'; 
  idx_P = (Cmax)*((1:N)'- 1) + classes;
  % selects the corresponding log probabilites for each data_point
  % And sums over the weighted log-likelihoods
  logL(j)  = sum(weights.*log(P(idx_P)));  
end
loglike = sum(logL);


% We add penalty term that discourages large weight parameters
loglike = loglike - lambda_w*(all_w*all_w');






return;