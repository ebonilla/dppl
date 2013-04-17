function [theta, ypred, idx_pred, Ltest, ltest] = ...
    fwd_softmax_class(all_w,nstates,ptr_func, xtest, ytest)
% Makes predictions on a softmax classification model
%
% INPUT:
%   - all_w: vector of all parameters of the softmax model
%       * The format is the followinng:
%       * [w_var1_class1, ..., w_var1_classC(var1), ...,
%          w_varL_class1, ..., w_varL_classC(varL) ]
%    setting from this set
%    corresponds to a multi-classed variabled
%    it asssumes that the target variable y(:,j) comes with values 
%    in the range 1 : nstates(j)
%   - nstates: L-vector with the number of states of each class in N
%    nstates(j) determines the number of classes for target variable j
%   - ptr_func: Pointer to evaluation function of sotfmax 
%   - xtest: NxD features for testing 
%   - ytest (optional): if given, it also returns the most likely 
%
% OUTPUT: 
%   - theta: cell array with corresponding probabilities
%       so that: theta{l} is matrix of P size NxCmax_l, where
%       P(i,j) = P(y_l = s_j | x_i)
%   - ypred: The predicted classes    
%
% Edwin V. Bonilla

[N D] = size(xtest);
L = length(nstates); % Number of target variables
Nparam = D + 1; % Number of parameters per class

low_var = zeros(L,1);
theta   = cell(L,1);

% for each target variable it figures out the point where the
% weight parameters start
low_var(1) = 1;
for j = 2 : L
  Cmax = nstates(j-1); % maximum number of classes for current variable
  low_var(j) = Cmax*(Nparam) + low_var(j-1);
end

% For every dimension of the target variables
ltest = zeros(N,L);
ypred = zeros(N,L);
for j = 1 : L % loop over target variavles
  Cmax = nstates(j); % Maximum number of current class
  low_pos = low_var(j) + ((1:Cmax)-1)*(Nparam); 
  high_pos = low_pos + D; 
  W = zeros(Cmax,Nparam);
  for i = 1 :Cmax
    W(i,:) = all_w(low_pos(i):high_pos(i)); % extracts param vectors 
  end

  
  P = softmax_func(ptr_func,W,xtest); % Computing the probabilities for all clases
  theta{j} = P;

  [foo ypred(:,j)] = max(P, [], 2);
  if (nargin == 5) % accumulates test likelihoods 
    classes = ytest(:,j);
    idx_P = (Cmax)*((1:N)'- 1) + classes;
    ltest(:,j) = P(idx_P);
  end
end

if (nargin == 5) % Necessary to rank likelihoods of test data
  Ltest = sum(ltest,2); % total test likelihoods
  [val_pred idx_pred] = max(Ltest);
  ypred = ytest(idx_pred,:);
end


