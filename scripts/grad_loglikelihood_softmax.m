function grad_Lw = grad_loglikelihood_softmax(all_w,x,y,nstates, weights,...
					      ptr_func,ptr_gradfunc, lambda_w)
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
%   - grad_Lw: Gradient of the log likelihood wrt w
%
% Edwin V. Bonilla


[N D] = size(x);
L = size(y,2); % Number of target variables
Nparam = D + 1; % Number of parameters per class
grad_Lw = zeros(size(all_w));
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
  for i = 1 : Cmax
    W(i,:) = all_w(low_pos(i):high_pos(i)); % extracts param vectors 
  end
  P = softmax_func(ptr_func,W,x); % Computing the probabilities for all clases
 for i = 1 : Cmax
   range = low_pos(i):high_pos(i);
   grad_fw = feval(ptr_gradfunc,all_w(range),x);
   grad_major = weights.*( (classes==i) - P(:,i) );
   
   grad_Lw(range) = sum (repmat(grad_major,1,Nparam).*grad_fw, 1);
   
 end
 
end


%  add penalty term
grad_Lw = grad_Lw - 2*lambda_w*all_w;


 
return;


