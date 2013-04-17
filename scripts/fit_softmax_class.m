function [w, flog] = fit_softmax_class(w0, x, y, nstates, weights, ptr_func, ...
				       ptr_gradfunc, maxiter, tol, lambda_w, verbose)
% Fits a softmax multi-label multi-class model
% Input:
%   - w0: Initial vector of weights
%   - x: NxD features for training
%   - y: NxL(possible multidimensional) target variables. Each column
%       corresponds to a multi-classed variabled
%       it asssumes that the target variable y(:,j) comes with values 
%       in the range 1 : nstates(j)
%   - nstates: L-vector with the number of states of each class in N
%       nstates(j) determines the number of classes for target variable j
%   - weights: Nx1 weight values on each data-point
%   - ptr_func: Pointer to evaluation function of sotfmax 
%   - ptr_gradfunc: Pointer to the gradient of the evaluation function of sotfmax 
%   - maxiter: Maximum number of iterations
%   - tol: Tolerance
%   - lambda_w: Regularization parameter
%   - verbose: flag to show various things
%
% OUTPUT:
%   - w: vector of all parameters of the softmax model
%       * The format is the followinng:
%       * [w_var1_class1, ..., w_var1_classC(var1), ...,
%       w_varL_class1, ..., w_varL_classC(varL) ]
%   - flog: log of negative likelihood values
%
% Edwin V. Bonilla

if (nargin == 9)
    verbose = -1;
end

ptr_error = @negloglikelihood_softmax; 
ptr_grad = @neggrad_loglikelihood_softmax;
D = size(x,2);

% **** Optimization options by Default
% w = randn(1,Nparam)/sqrt(D);
% w = ones(1,Nparam);
 w = w0;
 
options = foptions;
options(1) = verbose; % display error values
%options(1) = -1; % display nothing
options(14) = 1000; % Maximum number of iterations
% ****

if ( isempty(weights) )
    weights=ones(size(x,1),1);
end

if (nargin >= 7)
  options(14) = maxiter; 
end

if (nargin == 8) % puts tolerance
  options(3) = tol;
  options(2) = tol;
end
[w,options,flog] = conjgrad(ptr_error,w,options,ptr_grad,x,y, nstates,...
		       weights,ptr_func, ptr_gradfunc, lambda_w);


return;




  
  