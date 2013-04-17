function [Mu, flog] = learn_softmax_class_mu(Mu0, w, all_x, low, up, beta_x, ytrain, ...
                    C, weights, maxiter, tol, lambda_w, verbose)
% learn softmax codebook model wrt the centers mu
% Below  D is the input dimensionality
% Let N be the number of training input, D the dimensionality of 
% input space, K the number of prototypes, 
% and n >> N the total number of training vectors
%
% INPUT:
%   - Mu0:  (KxD) matrix of initial centers
%   - w: (C(D+1)x1) Vector of all initial weights for softmax model 
%   - all_x: (n x D) Matrix of all training vectors stacked together
%   - low: (Nx1) vector of indices indicating where the corresponding
%           input vectors of each training point start
%   - up:  (Nx1) vector of indices indicating where the corresponding
%           input vectors of each training point end
%   - beta_x: initial inverse temperature parameter o prototype function
%   - ytrain: NxL(possible multidimensional) target variables. Each column
%       corresponds to a multi-classed variabled
%   - C: The number of classes 
%   - weights: Nx1 weight values on each data-point
%       it asssumes that the target variable y(:,j) comes with values 
%       in the range 1 : nstates(j)
%   - max_iter: Maximum number of iterations for each sub-problem
%   - tol: Tolerance
%   - lambda_w: regularization parameter on w
%   - verbose: Verbose to show various things
% OUTPUT:
%   - w: vector of all parameters of the softmax model
%       - The format is the followinng:
%       - [w_var1_class1, ..., w_var1_classC(var1), ...,
%       w_varL_class1, ..., w_varL_classC(varL) ]
%
% Edwin V. Bonilla

if (nargin == 12)
    verbose = -1;
end
                
ptr_error = @negloglikelihood_softmax_codebook_mu; 
ptr_grad = @neggrad_loglikelihood_softmax_codebook_mu;


[K D_x]   = size(Mu0);
options = foptions;
options(1) = verbose; % display error values
%options(1) = -1; % display nothing
options(14) = 1000; % Maximum number of iterations
% ****

if ( isempty(weights) )
    weights=ones(size(x,1),1);
end

if (nargin >= 10)
  options(14) = maxiter; 
end

if (nargin >= 11) % puts tolerance
  options(3) = tol;
  options(2) = tol;
end

all_mu0 = Mu0';
all_mu0 = all_mu0(:)';

%[all_mu,options,flog] = conjgrad(ptr_error, all_mu0, options,ptr_grad, beta_x, ...
%                        w, cell_x,  ytrain, C, weights, @linear_map, [],...
%                        K);%

[all_mu, options, flog] = quasinew(ptr_error, all_mu0, options, ptr_grad, beta_x, ...
                        w, all_x, low, up,  ytrain, C, weights, @linear_map, [],...
                        K, lambda_w);

                    
Mu = reshape(all_mu', D_x, K )'; 

                

                    
return;






