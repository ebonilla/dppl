function [logbeta_x, flog] = learn_softmax_class_beta(logbeta0, w, all_x, low, up, Mu, ytrain, ...
                    C, weights, maxiter, tol, lambda_w, verbose)
% learn softmax codebook model wrt beta
% Below  D is the input dimensionality
% Let N be the number of training input, D the dimensionality of 
% input space, K the number of prototypes, 
% and n >> N the total number of training vectors
%
% INPUT:
%   - logbeta0: initial inverse temperature parameter o prototype function
%   - w: (C(D+1)x1) Vector of all initial weights for softmax model 
%   - all_x: (n x D) Matrix of all training vectors stacked together
%   - low: (Nx1) vector of indices indicating where the corresponding
%           input vectors of each training point start
%   - up:  (Nx1) vector of indices indicating where the corresponding
%           input vectors of each training point end
%   - Mu:  (KxD) matrix of  centers
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
%   - logbeta_x: the learned log beta
%   - flog: value of negative log likelihood function
%
% Edwin V. Bonilla


if (nargin == 12)
    verbose = -1;
end

ptr_error = @negloglikelihood_softmax_codebook_beta; 
ptr_grad = @neggrad_loglikelihood_softmax_codebook_beta;


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

%[logbeta_x,options,flog] = conjgrad(ptr_error, logbeta0, options,ptr_grad, w, ...
%    all_x, low, up, Mu, ytrain, C, weights, @linear_map, []);
%
% Beta optimizer Changed on 10/08/2011
[logbeta_x,options,flog] = quasinew(ptr_error, logbeta0, options,ptr_grad, w, ...
    all_x, low, up, Mu, ytrain, C, weights, @linear_map, [], lambda_w);



return;


                

  


    
