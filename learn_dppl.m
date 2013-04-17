function [Mu, beta_x, w] = learn_dppl(xtrain, ytrain, weights, low, up, C, Mu, beta_x, w, lambda_w, ...
               max_iter, total_iter, tol, verbose)
% Learns Discriminative Probabilistic Prototype Learning (DPPL) model
% of Bonilla and Robles-Kelly (ICML, 2012)
%
% For each training input we have a set of points and its corresponding
% label. 
% Let N be the number of training input, D the dimensionality of 
% input space, K the number of prototypes, 
% and n >> N the total number of training vectors
% 
%
% INPUT:
%   - xtrain: (n x D) Matrix of all training vectors stacked together
%   - ytrain: (N x 1) training labels 
%   - weights:( N x 1) weights on each data-point 
%   - low: (Nx1) vector of indices indicating where the corresponding
%           input vectors of each training point start
%   - up:  (Nx1) vector of indices indicating where the corresponding
%           input vectors of each training point end
%   - C: The number of classes 
%   - Mu:  (KxD) matrix of initial centers
%   - beta_x: initial inverse temperature parameter o prototype function
%   - w: (C(D+1)x1) Vector of all initial weights for softmax model 
%   - lambda_w: regularization parameter on w
%   - max_iter: Maximum number of iterations for each sub-problem
%   - total_iter: Total number of global iterations
%   - tol: Tolerance
%   - verbose: Verbose to show various things
%
% OUTPUT:
%   - Mu: learned matrix of centers
%   - beta_x: learned inverse temperature
%   - w: learned weight vector
%
% Edwin V. Bonilla 

logbeta_x = log(beta_x);
for iter = 1 : total_iter
    

    %% Learning the centers Mu
    fprintf('Learning Mu ... \n');
    [Mu, flog] = learn_softmax_class_mu(Mu, w, xtrain, low, up,  beta_x, ytrain, ...
                        C, weights, max_iter, tol, lambda_w, verbose);    
    fprintf('Learning Mu ... done\n');
       
    %% gets new codebook with learned beta and Mu and re-learn weights
    ztrain_new = get_soft_codebook(xtrain, low, up, Mu, beta_x);
    fprintf('Fitting W with new ztrain ...\n');
    [w_new, flog] = fit_softmax_class(w, ztrain_new, ytrain, C, [], @linear_map, ...
                           @grad_linear_map, max_iter, tol, lambda_w, verbose);  
    fprintf('Fitting W done\n');
    w     = w_new;

        
    %% Learning beta for everything else fixed
    fprintf('Learning Beta with beta0=');
    fprintf('%.2f ',  beta_x);
    fprintf('\n');
    [logbeta_x, flog_beta] = learn_softmax_class_beta(logbeta_x, w, xtrain, low, up,  Mu, ytrain, ...
                        C, weights, max_iter, tol, lambda_w, verbose);
    beta_x = exp(logbeta_x);
    fprintf('Learning Beta done\n');
    
 
    
 
   
end




return;