function neggrad_mu = neggrad_loglikelihood_softmax_codebook_mu(all_mu, ...
                beta_x, all_w, all_x, low, up, y, nstates, ...
                weights, ptr_func, ptr_gradfunc, K, lambda_w)
% Computes the negative gradient of the loglikelihood of softmax codebook  model
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
%   - negloglike: The negative gradient of the likelihood wrt Mu
%
% Edwin V. Bonilla

D_x = size(all_x,2);
N       = size(y,1);
Nparam  = K + 1; % Number of parameters per class

Mu = reshape(all_mu', D_x, K)'; 
z = get_soft_codebook(all_x,low, up, Mu, beta_x);


low_var = 1;
Cmax = nstates; % Maximum number of current class
low_pos = low_var + ((1:Cmax)-1)*(Nparam); 
high_pos = low_pos + K; 
W = zeros(Cmax,Nparam);
for i = 1 : Cmax % can replace this with a reshape
    W(i,:) = all_w(low_pos(i):high_pos(i)); % extracts param vectors 
end
P = softmax_func(ptr_func,W,z); % Computing the probabilities for all clases


%get_gradient_codebook_mu_matlab(all_mu, Mu, ...
%                beta_x, all_x, low, up, y, K, W, P);
            

low = low - 1; up  = up  - 1;
%% Single threaded version
%grad_mu = get_gradient_codebook_mu(all_x, low, up, Mu, P, W, N, D_x, K, ...
%                beta_x, y, Cmax);
%% multi-thread without limit in the number of threads
%grad_mu = get_gradient_codebook_mu_pthread(all_x, low, up, Mu, P, W, N, D_x, K, ...
%                beta_x, y, Cmax);
%
%% Multi-thread with limit in the number of thread,  now with a pool of threads

if (length(beta_x)>1)
   grad_mu = get_gradient_codebook_mu_pthread_pool_beta_vector(all_x, low, up, Mu, P, W, N, D_x, K, ...
                beta_x, y, Cmax); 
else
grad_mu = get_gradient_codebook_mu_pthread_pool_beta_scalar(all_x, low, up, Mu, P, W, N, D_x, K, ...
                beta_x, y, Cmax);
end
  
neggrad_mu = - grad_mu;
   


return;  


%% grad_mu = get_gradient_codebook_mu_matlab
function grad_mu = get_gradient_codebook_mu_matlab(all_mu, Mu, ...
                beta_x, all_x, low, up, y, K, W, P)
grad_mu  = zeros(size(all_mu));
classes = y;
D_x = size(all_x,2);
N       = size(y,1);


Lmu     = length(all_mu);
val     = zeros(1, Lmu); 
idx_all = 1 : Lmu; 
for n = 1 : N    
    X    = all_x(low(n):up(n),:);
    G    = get_softmax_generic_mex(Mu, X, beta_x);
    M = size(X,1);
    
    for k = 1 : K % for each z_k there is a different gradient
        val(idx_all) = 0;
        
        for l = 1 : K
            idx_l = (l-1)*D_x+1 : l*D_x;
        
            for m = 1 : M % sum over all the x \in S_m
                if (l==k)
                  val(idx_l) = val(idx_l) + G(k,m)*(1-G(k,m))*(X(m,:)-Mu(k,:));
                else
                  val(idx_l) = val(idx_l) + G(k,m)*G(l,m)*(Mu(l,:)-X(m,:));
                end
            end
            
            
        end
        dz_dmu = 2*beta_x*val;
        dz_dmu  = dz_dmu/M; % added on 10/08/2011
        
        c_class = classes(n); % Current class
        
        dL_dz = W((c_class),k) -  P(n,:)*W(:,k);
        % fprintf(' W((c_class),k)=%.3f, dldz_matlab=%.3f\n', W((c_class),k), dL_dz);
        
        grad_mu = grad_mu + dL_dz*dz_dmu;
    end
    
fprintf('Neggrad mu:: n= %d\n', n);    
end



