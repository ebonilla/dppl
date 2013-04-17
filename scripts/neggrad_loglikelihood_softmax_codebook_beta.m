function neggrad_beta = neggrad_loglikelihood_softmax_codebook_beta(logbeta_x,...
                            all_w, all_x, low, up,  Mu, y, nstates, ...
                            weights, ptr_func, ptr_gradfunc, lambda_w)
% function grad_loglikelihood_softmax(all_w,x,y,nstates,ptr_func,weights)
% Computes the gradient of the loglikelihood of softmax model
% x: NxD features for training
% y: NxL(possible multidimensional) target variables. Each column
% corresponds to a multi-classed variabled
% nstates: L-vector with the number of states of each class in N
% ptr_func: Pointer to evaluation function of sotfmax 
% weights: Nx1 weight values on each data-point
% it asssumes that the target variable y(:,j) comes with values 
% in the range 1 : nstates(j)
% all_w: vector of all parameters of the softmax model
%    - The format is the followinng:
%    - [w_var1_class1, ..., w_var1_classC(var1), ...,
%    w_varL_class1, ..., w_varL_classC(varL) ]
% nstates(j) determines the number of classes for target variable j
%
% Edwin V. Bonilla

                        
if (length(logbeta_x) == 1)
        grad_beta = get_gradient_beta_scalar(logbeta_x,...
                            all_w, all_x, low, up,  Mu, y, nstates, ...
                            weights, ptr_func, ptr_gradfunc, lambda_w);
else
        grad_beta = get_gradient_beta_vector(logbeta_x,...
                            all_w, all_x, low, up,  Mu, y, nstates, ...
                            weights, ptr_func, ptr_gradfunc, lambda_w);    
end



neggrad_beta = - grad_beta;






return;




%% 
function grad_beta = get_gradient_beta_vector(logbeta_x,...
                            all_w, all_x, low, up,  Mu, y, nstates, ...
                            weights, ptr_func, ptr_gradfunc, lambda_w)

beta_x = exp(logbeta_x);                     
D_x = size(all_x,2);
N       = size(y,1);
K  = size(Mu,1);
Nparam  = K + 1; % Number of parameters per class
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


low = low - 1; up  = up  - 1;                                                
grad_beta = get_gradient_codebook_beta_vector_pthread_pool(all_x, low, up, Mu, P, W, N, D_x, K, ...
                beta_x, y, Cmax); 
%grad_beta = get_gradient_codebook_beta_vector_matlab(all_x, low, up, ...
%                    Mu, P, W, N, D_x, K, beta_x, y, Cmax);                        

% derivatives wrt log(beta_x)
grad_beta  = (beta_x) .* grad_beta;

return;



%% function grad_beta = get_gradient_codebook_beta_vector_matlab
function grad_beta = get_gradient_codebook_beta_vector_matlab(all_x, low, up, ...
                    Mu, P, W, N, D_x, K, beta_x, y, Cmax)
grad_beta  = zeros(size(beta_x));
classes    = y;

dzkn_dbetal     = zeros(1, K); 
for n = 1 : N    
    X    = all_x(low(n):up(n),:);
    G    = sq_dist(Mu', X');
        F    = get_softmax_generic(Mu, X, beta_x);
    M = size(X,1);
    c_class = classes(n); % Current class        
    
    for k = 1 : K % for each z_k there is a different gradient        
        for l = 1 : K
            dzkn_dbetal(l) = 0;
            for m = 1 : M % sum over all the x \in S_m
                if (l==k)
                  dzkn_dbetal(l) = dzkn_dbetal(l) + F(k,m)*(F(k,m) - 1)*G(k,m);
                else
                  dzkn_dbetal(l) = dzkn_dbetal(l) + F(k,m)*F(l,m)*G(l,m);
                end
            end
        end
        dzkn_dbetal  = dzkn_dbetal/M; % added on 10/08/2011
        dL_dzkn = W((c_class),k) -  P(n,:)*W(:,k);
        
        grad_beta = grad_beta + dL_dzkn*dzkn_dbetal;
    end
    
fprintf('Neggrad mu:: n= %d\n', n);    
end


  
return;



%% function get_gradient_beta_scalar
function grad_beta = get_gradient_beta_scalar(logbeta_x,...
                            all_w, all_x, low, up,  Mu, y, nstates, ...
                            weights, ptr_func, ptr_gradfunc, lambda_w)

beta_x = exp(logbeta_x);
N       = size(y,1);
D       = size(Mu,1);
L       = size(y,2);  % Number of target variables
Nparam  = D + 1; % Number of parameters per class
low_var = zeros(L,1);

z = get_soft_codebook(all_x, low, up, Mu, beta_x); 


% dz_dbeta
dz_dbeta = zeros(N,D); % dznk_dbeta
for n = 1 : N
    X    = all_x(low(n):up(n),:);
    
    G    = get_softmax_generic_mex(Mu, X, beta_x);
    
    
    dist = sq_dist(Mu', X');
    
    
    dz_dbeta(n,:) = sum(G.*(repmat(sum(dist.*G,1),D,1) - dist),2)';

    % Added on 28/05/2011
    N_x    = size(X,1);
    dz_dbeta(n,:) = dz_dbeta(n,:)/N_x;
    
end

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
  P = softmax_func(ptr_func,W,z); % Computing the probabilities for all clases
 
  
 
  grad_beta = 0;
  for k = 1 : (Nparam-1) % last dimension is the bias 
    for n = 1 : N
        c_class = classes(n); % Current class
        grad_beta = grad_beta + ( W((c_class),k) -  P(n,:)*W(:,k) )*dz_dbeta(n,k);   
        
    end
  end
 
end

% also penalizing beta: 28/05/2011
% grad_beta = grad_beta - 2*LAMBDA*(beta_x);

% derivatives wrt log(beta_x)
grad_beta  = (beta_x) * grad_beta;



return;


















 


